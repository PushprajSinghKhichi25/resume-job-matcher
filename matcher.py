"""
Core RAG matching pipeline.

Stage 1  — Embed Resume     : VoyageAI dense vector (1024-dim)
Stage 2  — Dense Retrieval  : ChromaDB HNSW approximate nearest-neighbor (top-K)
Stage 3  — Cross-Encoder Reranking : Cohere rerank-english-v3.0 (top-N)
Stage 4  — Grounded Explanation   : Ollama LLaMA 3.2 3B (local, anti-hallucination)
"""

from __future__ import annotations

import logging
import re
from typing import TypedDict

import cohere
import numpy as np
import requests
import voyageai
import chromadb

from config import (
    CHROMA_COLLECTION,
    CHROMA_DIR,
    COHERE_API_KEY,
    COHERE_RERANK_MODEL,
    DEGREE_UNKNOWN,
    EMBEDDING_DIM,
    HNSW_EF,
    HNSW_EF_CONSTRUCTION,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    RERANK_TOP_N,
    RETRIEVE_TOP_K,
    SENIORITY_UNKNOWN,
    VOYAGE_API_KEY,
    VOYAGE_MODEL,
    YEARS_UNKNOWN,
)
from preprocess import build_where_filter

logger = logging.getLogger(__name__)


# ── TypedDicts ────────────────────────────────────────────────────────────────

class JobMatch(TypedDict):
    """One ranked result returned to the caller."""
    job_id: str
    title: str
    company: str
    location: str
    score: float          # Cohere relevance score (higher = better)
    explanation: str      # LLM-generated fit explanation
    url: str              # Link to the original posting


# ── Client helpers ────────────────────────────────────────────────────────────

def _voyage_client() -> voyageai.Client:
    if not VOYAGE_API_KEY:
        raise RuntimeError("VOYAGE_API_KEY is not set in .env")
    return voyageai.Client(api_key=VOYAGE_API_KEY)


def _cohere_client() -> cohere.ClientV2:
    if not COHERE_API_KEY:
        raise RuntimeError("COHERE_API_KEY is not set in .env")
    return cohere.ClientV2(api_key=COHERE_API_KEY)


def _chroma_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={
            "hnsw:space": "cosine",
            "hnsw:construction_ef": HNSW_EF_CONSTRUCTION,
            "hnsw:search_ef": HNSW_EF,
        },
    )


# ── Stage 1: Embed Resume ─────────────────────────────────────────────────────

def embed_resume(resume_text: str) -> np.ndarray:
    """
    Convert resume text into a 1024-dim float32 embedding via VoyageAI.

    Uses input_type="query" so the embedding is optimised for retrieval
    (versus "document" which is used when indexing job descriptions).
    """
    client = _voyage_client()
    result = client.embed([resume_text], model=VOYAGE_MODEL, input_type="query")
    vec = np.array(result.embeddings[0], dtype=np.float32)
    if vec.shape[0] != EMBEDDING_DIM:
        raise ValueError(f"Unexpected embedding dim {vec.shape[0]}, expected {EMBEDDING_DIM}")
    return vec


# ── Stage 2: Dense Retrieval (HNSW in ChromaDB) ───────────────────────────────

def dense_retrieve(
    query_embedding: np.ndarray,
    top_k: int = RETRIEVE_TOP_K,
    degree: int = DEGREE_UNKNOWN,
    seniority: int = SENIORITY_UNKNOWN,
    min_years: int = YEARS_UNKNOWN,
) -> list[dict]:
    """
    Query ChromaDB HNSW index and return top-K candidate jobs.

    Metadata filters (degree, seniority, years) are applied *inside* ChromaDB
    so only matching jobs are scored against the query vector.

    Returns a list of dicts with keys: id, title, company, location,
    description, url, distance, required_degree, seniority_level, min_years_experience.
    """
    collection = _chroma_collection()

    if collection.count() == 0:
        logger.warning("ChromaDB collection is empty — run ingest.py first")
        return []

    where = build_where_filter(degree, seniority, min_years)

    query_kwargs: dict = {
        "query_embeddings": [query_embedding.tolist()],
        "n_results": min(top_k, collection.count()),
        "include": ["metadatas", "distances", "documents"],
    }
    if where:
        query_kwargs["where"] = where

    result = collection.query(**query_kwargs)

    candidates: list[dict] = []
    ids       = result["ids"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]
    documents = result["documents"][0]

    for job_id, meta, dist, doc in zip(ids, metadatas, distances, documents):
        candidates.append({
            "id":                  job_id,
            "title":               meta.get("title", ""),
            "company":             meta.get("company", ""),
            "location":            meta.get("location", ""),
            "description":         doc,
            "url":                 meta.get("url", ""),
            "distance":            dist,
            "required_degree":     meta.get("required_degree", 0),
            "seniority_level":     meta.get("seniority_level", 0),
            "min_years_experience":meta.get("min_years_experience", -1),
        })

    logger.info("Dense retrieval returned %d candidates", len(candidates))
    return candidates


# ── Stage 3: Cross-Encoder Reranking ─────────────────────────────────────────

def rerank(resume_text: str, candidates: list[dict], top_n: int = RERANK_TOP_N) -> list[dict]:
    """
    Re-score candidates using Cohere's cross-encoder.

    The cross-encoder jointly reads (resume, job_description) together,
    unlike the bi-encoder used for retrieval which scores each in isolation.
    This catches subtle mismatches that cosine similarity misses.

    Returns candidates sorted by Cohere relevance score (highest first),
    trimmed to top_n, with a "rerank_score" field added.
    """
    if not candidates:
        return []

    client = _cohere_client()

    # Format each job for the cross-encoder
    documents = [
        f"{c['title']} | {c['company']} | {c['location']}\n{c['description']}"
        for c in candidates
    ]

    response = client.rerank(
        model=COHERE_RERANK_MODEL,
        query=resume_text,
        documents=documents,
        top_n=top_n,
    )

    reranked: list[dict] = []
    for result in response.results:
        job = dict(candidates[result.index])
        job["rerank_score"] = result.relevance_score
        reranked.append(job)

    logger.info("Reranking produced %d final matches", len(reranked))
    return reranked


# ── Stage 4: Grounded LLM Explanation ────────────────────────────────────────

_SKILL_EXTRACT_PROMPT = """\
You are a job requirements extractor.

Read the job description below and return a JSON array of the 5 most important required skills or qualifications (short phrases, e.g. "Python", "5 years backend experience", "REST API design").

Job description:
{description}

Return ONLY the JSON array. No explanations. Example:
["Python", "REST APIs", "Docker", "3+ years experience", "SQL"]
"""

_MATCH_EXPLAIN_PROMPT = """\
You are a career advisor. Explain in 2-3 sentences why the candidate is a good match for the job.
Only mention skills that EXPLICITLY appear in both the resume and the job description.

Verified matching skills: {skills}

Be concise and specific. Start with "Your".
"""

_SKILL_RE = re.compile(r'\[.*?\]', re.DOTALL)


def _ollama_chat(prompt: str, temperature: float = 0.3) -> str:
    """Call the local Ollama REST API and return the model's text response."""
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model":  OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": 300},
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json().get("response", "").strip()


def _extract_skills(description: str) -> list[str]:
    """Ask the LLM to extract key skills from a job description."""
    try:
        raw = _ollama_chat(_SKILL_EXTRACT_PROMPT.format(description=description[:3000]))
        m = _SKILL_RE.search(raw)
        if m:
            import json
            skills = json.loads(m.group())
            return [s for s in skills if isinstance(s, str)]
    except Exception as e:
        logger.warning("Skill extraction failed: %s", e)
    return []


def _verify_skills(skills: list[str], resume: str, description: str) -> list[str]:
    """
    Anti-hallucination guard: keep only skills that appear verbatim (case-insensitive)
    in BOTH the resume and the job description.
    """
    resume_lower = resume.lower()
    desc_lower   = description.lower()
    return [s for s in skills if s.lower() in resume_lower and s.lower() in desc_lower]


def generate_explanation(resume: str, job: dict) -> str:
    """
    Generate a grounded explanation of why the resume matches this job.

    1. Extract required skills from the job description (LLM)
    2. Verify skills appear in both resume and description (substring check)
    3. Generate explanation using ONLY verified skills (LLM)
    4. Fall back to a simpler prompt if the pipeline fails
    """
    try:
        raw_skills    = _extract_skills(job["description"])
        verified      = _verify_skills(raw_skills, resume, job["description"])

        if verified:
            skills_str = ", ".join(verified)
            explanation = _ollama_chat(
                _MATCH_EXPLAIN_PROMPT.format(skills=skills_str),
                temperature=0.5,
            )
            return explanation

        # No verified overlapping skills found — fall back to direct explanation
        logger.debug("No verified skills for job %s — using direct fallback", job["id"])

    except Exception as e:
        logger.warning("Generation pipeline error for job %s: %s", job["id"], e)

    # ── Fallback: direct prompt without skill verification ──
    try:
        fallback_prompt = (
            f"You are a career advisor. In 2-3 sentences explain why this resume is a "
            f"good match for the job '{job['title']}' at {job['company']}.\n\n"
            f"Resume summary: {resume[:400]}\n\n"
            f"Job highlights: {job['description'][:400]}\n\n"
            f"Start with 'Your'."
        )
        return _ollama_chat(fallback_prompt, temperature=0.5)
    except Exception as e:
        logger.error("Fallback explanation also failed: %s", e)
        return "Could not generate explanation (Ollama unavailable)."


# ── Public API — single entry point ──────────────────────────────────────────

def match_resume(
    resume: str,
    top_k: int = RETRIEVE_TOP_K,
    top_n: int = RERANK_TOP_N,
    degree: int = DEGREE_UNKNOWN,
    seniority: int = SENIORITY_UNKNOWN,
    min_years: int = YEARS_UNKNOWN,
) -> list[JobMatch]:
    """
    End-to-end resume matching pipeline.

    Args:
        resume:    Raw resume text (plain text or extracted from PDF).
        top_k:     Candidates fetched from ChromaDB (dense retrieval pool).
        top_n:     Final results after cross-encoder reranking.
        degree:    Required degree filter (0=any, 1=bachelor, 2=master, 3=phd).
        seniority: Seniority filter (0=any, 1=entry, 2=mid, 3=senior).
        min_years: Minimum years of experience the candidate has (-1=skip filter).

    Returns:
        List of JobMatch TypedDicts, ranked by Cohere relevance score.
    """
    if len(resume.strip()) < 50:
        raise ValueError("Resume is too short (minimum 50 characters).")

    # Stage 1 — Embed
    logger.info("Stage 1: Embedding resume (%d chars)", len(resume))
    query_vec = embed_resume(resume)

    # Stage 2 — Dense Retrieve
    logger.info("Stage 2: Dense retrieval (top-%d, filters: degree=%d seniority=%d years=%d)",
                top_k, degree, seniority, min_years)
    candidates = dense_retrieve(query_vec, top_k=top_k, degree=degree,
                                seniority=seniority, min_years=min_years)

    if not candidates:
        logger.warning("No candidates found — returning empty results")
        return []

    # Stage 3 — Rerank
    logger.info("Stage 3: Cross-encoder reranking (top-%d)", top_n)
    reranked = rerank(resume, candidates, top_n=top_n)

    # Stage 4 — Explain
    logger.info("Stage 4: Generating grounded explanations for %d jobs", len(reranked))
    matches: list[JobMatch] = []
    for job in reranked:
        explanation = generate_explanation(resume, job)
        matches.append(JobMatch(
            job_id      = job["id"],
            title       = job["title"],
            company     = job["company"],
            location    = job["location"],
            score       = round(job["rerank_score"], 4),
            explanation = explanation,
            url         = job["url"],
        ))

    return matches
