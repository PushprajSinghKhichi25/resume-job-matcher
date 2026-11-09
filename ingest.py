"""
Job ingestion CLI.

Reads a JSON file of job postings, cleans the text, extracts metadata,
embeds each description with VoyageAI, and upserts everything into ChromaDB.

Usage
─────
  python ingest.py                           # uses sample_jobs.json
  python ingest.py --file my_jobs.json
  python ingest.py --file my_jobs.json --reset   # wipe collection first

JSON schema (each object in the array)
──────────────────────────────────────
  {
    "id":          "unique-string",          # required
    "title":       "Software Engineer",      # required
    "company":     "Acme Corp",              # required
    "location":    "Remote",
    "description": "<p>HTML or plain text</p>",  # required
    "url":         "https://jobs.example.com/123"
  }
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import chromadb
import numpy as np
import voyageai
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env")
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    CHROMA_COLLECTION,
    CHROMA_DIR,
    EMBEDDING_DIM,
    HNSW_EF_CONSTRUCTION,
    HNSW_EF,
    VOYAGE_API_KEY,
    VOYAGE_MODEL,
)
from preprocess import clean_html, extract_degree, extract_seniority, extract_min_years

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

BATCH_SIZE = 64   # VoyageAI max batch size for document embedding


# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_documents(texts: list[str], voyage_client: voyageai.Client) -> list[list[float]]:
    """
    Embed a list of texts in batches using VoyageAI.

    Uses input_type="document" (optimised for indexing, not querying).
    Retries once on transient API failures.
    """
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        for attempt in range(2):
            try:
                result = voyage_client.embed(batch, model=VOYAGE_MODEL, input_type="document")
                all_embeddings.extend(result.embeddings)
                break
            except Exception as e:
                if attempt == 0:
                    logger.warning("Embedding batch %d failed (%s) — retrying…", i // BATCH_SIZE, e)
                    time.sleep(2)
                else:
                    raise

        logger.info("Embedded %d / %d descriptions", min(i + BATCH_SIZE, len(texts)), len(texts))

    return all_embeddings


# ── Ingestion ─────────────────────────────────────────────────────────────────

def ingest(jobs_path: str, reset: bool = False) -> None:
    """
    Load jobs from a JSON file and upsert them into ChromaDB.

    Args:
        jobs_path: Path to the JSON file containing job objects.
        reset:     If True, delete the collection and rebuild from scratch.
    """
    # Validate API key
    if not VOYAGE_API_KEY:
        logger.error("VOYAGE_API_KEY is not set in .env — cannot embed jobs")
        sys.exit(1)

    # Load jobs file
    jobs_file = Path(jobs_path)
    if not jobs_file.exists():
        logger.error("Jobs file not found: %s", jobs_path)
        sys.exit(1)

    with open(jobs_file) as f:
        raw_jobs: list[dict] = json.load(f)

    logger.info("Loaded %d jobs from %s", len(raw_jobs), jobs_path)

    # Validate required fields
    valid_jobs = []
    for job in raw_jobs:
        if not all(k in job for k in ("id", "title", "company", "description")):
            logger.warning("Skipping job missing required fields: %s", job.get("id", "<no id>"))
            continue
        valid_jobs.append(job)

    logger.info("%d jobs passed validation", len(valid_jobs))

    if not valid_jobs:
        logger.error("No valid jobs to ingest — check your JSON file")
        sys.exit(1)

    # Preprocess descriptions
    logger.info("Preprocessing job descriptions…")
    cleaned: list[str] = [clean_html(j["description"]) for j in valid_jobs]

    # Extract metadata
    metadatas: list[dict] = []
    for job, desc in zip(valid_jobs, cleaned):
        metadatas.append({
            "title":               job.get("title", ""),
            "company":             job.get("company", ""),
            "location":            job.get("location", ""),
            "url":                 job.get("url", ""),
            "required_degree":     extract_degree(desc),
            "seniority_level":     extract_seniority(desc, title=job.get("title", "")),
            "min_years_experience":extract_min_years(desc),
        })

    # Embed
    logger.info("Embedding %d job descriptions via VoyageAI…", len(cleaned))
    voyage_client = voyageai.Client(api_key=VOYAGE_API_KEY)
    embeddings = embed_documents(cleaned, voyage_client)

    # Upsert into ChromaDB
    Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

    if reset:
        try:
            chroma_client.delete_collection(name=CHROMA_COLLECTION)
            logger.info("Deleted existing collection '%s'", CHROMA_COLLECTION)
        except Exception:
            pass

    collection = chroma_client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={
            "hnsw:space":           "cosine",
            "hnsw:construction_ef": HNSW_EF_CONSTRUCTION,
            "hnsw:search_ef":       HNSW_EF,
        },
    )

    collection.upsert(
        ids        =[j["id"] for j in valid_jobs],
        embeddings =embeddings,
        documents  =cleaned,
        metadatas  =metadatas,
    )

    logger.info(
        "Upserted %d jobs into ChromaDB collection '%s' (total: %d)",
        len(valid_jobs),
        CHROMA_COLLECTION,
        collection.count(),
    )
    logger.info("Ingestion complete ✓")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest jobs into ChromaDB")
    parser.add_argument(
        "--file",
        default=str(Path(__file__).parent / "sample_jobs.json"),
        help="Path to JSON file of jobs (default: sample_jobs.json)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Wipe the ChromaDB collection before ingesting",
    )
    args = parser.parse_args()
    ingest(args.file, reset=args.reset)
