"""
Job ingestion CLI.

Reads job postings from a JSON file OR fetches them live from Greenhouse / Lever,
cleans the text, extracts metadata, embeds each description with VoyageAI,
and upserts everything into ChromaDB.

Usage
─────
  # Ingest from JSON file:
  python ingest.py                             # uses real_jobs.json if exists, else sample_jobs.json
  python ingest.py --file real_jobs.json
  python ingest.py --file real_jobs.json --reset   # wipe collection first

  # Fetch live from job boards and ingest directly:
  python ingest.py --fetch greenhouse --companies cloudflare,figma --reset
  python ingest.py --fetch lever --companies spotify,palantir
  python ingest.py --fetch all --limit 40 --reset
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
from fetchers import (
    GreenhouseFetcher,
    LeverFetcher,
    RemotiveFetcher,
    fetch_all_jobs,
)

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


# ── Ingestion Core ────────────────────────────────────────────────────────────

def ingest_jobs_list(raw_jobs: list[dict], reset: bool = False) -> int:
    """
    Preprocess, embed, and upsert a list of job dictionaries into ChromaDB.

    Args:
        raw_jobs: List of dicts with id, title, company, description, etc.
        reset: If True, deletes existing collection before indexing.

    Returns:
        Number of jobs successfully indexed.
    """
    if not VOYAGE_API_KEY:
        logger.error("VOYAGE_API_KEY is not set in .env — cannot embed jobs")
        raise RuntimeError("VOYAGE_API_KEY is not set in .env")

    # Validate required fields
    valid_jobs = []
    for job in raw_jobs:
        if not all(k in job for k in ("id", "title", "company", "description")):
            logger.warning("Skipping job missing required fields: %s", job.get("id", "<no id>"))
            continue
        valid_jobs.append(job)

    logger.info("%d / %d jobs passed validation", len(valid_jobs), len(raw_jobs))

    if not valid_jobs:
        logger.error("No valid jobs to ingest")
        return 0

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
            "source":              job.get("source", "sample"),
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

    total_count = collection.count()
    logger.info(
        "Upserted %d jobs into ChromaDB collection '%s' (total in collection: %d)",
        len(valid_jobs),
        CHROMA_COLLECTION,
        total_count,
    )
    logger.info("Ingestion complete ✓")
    return len(valid_jobs)


def ingest(jobs_path: str, reset: bool = False) -> None:
    """
    Load jobs from a JSON file and upsert them into ChromaDB.
    """
    jobs_file = Path(jobs_path)
    if not jobs_file.exists():
        logger.error("Jobs file not found: %s", jobs_path)
        sys.exit(1)

    with open(jobs_file, "r", encoding="utf-8") as f:
        raw_jobs: list[dict] = json.load(f)

    logger.info("Loaded %d jobs from %s", len(raw_jobs), jobs_path)
    ingest_jobs_list(raw_jobs, reset=reset)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    default_file = "real_jobs.json" if Path("real_jobs.json").exists() else "sample_jobs.json"

    parser = argparse.ArgumentParser(description="Ingest jobs into ChromaDB")
    parser.add_argument(
        "--file",
        default=str(Path(__file__).parent / default_file),
        help=f"Path to JSON file of jobs (default: {default_file})",
    )
    parser.add_argument(
        "--fetch",
        choices=["greenhouse", "lever", "remotive", "all"],
        default=None,
        help="Fetch real jobs live from a job portal and ingest directly",
    )
    parser.add_argument(
        "--companies",
        type=str,
        default=None,
        help="Comma-separated company slugs/boards when using --fetch (e.g. cloudflare,figma,spotify)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=40,
        help="Max jobs to fetch when using --fetch (default: 40)",
    )
    parser.add_argument(
        "--save-fetched",
        type=str,
        default=None,
        help="Optional path to save fetched jobs to a JSON file",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Wipe the ChromaDB collection before ingesting",
    )

    args = parser.parse_args()

    if args.fetch:
        company_list = [c.strip() for c in args.companies.split(",")] if args.companies else None
        logger.info("Fetching real jobs [source=%s, companies=%s, limit=%d]...", args.fetch, company_list, args.limit)

        jobs: list[dict] = []
        if args.fetch == "greenhouse":
            gh = GreenhouseFetcher()
            jobs = gh.fetch_jobs(boards=company_list, limit=args.limit)
        elif args.fetch == "lever":
            lever = LeverFetcher()
            jobs = lever.fetch_jobs(companies=company_list, limit=args.limit)
        elif args.fetch == "remotive":
            remotive = RemotiveFetcher()
            jobs = remotive.fetch_jobs(limit=args.limit)
        elif args.fetch == "all":
            half = max(1, args.limit // 2)
            gh_boards = company_list[:len(company_list)//2] if company_list else None
            lever_comps = company_list[len(company_list)//2:] if company_list else None
            jobs = fetch_all_jobs(limit_per_source=half, greenhouse_boards=gh_boards, lever_companies=lever_comps)

        if not jobs:
            logger.error("No jobs fetched — aborting ingestion")
            sys.exit(1)

        if args.save_fetched:
            with open(args.save_fetched, "w", encoding="utf-8") as f:
                json.dump(jobs, f, indent=2, ensure_ascii=False)
            logger.info("Saved %d fetched jobs to %s", len(jobs), args.save_fetched)

        ingest_jobs_list(jobs, reset=args.reset)
    else:
        ingest(args.file, reset=args.reset)
