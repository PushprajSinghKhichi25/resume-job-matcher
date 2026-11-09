"""
Central configuration — all constants and env-var loading live here.
Every other module imports from this file; nothing reads os.environ directly.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root (the folder this file lives in)
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# ── API Keys ────────────────────────────────────────────────────────────────
VOYAGE_API_KEY: str = os.environ.get("VOYAGE_API_KEY", "")
COHERE_API_KEY: str = os.environ.get("COHERE_API_KEY", "")

# ── Embedding ────────────────────────────────────────────────────────────────
VOYAGE_MODEL: str = "voyage-3.5-lite"
EMBEDDING_DIM: int = 1024

# ── ChromaDB ─────────────────────────────────────────────────────────────────
# Persistent storage directory for the vector index
CHROMA_DIR: str = str(Path(__file__).parent / "data" / "chroma")
CHROMA_COLLECTION: str = "jobs"

# HNSW index quality params
#   ef_construction — higher = better index quality, slower build (default 100)
#   ef              — higher = better recall at query time, slightly slower
HNSW_EF_CONSTRUCTION: int = 100
HNSW_EF: int = 200

# ── Retrieval & Reranking ────────────────────────────────────────────────────
RETRIEVE_TOP_K: int = 50   # Candidates fetched from ChromaDB (dense search)
RERANK_TOP_N: int = 10     # Final results returned after cross-encoder reranking
COHERE_RERANK_MODEL: str = "rerank-english-v3.0"

# ── LLM (local via Ollama) ───────────────────────────────────────────────────
OLLAMA_BASE_URL: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = "llama3.2:3b-instruct-q4_K_M"

# ── Metadata filter sentinel values ─────────────────────────────────────────
# When a filter is "unknown" / not set, we store these sentinels so ChromaDB
# where-filters can safely skip that field.
DEGREE_UNKNOWN: int = 0
SENIORITY_UNKNOWN: int = 0
YEARS_UNKNOWN: int = -1

# Human-readable labels used in the UI
SENIORITY_LABELS: dict[int, str] = {
    0: "Any",
    1: "Entry-level",
    2: "Mid-level",
    3: "Senior",
}
DEGREE_LABELS: dict[int, str] = {
    0: "Any",
    1: "Bachelor's",
    2: "Master's",
    3: "PhD",
}
