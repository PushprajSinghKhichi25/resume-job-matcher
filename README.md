# Resume–Job Matcher

A production-grade RAG-based job recommendation system with live integration for real job openings from **Greenhouse** and **Lever** (plus Remotive).

## How It Works

```
Your Resume
   │
   ├─► 1. Embed with VoyageAI (voyage-3.5-lite, 1024-dim)
   │
   ├─► 2. Dense Retrieval from ChromaDB (HNSW, top-50)
   │       └── Optional metadata filters: degree, seniority, years
   │
   ├─► 3. Cross-Encoder Reranking via Cohere (rerank-english-v3.0, top-10)
   │       └── Jointly reads (resume, job) together for contextual scoring
   │
   └─► 4. Grounded Explanation via Ollama (LLaMA 3.2 3B, local)
           └── Extracts skills → verifies against both texts → explains match
```

## Project Structure

```
resume-job-matcher/
├── config.py            # All constants and env-var loading
├── preprocess.py        # HTML cleaning + regex metadata extraction
├── matcher.py           # Core RAG pipeline (retrieval, rerank, explain)
├── fetch_jobs.py        # CLI: fetch real job openings from Greenhouse & Lever
├── ingest.py            # CLI: load jobs from JSON or live boards into ChromaDB
├── app.py               # FastAPI server + interactive HTML UI with live fetcher
├── real_jobs.json       # 20 pre-fetched real jobs from Greenhouse & Lever
├── sample_jobs.json     # 10 synthetic jobs
├── fetchers/            # Modular job board integrations
│   ├── __init__.py      # Fetcher registry & multi-source fetch helper
│   ├── base.py          # BaseJobFetcher abstract class + HTTP utilities
│   ├── greenhouse.py    # Greenhouse ATS Job Board API integration
│   ├── lever.py         # Lever ATS Postings API integration
│   └── remotive.py      # Remotive public tech job aggregator
└── requirements.txt
```

---

## Real Job Portals Integration

The system natively connects to the public, unauthenticated APIs of **Greenhouse** and **Lever**:

### 1. Greenhouse ATS
Fetches active job openings from companies hosted on Greenhouse (`boards-api.greenhouse.io`):
- **Featured Default Boards**: Cloudflare, Figma, GitLab, Stripe, Airbnb, Databricks, DoorDash, etc.
- **Custom Boards**: Any company token can be passed (e.g. `cloudflare`, `figma`, `gitlab`).

### 2. Lever ATS
Fetches active job openings from companies hosted on Lever (`api.lever.co`):
- **Featured Default Companies**: Spotify, Palantir, Wealthfront, etc.
- **Custom Companies**: Any company slug on Lever can be queried.

### 3. Remotive (Bonus Multi-Company Aggregator)
Fetches live remote tech jobs across hundreds of companies without needing company tokens.

---

## Quickstart

### 1. Install dependencies

```bash
cd ~/Downloads/resume-job-matcher
pip install -r requirements.txt
```

### 2. Set API keys

```bash
cp .env.example .env
# Edit .env and fill in VOYAGE_API_KEY and COHERE_API_KEY
```

### 3. Pull Ollama model (for explanations)

```bash
ollama pull llama3.2:3b-instruct-q4_K_M
```

### 4. Fetch Real Jobs & Ingest

You have multiple flexible ways to fetch and index real jobs:

#### Option A: Fetch & Ingest in One Step
```bash
# Fetch from all default boards (Greenhouse + Lever) and ingest directly:
python ingest.py --fetch all --limit 40 --reset

# Fetch from Greenhouse for Cloudflare and Figma:
python ingest.py --fetch greenhouse --companies cloudflare,figma --reset

# Fetch from Lever for Spotify and Palantir:
python ingest.py --fetch lever --companies spotify,palantir --reset
```

#### Option B: Fetch to JSON File, then Ingest
```bash
# Fetch and save real jobs:
python fetch_jobs.py --source all --limit 50 --output real_jobs.json --print-sample

# Ingest the saved file into ChromaDB:
python ingest.py --file real_jobs.json --reset
```

#### Option C: Use Pre-fetched Real Jobs
The repository includes `real_jobs.json` with 20 live jobs:
```bash
python ingest.py --file real_jobs.json --reset
```

### 5. Start the Web Server & UI

```bash
uvicorn app:app --reload --port 8000
```

Then open **http://localhost:8000** in your browser.

---

## Web UI Features

1. **Live Job Ingestion Panel**:
   - Quick one-click tags for **Cloudflare**, **Figma**, **GitLab**, **Stripe**, **Spotify**, **Palantir**, **Wealthfront**, and **All Remote**.
   - Custom company input to query any Greenhouse or Lever board on the fly.
   - Checkbox to auto-embed and index new openings straight into ChromaDB.
2. **Real-time Matcher**:
   - Paste any candidate resume.
   - Filters for Seniority, Minimum Degree, and Candidate Experience Years.
   - Displays Cohere cross-encoder match score, grounded explanation, and direct link to apply on the official career portal (`Greenhouse` / `Lever`).

---

## API Reference

### `POST /match` — Match Resume Against Jobs

```json
{
  "resume":    "string (min 50 chars)",
  "top_k":     50,          // retrieval pool size (default 50)
  "top_n":     10,          // final results (default 10)
  "degree":    0,           // 0=any, 1=bachelor, 2=master, 3=phd
  "seniority": 0,           // 0=any, 1=entry, 2=mid, 3=senior
  "min_years": -1           // candidate's years of exp (-1=skip filter)
}
```

**Response:**
```json
{
  "resume_id": "a1b2c3d4",
  "matches": [
    {
      "job_id":      "gh_cloudflare_8097321",
      "title":       "AI Security Research & Red Team Engineer",
      "company":     "Cloudflare",
      "location":    "Hybrid",
      "score":       0.9124,
      "explanation": "Your 4+ years in offensive security, Python skills, and experience with LLMs and prompt security directly match the key responsibilities of this role.",
      "url":         "https://boards.greenhouse.io/cloudflare/jobs/8097321",
      "source":      "greenhouse"
    }
  ]
}
```

### `POST /api/fetch-jobs` — Fetch Real Jobs On-Demand

```json
{
  "source": "greenhouse",      // "greenhouse", "lever", "remotive", or "all"
  "company": "cloudflare",     // optional company slug
  "limit": 20,                 // number of jobs
  "auto_ingest": true          // whether to index into ChromaDB
}
```

### `GET /api/sources` — List Supported Job Portals

Returns configuration and default company handles for Greenhouse, Lever, and Remotive.

### `GET /api/stats` — Collection Job Counts

Returns total number of indexed jobs in ChromaDB.

### `GET /health` — Liveness probe
### `GET /ready` — Deep readiness check (ChromaDB, VoyageAI, Cohere, Ollama)

---

## Tech Stack

| Component | Technology |
|---|---|
| Job Portals | **Greenhouse** API + **Lever** API (+ Remotive) |
| Embeddings | VoyageAI `voyage-3.5-lite` (1024-dim) |
| Vector Store | ChromaDB (HNSW index) |
| Reranking | Cohere `rerank-english-v3.0` |
| LLM Explanations | Ollama `llama3.2:3b-instruct-q4_K_M` (local) |
| Web Framework | FastAPI + Uvicorn |
| Text Cleaning | BeautifulSoup4 + regex |
