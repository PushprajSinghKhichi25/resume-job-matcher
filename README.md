# Resume–Job Matcher

A clean, minimal RAG-based job recommendation system.  

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
├── config.py          # All constants and env-var loading
├── preprocess.py      # HTML cleaning + regex metadata extraction
├── matcher.py         # Core RAG pipeline (the brain)
├── ingest.py          # CLI: load jobs from JSON into ChromaDB
├── app.py             # FastAPI server + inline HTML UI
├── sample_jobs.json   # 10 sample jobs to test immediately
└── requirements.txt
```

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

### 4. Ingest jobs

```bash
# Uses sample_jobs.json by default
python ingest.py

# To use your own jobs file:
python ingest.py --file my_jobs.json

# To reset and re-ingest:
python ingest.py --reset
```

### 5. Start the server

```bash
uvicorn app:app --reload --port 8000
```

Then open **http://localhost:8000** in your browser.

---

## API Reference

### `POST /match`

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
      "job_id":      "job_002",
      "title":       "Machine Learning Engineer",
      "company":     "Nexus AI",
      "location":    "San Francisco, CA",
      "score":       0.9312,
      "explanation": "Your 3+ years of Python and PyTorch experience aligns...",
      "url":         "https://example.com/jobs/002"
    }
  ]
}
```

### `GET /health` — Liveness probe  
### `GET /ready` — Checks ChromaDB, VoyageAI, Cohere, and Ollama

---

## Adding Your Own Jobs

The ingestion script accepts any JSON file with this schema:

```json
[
  {
    "id":          "unique-string",
    "title":       "Software Engineer",
    "company":     "Acme Corp",
    "location":    "Remote",
    "description": "Plain text or HTML job description...",
    "url":         "https://jobs.example.com/123"
  }
]
```

```bash
python ingest.py --file path/to/your_jobs.json
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Embeddings | VoyageAI `voyage-3.5-lite` (1024-dim) |
| Vector Store | ChromaDB (HNSW index) |
| Reranking | Cohere `rerank-english-v3.0` |
| LLM Explanations | Ollama `llama3.2:3b-instruct-q4_K_M` (local) |
| API | FastAPI + Uvicorn |
| Text Cleaning | BeautifulSoup4 + regex |
