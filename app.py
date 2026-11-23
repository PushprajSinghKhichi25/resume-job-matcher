"""
FastAPI web application — single file for the full web layer.

Endpoints
─────────
  GET  /                → Interactive HTML UI
  GET  /health          → Liveness probe
  GET  /ready           → Deep readiness check (ChromaDB, VoyageAI, Cohere, Ollama)
  POST /match           → Resume matching endpoint (JSON in, ranked jobs out)
  GET  /api/sources     → List integrated job board sources (Greenhouse, Lever, Remotive)
  GET  /api/stats       → Return collection job counts and status
  POST /api/fetch-jobs  → Fetch real job openings live from Greenhouse / Lever and optionally ingest

Run
───
  uvicorn app:app --reload --port 8000
"""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path

import chromadb
import requests
import voyageai
import cohere
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

from config import (
    CHROMA_COLLECTION,
    CHROMA_DIR,
    COHERE_API_KEY,
    DEGREE_LABELS,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    RERANK_TOP_N,
    RETRIEVE_TOP_K,
    SENIORITY_LABELS,
    VOYAGE_API_KEY,
    VOYAGE_MODEL,
    DEGREE_UNKNOWN,
    SENIORITY_UNKNOWN,
    YEARS_UNKNOWN,
)
from matcher import match_resume, JobMatch
from ingest import ingest_jobs_list
from fetchers import (
    GreenhouseFetcher,
    LeverFetcher,
    RemotiveFetcher,
    fetch_all_jobs,
    DEFAULT_GREENHOUSE_BOARDS,
    DEFAULT_LEVER_COMPANIES,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Resume–Job Matcher", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Pydantic Schemas ──────────────────────────────────────────────────────────

class MatchRequest(BaseModel):
    resume:    str = Field(..., min_length=50, description="Plain text resume content")
    top_k:     int = Field(RETRIEVE_TOP_K, ge=1, le=200, description="Retrieval pool size")
    top_n:     int = Field(RERANK_TOP_N,   ge=1, le=50,  description="Final results count")
    degree:    int = Field(DEGREE_UNKNOWN,    ge=0, le=3,  description="Min required degree (0=any)")
    seniority: int = Field(SENIORITY_UNKNOWN, ge=0, le=3,  description="Seniority level (0=any)")
    min_years: int = Field(YEARS_UNKNOWN,     ge=-1,       description="Candidate's years of experience (-1=skip)")


class MatchResponse(BaseModel):
    resume_id: str
    matches:   list[JobMatch]


class FetchJobsRequest(BaseModel):
    source:      str = Field("all", description="Job board source: greenhouse, lever, remotive, or all")
    company:     str | None = Field(None, description="Company slug / board token (e.g. cloudflare, spotify)")
    limit:       int = Field(20, ge=1, le=100, description="Max jobs to fetch")
    auto_ingest: bool = Field(False, description="Automatically index into ChromaDB vector database")


# ── Health & Readiness ────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Liveness probe — always 200 if the server is up."""
    return {"status": "ok"}


@app.get("/ready")
def ready():
    """
    Deep readiness check.
    Returns 200 only when all dependencies are reachable and configured.
    Returns 503 with a detailed error list otherwise.
    """
    errors: list[str] = []

    # ChromaDB
    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        col = client.get_or_create_collection(CHROMA_COLLECTION)
        count = col.count()
        chroma_status = f"ok ({count} jobs indexed)"
    except Exception as e:
        errors.append(f"ChromaDB: {e}")
        chroma_status = "error"

    # VoyageAI — live test embed
    try:
        if not VOYAGE_API_KEY:
            raise ValueError("VOYAGE_API_KEY not set")
        vc = voyageai.Client(api_key=VOYAGE_API_KEY)
        vc.embed(["test"], model=VOYAGE_MODEL, input_type="query")
        voyage_status = "ok"
    except Exception as e:
        errors.append(f"VoyageAI: {e}")
        voyage_status = "error"

    # Cohere
    try:
        if not COHERE_API_KEY:
            raise ValueError("COHERE_API_KEY not set")
        cohere_status = "ok (key present)"
    except Exception as e:
        errors.append(f"Cohere: {e}")
        cohere_status = "error"

    # Ollama
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        if OLLAMA_MODEL not in models:
            errors.append(f"Ollama: model '{OLLAMA_MODEL}' not loaded (run: ollama pull {OLLAMA_MODEL})")
            ollama_status = "model missing"
        else:
            ollama_status = f"ok ({OLLAMA_MODEL})"
    except Exception as e:
        errors.append(f"Ollama: {e}")
        ollama_status = "error"

    payload = {
        "chroma":  chroma_status,
        "voyage":  voyage_status,
        "cohere":  cohere_status,
        "ollama":  ollama_status,
    }

    if errors:
        payload["errors"] = errors
        return JSONResponse(status_code=503, content=payload)

    return payload


# ── Job Board Integration Endpoints ───────────────────────────────────────────

@app.get("/api/sources")
def get_sources():
    """Return available job portal integrations and featured companies."""
    return {
        "sources": [
            {
                "id": "greenhouse",
                "name": "Greenhouse",
                "badge": "greenhouse",
                "description": "Public Job Board API for top tech companies using Greenhouse ATS",
                "default_companies": DEFAULT_GREENHOUSE_BOARDS,
            },
            {
                "id": "lever",
                "name": "Lever",
                "badge": "lever",
                "description": "Public Postings API for high-growth tech companies using Lever ATS",
                "default_companies": DEFAULT_LEVER_COMPANIES,
            },
            {
                "id": "remotive",
                "name": "Remotive",
                "badge": "remotive",
                "description": "Open remote tech job aggregator across hundreds of companies",
                "default_companies": [],
            },
        ]
    }


@app.get("/api/stats")
def get_stats():
    """Return current collection stats."""
    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        col = client.get_or_create_collection(CHROMA_COLLECTION)
        return {"collection": CHROMA_COLLECTION, "total_jobs": col.count()}
    except Exception as e:
        return {"collection": CHROMA_COLLECTION, "total_jobs": 0, "error": str(e)}


@app.post("/api/fetch-jobs")
def fetch_jobs_api(req: FetchJobsRequest):
    """
    Fetch real job openings on-demand from Greenhouse, Lever, or Remotive.
    If auto_ingest is true, embeds and upserts them directly into ChromaDB.
    """
    source = req.source.lower().strip()
    company = req.company.strip().lower() if req.company else None
    limit = req.limit

    logger.info("API Fetch Jobs request: source=%s, company=%s, limit=%d, auto_ingest=%s",
                source, company, limit, req.auto_ingest)

    fetched_jobs: list[dict] = []
    try:
        if source == "greenhouse":
            gh = GreenhouseFetcher()
            fetched_jobs = gh.fetch_jobs(company=company, limit=limit)
        elif source == "lever":
            lever = LeverFetcher()
            fetched_jobs = lever.fetch_jobs(company=company, limit=limit)
        elif source == "remotive":
            remotive = RemotiveFetcher()
            fetched_jobs = remotive.fetch_jobs(limit=limit)
        elif source == "all":
            half = max(1, limit // 2)
            gh_jobs = GreenhouseFetcher().fetch_jobs(company=company, limit=half)
            lever_jobs = LeverFetcher().fetch_jobs(company=company, limit=half)
            fetched_jobs = gh_jobs + lever_jobs
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported source '{source}'")
    except Exception as e:
        logger.exception("Error while fetching jobs")
        raise HTTPException(status_code=500, detail=f"Failed to fetch jobs: {str(e)}")

    indexed_count = None
    if req.auto_ingest and fetched_jobs:
        try:
            ingest_jobs_list(fetched_jobs, reset=False)
            client = chromadb.PersistentClient(path=CHROMA_DIR)
            col = client.get_or_create_collection(CHROMA_COLLECTION)
            indexed_count = col.count()
        except Exception as e:
            logger.exception("Failed to auto-ingest fetched jobs")
            raise HTTPException(status_code=500, detail=f"Fetched {len(fetched_jobs)} jobs, but ingestion failed: {e}")

    # Return preview list (first 10 jobs) and summary
    previews = [
        {
            "id": j["id"],
            "title": j["title"],
            "company": j["company"],
            "location": j["location"],
            "url": j["url"],
            "source": j["source"],
        }
        for j in fetched_jobs[:10]
    ]

    return {
        "count": len(fetched_jobs),
        "source": source,
        "company": company,
        "auto_ingested": req.auto_ingest,
        "total_in_collection": indexed_count,
        "sample_jobs": previews,
    }


# ── Match Endpoint ────────────────────────────────────────────────────────────

@app.post("/match", response_model=MatchResponse)
def match(req: MatchRequest):
    """
    Run the full RAG pipeline:
      1. Embed resume (VoyageAI)
      2. Dense retrieval from ChromaDB (HNSW, top-K)
      3. Cross-encoder reranking (Cohere, top-N)
      4. Grounded explanation per match (Ollama LLaMA 3.2)
    """
    resume_id = str(uuid.uuid4())[:8]
    logger.info("[%s] Match request — resume length: %d chars", resume_id, len(req.resume))

    try:
        matches = match_resume(
            resume    = req.resume,
            top_k     = req.top_k,
            top_n     = req.top_n,
            degree    = req.degree,
            seniority = req.seniority,
            min_years = req.min_years,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("[%s] Unexpected error during matching", resume_id)
        raise HTTPException(status_code=500, detail="Internal matching error")

    logger.info("[%s] Returning %d matches", resume_id, len(matches))
    return MatchResponse(resume_id=resume_id, matches=matches)


# ── Interactive HTML UI ───────────────────────────────────────────────────────

_UI = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Resume–Job Matcher · Real Jobs (Greenhouse & Lever)</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;
       background:linear-gradient(135deg,#0f172a 0%,#1e1b4b 50%,#090d16 100%);
       min-height:100vh;padding:30px 20px;color:#e2e8f0}
  .wrap{max-width:920px;margin:0 auto}
  header{text-align:center;margin-bottom:28px}
  h1{font-size:2.2rem;font-weight:800;color:#f8fafc;letter-spacing:-0.02em;margin-bottom:8px}
  .gradient-text{background:linear-gradient(90deg,#38bdf8,#818cf8,#c084fc);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
  .subtitle{color:#94a3b8;font-size:.95rem}
  
  .card{background:#1e293b;border:1px solid #334155;border-radius:14px;padding:24px;margin-bottom:20px;
        box-shadow:0 10px 25px -5px rgba(0,0,0,0.5)}
  
  /* Collapsible / Accordion styles */
  .section-toggle{display:flex;justify-content:space-between;align-items:center;cursor:pointer;user-select:none}
  .section-title{font-size:1.05rem;font-weight:700;color:#f1f5f9;display:flex;align-items:center;gap:8px}
  .toggle-icon{color:#94a3b8;font-size:1.1rem;transition:transform .2s}
  
  label{display:block;font-size:.8rem;font-weight:600;color:#94a3b8;margin-bottom:6px;
        text-transform:uppercase;letter-spacing:.05em}
  textarea{width:100%;padding:14px;border:1px solid #334155;border-radius:8px;
            background:#0f172a;color:#f8fafc;font-size:.92rem;resize:vertical;
            min-height:170px;line-height:1.55;outline:none;transition:border-color .2s}
  textarea:focus{border-color:#38bdf8}
  
  .filters{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px;margin-top:16px}
  select,input[type=number],input[type=text]{width:100%;padding:10px 12px;border:1px solid #334155;
    border-radius:8px;background:#0f172a;color:#f8fafc;font-size:.9rem;outline:none;transition:border-color .2s}
  select:focus,input[type=number]:focus,input[type=text]:focus{border-color:#38bdf8}
  
  .btn{display:block;width:100%;padding:13px;background:#2563eb;color:#fff;
       border:none;border-radius:10px;font-size:1rem;font-weight:600;cursor:pointer;
       margin-top:18px;transition:all .2s;text-align:center}
  .btn:hover{background:#1d4ed8}
  .btn:disabled{background:#475569;cursor:not-allowed}
  .btn-secondary{background:#334155;color:#e2e8f0;margin-top:0}
  .btn-secondary:hover{background:#475569}
  .btn-green{background:#059669}
  .btn-green:hover{background:#047857}
  
  /* Quick Chips */
  .chips{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0 16px 0}
  .chip{background:#0f172a;border:1px solid #334155;border-radius:20px;padding:5px 12px;
        font-size:.78rem;color:#cbd5e1;cursor:pointer;transition:all .15s}
  .chip:hover{border-color:#38bdf8;color:#38bdf8}
  .chip.active{background:#1e3a5f;border-color:#38bdf8;color:#7dd3fc;font-weight:600}
  
  .spinner{display:none;text-align:center;padding:20px;color:#38bdf8;font-weight:600}
  .alert{border-radius:10px;padding:12px 16px;margin-bottom:16px;font-size:.9rem}
  .alert-error{background:#451a1a;border:1px solid #991b1b;color:#fca5a5}
  .alert-success{background:#064e3b;border:1px solid #065f46;color:#a7f3d0}
  
  .results-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:16px}
  .results-title{font-size:1.15rem;font-weight:700;color:#f8fafc}
  
  .job{background:#0f172a;border:1px solid #334155;border-radius:12px;
       padding:20px;margin-bottom:14px;transition:border-color .2s}
  .job:hover{border-color:#38bdf8}
  .job-header{display:flex;justify-content:space-between;align-items:flex-start;gap:12px}
  .job-title{font-size:1.1rem;font-weight:700;color:#7dd3fc}
  .job-sub{display:flex;align-items:center;gap:10px;margin-top:6px;flex-wrap:wrap}
  .job-company{color:#cbd5e1;font-size:.9rem;font-weight:600}
  .job-location{color:#94a3b8;font-size:.85rem}
  
  /* Source Badges */
  .badge{display:inline-block;padding:2px 8px;border-radius:6px;font-size:.72rem;font-weight:700;text-transform:uppercase;letter-spacing:.05em}
  .badge-gh{background:rgba(16,185,129,0.15);border:1px solid #10b981;color:#6ee7b7}
  .badge-lever{background:rgba(99,102,241,0.15);border:1px solid #6366f1;color:#a5b4fc}
  .badge-remotive{background:rgba(245,158,11,0.15);border:1px solid #f59e0b;color:#fcd34d}
  .badge-sample{background:rgba(148,163,184,0.15);border:1px solid #64748b;color:#cbd5e1}
  
  .score-badge{background:#1e3a5f;border:1px solid #0284c7;color:#7dd3fc;border-radius:20px;
               padding:5px 14px;font-size:.82rem;font-weight:700;white-space:nowrap}
  .explanation{margin-top:14px;font-size:.9rem;color:#cbd5e1;line-height:1.6;
               border-left:3px solid #38bdf8;padding-left:14px;background:rgba(15,23,42,.6);padding:10px 14px;border-radius:0 8px 8px 0}
  .job-link{display:inline-flex;align-items:center;gap:6px;margin-top:14px;font-size:.85rem;color:#38bdf8;
            text-decoration:none;font-weight:600}
  .job-link:hover{text-decoration:underline}
  .empty{text-align:center;padding:40px;color:#94a3b8}
  .stats-pill{background:#0f172a;border:1px solid #334155;border-radius:20px;padding:4px 12px;font-size:.78rem;color:#94a3b8}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>🎯 <span class="gradient-text">Resume–Job Matcher</span></h1>
    <p class="subtitle">AI-Powered Job Recommendation System with Real Openings from <strong>Greenhouse</strong> & <strong>Lever</strong></p>
  </header>

  <!-- Live Job Fetching Integration Card -->
  <div class="card" id="fetchCard">
    <div class="section-toggle" onclick="toggleSection('fetchBody', 'fetchToggleIcon')">
      <div class="section-title">
        <span>📡 Fetch Real Job Openings</span>
        <span class="stats-pill" id="indexStatus">Checking database…</span>
      </div>
      <span class="toggle-icon" id="fetchToggleIcon">▼</span>
    </div>

    <div id="fetchBody" style="margin-top:18px;">
      <p style="color:#94a3b8;font-size:.88rem;margin-bottom:12px">
        Fetch live open positions directly via the public APIs of <strong>Greenhouse</strong> and <strong>Lever</strong>.
      </p>

      <div style="margin-bottom:8px">
        <label>Popular Tech Boards (Click to select)</label>
        <div class="chips">
          <span class="chip" onclick="selectCompany('greenhouse', 'cloudflare')">🟢 Cloudflare (Greenhouse)</span>
          <span class="chip" onclick="selectCompany('greenhouse', 'figma')">🟢 Figma (Greenhouse)</span>
          <span class="chip" onclick="selectCompany('greenhouse', 'gitlab')">🟢 GitLab (Greenhouse)</span>
          <span class="chip" onclick="selectCompany('greenhouse', 'stripe')">🟢 Stripe (Greenhouse)</span>
          <span class="chip" onclick="selectCompany('lever', 'spotify')">🟣 Spotify (Lever)</span>
          <span class="chip" onclick="selectCompany('lever', 'palantir')">🟣 Palantir (Lever)</span>
          <span class="chip" onclick="selectCompany('lever', 'wealthfront')">🟣 Wealthfront (Lever)</span>
          <span class="chip" onclick="selectCompany('remotive', '')">🟠 All Remote (Remotive)</span>
        </div>
      </div>

      <div class="filters">
        <div>
          <label for="fetchSource">Job Portal</label>
          <select id="fetchSource">
            <option value="all">All (Greenhouse + Lever)</option>
            <option value="greenhouse">Greenhouse ATS</option>
            <option value="lever">Lever ATS</option>
            <option value="remotive">Remotive (Remote Tech)</option>
          </select>
        </div>
        <div>
          <label for="fetchCompany">Company Board Slug</label>
          <input type="text" id="fetchCompany" placeholder="e.g. cloudflare, spotify (optional)">
        </div>
        <div>
          <label for="fetchLimit">Number of Jobs</label>
          <input type="number" id="fetchLimit" min="1" max="100" value="20">
        </div>
      </div>

      <div style="display:flex;align-items:center;gap:10px;margin-top:14px">
        <input type="checkbox" id="autoIngest" checked style="width:16px;height:16px;cursor:pointer">
        <label for="autoIngest" style="margin-bottom:0;cursor:pointer;text-transform:none;font-size:.9rem;color:#cbd5e1">
          Automatically embed and index fetched jobs into ChromaDB vector database
        </label>
      </div>

      <button class="btn btn-green" id="fetchBtn" onclick="runFetchJobs()">Fetch & Index Real Jobs</button>
      <div id="fetchAlert" style="display:none;margin-top:14px"></div>
    </div>
  </div>

  <!-- Resume Matcher Card -->
  <div class="card">
    <label for="resume">Your Resume</label>
    <textarea id="resume" placeholder="Paste your resume here (at least 50 characters)…"></textarea>

    <div class="filters">
      <div>
        <label for="seniority">Seniority Level</label>
        <select id="seniority">
          <option value="0">Any</option>
          <option value="1">Entry-level</option>
          <option value="2">Mid-level</option>
          <option value="3">Senior</option>
        </select>
      </div>
      <div>
        <label for="degree">Min. Degree</label>
        <select id="degree">
          <option value="0">Any</option>
          <option value="1">Bachelor's</option>
          <option value="2">Master's</option>
          <option value="3">PhD</option>
        </select>
      </div>
      <div>
        <label for="years">My Experience (years)</label>
        <input type="number" id="years" min="-1" value="-1" placeholder="-1 = skip">
      </div>
      <div>
        <label for="topn">Top Matches</label>
        <input type="number" id="topn" min="1" max="20" value="5">
      </div>
    </div>

    <button class="btn" id="matchBtn" onclick="runMatch()">Find Best Matching Real Jobs</button>
  </div>

  <div class="spinner" id="spinner">⏳ Running RAG pipeline (Retrieval → Cross-Encoder Rerank → LLM Explanation)…</div>
  <div id="error" style="display:none"></div>
  <div id="results"></div>
</div>

<script>
function toggleSection(bodyId, iconId){
  const body = document.getElementById(bodyId);
  const icon = document.getElementById(iconId);
  if(body.style.display === "none"){
    body.style.display = "block";
    icon.textContent = "▼";
  } else {
    body.style.display = "none";
    icon.textContent = "▶";
  }
}

function selectCompany(source, company){
  document.getElementById("fetchSource").value = source;
  document.getElementById("fetchCompany").value = company;
}

async function refreshStats(){
  try {
    const res = await fetch("/api/stats");
    const data = await res.json();
    const el = document.getElementById("indexStatus");
    if(data.total_jobs !== undefined){
      el.textContent = `${data.total_jobs} active jobs in ChromaDB`;
    }
  } catch(e){}
}
refreshStats();

async function runFetchJobs(){
  const btn = document.getElementById("fetchBtn");
  const alertEl = document.getElementById("fetchAlert");
  btn.disabled = true;
  btn.textContent = "⏳ Fetching live jobs from portal API…";
  alertEl.style.display = "none";

  const source = document.getElementById("fetchSource").value;
  const company = document.getElementById("fetchCompany").value.trim();
  const limit = parseInt(document.getElementById("fetchLimit").value) || 20;
  const autoIngest = document.getElementById("autoIngest").checked;

  try {
    const res = await fetch("/api/fetch-jobs", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({source, company: company || null, limit, auto_ingest: autoIngest})
    });
    const data = await res.json();
    if(!res.ok){
      alertEl.className = "alert alert-error";
      alertEl.textContent = "Error: " + (data.detail || "Fetch failed");
      alertEl.style.display = "block";
    } else {
      alertEl.className = "alert alert-success";
      let msg = `Successfully fetched ${data.count} real jobs from ${data.source.toUpperCase()}!`;
      if(data.auto_ingested){
        msg += ` Embedded & indexed into ChromaDB (Total in DB: ${data.total_in_collection}).`;
      }
      alertEl.textContent = msg;
      alertEl.style.display = "block";
      refreshStats();
    }
  } catch(e){
    alertEl.className = "alert alert-error";
    alertEl.textContent = "Network error: " + e.message;
    alertEl.style.display = "block";
  } finally {
    btn.disabled = false;
    btn.textContent = "Fetch & Index Real Jobs";
  }
}

async function runMatch(){
  const resume = document.getElementById("resume").value.trim();
  if(resume.length < 50){ alert("Please paste a longer resume (at least 50 characters)."); return; }

  const btn = document.getElementById("matchBtn");
  btn.disabled = true;
  document.getElementById("spinner").style.display = "block";
  document.getElementById("error").style.display   = "none";
  document.getElementById("results").innerHTML = "";

  const body = {
    resume:    resume,
    top_n:     parseInt(document.getElementById("topn").value) || 5,
    seniority: parseInt(document.getElementById("seniority").value),
    degree:    parseInt(document.getElementById("degree").value),
    min_years: parseInt(document.getElementById("years").value),
  };

  try {
    const res  = await fetch("/match", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)});
    const data = await res.json();

    if(!res.ok){ showError(data.detail || "Unknown matching error"); return; }
    renderResults(data.matches, data.resume_id);
  } catch(e){
    showError("Network error: " + e.message);
  } finally {
    btn.disabled = false;
    document.getElementById("spinner").style.display = "none";
  }
}

function showError(msg){
  const el = document.getElementById("error");
  el.className = "alert alert-error"; el.textContent = "Error: " + msg; el.style.display = "block";
}

function getSourceBadge(source){
  const s = (source || "").toLowerCase();
  if(s === "greenhouse") return '<span class="badge badge-gh">Greenhouse</span>';
  if(s === "lever") return '<span class="badge badge-lever">Lever</span>';
  if(s === "remotive") return '<span class="badge badge-remotive">Remotive</span>';
  return '<span class="badge badge-sample">Verified Job</span>';
}

function renderResults(matches, resumeId){
  const el = document.getElementById("results");
  if(!matches || matches.length === 0){
    el.innerHTML = '<div class="empty">No matches found. Try adjusting filters or clicking "Fetch Real Job Openings" above to index jobs into the database.</div>'; return;
  }
  let html = `<div class="card"><div class="results-header">
    <div class="results-title">🏆 Top ${matches.length} Matches</div>
    <span style="color:#94a3b8;font-size:.85rem">ID: ${resumeId}</span>
  </div>`;
  matches.forEach((m,i)=>{
    const pct = (m.score*100).toFixed(1);
    const badge = getSourceBadge(m.source);
    html += `
    <div class="job">
      <div class="job-header">
        <div>
          <div class="job-title">${i+1}. ${m.title}</div>
          <div class="job-sub">
            <span class="job-company">${m.company}</span>
            ${m.location ? `<span class="job-location">📍 ${m.location}</span>` : ""}
            ${badge}
          </div>
        </div>
        <div class="score-badge">Match: ${pct}%</div>
      </div>
      <div class="explanation">${m.explanation}</div>
      ${m.url ? `<a class="job-link" href="${m.url}" target="_blank" rel="noopener noreferrer">View & Apply on Official Portal →</a>` : ""}
    </div>`;
  });
  html += "</div>";
  el.innerHTML = html;
}

document.getElementById("resume").addEventListener("keydown", e => {
  if((e.ctrlKey||e.metaKey) && e.key==="Enter") runMatch();
});
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def ui():
    """Serve the interactive matching UI."""
    return HTMLResponse(content=_UI)
