"""
FastAPI web application — single file for the full web layer.

Endpoints
─────────
  GET  /           → Interactive HTML UI (no extra template files needed)
  GET  /health     → Liveness probe
  GET  /ready      → Deep readiness check (ChromaDB, VoyageAI, Cohere, Ollama)
  POST /match      → Resume matching endpoint (JSON in, ranked jobs out)

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Resume–Job Matcher", version="1.0.0")
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


# ── Match Endpoint ────────────────────────────────────────────────────────────

@app.post("/match", response_model=MatchResponse)
def match(req: MatchRequest):
    """
    Run the full RAG pipeline:
      1. Embed resume (VoyageAI)
      2. Dense retrieval from ChromaDB (HNSW, top-K)
      3. Cross-encoder reranking (Cohere, top-N)
      4. Grounded explanation per match (Ollama LLaMA 3.2)

    Accepts metadata filters for degree, seniority, and years of experience.
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
<title>Resume–Job Matcher</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
       background:linear-gradient(135deg,#1a1a2e 0%,#16213e 60%,#0f3460 100%);
       min-height:100vh;padding:30px 20px;color:#e0e0e0}
  .wrap{max-width:860px;margin:0 auto}
  h1{text-align:center;font-size:2rem;font-weight:700;color:#fff;margin-bottom:6px}
  .subtitle{text-align:center;color:#90b4ce;margin-bottom:30px;font-size:.95rem}
  .card{background:#1e2a3a;border-radius:14px;padding:28px;margin-bottom:24px;
        box-shadow:0 4px 24px rgba(0,0,0,.4)}
  label{display:block;font-size:.85rem;font-weight:600;color:#90b4ce;margin-bottom:6px;
        text-transform:uppercase;letter-spacing:.04em}
  textarea{width:100%;padding:14px;border:1px solid #2d4055;border-radius:8px;
           background:#0d1b2a;color:#dde;font-size:.93rem;resize:vertical;
           min-height:180px;line-height:1.5;outline:none;transition:border .2s}
  textarea:focus{border-color:#4a90d9}
  .filters{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:14px;margin-top:16px}
  select,input[type=number]{width:100%;padding:10px 12px;border:1px solid #2d4055;
    border-radius:8px;background:#0d1b2a;color:#dde;font-size:.9rem;outline:none}
  select:focus,input[type=number]:focus{border-color:#4a90d9}
  .btn{display:block;width:100%;padding:14px;background:#2563eb;color:#fff;
       border:none;border-radius:10px;font-size:1rem;font-weight:600;cursor:pointer;
       margin-top:20px;transition:background .2s}
  .btn:hover{background:#1d4ed8}
  .btn:disabled{background:#374151;cursor:not-allowed}
  .spinner{display:none;text-align:center;padding:20px;color:#90b4ce}
  .error{background:#3b1a1a;border:1px solid #7f1d1d;border-radius:10px;
         padding:14px 18px;color:#fca5a5;margin-bottom:16px}
  .results-title{font-size:1.1rem;font-weight:600;color:#fff;margin-bottom:16px}
  .job{background:#0d1b2a;border:1px solid #2d4055;border-radius:12px;
       padding:20px;margin-bottom:14px;transition:border .2s}
  .job:hover{border-color:#4a90d9}
  .job-header{display:flex;justify-content:space-between;align-items:flex-start;gap:10px}
  .job-title{font-size:1.05rem;font-weight:700;color:#7dd3fc}
  .job-company{color:#90b4ce;font-size:.88rem;margin-top:3px}
  .score-badge{background:#1e3a5f;color:#7dd3fc;border-radius:20px;
               padding:4px 12px;font-size:.8rem;font-weight:700;white-space:nowrap}
  .job-meta{display:flex;gap:10px;margin-top:10px;flex-wrap:wrap}
  .tag{background:#1e2a3a;border:1px solid #2d4055;border-radius:6px;
       padding:3px 10px;font-size:.78rem;color:#94a3b8}
  .explanation{margin-top:12px;font-size:.9rem;color:#c0d4e8;line-height:1.6;
               border-left:3px solid #2563eb;padding-left:12px}
  .job-link{display:inline-block;margin-top:12px;font-size:.82rem;color:#60a5fa;
            text-decoration:none}
  .job-link:hover{text-decoration:underline}
  .empty{text-align:center;padding:40px;color:#90b4ce}
</style>
</head>
<body>
<div class="wrap">
  <h1>🎯 Resume–Job Matcher</h1>
  <p class="subtitle">RAG pipeline · Dense retrieval · Cross-encoder reranking · Grounded explanations</p>

  <div class="card">
    <label for="resume">Your Resume</label>
    <textarea id="resume" placeholder="Paste your resume here (plain text)…"></textarea>

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
        <label for="topn">Results Count</label>
        <input type="number" id="topn" min="1" max="20" value="5">
      </div>
    </div>

    <button class="btn" id="btn" onclick="runMatch()">Find Matching Jobs</button>
  </div>

  <div class="spinner" id="spinner">⏳ Running RAG pipeline…</div>
  <div id="error" style="display:none"></div>
  <div id="results"></div>
</div>

<script>
const SENIORITY = {0:"Any",1:"Entry-level",2:"Mid-level",3:"Senior"};
const DEGREE    = {0:"Any",1:"Bachelor's",2:"Master's",3:"PhD"};

async function runMatch(){
  const resume = document.getElementById("resume").value.trim();
  if(resume.length < 50){ alert("Please paste a longer resume (at least 50 characters)."); return; }

  const btn = document.getElementById("btn");
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

    if(!res.ok){ showError(data.detail || "Unknown error"); return; }

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
  el.className = "error"; el.textContent = "Error: " + msg; el.style.display = "block";
}

function renderResults(matches, resumeId){
  const el = document.getElementById("results");
  if(!matches || matches.length === 0){
    el.innerHTML = '<div class="empty">No matches found. Try adjusting filters or ingesting more jobs.</div>'; return;
  }
  let html = `<div class="card"><p class="results-title">🏆 ${matches.length} Matches  <span style="color:#4a90d9;font-weight:400;font-size:.85rem">(resume_id: ${resumeId})</span></p>`;
  matches.forEach((m,i)=>{
    const pct = (m.score*100).toFixed(1);
    html += `
    <div class="job">
      <div class="job-header">
        <div>
          <div class="job-title">${i+1}. ${m.title}</div>
          <div class="job-company">${m.company}${m.location ? " · " + m.location : ""}</div>
        </div>
        <div class="score-badge">Score: ${pct}%</div>
      </div>
      <div class="explanation">${m.explanation}</div>
      ${m.url ? `<a class="job-link" href="${m.url}" target="_blank">View Job Posting →</a>` : ""}
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
