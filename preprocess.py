"""
Text preprocessing utilities.

Responsibilities
────────────────
• Clean raw HTML job descriptions → plain text
• Extract structured metadata (seniority, degree, years) via regex
• Build ChromaDB where-filters from user-supplied filter values
"""

import re
from html import unescape
from bs4 import BeautifulSoup


# ── HTML Cleaning ─────────────────────────────────────────────────────────────

def clean_html(raw: str) -> str:
    """
    Strip HTML tags, scripts, iframes and normalize whitespace.

    Steps
    1. Unescape HTML entities  (&amp; → &, &#39; → ')
    2. Remove <script>, <style>, <iframe> and <img> tags entirely
    3. Extract plain text via BeautifulSoup
    4. Normalize whitespace (collapse runs of spaces/newlines)
    5. Normalize unicode punctuation (smart quotes → straight quotes)
    """
    if not raw:
        return ""

    # Step 1 — unescape entities
    text = unescape(raw)

    # Step 2 — drop tags we never want text from
    soup = BeautifulSoup(text, "html.parser")
    for tag in soup(["script", "style", "iframe", "img"]):
        tag.decompose()

    # Step 3 — extract visible text
    text = soup.get_text(separator=" ")

    # Step 4 — normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    # Step 5 — normalize unicode punctuation
    text = text.replace("\u2018", "'").replace("\u2019", "'")  # curly single quotes
    text = text.replace("\u201c", '"').replace("\u201d", '"')  # curly double quotes
    text = text.replace("\u2013", "-").replace("\u2014", "-")  # en/em dashes

    return text


# ── Metadata Extraction ───────────────────────────────────────────────────────

_DEGREE_PATTERNS: list[tuple[int, list[str]]] = [
    (3, [r"\bph\.?d\b", r"\bdoctorate\b"]),
    (2, [r"\bmaster'?s?\b", r"\bm\.?s\.?\b", r"\bm\.?eng\b", r"\bmba\b"]),
    (1, [r"\bbachelor'?s?\b", r"\bb\.?s\.?\b", r"\bb\.?e\.?\b", r"\bb\.?tech\b", r"\bundergraduate\b"]),
]

_SENIORITY_PATTERNS: list[tuple[int, list[str]]] = [
    (3, [r"\bstaff\b", r"\bprincipal\b", r"\bsenior\b", r"\bsr\.\b", r"\blead\b"]),
    (2, [r"\bmid[- ]?level\b", r"\bintermediate\b"]),
    (1, [r"\bjunior\b", r"\bjr\.\b", r"\bentry[- ]?level\b", r"\bassociate\b", r"\bnew grad\b"]),
]

_YEARS_RE = re.compile(
    r"(\d+)\+?\s*(?:to|[-–])\s*(\d+)\s*(?:years?|yrs?)"  # "3 to 5 years"
    r"|(\d+)\+\s*(?:years?|yrs?)"                           # "3+ years"
    r"|(\d+)\s*(?:years?|yrs?)\s+(?:of\s+)?experience",    # "3 years experience"
    re.IGNORECASE,
)


def extract_degree(text: str) -> int:
    """Return 0–3 (0=unknown, 1=bachelor, 2=master, 3=phd)."""
    lower = text.lower()
    for level, patterns in _DEGREE_PATTERNS:
        if any(re.search(p, lower) for p in patterns):
            return level
    return 0


def extract_seniority(text: str, title: str = "") -> int:
    """Return 0–3 (0=unknown, 1=entry, 2=mid, 3=senior)."""
    combined = (title + " " + text).lower()
    for level, patterns in _SENIORITY_PATTERNS:
        if any(re.search(p, combined) for p in patterns):
            return level
    return 0


def extract_min_years(text: str) -> int:
    """Return minimum required years of experience, or -1 if not found."""
    match = _YEARS_RE.search(text)
    if not match:
        return -1
    groups = match.groups()
    # group 0: lower bound of range | group 2: X+ | group 3: plain X years
    for g in [groups[0], groups[2], groups[3]]:
        if g is not None:
            return int(g)
    return -1


# ── ChromaDB Metadata Filters ─────────────────────────────────────────────────

def build_where_filter(
    degree: int,
    seniority: int,
    min_years: int,
) -> dict | None:
    """
    Build a ChromaDB `where` clause from user-supplied filter values.

    Fields set to their sentinel (0 / -1) are skipped so they don't
    over-constrain the query.

    Returns None when no filter is active (ChromaDB won't accept an empty
    $and clause).
    """
    clauses: list[dict] = []

    if degree > 0:
        # Jobs that require exactly this degree OR don't specify (0)
        clauses.append({"$or": [{"required_degree": {"$eq": degree}}, {"required_degree": {"$eq": 0}}]})

    if seniority > 0:
        clauses.append({"$or": [{"seniority_level": {"$eq": seniority}}, {"seniority_level": {"$eq": 0}}]})

    if min_years >= 0:
        clauses.append({"$or": [{"min_years_experience": {"$lte": min_years}}, {"min_years_experience": {"$eq": -1}}]})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}
