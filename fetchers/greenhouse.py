"""
Greenhouse Job Board API integration.

Fetches live job postings from companies using Greenhouse ATS.
Greenhouse provides a public, unauthenticated Job Board API:
  https://boards-api.greenhouse.io/v1/boards/{board_token}/jobs?content=true
"""

from __future__ import annotations

import logging
from typing import Any

from fetchers.base import BaseJobFetcher, fetch_json

logger = logging.getLogger(__name__)

# A curated selection of popular tech companies using Greenhouse
DEFAULT_GREENHOUSE_BOARDS: list[str] = [
    "cloudflare",
    "figma",
    "gitlab",
    "stripe",
]


class GreenhouseFetcher(BaseJobFetcher):
    """
    Fetches real job postings from Greenhouse Job Boards.
    """

    source_name: str = "greenhouse"

    def __init__(self, default_boards: list[str] | None = None):
        self.boards = default_boards or DEFAULT_GREENHOUSE_BOARDS

    def fetch_company_jobs(
        self,
        board_token: str,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Fetch jobs for a specific company board on Greenhouse.

        Args:
            board_token: Company identifier on Greenhouse (e.g., 'cloudflare', 'figma').
            limit: Maximum number of jobs to return for this company.

        Returns:
            List of standardized job dictionaries.
        """
        board_token = board_token.strip().lower()
        url = f"https://boards-api.greenhouse.io/v1/boards/{board_token}/jobs?content=true"
        logger.info("[Greenhouse] Fetching jobs for board '%s' from %s", board_token, url)

        try:
            data = fetch_json(url)
        except Exception as e:
            logger.error("[Greenhouse] Failed to fetch jobs for board '%s': %s", board_token, e)
            return []

        raw_jobs = data.get("jobs", [])
        logger.info("[Greenhouse] Retrieved %d postings for '%s'", len(raw_jobs), board_token)

        jobs: list[dict[str, Any]] = []
        company_name = board_token.capitalize()

        for raw in raw_jobs:
            if limit is not None and len(jobs) >= limit:
                break

            job_id = str(raw.get("id", ""))
            title = (raw.get("title") or "").strip()
            content = raw.get("content") or ""

            # Location formatting
            loc_data = raw.get("location")
            if isinstance(loc_data, dict):
                location = loc_data.get("name", "")
            elif isinstance(loc_data, str):
                location = loc_data
            else:
                location = "Remote / Unspecified"

            # URL
            url = raw.get("absolute_url") or f"https://boards.greenhouse.io/{board_token}/jobs/{job_id}"

            # Departments / Offices metadata
            departments = [d.get("name") for d in raw.get("departments", []) if d.get("name")]
            offices = [o.get("name") for o in raw.get("offices", []) if o.get("name")]

            # Ensure minimal required fields
            if not job_id or not title or not content:
                continue

            jobs.append({
                "id": f"gh_{board_token}_{job_id}",
                "title": title,
                "company": company_name,
                "location": location or "Remote",
                "description": content,
                "url": url,
                "source": "greenhouse",
                "metadata": {
                    "board": board_token,
                    "departments": departments,
                    "offices": offices,
                    "updated_at": raw.get("updated_at"),
                },
            })

        logger.info("[Greenhouse] Standardized %d valid jobs from '%s'", len(jobs), board_token)
        return jobs

    def fetch_jobs(
        self,
        company: str | None = None,
        limit: int | None = None,
        boards: list[str] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """
        Fetch jobs from one or multiple Greenhouse boards.

        Args:
            company: Single company board token (e.g. 'cloudflare').
            limit: Maximum total number of jobs to return.
            boards: Custom list of company board tokens.
        """
        target_boards = [company] if company else (boards or self.boards)
        all_jobs: list[dict[str, Any]] = []

        for i, b in enumerate(target_boards):
            if not b:
                continue
            remaining_boards = len(target_boards) - i
            needed = (limit - len(all_jobs)) if limit is not None else None
            per_board = max(1, (needed + remaining_boards - 1) // remaining_boards) if needed is not None else None

            b_jobs = self.fetch_company_jobs(b, limit=per_board)
            all_jobs.extend(b_jobs)
            if limit is not None and len(all_jobs) >= limit:
                all_jobs = all_jobs[:limit]
                break

        logger.info("[Greenhouse] Finished fetching: %d total jobs across boards: %s", len(all_jobs), target_boards)
        return all_jobs
