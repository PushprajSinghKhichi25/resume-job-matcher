"""
Remotive Job Board API integration.

Remotive provides a free public API for remote tech jobs across hundreds of companies:
  https://remotive.com/api/remote-jobs
"""

from __future__ import annotations

import logging
from typing import Any
import urllib.parse

from fetchers.base import BaseJobFetcher, fetch_json

logger = logging.getLogger(__name__)


class RemotiveFetcher(BaseJobFetcher):
    """
    Fetches real remote job openings from the Remotive API across various companies.
    """

    source_name: str = "remotive"

    def fetch_jobs(
        self,
        category: str | None = "software-dev",
        search: str | None = None,
        limit: int | None = 25,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """
        Fetch jobs from Remotive.

        Args:
            category: Job category (e.g. 'software-dev', 'data', 'devops', 'product').
            search: Search keyword.
            limit: Maximum number of jobs.
        """
        params: dict[str, Any] = {}
        if category:
            params["category"] = category
        if search:
            params["search"] = search
        if limit:
            params["limit"] = min(limit, 100)

        query_string = urllib.parse.urlencode(params)
        url = f"https://remotive.com/api/remote-jobs?{query_string}" if query_string else "https://remotive.com/api/remote-jobs"

        logger.info("[Remotive] Fetching jobs from %s", url)
        try:
            data = fetch_json(url)
        except Exception as e:
            logger.error("[Remotive] Failed to fetch jobs: %s", e)
            return []

        raw_jobs = data.get("jobs", [])
        logger.info("[Remotive] Retrieved %d postings", len(raw_jobs))

        jobs: list[dict[str, Any]] = []
        for raw in raw_jobs:
            if limit is not None and len(jobs) >= limit:
                break

            job_id = str(raw.get("id", ""))
            title = (raw.get("title") or "").strip()
            company = (raw.get("company_name") or "Unknown").strip()
            description = raw.get("description") or ""
            url_link = raw.get("url") or ""
            location = raw.get("candidate_required_location") or "Remote"

            if not job_id or not title or not description:
                continue

            jobs.append({
                "id": f"remotive_{job_id}",
                "title": title,
                "company": company,
                "location": location,
                "description": description,
                "url": url_link,
                "source": "remotive",
                "metadata": {
                    "category": raw.get("category"),
                    "tags": raw.get("tags", []),
                    "job_type": raw.get("job_type"),
                    "salary": raw.get("salary"),
                    "publication_date": raw.get("publication_date"),
                },
            })

        logger.info("[Remotive] Standardized %d valid jobs", len(jobs))
        return jobs
