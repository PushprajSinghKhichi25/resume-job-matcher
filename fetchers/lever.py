"""
Lever Job Board API integration.

Fetches live job postings from companies using Lever ATS.
Lever provides a public, unauthenticated Postings API:
  https://api.lever.co/v0/postings/{company}?mode=json
"""

from __future__ import annotations

import logging
from typing import Any

from fetchers.base import BaseJobFetcher, fetch_json

logger = logging.getLogger(__name__)

# Curated selection of tech companies actively using Lever
DEFAULT_LEVER_COMPANIES: list[str] = [
    "spotify",
    "palantir",
    "wealthfront",
]


class LeverFetcher(BaseJobFetcher):
    """
    Fetches real job postings from Lever Job Boards.
    """

    source_name: str = "lever"

    def __init__(self, default_companies: list[str] | None = None):
        self.companies = default_companies or DEFAULT_LEVER_COMPANIES

    def fetch_company_jobs(
        self,
        company: str,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Fetch jobs for a specific company on Lever.

        Args:
            company: Company slug on Lever (e.g., 'spotify', 'palantir').
            limit: Maximum number of jobs to return.

        Returns:
            List of standardized job dictionaries.
        """
        company_slug = company.strip().lower()
        url = f"https://api.lever.co/v0/postings/{company_slug}?mode=json"
        logger.info("[Lever] Fetching jobs for company '%s' from %s", company_slug, url)

        try:
            data = fetch_json(url)
        except Exception as e:
            logger.error("[Lever] Failed to fetch jobs for company '%s': %s", company_slug, e)
            return []

        if not isinstance(data, list):
            logger.warning("[Lever] Expected list of postings, got %s", type(data))
            return []

        logger.info("[Lever] Retrieved %d postings for '%s'", len(data), company_slug)

        jobs: list[dict[str, Any]] = []
        company_name = company_slug.capitalize()

        for raw in data:
            if limit is not None and len(jobs) >= limit:
                break

            job_id = str(raw.get("id", ""))
            title = (raw.get("text") or "").strip()

            # Compose rich description from description + lists + additional
            desc_parts: list[str] = []
            if raw.get("description"):
                desc_parts.append(raw["description"])
            elif raw.get("descriptionPlain"):
                desc_parts.append(f"<p>{raw['descriptionPlain']}</p>")

            for item in raw.get("lists", []):
                list_title = item.get("text", "")
                list_content = item.get("content", "")
                if list_title or list_content:
                    desc_parts.append(f"<h4>{list_title}</h4>\n{list_content}")

            if raw.get("additional"):
                desc_parts.append(raw["additional"])
            elif raw.get("additionalPlain"):
                desc_parts.append(f"<p>{raw['additionalPlain']}</p>")

            full_description = "\n\n".join(desc_parts)

            # Location formatting
            categories = raw.get("categories") or {}
            location = categories.get("location")
            if not location:
                location = raw.get("workplaceType") or "Remote / Unspecified"

            # Direct URL
            url = raw.get("hostedUrl") or raw.get("applyUrl") or f"https://jobs.lever.co/{company_slug}/{job_id}"

            if not job_id or not title or not full_description:
                continue

            jobs.append({
                "id": f"lever_{company_slug}_{job_id}",
                "title": title,
                "company": company_name,
                "location": location,
                "description": full_description,
                "url": url,
                "source": "lever",
                "metadata": {
                    "company_slug": company_slug,
                    "categories": categories,
                    "workplace_type": raw.get("workplaceType"),
                    "created_at": raw.get("createdAt"),
                },
            })

        logger.info("[Lever] Standardized %d valid jobs from '%s'", len(jobs), company_slug)
        return jobs

    def fetch_jobs(
        self,
        company: str | None = None,
        limit: int | None = None,
        companies: list[str] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """
        Fetch jobs from one or multiple Lever companies.

        Args:
            company: Single company slug (e.g. 'spotify').
            limit: Maximum total number of jobs to return.
            companies: Custom list of company slugs.
        """
        target_companies = [company] if company else (companies or self.companies)
        all_jobs: list[dict[str, Any]] = []

        for i, c in enumerate(target_companies):
            if not c:
                continue
            remaining_companies = len(target_companies) - i
            needed = (limit - len(all_jobs)) if limit is not None else None
            per_company = max(1, (needed + remaining_companies - 1) // remaining_companies) if needed is not None else None

            c_jobs = self.fetch_company_jobs(c, limit=per_company)
            all_jobs.extend(c_jobs)
            if limit is not None and len(all_jobs) >= limit:
                all_jobs = all_jobs[:limit]
                break

        logger.info("[Lever] Finished fetching: %d total jobs across companies: %s", len(all_jobs), target_companies)
        return all_jobs
