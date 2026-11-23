"""
Job Fetchers package — unified interface for fetching real job postings
from Greenhouse, Lever, and other job boards.
"""

from __future__ import annotations

import logging
from typing import Any

from fetchers.base import BaseJobFetcher
from fetchers.greenhouse import GreenhouseFetcher, DEFAULT_GREENHOUSE_BOARDS
from fetchers.lever import LeverFetcher, DEFAULT_LEVER_COMPANIES
from fetchers.remotive import RemotiveFetcher

logger = logging.getLogger(__name__)

AVAILABLE_FETCHERS: dict[str, type[BaseJobFetcher]] = {
    "greenhouse": GreenhouseFetcher,
    "lever": LeverFetcher,
    "remotive": RemotiveFetcher,
}


def get_fetcher(name: str) -> BaseJobFetcher:
    """Instantiate a fetcher by its source name."""
    name_clean = name.strip().lower()
    fetcher_cls = AVAILABLE_FETCHERS.get(name_clean)
    if not fetcher_cls:
        raise ValueError(f"Unknown job source '{name}'. Available: {list(AVAILABLE_FETCHERS.keys())}")
    return fetcher_cls()


def fetch_all_jobs(
    sources: list[str] | None = None,
    limit_per_source: int = 25,
    greenhouse_boards: list[str] | None = None,
    lever_companies: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Fetch jobs across multiple sources (defaults to Greenhouse + Lever).

    Args:
        sources: List of sources to query (e.g. ['greenhouse', 'lever']).
        limit_per_source: Maximum jobs to fetch from each source.
        greenhouse_boards: Optional specific Greenhouse boards.
        lever_companies: Optional specific Lever companies.

    Returns:
        Combined list of standardized job postings.
    """
    sources_to_query = sources or ["greenhouse", "lever"]
    combined_jobs: list[dict[str, Any]] = []

    for src in sources_to_query:
        src = src.strip().lower()
        try:
            fetcher = get_fetcher(src)
            if src == "greenhouse":
                jobs = fetcher.fetch_jobs(boards=greenhouse_boards, limit=limit_per_source)
            elif src == "lever":
                jobs = fetcher.fetch_jobs(companies=lever_companies, limit=limit_per_source)
            else:
                jobs = fetcher.fetch_jobs(limit=limit_per_source)
            combined_jobs.extend(jobs)
        except Exception as e:
            logger.error("Error fetching from '%s': %s", src, e)

    logger.info("Combined %d total jobs across %s", len(combined_jobs), sources_to_query)
    return combined_jobs


__all__ = [
    "BaseJobFetcher",
    "GreenhouseFetcher",
    "LeverFetcher",
    "RemotiveFetcher",
    "get_fetcher",
    "fetch_all_jobs",
    "DEFAULT_GREENHOUSE_BOARDS",
    "DEFAULT_LEVER_COMPANIES",
]
