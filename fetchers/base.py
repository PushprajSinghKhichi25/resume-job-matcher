"""
Base classes and utilities for job board fetchers.
"""

from __future__ import annotations

import json
import logging
import urllib.request
import urllib.error
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
}


def fetch_json(url: str, headers: dict[str, str] | None = None, timeout: int = 15) -> Any:
    """
    Fetch and parse JSON from a URL using standard urllib with custom headers.
    Falls back gracefully or raises with helpful error messages.
    """
    req_headers = dict(DEFAULT_HEADERS)
    if headers:
        req_headers.update(headers)

    req = urllib.request.Request(url, headers=req_headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            encoding = response.headers.get_content_charset() or "utf-8"
            raw_data = response.read().decode(encoding, errors="replace")
            return json.loads(raw_data)
    except urllib.error.HTTPError as e:
        logger.warning("HTTP error %d fetching %s: %s", e.code, url, e.reason)
        raise
    except urllib.error.URLError as e:
        logger.warning("Network error fetching %s: %s", url, e.reason)
        raise
    except json.JSONDecodeError as e:
        logger.warning("Failed to decode JSON from %s: %s", url, e)
        raise


class BaseJobFetcher(ABC):
    """
    Abstract Base Class for job board integrations.
    """

    source_name: str = "base"

    @abstractmethod
    def fetch_jobs(
        self,
        company: str | None = None,
        limit: int | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """
        Fetch real job postings and return them formatted according to the project's job schema:
        [
            {
                "id": str,
                "title": str,
                "company": str,
                "location": str,
                "description": str,
                "url": str,
                "source": str,
                "metadata": dict
            }, ...
        ]
        """
        pass
