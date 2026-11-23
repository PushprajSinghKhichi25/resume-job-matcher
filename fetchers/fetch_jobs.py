"""
CLI script to fetch real job postings from Greenhouse, Lever, and other job portals.

Usage Examples:
  # Fetch from both Greenhouse and Lever (default) and save to real_jobs.json:
  python fetch_jobs.py --source all --limit 50 --output real_jobs.json

  # Fetch only from Greenhouse for Cloudflare:
  python fetch_jobs.py --source greenhouse --company cloudflare --limit 20 --output greenhouse_jobs.json

  # Fetch only from Lever for Spotify:
  python fetch_jobs.py --source lever --company spotify --limit 20 --output lever_jobs.json

  # Fetch from Remotive (remote tech jobs):
  python fetch_jobs.py --source remotive --limit 25 --output remotive_jobs.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from fetchers import (
    GreenhouseFetcher,
    LeverFetcher,
    RemotiveFetcher,
    fetch_all_jobs,
    get_fetcher,
    DEFAULT_GREENHOUSE_BOARDS,
    DEFAULT_LEVER_COMPANIES,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch real job postings from Greenhouse, Lever, and Remotive"
    )
    parser.add_argument(
        "--source",
        choices=["greenhouse", "lever", "remotive", "all"],
        default="all",
        help="Job portal to fetch from (default: all)",
    )
    parser.add_argument(
        "--company",
        type=str,
        default=None,
        help="Specific company slug / board token (e.g. cloudflare, spotify, figma)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum total jobs to fetch (default: 50)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="real_jobs.json",
        help="Output JSON file path (default: real_jobs.json)",
    )
    parser.add_argument(
        "--print-sample",
        action="store_true",
        help="Print the first 3 fetched jobs to console",
    )

    args = parser.parse_args()

    logger.info("Starting job fetch [source=%s, company=%s, limit=%d]...", args.source, args.company, args.limit)

    jobs: list[dict] = []
    if args.source == "all":
        # Fetch from Greenhouse and Lever
        half_limit = max(1, args.limit // 2)
        gh = GreenhouseFetcher()
        lever = LeverFetcher()
        gh_jobs = gh.fetch_jobs(company=args.company, limit=half_limit)
        lever_jobs = lever.fetch_jobs(company=args.company, limit=half_limit)
        jobs = gh_jobs + lever_jobs
    elif args.source == "greenhouse":
        fetcher = GreenhouseFetcher()
        jobs = fetcher.fetch_jobs(company=args.company, limit=args.limit)
    elif args.source == "lever":
        fetcher = LeverFetcher()
        jobs = fetcher.fetch_jobs(company=args.company, limit=args.limit)
    elif args.source == "remotive":
        fetcher = RemotiveFetcher()
        jobs = fetcher.fetch_jobs(limit=args.limit)

    logger.info("Fetched %d real job openings", len(jobs))

    if not jobs:
        logger.warning("No jobs were found with the given parameters.")
        return

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(jobs, f, indent=2, ensure_ascii=False)

    logger.info("Saved %d jobs to %s ✓", len(jobs), out_path.resolve())

    if args.print_sample:
        print("\n--- Sample Fetched Jobs ---")
        for i, j in enumerate(jobs[:3], 1):
            print(f"{i}. [{j.get('source', '').upper()}] {j.get('title')} @ {j.get('company')} ({j.get('location')})")
            print(f"   URL: {j.get('url')}\n")


if __name__ == "__main__":
    main()
