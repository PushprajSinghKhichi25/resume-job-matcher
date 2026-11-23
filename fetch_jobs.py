"""
Top-level entrypoint for fetching real job postings.

Usage:
  python fetch_jobs.py --source all --limit 50 --output real_jobs.json
  python fetch_jobs.py --source greenhouse --company cloudflare --limit 20
  python fetch_jobs.py --source lever --company spotify --limit 20
"""

from fetchers.fetch_jobs import main

if __name__ == "__main__":
    main()
