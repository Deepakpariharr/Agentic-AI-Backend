"""search_jobs: live job listings from JSearch (RapidAPI).

Runs inside the stdio MCP subprocess, where stdout carries the protocol —
plain stdlib logging goes to stderr, so it can't corrupt the stream.
"""
import json
import logging
import urllib.parse
from typing import Annotated, Literal

import httpx
from pydantic import Field

from config import RAPIDAPI_KEY
from prompts import wrap_untrusted

logger = logging.getLogger(__name__)

RAPIDAPI_HOST = "jsearch.p.rapidapi.com"
MAX_RESULTS = 5


async def search_jobs(
    query: Annotated[
        str,
        Field(description="Role and optional location in plain words, e.g. 'Python developer in Bangalore'."),
    ],
    country: Annotated[
        str,
        Field(description="ISO 3166-1 alpha-2 country code to search in, e.g. 'in', 'us', 'ca'."),
    ] = "in",
    date_posted: Literal["all", "today", "3days", "week", "month"] = "week",
    experience_level: Literal[
        "no_experience",
        "under_3_years_experience",
        "more_than_3_years_experience",
        "no_degree",
    ] = "under_3_years_experience",
) -> str:
    """Find open job listings for a role, optionally in a specific location.

    Returns up to 5 listings, each with title, company, location, the employer's
    apply link when one exists, and LinkedIn / Naukri / Indeed search links.
    Use this for any request to find jobs, openings or vacancies.
    """
    if not RAPIDAPI_KEY:
        return "Job search is unavailable: RAPIDAPI_KEY is not configured."
    if not query.strip():
        return "Provide a role or keywords to search for, e.g. 'data analyst in Pune'."

    params = {
        "query": query,
        "page": 1,
        "num_pages": 1,
        "country": country.lower(),
        "date_posted": date_posted,
        "job_requirements": experience_level,
    }
    try:
        # JSearch routinely takes 10-20s; stay under Lambda's 30s limit.
        async with httpx.AsyncClient(timeout=25) as client:
            response = await client.get(
                f"https://{RAPIDAPI_HOST}/search",
                params=params,
                headers={"x-rapidapi-host": RAPIDAPI_HOST, "x-rapidapi-key": RAPIDAPI_KEY},
            )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        logger.error(
            "JSearch request failed status_code=%s body=%s",
            exc.response.status_code,
            exc.response.text[:500],
        )
        return f"Job search failed (JSearch returned HTTP {exc.response.status_code}). Try again later."
    except httpx.HTTPError as exc:
        logger.error("JSearch request failed: %s", exc)
        return "Job search failed (JSearch could not be reached). Try again later."

    jobs = response.json().get("data", [])
    if not jobs:
        return (
            f"No listings found for {query!r}. Try broader keywords, "
            "date_posted='month' or 'all', or a different experience_level."
        )

    listings = [_to_listing(job) for job in jobs[:MAX_RESULTS]]
    return wrap_untrusted("job_listings", json.dumps(listings))


def _to_listing(job: dict) -> dict:
    """Keep only the fields the agent needs from one JSearch result."""
    title = job.get("job_title") or "Untitled role"
    company = job.get("employer_name") or "Unknown company"
    location = ", ".join(filter(None, [job.get("job_city"), job.get("job_country")]))
    if not location:
        location = "Remote" if job.get("job_is_remote") else "Unspecified"

    search_term = f"{title} {company}"
    return {
        "title": title,
        "company": company,
        "location": location,
        "apply_link": job.get("job_apply_link"),
        "search_links": {
            "LinkedIn": "https://www.linkedin.com/jobs/search/?"
            + urllib.parse.urlencode({"keywords": search_term}),
            "Naukri": "https://www.naukri.com/" + urllib.parse.quote_plus(search_term),
            "Indeed": "https://www.indeed.com/jobs?" + urllib.parse.urlencode({"q": search_term}),
        },
    }
