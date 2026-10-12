"""search_web: live web search through Tavily's REST API."""
import json
import logging
from typing import Annotated

import httpx
from pydantic import Field

from config import TAVILY_API_KEY
from prompts import wrap_untrusted

logger = logging.getLogger(__name__)

TAVILY_SEARCH_URL = "https://api.tavily.com/search"
MAX_RESULTS = 3


async def search_web(
    query: Annotated[
        str,
        Field(description="What to look up, phrased as a search query, e.g. 'latest stable Python release'."),
    ],
) -> str:
    """Search the live internet for current information.

    Use for news, company details, recent releases, market updates or any fact
    that may have changed recently. Returns a short answer plus up to 3 results
    (title, URL, snippet); cite the URLs you rely on.
    """
    if not TAVILY_API_KEY:
        return "Web search is unavailable: TAVILY_API_KEY is not configured."

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                TAVILY_SEARCH_URL,
                json={
                    "api_key": TAVILY_API_KEY,
                    "query": query,
                    "max_results": MAX_RESULTS,
                    "search_depth": "basic",
                    "include_answer": True,
                    "include_raw_content": False,
                    "include_images": False,
                },
            )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        logger.error("Tavily search failed: %s", exc)
        return "Web search failed. Try again later."

    payload = response.json()
    results = {
        "answer": payload.get("answer"),
        "results": [
            {"title": r.get("title"), "url": r.get("url"), "content": r.get("content")}
            for r in payload.get("results", [])
        ],
    }
    return wrap_untrusted("web_search", json.dumps(results))
