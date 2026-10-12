import base64
import json

import fitz
import httpx
import pytest

from mcp_server.tools import jobs, web_search
from mcp_server.tools.calculator import calculate
from services.pdf_reader import read_pdf


def _pdf_base64(text: str | None) -> str:
    doc = fitz.open()
    page = doc.new_page()
    if text:
        page.insert_text((72, 72), text)
    return base64.b64encode(doc.tobytes()).decode()


def _unwrap(result: str, source: str):
    """Body of a wrap_untrusted() block, parsed as JSON."""
    prefix = f'<untrusted_data source="{source}">\n'
    assert result.startswith(prefix) and result.endswith("\n</untrusted_data>")
    return json.loads(result[len(prefix) : -len("\n</untrusted_data>")])


@pytest.fixture
def http(monkeypatch):
    """Route every httpx.AsyncClient the tools open to a fake transport."""
    state = {"requests": [], "response": httpx.Response(200, json={})}

    def handler(request):
        state["requests"].append(request)
        response = state["response"]
        if isinstance(response, Exception):
            raise response
        return response

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs)
    )
    return state


# ── calculator ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "expression, expected",
    [
        ("2 + 3 * 5", "17"),
        ("(2 + 3) * 5", "25"),
        ("10 / 4", "2.5"),
        ("2.5 - 0.5", "2.0"),
    ],
)
def test_calculate_evaluates_arithmetic(expression, expected):
    assert calculate(expression) == expected


@pytest.mark.parametrize(
    "expression",
    ["__import__('os')", "abs(-1)", "2 ** a", "1; 2", "9**9**9**9"],
)
def test_calculate_rejects_non_arithmetic_input(expression):
    assert calculate(expression).startswith("❌ Invalid expression")


def test_calculate_reports_evaluation_errors():
    assert calculate("1 / 0").startswith("⚠️ Error evaluating expression")


# ── pdf reader ───────────────────────────────────────────────────────────────

def test_read_pdf_extracts_text():
    assert "Senior Python Developer" in read_pdf(_pdf_base64("Senior Python Developer"))


def test_read_pdf_without_text():
    assert read_pdf(_pdf_base64(None)) == "No readable text found in the PDF."


def test_read_pdf_with_invalid_content():
    assert read_pdf(base64.b64encode(b"not a pdf").decode()).startswith("Error reading PDF")


# ── search_jobs ──────────────────────────────────────────────────────────────

@pytest.fixture
def rapidapi_key(monkeypatch):
    monkeypatch.setattr(jobs, "RAPIDAPI_KEY", "rapid-key")


async def test_search_jobs_returns_top_five_listings(http, rapidapi_key):
    listings = [
        {
            "job_title": f"Dev {i}",
            "employer_name": "Acme",
            "job_city": "Pune",
            "job_country": "IN",
            "job_apply_link": f"https://apply/{i}",
        }
        for i in range(8)
    ]
    http["response"] = httpx.Response(200, json={"data": listings})

    result = _unwrap(
        await jobs.search_jobs("python developer", country="IN", date_posted="today"),
        "job_listings",
    )

    assert [job["title"] for job in result] == [f"Dev {i}" for i in range(5)]
    assert result[0]["location"] == "Pune, IN"
    assert result[0]["apply_link"] == "https://apply/0"
    assert set(result[0]["search_links"]) == {"LinkedIn", "Naukri", "Indeed"}
    assert "Dev+0+Acme" in result[0]["search_links"]["Indeed"]

    request = http["requests"][0]
    assert request.headers["x-rapidapi-key"] == "rapid-key"
    assert request.url.params["country"] == "in"
    assert request.url.params["date_posted"] == "today"
    assert request.url.params["job_requirements"] == "under_3_years_experience"


async def test_search_jobs_fills_missing_fields(http, rapidapi_key):
    http["response"] = httpx.Response(200, json={"data": [{"job_is_remote": True}, {}]})

    remote, unknown = _unwrap(await jobs.search_jobs("anything"), "job_listings")

    assert remote["location"] == "Remote"
    assert unknown == {**unknown, "title": "Untitled role", "company": "Unknown company", "location": "Unspecified"}
    assert unknown["apply_link"] is None


async def test_search_jobs_suggests_broader_filters_when_empty(http, rapidapi_key):
    http["response"] = httpx.Response(200, json={"data": []})
    result = await jobs.search_jobs("rare role")
    assert result.startswith("No listings found for 'rare role'")
    assert "date_posted" in result


async def test_search_jobs_reports_upstream_status(http, rapidapi_key):
    http["response"] = httpx.Response(429, text="Too many requests")
    assert await jobs.search_jobs("python") == (
        "Job search failed (JSearch returned HTTP 429). Try again later."
    )


async def test_search_jobs_reports_network_errors(http, rapidapi_key):
    http["response"] = httpx.ConnectTimeout("slow")
    assert (await jobs.search_jobs("python")).startswith("Job search failed (JSearch could not be reached)")


async def test_search_jobs_needs_a_query(http, rapidapi_key):
    assert (await jobs.search_jobs("   ")).startswith("Provide a role or keywords")
    assert http["requests"] == []


async def test_search_jobs_without_key(http, monkeypatch):
    monkeypatch.setattr(jobs, "RAPIDAPI_KEY", None)
    assert "RAPIDAPI_KEY is not configured" in await jobs.search_jobs("python")
    assert http["requests"] == []


# ── search_web ───────────────────────────────────────────────────────────────

@pytest.fixture
def tavily_key(monkeypatch):
    monkeypatch.setattr(web_search, "TAVILY_API_KEY", "tvly-key")


async def test_search_web_returns_answer_and_trimmed_results(http, tavily_key):
    http["response"] = httpx.Response(
        200,
        json={
            "answer": "Python 3.14",
            "results": [{"title": "Python", "url": "https://python.org", "content": "3.14", "score": 0.9}],
        },
    )

    result = _unwrap(await web_search.search_web("latest python"), "web_search")

    assert result == {
        "answer": "Python 3.14",
        "results": [{"title": "Python", "url": "https://python.org", "content": "3.14"}],
    }
    body = json.loads(http["requests"][0].content)
    assert body["query"] == "latest python"
    assert body["api_key"] == "tvly-key"
    assert body["max_results"] == web_search.MAX_RESULTS


async def test_search_web_reports_failures(http, tavily_key):
    http["response"] = httpx.Response(500)
    assert await web_search.search_web("anything") == "Web search failed. Try again later."


async def test_search_web_without_key(http, monkeypatch):
    monkeypatch.setattr(web_search, "TAVILY_API_KEY", None)
    assert "TAVILY_API_KEY is not configured" in await web_search.search_web("anything")
    assert http["requests"] == []


# ── MCP server ───────────────────────────────────────────────────────────────

async def test_mcp_server_exposes_expected_tools():
    from mcp_server.server import mcp

    tools = {tool.name: tool for tool in await mcp.list_tools()}

    assert set(tools) == {"calculate", "search_jobs", "search_web"}
    assert all(tool.annotations.readOnlyHint for tool in tools.values())
    assert set(tools["search_jobs"].inputSchema["properties"]) == {
        "query", "country", "date_posted", "experience_level",
    }
