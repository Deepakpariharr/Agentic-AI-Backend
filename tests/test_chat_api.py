"""Endpoint tests for the FastAPI app. The agent and OpenAI calls are mocked —
these check request validation, response shape and rate limiting.

Note: handlers return a ResponseModel, so validation/server errors come back as
HTTP 200 with the real status in the body's `statusCode`.
"""
from unittest.mock import AsyncMock

import pytest

from controllers import chat_controller
from utility.request_limits import MAX_FILE_SIZE_BYTES


@pytest.fixture
def mock_agent(monkeypatch):
    mock = AsyncMock(return_value="Here are some jobs")
    monkeypatch.setattr(chat_controller, "agent_response", mock)
    return mock


def test_health_check(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["data"]["status"] == "healthy"


def test_agent_chat_success(client, mock_agent):
    response = client.post("/chat/mcpAgentChat", data={"message": "  find python jobs  "})

    body = response.json()
    assert body["statusCode"] == 200
    assert body["data"] == {"response": "Here are some jobs"}
    args, kwargs = mock_agent.call_args
    assert args[0] == "find python jobs"
    assert args[1] is None
    assert kwargs["user_id"] == "testclient"  # falls back to client IP without auth


def test_agent_chat_passes_pdf_through(client, mock_agent):
    files = {"file": ("resume.pdf", b"%PDF-1.4 fake", "application/pdf")}
    response = client.post("/chat/mcpAgentChat", data={"message": "review"}, files=files)

    assert response.json()["statusCode"] == 200
    uploaded = mock_agent.call_args.args[1]
    assert uploaded.filename == "resume.pdf"


@pytest.mark.parametrize(
    "message, expected",
    [
        ("   ", "Message should not be blank"),
        ("x" * 4001, "Message must be at most 4000 characters"),
    ],
)
def test_agent_chat_rejects_bad_messages(client, mock_agent, message, expected):
    body = client.post("/chat/mcpAgentChat", data={"message": message}).json()
    assert body["statusCode"] == 400
    assert body["message"] == expected
    mock_agent.assert_not_called()


def test_agent_chat_rejects_non_pdf(client, mock_agent):
    files = {"file": ("resume.docx", b"data", "application/octet-stream")}
    body = client.post("/chat/mcpAgentChat", data={"message": "hi"}, files=files).json()
    assert body == {"statusCode": 400, "message": "Only PDF files are allowed", "data": None}
    mock_agent.assert_not_called()


def test_chat_limits_match_what_is_enforced(client):
    body = client.get("/chat/limits").json()

    assert body["statusCode"] == 200
    assert body["data"] == {
        "maxQueryChars": 4000,
        "maxFileSizeBytes": MAX_FILE_SIZE_BYTES,
        "maxFiles": 3,
        "allowedFileTypes": [".pdf"],
    }


def test_agent_chat_rejects_large_pdf(client, mock_agent):
    too_big = b"0" * (MAX_FILE_SIZE_BYTES + 1)
    files = {"file": ("resume.pdf", too_big, "application/pdf")}
    body = client.post("/chat/mcpAgentChat", data={"message": "hi"}, files=files).json()
    assert body["statusCode"] == 400
    assert body["message"] == "'resume.pdf' is larger than 1 MB"
    mock_agent.assert_not_called()


def test_agent_chat_empty_result_is_an_error(client, mock_agent):
    mock_agent.return_value = ""
    body = client.post("/chat/mcpAgentChat", data={"message": "hi"}).json()
    assert body["statusCode"] == 500


def test_agent_chat_hides_exception_details(client, mock_agent):
    mock_agent.side_effect = RuntimeError("secret internal detail")
    body = client.post("/chat/mcpAgentChat", data={"message": "hi"}).json()
    assert body["statusCode"] == 500
    assert "secret internal detail" not in body["message"]


def test_agent_chat_is_rate_limited(client, mock_agent):
    for _ in range(5):
        assert client.post("/chat/mcpAgentChat", data={"message": "hi"}).status_code == 200

    response = client.post("/chat/mcpAgentChat", data={"message": "hi"})
    assert response.status_code == 429
    assert response.headers["Retry-After"] == "60"


def test_openai_chat_success(client, monkeypatch):
    monkeypatch.setattr(chat_controller, "openai_response", lambda message: f"echo: {message}")
    body = client.post("/chat/openAIChat", json={"message": "hello"}).json()
    assert body["statusCode"] == 200
    assert body["data"] == {"response": "echo: hello"}


def test_openai_chat_reports_failures(client, monkeypatch):
    def fail(message):
        raise RuntimeError("upstream down")

    monkeypatch.setattr(chat_controller, "openai_response", fail)
    body = client.post("/chat/openAIChat", json={"message": "hello"}).json()
    assert body["statusCode"] == 500
    assert body["data"] is None


def test_openai_chat_requires_message(client):
    assert client.post("/chat/openAIChat", json={}).status_code == 422
