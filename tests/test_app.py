"""App wiring: startup/shutdown (main.lifespan), the Lambda entry point, the
Depends() singletons and the request-id middleware's error path."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from mangum import Mangum

import dependencies
import main
from middleware.request_id import request_id_middleware
from services.rag.retrieval import CollectionSchemaError

# ── main.lifespan ────────────────────────────────────────────────────────────

@pytest.fixture
def close_mcp(monkeypatch):
    close = AsyncMock()
    monkeypatch.setattr(main, "close_mcp_client", close)
    return close


async def test_lifespan_starts_rag_and_closes_mcp_on_shutdown(monkeypatch, close_mcp):
    get_rag_service = MagicMock()
    monkeypatch.setattr(main, "get_rag_service", get_rag_service)

    async with main.lifespan(main.app):
        get_rag_service.assert_called_once()
        close_mcp.assert_not_awaited()

    close_mcp.assert_awaited_once()


async def test_lifespan_runs_without_rag_when_qdrant_is_unreachable(monkeypatch, close_mcp):
    monkeypatch.setattr(main, "get_rag_service", MagicMock(side_effect=ConnectionError("refused")))

    async with main.lifespan(main.app):
        pass

    close_mcp.assert_awaited_once()


async def test_lifespan_refuses_to_start_on_incompatible_collection(monkeypatch, close_mcp):
    monkeypatch.setattr(
        main, "get_rag_service", MagicMock(side_effect=CollectionSchemaError("wrong model"))
    )

    with pytest.raises(CollectionSchemaError):
        async with main.lifespan(main.app):
            pytest.fail("app must not start")

    close_mcp.assert_not_awaited()


# ── lambda_handler ───────────────────────────────────────────────────────────

def _http_api_event(path, request_id):
    """Minimal API Gateway HTTP API (payload v2.0) event."""
    return {
        "version": "2.0",
        "routeKey": "$default",
        "rawPath": path,
        "rawQueryString": "",
        "headers": {"host": "api.example.com"},
        "requestContext": {
            "http": {
                "method": "GET",
                "path": path,
                "protocol": "HTTP/1.1",
                "sourceIp": "203.0.113.7",
                "userAgent": "pytest",
            },
            "requestId": request_id,
            "stage": "$default",
        },
        "isBase64Encoded": False,
    }


_LAMBDA_CONTEXT = SimpleNamespace(
    function_name="mcpbe",
    function_version="$LATEST",
    memory_limit_in_mb=512,
    invoked_function_arn="arn:aws:lambda:us-east-1:123456789012:function:mcpbe",
    aws_request_id="lambda-req-1",
)


@pytest.fixture
def lambda_event_loop():
    """Mangum drives the app on `asyncio.get_event_loop()`. The Lambda runtime
    always has one; Python 3.14 no longer creates it on demand, so set one up."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield
    asyncio.set_event_loop(None)
    loop.close()


def test_lambda_handler_serves_the_app_with_the_gateway_request_id(monkeypatch, lambda_event_loop):
    import lambda_handler

    # lifespan off: startup would try to reach Qdrant.
    monkeypatch.setattr(lambda_handler, "mangum_handler", Mangum(main.app, lifespan="off"))

    result = lambda_handler.handler(_http_api_event("/", "gw-req-42"), _LAMBDA_CONTEXT)

    assert result["statusCode"] == 200
    headers = {name.lower(): value for name, value in result["headers"].items()}
    assert headers["x-request-id"] == "gw-req-42"


# ── dependencies ─────────────────────────────────────────────────────────────

@pytest.fixture
def fresh_singletons(monkeypatch):
    monkeypatch.setattr(dependencies, "_rag_service_instance", None)
    monkeypatch.setattr(dependencies, "_rag_agent_instance", None)
    rag_service = MagicMock(side_effect=lambda: SimpleNamespace(kind="rag"))
    rag_agent = MagicMock(side_effect=lambda rag_service: SimpleNamespace(rag=rag_service))
    monkeypatch.setattr(dependencies, "RAGService", rag_service)
    monkeypatch.setattr(dependencies, "RagAgentService", rag_agent)
    return rag_service, rag_agent


def test_rag_service_is_built_once(fresh_singletons):
    rag_service, _ = fresh_singletons
    assert dependencies.get_rag_service() is dependencies.get_rag_service()
    rag_service.assert_called_once()


def test_rag_agent_is_built_once_on_the_shared_rag_service(fresh_singletons):
    _, rag_agent = fresh_singletons

    agent = dependencies.get_rag_agent()

    assert dependencies.get_rag_agent() is agent
    assert agent.rag is dependencies.get_rag_service()
    rag_agent.assert_called_once()


# ── request_id middleware ────────────────────────────────────────────────────

def test_request_id_middleware_reraises_unhandled_errors():
    app = FastAPI()
    app.middleware("http")(request_id_middleware)

    @app.get("/boom")
    async def boom():
        raise RuntimeError("kaboom")

    with pytest.raises(RuntimeError, match="kaboom"):
        TestClient(app).get("/boom")
