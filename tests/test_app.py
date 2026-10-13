"""App wiring: startup/shutdown, Depends() singletons and request-id middleware."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

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
