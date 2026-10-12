"""GET /documents and DELETE /documents/{source_id}. The RAG service is mocked;
these check auth, per-user scoping and response shape.

Handlers return a ResponseModel, so errors come back as HTTP 200 with the real
status in the body's `statusCode`.
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from dependencies import get_rag_service


@pytest.fixture
def documents():
    """TestClient whose requests carry `user` the way firebase_auth_middleware
    would set it, with the RAG service swapped for a mock."""
    from main import app

    rag = SimpleNamespace(list_documents=AsyncMock(), delete_document=AsyncMock())
    app.dependency_overrides[get_rag_service] = lambda: rag

    def client_for(user):
        async def with_user(scope, receive, send):
            if scope["type"] == "http" and user is not None:
                scope = {**scope, "state": {"user": user}}
            await app(scope, receive, send)

        return TestClient(with_user)

    yield client_for, rag
    app.dependency_overrides.pop(get_rag_service)


# ── GET /documents ───────────────────────────────────────────────────────────

def test_list_documents_returns_the_users_documents(documents):
    client_for, rag = documents
    rag.list_documents.return_value = [{"source_id": "resume.pdf", "chunk_count": 12}]

    body = client_for("uid-1").get("/documents").json()

    assert body["statusCode"] == 200
    assert body["data"] == {"documents": [{"source_id": "resume.pdf", "chunk_count": 12}]}
    rag.list_documents.assert_awaited_once_with(session_id="uid-1")


def test_list_documents_reports_storage_errors(documents):
    client_for, rag = documents
    rag.list_documents.side_effect = RuntimeError("qdrant down")

    body = client_for("uid-1").get("/documents").json()

    assert body["statusCode"] == 500
    assert body["message"] == "Failed to list documents."


def test_list_documents_requires_a_user(documents):
    client_for, rag = documents
    assert client_for(None).get("/documents").status_code == 401
    rag.list_documents.assert_not_awaited()


# ── DELETE /documents/{source_id} ───────────────────────────────────────────

def test_delete_document_removes_only_the_users_copy(documents):
    client_for, rag = documents
    rag.delete_document.return_value = 12

    body = client_for("uid-1").delete("/documents/resume.pdf").json()

    assert body["statusCode"] == 200
    assert body["data"] == {"source_id": "resume.pdf", "deleted_chunks": 12}
    rag.delete_document.assert_awaited_once_with(session_id="uid-1", source_id="resume.pdf")


def test_delete_document_accepts_names_with_slashes(documents):
    client_for, rag = documents
    rag.delete_document.return_value = 1

    client_for("uid-1").delete("/documents/cv/2026/resume.pdf")

    assert rag.delete_document.await_args.kwargs["source_id"] == "cv/2026/resume.pdf"


def test_delete_unknown_document_is_not_found(documents):
    client_for, rag = documents
    rag.delete_document.return_value = 0

    body = client_for("uid-1").delete("/documents/missing.pdf").json()

    assert body["statusCode"] == 404
    assert "missing.pdf" in body["message"]


def test_delete_document_reports_storage_errors(documents):
    client_for, rag = documents
    rag.delete_document.side_effect = RuntimeError("qdrant down")

    body = client_for("uid-1").delete("/documents/resume.pdf").json()

    assert body["statusCode"] == 500
    assert body["message"] == "Failed to delete document."


def test_delete_document_requires_a_user(documents):
    client_for, rag = documents
    assert client_for(None).delete("/documents/resume.pdf").status_code == 401
    rag.delete_document.assert_not_awaited()
