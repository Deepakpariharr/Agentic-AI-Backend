from unittest.mock import MagicMock

import pytest

from services.rag import retrieval


def _stub_retriever_setup(monkeypatch):
    monkeypatch.setattr(retrieval, "get_embeddings", lambda: object())
    monkeypatch.setattr(retrieval, "FastEmbedSparse", lambda **kwargs: object())
    monkeypatch.setattr(retrieval.HybridRetriever, "_ensure_collection", lambda self: None)
    monkeypatch.setattr(retrieval, "QdrantVectorStore", MagicMock())


def test_retriever_connects_to_cloud_api_with_configured_key(monkeypatch):
    _stub_retriever_setup(monkeypatch)
    client = MagicMock()
    client_factory = MagicMock(return_value=client)
    monkeypatch.setattr(retrieval, "QdrantClient", client_factory)

    retriever = retrieval.HybridRetriever(
        url="https://cluster.cloud.qdrant.io:6333",
        api_key="test-api-key",
    )

    client_factory.assert_called_once_with(
        url="https://cluster.cloud.qdrant.io:6333",
        api_key="test-api-key",
    )
    assert retriever.client is client


def test_retriever_requires_an_explicit_url(monkeypatch):
    client_factory = MagicMock()
    monkeypatch.setattr(retrieval, "QdrantClient", client_factory)

    with pytest.raises(ValueError, match="QDRANT_URL must be set"):
        retrieval.HybridRetriever(url=None)

    client_factory.assert_not_called()


def test_retriever_requires_api_key_for_qdrant_cloud(monkeypatch):
    client_factory = MagicMock()
    monkeypatch.setattr(retrieval, "QdrantClient", client_factory)

    with pytest.raises(ValueError, match="QDRANT_API_KEY is required"):
        retrieval.HybridRetriever(url="https://cluster.cloud.qdrant.io:6333")

    client_factory.assert_not_called()
