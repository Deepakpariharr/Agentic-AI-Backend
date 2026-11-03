"""RAGService's Redis retrieval cache: keys, hits, invalidation, and failing safe."""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.documents import Document

from config import RAG_CACHE_TTL_SECONDS, REDIS_KEY_PREFIX, RERANK_TOP_N, RETRIEVAL_TOP_K
from services.rag import RAGService
from services.rag import service as rag_module
from services.rag.service import _doc_version_key, _generate_cache_key, citation_label
from tests.local_redis import BrokenRedis, FakeRedis


def _service(redis, candidates=None):
    retriever = SimpleNamespace(
        search=AsyncMock(return_value=candidates or [Document(page_content="chunk", metadata={"source_id": "d"})]),
        add_chunks=AsyncMock(return_value=True),
        delete_document=AsyncMock(return_value=0),
    )
    service = RAGService(retriever=retriever, redis_url=None)
    service.redis = redis
    service.reranker = SimpleNamespace(rerank=AsyncMock(side_effect=lambda q, docs, top_n: docs))
    return service, retriever


def test_keys_are_namespaced_by_environment():
    key = _generate_cache_key("u1", " Hello ", RETRIEVAL_TOP_K, RERANK_TOP_N, 3)
    assert key.startswith(f"{REDIS_KEY_PREFIX}:rag:u1:v3:")
    assert key == _generate_cache_key("u1", "hello", RETRIEVAL_TOP_K, RERANK_TOP_N, 3)  # normalised
    assert _doc_version_key("u1").startswith(f"{REDIS_KEY_PREFIX}:session:u1:")


def test_users_never_share_cache_keys():
    assert _generate_cache_key("u1", "q", 40, 5, 0) != _generate_cache_key("u2", "q", 40, 5, 0)


def test_no_redis_url_means_no_cache():
    assert RAGService(retriever=SimpleNamespace(), redis_url=None).redis is None


def test_redis_client_gets_timeouts(monkeypatch):
    from_url = MagicMock()
    monkeypatch.setattr(rag_module.Redis, "from_url", from_url)

    rag_module._build_redis("rediss://host:6379")

    kwargs = from_url.call_args.kwargs
    assert kwargs["socket_connect_timeout"] and kwargs["socket_timeout"]  # an outage must not hang a request
    assert kwargs["health_check_interval"] == 30


async def test_miss_then_hit():
    redis = FakeRedis()
    service, retriever = _service(redis)

    first = await service.retrieve("skills?", "u1")
    second = await service.retrieve("skills?", "u1")

    assert [d.page_content for d in first] == [d.page_content for d in second] == ["chunk"]
    retriever.search.assert_awaited_once()  # the second call came from Redis
    cached_key = next(k for k in redis.store if ":rag:u1:" in k)
    assert redis.ttls[cached_key] == RAG_CACHE_TTL_SECONDS


async def test_cache_hit_skips_search_and_rerank():
    redis = FakeRedis()
    service, retriever = _service(redis)
    key = _generate_cache_key("u1", "q", RETRIEVAL_TOP_K, RERANK_TOP_N, 0)
    redis.store[key] = json.dumps([{"page_content": "cached", "metadata": {"source_id": "d"}}])

    docs = await service.retrieve("q", "u1")

    assert docs[0].page_content == "cached"
    retriever.search.assert_not_awaited()
    service.reranker.rerank.assert_not_awaited()


async def test_other_users_do_not_hit_this_users_cache():
    redis = FakeRedis()
    service, retriever = _service(redis)

    await service.retrieve("q", "u1")
    await service.retrieve("q", "u2")

    assert retriever.search.await_count == 2


async def test_ingest_invalidates_the_users_cache():
    redis = FakeRedis()
    service, retriever = _service(redis)
    await service.retrieve("q", "u1")

    await service.ingest_document("some text to index", "u1", "hash1", file_name="cv.pdf")
    await service.retrieve("q", "u1")

    assert redis.store[_doc_version_key("u1")] == "1"
    assert retriever.search.await_count == 2  # the new version made the old entry unreachable


async def test_ingest_passes_the_file_name_to_the_chunks():
    service, retriever = _service(FakeRedis())

    await service.ingest_document("some text to index", "u1", "hash1", file_name="cv.pdf")

    docs = retriever.add_chunks.await_args.args[0]
    assert {d.metadata["file_name"] for d in docs} == {"cv.pdf"}
    assert citation_label(docs[0]) == "cv.pdf#0"


def test_citations_fall_back_to_the_source_id():
    doc = Document(page_content="x", metadata={"source_id": "abc123", "chunk_index": 2})
    assert citation_label(doc) == "abc123#2"


async def test_delete_removes_cached_text_immediately():
    redis = FakeRedis()
    service, retriever = _service(redis)
    await service.retrieve("q", "u1")
    await service.retrieve("q", "u2")
    retriever.delete_document.return_value = 3

    assert await service.delete_document("u1", "hash1") == 3

    assert not any(":rag:u1:" in key for key in redis.store)  # u1's cached chunks are gone
    assert any(":rag:u2:" in key for key in redis.store)  # other users untouched
    assert redis.store[_doc_version_key("u1")] == "1"


async def test_delete_of_unknown_document_leaves_the_cache_alone():
    redis = FakeRedis()
    service, _ = _service(redis)
    await service.retrieve("q", "u1")

    assert await service.delete_document("u1", "nope") == 0

    assert any(":rag:u1:" in key for key in redis.store)


async def test_retrieval_still_works_when_redis_is_down():
    service, retriever = _service(BrokenRedis())

    docs = await service.retrieve("q", "u1")

    assert docs[0].page_content == "chunk"
    retriever.search.assert_awaited_once()


async def test_delete_still_works_when_redis_is_down():
    service, retriever = _service(BrokenRedis())
    retriever.delete_document.return_value = 2

    assert await service.delete_document("u1", "hash1") == 2


async def test_unreadable_cache_entry_falls_back_to_search():
    redis = FakeRedis()
    service, retriever = _service(redis)
    redis.store[_generate_cache_key("u1", "q", RETRIEVAL_TOP_K, RERANK_TOP_N, 0)] = "not json"

    docs = await service.retrieve("q", "u1")

    assert docs[0].page_content == "chunk"
    retriever.search.assert_awaited_once()
