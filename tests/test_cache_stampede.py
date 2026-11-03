import asyncio
import uuid

import pytest
from langchain_core.documents import Document

from services.rag import RAGService
from tests.local_redis import FakeRedis


class MockRetriever:
    def __init__(self):
        self.search_call_count = 0

    async def search(self, query: str, session_id: str, top_k: int):
        self.search_call_count += 1
        return [
            Document(
                page_content=f"Mocked data for {query}",
                metadata={"source_id": "doc_1"},
            )
        ]


class MockReranker:
    def __init__(self):
        self.rerank_call_count = 0

    async def rerank(self, query: str, chunks: list, top_n: int):
        self.rerank_call_count += 1
        await asyncio.sleep(0.1)
        return chunks


@pytest.mark.asyncio
async def test_cache_stampede_prevention():
    """10 simultaneous identical requests must trigger only 1 search + 1 rerank."""
    mock_retriever = MockRetriever()
    mock_reranker = MockReranker()

    rag_service = RAGService(retriever=mock_retriever, redis_url=None)
    rag_service.redis = FakeRedis()  # shared lock + cache, no real server needed
    rag_service.reranker = mock_reranker

    session_id = f"test_user_{uuid.uuid4().hex}"
    query = "What is the holiday policy?"

    results = await asyncio.gather(*[rag_service.retrieve(query, session_id) for _ in range(10)])

    assert len(results) == 10
    for res in results:
        assert res[0].page_content == "Mocked data for What is the holiday policy?"

    assert mock_retriever.search_call_count == 1, (
        f"Expected 1 DB call, got {mock_retriever.search_call_count}"
    )
    assert mock_reranker.rerank_call_count == 1
