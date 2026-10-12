import hashlib
import json
import logging
import typing

from langchain_core.documents import Document
from redis import exceptions as redis_exceptions
from redis.asyncio import Redis

from config import REDIS_URL, RERANK_TOP_N, RETRIEVAL_TOP_K
from prompts import wrap_untrusted

from .ingestion import DocumentIngestor
from .reranker import get_reranker
from .retrieval import HybridRetriever

logger = logging.getLogger(__name__)

NO_CONTEXT_MESSAGE = "No relevant context found in uploaded documents."


def citation_label(doc: Document) -> str:
    """Stable, human-readable label the model must cite, e.g. 'resume.pdf#3'."""
    source = doc.metadata.get("source_id", "unknown")
    chunk = doc.metadata.get("chunk_index", "?")
    return f"{source}#{chunk}"


def format_context(docs: list[Document]) -> str:
    """Render reranked chunks as labelled excerpts inside the injection guard."""
    excerpts = [f"[{citation_label(doc)}]\n{doc.page_content}" for doc in docs]
    return wrap_untrusted("uploaded_documents", "\n\n---\n\n".join(excerpts))


def _generate_cache_key(
    session_id: str,
    query: str,
    retrieval_top_k: int,
    top_n: int,
    version: int,
) -> str:
    normalized_query = query.strip().lower()
    query_hash = hashlib.sha256(normalized_query.encode()).hexdigest()
    return f"rag:{session_id}:v{version}:{query_hash}:k{retrieval_top_k}:n{top_n}"


class RAGService:
    def __init__(
        self,
        retriever: typing.Optional[HybridRetriever] = None,
        redis_url: str | None = REDIS_URL,
    ):
        # `retriever` lets callers (e.g. the offline eval runner) inject a
        # HybridRetriever pointed at an isolated Qdrant instance. Normal app
        # startup omits it and gets the real, configured one.
        self.ingestor = DocumentIngestor()
        self.retriever = retriever if retriever is not None else HybridRetriever()
        self.reranker = get_reranker()
        self.redis = (
            Redis.from_url(redis_url, decode_responses=True) if redis_url else None
        )

    async def _get_doc_version(self, session_id: str) -> int:
        if self.redis is None:
            return 0
        try:
            version = await self.redis.get(f"session:{session_id}:doc_version")
        except redis_exceptions.RedisError:
            logger.exception("Redis version lookup failed; bypassing cache.")
            return 0
        return int(version) if version else 0

    async def _bump_doc_version(self, session_id: str) -> None:
        if self.redis is None:
            return
        try:
            await self.redis.incr(f"session:{session_id}:doc_version")
        except redis_exceptions.RedisError:
            logger.exception("Redis version update failed; cached results may persist.")

    async def ingest_document(
        self, raw_text: str, session_id: str, source_id: str
    ) -> str:
        try:
            docs = self.ingestor.process_document(raw_text, session_id, source_id)
            success = await self.retriever.add_chunks(docs)
            if success:
                await self._bump_doc_version(session_id)
                return f"Successfully ingested {len(docs)} chunks from {source_id}."
            return "Failed to ingest document into vector database."
        except Exception as e:
            logger.exception(f"Ingestion pipeline error: {e}")
            return "Critical error during document ingestion."

    async def retrieve(self, query: str, session_id: str) -> list[Document]:
        """Hybrid search + rerank, returning the ranked Document chunks themselves.

        Used both by `search_context` (which formats them for the chat prompt)
        and by the eval runner (which needs the raw, ordered chunks to score
        retrieval quality).
        """
        version = await self._get_doc_version(session_id)
        cache_key = _generate_cache_key(
            session_id, query, RETRIEVAL_TOP_K, RERANK_TOP_N, version
        )

        if self.redis is not None:
            try:
                cached_data = await self.redis.get(cache_key)
                if cached_data:
                    logger.info("Cache hit for query in session %s", session_id)
                    return self._deserialize_documents(cached_data)
            except (redis_exceptions.RedisError, json.JSONDecodeError, KeyError, TypeError):
                logger.exception("Redis cache read failed; bypassing cache.")

        lock = None
        acquired = False
        try:
            if self.redis is not None:
                lock = self.redis.lock(
                    f"lock:{cache_key}", timeout=10, blocking_timeout=5
                )
                try:
                    acquired = await lock.acquire()
                except redis_exceptions.RedisError:
                    logger.exception("Redis lock failed; proceeding without cache.")

                if acquired:
                    try:
                        cached_data = await self.redis.get(cache_key)
                        if cached_data:
                            logger.info(
                                "Cache hit after lock for session %s", session_id
                            )
                            return self._deserialize_documents(cached_data)
                    except (
                        redis_exceptions.RedisError,
                        json.JSONDecodeError,
                        KeyError,
                        TypeError,
                    ):
                        logger.exception(
                            "Redis cache recheck failed; proceeding with retrieval."
                        )

            logger.info("Cache miss. Executing Qdrant search and reranker.")
            candidates = await self.retriever.search(
                query, session_id, top_k=RETRIEVAL_TOP_K
            )
            if not candidates:
                return []

            docs = await self.reranker.rerank(
                query, candidates, top_n=RERANK_TOP_N
            )

            if self.redis is not None:
                try:
                    serialized_docs = json.dumps(
                        [
                            {
                                "page_content": doc.page_content,
                                "metadata": doc.metadata,
                            }
                            for doc in docs
                        ]
                    )
                    await self.redis.set(cache_key, serialized_docs, ex=7200)
                except (redis_exceptions.RedisError, TypeError, ValueError):
                    logger.exception("Redis cache write failed.")

            return docs

        finally:
            if acquired and lock is not None:
                try:
                    await lock.release()
                except redis_exceptions.LockError:
                    logger.warning("Redis cache lock expired before it was released.")

    @staticmethod
    def _deserialize_documents(cached_data: str) -> list[Document]:
        return [
            Document(page_content=doc["page_content"], metadata=doc["metadata"])
            for doc in json.loads(cached_data)
        ]

    async def list_documents(self, session_id: str) -> list[dict]:
        """Documents this session has uploaded, as [{source_id, chunk_count}, ...]."""
        return await self.retriever.list_documents(session_id)

    async def delete_document(self, session_id: str, source_id: str) -> int:
        """Delete a previously uploaded document's chunks for one user.

        Returns the number of chunks deleted; 0 means no document with that
        source_id exists for this session_id. Tenant-scoped the same way
        search is, so a session can only ever delete its own documents.
        """
        deleted_count = await self.retriever.delete_document(session_id, source_id)
        if deleted_count > 0:
            await self._bump_doc_version(session_id)
        return deleted_count

    async def search_context(self, query: str, session_id: str) -> str:
        try:
            docs = await self.retrieve(query, session_id)
            if not docs:
                return NO_CONTEXT_MESSAGE
            return format_context(docs)
        except Exception as e:
            logger.exception(f"Search pipeline error: {e}")
            return "Critical error during context retrieval."
