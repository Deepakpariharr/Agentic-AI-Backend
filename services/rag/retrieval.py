import logging
from urllib.parse import urlparse

from langchain_core.documents import Document
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    KeywordIndexParams,
    KeywordIndexType,
    MatchValue,
    SparseVectorParams,
    VectorParams,
)

from config import (
    EMBEDDING_MODEL_NAME,
    QDRANT_API_KEY,
    QDRANT_COLLECTION_NAME,
    QDRANT_URL,
    SPARSE_EMBEDDING_MODEL,
)
from services.llm_factory import get_embedding_dimension, get_embeddings

logger = logging.getLogger(__name__)

SPARSE_VECTOR_NAME = "langchain-sparse"
EMBEDDING_MODEL_KEY = "embedding_model"
EMBEDDING_DIMENSION_KEY = "embedding_dimension"

# Payload fields that searches, listing and deletes filter on. Each needs an
# index: Qdrant Cloud's strict mode rejects filters on unindexed fields, and
# the index keeps per-user lookups fast as the collection grows.
PAYLOAD_INDEXES = {
    # is_tenant groups each user's points together on disk.
    "metadata.session_id": KeywordIndexParams(type=KeywordIndexType.KEYWORD, is_tenant=True),
    "metadata.source_id": KeywordIndexParams(type=KeywordIndexType.KEYWORD),
}


class CollectionSchemaError(RuntimeError):
    """An existing collection is incompatible with hybrid search."""


class EmbeddingMismatchError(CollectionSchemaError):
    """The collection uses a different embedding model or dimension."""


def _dense_vector_size(info) -> int | None:
    vectors = info.config.params.vectors
    if isinstance(vectors, dict):  # named-vector layout; langchain uses the "" name
        vectors = vectors.get("")
    return getattr(vectors, "size", None)


class HybridRetriever:
    def __init__(
        self,
        collection_name: str = QDRANT_COLLECTION_NAME,
        url: str | None = QDRANT_URL,
        api_key: str | None = QDRANT_API_KEY,
        client: QdrantClient | None = None,
    ):
        self.collection_name = collection_name
        # `client` lets callers (e.g. the offline eval runner) supply their own
        # QdrantClient, such as an in-memory instance, instead of connecting to
        # QDRANT_URL. Normal app startup never passes it.
        if client is None:
            if not url:
                raise ValueError(
                    "QDRANT_URL must be set; configure a Qdrant server instead of "
                    "falling back to localhost."
                )
            hostname = urlparse(url).hostname
            if hostname and hostname.endswith(".cloud.qdrant.io") and not api_key:
                raise ValueError("QDRANT_API_KEY is required for Qdrant Cloud.")
            client = QdrantClient(url=url, api_key=api_key)
        self.client = client
        self.dense_embeddings = get_embeddings()
        self.sparse_embeddings = FastEmbedSparse(model_name=SPARSE_EMBEDDING_MODEL)

        self._ensure_collection()

        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.dense_embeddings,
            sparse_embedding=self.sparse_embeddings,
            sparse_vector_name=SPARSE_VECTOR_NAME,
            retrieval_mode=RetrievalMode.HYBRID,
        )

    def _embedding_pin(self, dim: int) -> dict:
        return {EMBEDDING_MODEL_KEY: EMBEDDING_MODEL_NAME, EMBEDDING_DIMENSION_KEY: dim}

    def _verify_schema(self, expected_dim: int) -> None:
        info = self.client.get_collection(collection_name=self.collection_name)

        sparse_configs = info.config.params.sparse_vectors or {}
        if SPARSE_VECTOR_NAME not in sparse_configs:
            raise CollectionSchemaError(
                f"Collection '{self.collection_name}' lacks the required sparse vector. "
                "It was not modified. Use a new collection and re-ingest the documents."
            )

        stored_dim = _dense_vector_size(info)
        stored_model = (info.config.metadata or {}).get(EMBEDDING_MODEL_KEY)
        if stored_dim != expected_dim or (stored_model and stored_model != EMBEDDING_MODEL_NAME):
            raise EmbeddingMismatchError(
                f"Collection '{self.collection_name}' uses model '{stored_model or 'unknown'}' "
                f"(dimension {stored_dim}), expected '{EMBEDDING_MODEL_NAME}' ({expected_dim}). "
                "Use the original model or create a new collection and re-ingest."
            )

        if not stored_model:
            logger.warning(
                "Collection '%s' had no embedding pin; recording '%s' (dim %d). This assumes its "
                "vectors were created with that model.",
                self.collection_name, EMBEDDING_MODEL_NAME, expected_dim,
            )
            self.client.update_collection(
                collection_name=self.collection_name,
                metadata={**(info.config.metadata or {}), **self._embedding_pin(expected_dim)},
            )

    def _ensure_collection(self):
        # No try/except: the caller (main.py lifespan) decides whether a schema
        # or connection failure is fatal, and logs it once.
        dim = get_embedding_dimension()
        if self.client.collection_exists(collection_name=self.collection_name):
            self._verify_schema(dim)
        else:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                sparse_vectors_config={SPARSE_VECTOR_NAME: SparseVectorParams()},
                metadata=self._embedding_pin(dim),
            )
            logger.info(
                "Created hybrid Qdrant collection '%s' (model: %s, dim: %d)",
                self.collection_name, EMBEDDING_MODEL_NAME, dim,
            )
        self._ensure_payload_indexes()

    def _ensure_payload_indexes(self) -> None:
        """Create any missing filter index; existing points are indexed too."""
        info = self.client.get_collection(collection_name=self.collection_name)
        existing = info.payload_schema or {}
        for field_name, schema in PAYLOAD_INDEXES.items():
            if field_name in existing:
                continue
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=schema,
                wait=True,
            )
            logger.info("Created payload index '%s' on '%s'", field_name, self.collection_name)

    def _build_tenant_filter(self, session_id: str) -> Filter:
        return Filter(
            must=[
                FieldCondition(
                    key="metadata.session_id",
                    match=MatchValue(value=session_id)
                )
            ]
        )

    async def add_chunks(self, docs: list[Document]) -> bool:
        if not docs:
            return False
        try:
            await self.vector_store.aadd_documents(docs)
            return True
        except Exception:
            logger.exception("DB insertion failed")
            return False

    async def search(self, query: str, session_id: str, top_k: int = 40) -> list[Document]:
        # Errors propagate: returning [] here would make an outage look like
        # "no relevant documents". RAGService.search_context handles them.
        return await self.vector_store.asimilarity_search(
            query=query,
            k=top_k,
            filter=self._build_tenant_filter(session_id),
        )

    async def list_documents(self, session_id: str) -> list[dict]:
        """Distinct documents (source_id + chunk count) uploaded by this session.

        Scrolls the collection (payload-only, no vectors) rather than
        searching, since this needs an exhaustive, similarity-independent
        list of everything the tenant owns.
        """
        tenant_filter = self._build_tenant_filter(session_id)
        counts: dict = {}
        offset = None
        while True:
            records, offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=tenant_filter,
                limit=256,
                offset=offset,
                with_payload=["metadata.source_id"],
                with_vectors=False,
            )
            for record in records:
                source_id = (record.payload or {}).get("metadata", {}).get("source_id")
                if source_id:
                    counts[source_id] = counts.get(source_id, 0) + 1
            if offset is None:
                break

        return [
            {"source_id": source_id, "chunk_count": count}
            for source_id, count in sorted(counts.items())
        ]

    def _build_document_filter(self, session_id: str, source_id: str) -> Filter:
        return Filter(
            must=[
                FieldCondition(key="metadata.session_id", match=MatchValue(value=session_id)),
                FieldCondition(key="metadata.source_id", match=MatchValue(value=source_id)),
            ]
        )

    async def delete_document(self, session_id: str, source_id: str) -> int:
        """Delete every chunk of `source_id` that belongs to `session_id`.

        Scoped by both fields so one tenant can never delete another
        tenant's chunks, even if they guess a source_id. Returns the number
        of chunks that were deleted (0 if nothing matched).
        """
        doc_filter = self._build_document_filter(session_id, source_id)
        existing = self.client.count(
            collection_name=self.collection_name, count_filter=doc_filter
        ).count
        if existing == 0:
            return 0

        self.client.delete(collection_name=self.collection_name, points_selector=doc_filter)
        return existing