"""Pure, document-type-agnostic retrieval metrics.

A "chunk id" is whatever hashable value identifies one retrieved chunk
(the runner uses `(source_id, chunk_index)`). These functions know nothing
about documents, sessions, or the RAG pipeline itself -- they only compare
an ordered list of retrieved chunk ids against a set of relevant ones.
"""
import math
from typing import Hashable, List, Sequence, Set

ChunkId = Hashable


def recall_at_k(retrieved: Sequence[ChunkId], relevant: Set[ChunkId], k: int = 5) -> float:
    """Fraction of all relevant chunks that appear in the top-k retrieved chunks."""
    if not relevant:
        return 0.0
    top_k = set(retrieved[:k])
    return len(top_k & relevant) / len(relevant)


def reciprocal_rank(retrieved: Sequence[ChunkId], relevant: Set[ChunkId]) -> float:
    """1 / rank of the first relevant chunk in `retrieved` (1-indexed); 0 if none found."""
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[ChunkId], relevant: Set[ChunkId], k: int = 5) -> float:
    """Binary-relevance NDCG@k: DCG of the actual ranking over DCG of the ideal ranking."""
    if not relevant:
        return 0.0

    top_k = retrieved[:k]
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, chunk_id in enumerate(top_k, start=1)
        if chunk_id in relevant
    )

    ideal_hits = min(k, len(relevant))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))

    return dcg / idcg if idcg > 0 else 0.0
