import asyncio
import json
import logging
from functools import lru_cache
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from prompts import RERANK_SYSTEM_PROMPT, wrap_untrusted
from services.llm_factory import get_rerank_model

logger = logging.getLogger(__name__)

# Passages scored per LLM call. Keeps each request well inside the model's
# context window; batches run concurrently.
BATCH_SIZE = 10
MAX_SCORE = 10.0


class Reranker:
    """LLM-based reranker using an OpenAI chat model (no local model download).

    Candidates are scored 0-10 for relevance in concurrent batches. If scoring
    fails, reranking degrades to the retriever's own order instead of failing
    the request.
    """

    def __init__(self, llm=None):
        self._llm = llm

    @property
    def llm(self):
        if self._llm is None:
            self._llm = get_rerank_model()
        return self._llm

    async def _score_batch(self, query: str, batch: List[Document]) -> List[Optional[float]]:
        passages = "\n\n".join(
            f"[{i}]\n{doc.page_content}" for i, doc in enumerate(batch, start=1)
        )
        messages = [
            SystemMessage(content=RERANK_SYSTEM_PROMPT),
            HumanMessage(
                content=f"Question: {query}\n\nPassages:\n{wrap_untrusted('rerank_candidates', passages)}"
            ),
        ]
        response = await self.llm.ainvoke(messages)
        raw_scores = json.loads(response.content).get("scores", {})

        scores: List[Optional[float]] = []
        for i in range(1, len(batch) + 1):
            try:
                value = float(raw_scores[str(i)])
            except (KeyError, TypeError, ValueError):
                scores.append(None)
                continue
            scores.append(min(max(value, 0.0), MAX_SCORE))
        return scores

    async def rerank(self, query: str, docs: List[Document], top_n: int) -> List[Document]:
        if not docs:
            return []

        batches = [docs[i:i + BATCH_SIZE] for i in range(0, len(docs), BATCH_SIZE)]
        results = await asyncio.gather(
            *(self._score_batch(query, batch) for batch in batches),
            return_exceptions=True,
        )

        scores: List[Optional[float]] = []
        for batch, result in zip(batches, results):
            if isinstance(result, BaseException):
                logger.error("Rerank batch failed (%s); keeping retrieval order for it.", result)
                scores.extend([None] * len(batch))
            else:
                scores.extend(result)

        if all(score is None for score in scores):
            return docs[:top_n]

        # Stable sort: ties and unscored docs keep the retriever's order,
        # with unscored docs placed after every scored one.
        ranked = sorted(
            zip(docs, scores),
            key=lambda pair: pair[1] if pair[1] is not None else -1.0,
            reverse=True,
        )[:top_n]
        for doc, score in ranked:
            if score is not None:
                doc.metadata["rerank_score"] = score / MAX_SCORE
        return [doc for doc, _ in ranked]


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    """Process-wide singleton so multiple RAGService instances share one client."""
    return Reranker()
