"""The one evaluation runner: golden benchmark -> real RAG pipeline -> real metrics.

This module is the single source of truth for the evaluation logic. The CLI
(`evals/rag/run_eval.py`) calls `run_benchmark()` directly.

Requires the extra, pinned dependencies in `requirements-eval.txt` (ragas +
a compatible langchain-community) to be installed in whatever process calls
this module. It is not imported by the main app at startup, so a normal
deployment that only installs `requirements.txt` is unaffected.
"""
import asyncio
import json
import logging
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from qdrant_client import QdrantClient

from evals.rag.metrics import ndcg_at_k, recall_at_k, reciprocal_rank
from prompts import RAG_EVAL_ANSWER_PROMPT
from services.llm_factory import get_chat_model, get_embeddings
from services.rag.retrieval import HybridRetriever
from services.rag.service import RAGService, format_context

logger = logging.getLogger(__name__)

BENCHMARK_DIR = Path(__file__).parent / "benchmark"
DOCS_DIR = BENCHMARK_DIR / "docs"
QUESTIONS_PATH = BENCHMARK_DIR / "questions.json"

# The benchmark always runs against an isolated, in-memory Qdrant collection
# under a fixed session id -- it never touches production data.
EVAL_SESSION_ID = "eval-benchmark"
EVAL_COLLECTION_NAME = "eval_benchmark"
RETRIEVAL_K = 5  # matches RERANK_TOP_N, i.e. "top 5" in Recall@5 / NDCG@5


def _load_benchmark():
    questions = json.loads(QUESTIONS_PATH.read_text())
    documents = {
        path.name: path.read_text()
        for path in sorted(DOCS_DIR.glob("*.txt"))
    }
    return documents, questions


def _relevant_chunk_ids(question: dict, chunks_by_source: Dict[str, list]) -> set:
    """Chunks are "relevant" if they belong to the question's source document
    and contain at least one of the question's evidence quotes. This is
    computed from the real, current chunking output, so it stays correct
    even if chunk size/overlap changes -- and it never depends on document
    type or hand-picked chunk indices.
    """
    chunks = chunks_by_source.get(question["source_id"], [])
    relevant = set()
    for chunk in chunks:
        text = chunk.page_content.lower()
        if any(evidence.lower() in text for evidence in question["evidence"]):
            relevant.add((question["source_id"], chunk.metadata["chunk_index"]))
    return relevant


async def _generate_answer(llm, question: str, docs: List) -> str:
    context = format_context(docs)
    prompt = f"{RAG_EVAL_ANSWER_PROMPT}\n\nContext:\n{context}\n\nQuestion: {question}"
    response = await llm.ainvoke(prompt)
    return response.content


async def run_benchmark() -> Dict:
    """Run the full golden benchmark end-to-end and return the aggregate metrics.

    Retrieval metrics (Recall@5, MRR, NDCG@5) are computed with plain Python
    against the benchmark's ground-truth evidence. Answer metrics
    (Faithfulness, Answer Relevance, Answer Correctness) are computed with
    Ragas over the actual generated answers. No metric is hardcoded.
    """
    # Imported here, not at module load: these packages come from
    # requirements-eval.txt and are only needed when an eval actually runs.
    from ragas import EvaluationDataset, SingleTurnSample, aevaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import AnswerCorrectness, Faithfulness, ResponseRelevancy

    documents, questions = _load_benchmark()

    # Isolated, ephemeral Qdrant instance -- never the production collection.
    client = QdrantClient(location=":memory:")
    retriever = HybridRetriever(collection_name=EVAL_COLLECTION_NAME, client=client)
    rag = RAGService(retriever=retriever)

    chunks_by_source: Dict[str, list] = {}
    for source_id, text in documents.items():
        chunks = rag.ingestor.process_document(text, EVAL_SESSION_ID, source_id)
        await rag.retriever.add_chunks(chunks)
        chunks_by_source[source_id] = chunks

    llm = get_chat_model()
    recalls, rr_scores, ndcgs = [], [], []
    ragas_samples = []

    for question in questions:
        relevant_ids = _relevant_chunk_ids(question, chunks_by_source)
        if not relevant_ids:
            raise ValueError(
                f"Benchmark question '{question['id']}' has no matching chunk for its "
                "evidence quote(s) -- fix the question or the evidence text."
            )

        retrieved_docs = await rag.retrieve(question["question"], session_id=EVAL_SESSION_ID)
        retrieved_ids = [
            (doc.metadata["source_id"], doc.metadata["chunk_index"]) for doc in retrieved_docs
        ]

        recalls.append(recall_at_k(retrieved_ids, relevant_ids, k=RETRIEVAL_K))
        rr_scores.append(reciprocal_rank(retrieved_ids, relevant_ids))
        ndcgs.append(ndcg_at_k(retrieved_ids, relevant_ids, k=RETRIEVAL_K))

        answer = await _generate_answer(llm, question["question"], retrieved_docs)
        ragas_samples.append(
            SingleTurnSample(
                user_input=question["question"],
                response=answer,
                retrieved_contexts=[doc.page_content for doc in retrieved_docs],
                reference=question["expected_answer"],
            )
        )

    ragas_llm = LangchainLLMWrapper(llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(get_embeddings())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ragas' own deprecation noise, not ours
        eval_result = await aevaluate(
            dataset=EvaluationDataset(samples=ragas_samples),
            metrics=[
                Faithfulness(llm=ragas_llm),
                ResponseRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
                AnswerCorrectness(llm=ragas_llm, embeddings=ragas_embeddings),
            ],
            show_progress=False,
        )
    scores_df = eval_result.to_pandas()

    return {
        "retrieval": {
            "recall_at_5": sum(recalls) / len(recalls),
            "mrr": sum(rr_scores) / len(rr_scores),
            "ndcg_at_5": sum(ndcgs) / len(ndcgs),
        },
        "answer_quality": {
            "faithfulness": float(scores_df["faithfulness"].mean()),
            "answer_relevance": float(scores_df["answer_relevancy"].mean()),
            "answer_correctness": float(scores_df["answer_correctness"].mean()),
        },
        "num_questions": len(questions),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    print(json.dumps(asyncio.run(run_benchmark()), indent=2))
