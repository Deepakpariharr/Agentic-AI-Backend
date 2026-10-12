"""CLI entrypoint for the RAG evaluation benchmark.

    python -m evals.rag.run_eval
    python -m evals.rag.run_eval --output eval-results.json

Runs `run_benchmark()` from evals/rag/runner.py. Requires requirements-eval.txt
to be installed (see evals/rag/README.md).
"""
import argparse
import asyncio
import json
import logging
from pathlib import Path

from evals.rag.runner import run_benchmark


def _print_report(result: dict) -> None:
    retrieval = result["retrieval"]
    answer_quality = result["answer_quality"]

    print(f"\nBenchmark: {result['num_questions']} questions\n")
    print("Retrieval")
    print(f"Recall@5: {retrieval['recall_at_5']:.2f}")
    print(f"MRR:      {retrieval['mrr']:.2f}")
    print(f"NDCG@5:   {retrieval['ndcg_at_5']:.2f}")
    print("\nAnswer Quality")
    print(f"Faithfulness:       {answer_quality['faithfulness']:.2f}")
    print(f"Answer Relevance:   {answer_quality['answer_relevance']:.2f}")
    print(f"Answer Correctness: {answer_quality['answer_correctness']:.2f}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RAG evaluation benchmark.")
    parser.add_argument(
        "--output",
        type=Path,
        help="also write the metrics as JSON to this file",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    result = asyncio.run(run_benchmark())
    _print_report(result)

    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
