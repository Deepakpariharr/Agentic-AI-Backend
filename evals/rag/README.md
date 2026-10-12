# RAG evaluation

A small, real (no hardcoded numbers) evaluation layer over the existing RAG
pipeline: `Document → Retrieval → Reranking → Grounded Answer → Evaluation`.

It reuses the production pipeline as-is (`services/rag/`) — same chunking,
same hybrid retriever, same LLM reranker — and runs it against a fixed
golden benchmark instead of live user documents.

## What's here

- `benchmark/docs/` — 5 short, mixed-type documents (HR policy, API
  reference, financial report, product FAQ, resume).
- `benchmark/questions.json` — 30 questions across those docs. Each has an
  `expected_answer` and one or more `evidence` quotes (exact substrings from
  the source document).
- `metrics.py` — pure functions for Recall@5, MRR, NDCG@5. Relevance is
  chunk-level and computed from the evidence quotes against the *actual*
  chunks the pipeline produced for that run — nothing is hand-pinned to a
  chunk index, so it stays correct if chunk size/overlap ever changes.
- `runner.py` — `run_benchmark()`, the single implementation of the
  evaluation. Ingests the benchmark docs into an isolated,
  in-memory Qdrant collection (never the real `enterprise_docs`
  collection), runs retrieval + rerank + generation for every question, and
  scores the three answer-quality metrics with Ragas.
- `run_eval.py` — CLI entrypoint.

Uploaded user documents never get these metrics — there's no ground truth
for arbitrary documents. Recall/MRR/NDCG/Ragas only ever run against this
fixed benchmark.

## Setup

Ragas needs a `langchain-community` version older than what
`requirements.txt` resolves to on its own (see the comment in
`requirements-eval.txt` for why). Install both files into the same
environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-eval.txt
```

## Run it

```bash
source .venv/bin/activate
python -m evals.rag.run_eval
```

Example output:

```
Retrieval
Recall@5: 1.00
MRR:      1.00
NDCG@5:   1.00

Answer Quality
Faithfulness:       0.88
Answer Relevance:   0.76
Answer Correctness: 0.81
```

A full run makes real OpenAI calls (embeddings, reranking, generation, and
Ragas' own LLM-judge calls) for all 30 questions — expect it to take a
couple of minutes and cost a small amount of API usage. You may see
`LLM returned 1 generations instead of requested 3` warnings from Ragas;
that's Ragas' internal self-consistency sampling degrading gracefully on a
non-`n`-supporting model call, not a failure.

### Saving the results as JSON

```bash
python -m evals.rag.run_eval --output eval-results.json
```

Prints the same report and also writes the metrics to the file, so a run can
be kept, compared with a later one, or published from CI:

```json
{
  "retrieval": { "recall_at_5": 1.0, "mrr": 1.0, "ndcg_at_5": 1.0 },
  "answer_quality": { "faithfulness": 0.88, "answer_relevance": 0.76, "answer_correctness": 0.81 },
  "num_questions": 30,
  "generated_at": "2026-09-27T13:40:00+00:00"
}
```

The benchmark is CLI-only on purpose. It isn't exposed as an API route: a
run takes minutes (longer than the Lambda and API Gateway 30s limits),
`ragas` isn't installed in the Lambda image, and any logged-in user could
otherwise trigger a paid run.
