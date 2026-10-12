"""Prompt for the RAG benchmark's answer step (evals/rag/runner.py)."""

RAG_EVAL_ANSWER_PROMPT = (
    "Answer the user's question using ONLY the excerpts in the context below. "
    "If the excerpts do not contain the answer, say you don't know. Be concise."
)