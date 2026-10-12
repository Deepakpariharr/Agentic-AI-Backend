"""Every LLM prompt in the app, in one place.

One module per feature; import from the package, e.g.
`from prompts import RAG_SYSTEM_PROMPT`.
"""
from prompts.guard import wrap_untrusted
from prompts.mcp_agent import MCP_AGENT_SYSTEM_PROMPT
from prompts.openai_chat import OPENAI_CHAT_SYSTEM_PROMPT
from prompts.rag_agent import FALLBACK_SYSTEM_PROMPT, RAG_SYSTEM_PROMPT
from prompts.rag_eval import RAG_EVAL_ANSWER_PROMPT
from prompts.reranker import RERANK_SYSTEM_PROMPT

__all__ = [
    "FALLBACK_SYSTEM_PROMPT",
    "MCP_AGENT_SYSTEM_PROMPT",
    "OPENAI_CHAT_SYSTEM_PROMPT",
    "RAG_EVAL_ANSWER_PROMPT",
    "RAG_SYSTEM_PROMPT",
    "RERANK_SYSTEM_PROMPT",
    "wrap_untrusted",
]
