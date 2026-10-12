import logging
from functools import lru_cache

from config import (
    OPEN_API_KEY,
    LLM_PROVIDER,
    EMBEDDING_MODEL_NAME,
    LLM_TEMPERATURE,
    RERANK_MODEL_NAME,
)

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_chat_model(temperature: float = LLM_TEMPERATURE):
    """Returns the configured LangChain chat model."""
    if LLM_PROVIDER not in ("openai", None):
        raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER!r} (expected 'openai')")

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model="gpt-3.5-turbo",
        api_key=OPEN_API_KEY,
        temperature=temperature,
    )


@lru_cache(maxsize=1)
def get_rerank_model():
    """Chat model used to score retrieved passages (deterministic, JSON output)."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=RERANK_MODEL_NAME,
        api_key=OPEN_API_KEY,
        temperature=0,
        max_tokens=300,
        timeout=20,
        max_retries=1,
        model_kwargs={"response_format": {"type": "json_object"}},
    )


# Output size of each supported OpenAI embedding model. The collection's
# vector size must equal this, so it is pinned rather than guessed.
KNOWN_EMBEDDING_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


@lru_cache(maxsize=1)
def get_embeddings():
    """Returns the configured OpenAI embedding model.

    OpenAI is the only supported embedding provider. There is deliberately no
    fallback: switching models silently would mix incompatible vector spaces in
    one index. If OpenAI is unreachable, embedding calls fail loudly instead.
    """
    if LLM_PROVIDER not in ("openai", None):
        raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER!r} (expected 'openai')")

    if not EMBEDDING_MODEL_NAME.startswith("text-embedding"):
        raise ValueError(
            f"Unsupported EMBEDDING_MODEL_NAME {EMBEDDING_MODEL_NAME!r}: "
            "only OpenAI 'text-embedding-*' models are supported."
        )

    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME, api_key=OPEN_API_KEY)


@lru_cache(maxsize=None)
def get_embedding_dimension(model_name: str = EMBEDDING_MODEL_NAME) -> int:
    """Vector size produced by `model_name`.

    Known OpenAI models are answered from a table (no API call). For an unknown
    model the size is measured with one real embedding call; if that fails the
    error propagates instead of guessing a size.
    """
    if model_name in KNOWN_EMBEDDING_DIMENSIONS:
        return KNOWN_EMBEDDING_DIMENSIONS[model_name]
    return len(get_embeddings().embed_query("dimension probe"))
