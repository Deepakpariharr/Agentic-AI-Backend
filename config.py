"""Configuration for Application"""
import os

from dotenv import load_dotenv

load_dotenv()


# --- How settings are loaded ---------------------------------------------

SSM_PREFIX = os.getenv("SSM_PREFIX")  # "/mcpbe/prod" on Lambda, unset locally


def _load_from_ssm(prefix: str) -> dict[str, str]:
    import boto3
    paginator = boto3.client("ssm").get_paginator("get_parameters_by_path")
    values: dict[str, str] = {}
    for page in paginator.paginate(Path=prefix, Recursive=True, WithDecryption=True):
        for param in page["Parameters"]:
            name = param["Name"].rsplit("/", 1)[-1]  # "/mcpbe/prod/X" -> "X"
            values[name] = param["Value"]
    return values


_ssm_values: dict[str, str] = _load_from_ssm(SSM_PREFIX) if SSM_PREFIX else {}


def get(name: str, default: str | None = None) -> str | None:
    """SSM first, then environment / .env, then `default`."""
    return _ssm_values.get(name) or os.getenv(name, default)


def get_int(name: str, default: int) -> int:
    return int(get(name) or default)


def get_float(name: str, default: float) -> float:
    return float(get(name) or default)


def get_bool(name: str, default: bool = False) -> bool:
    value = get(name)
    return value.lower() == "true" if value else default


# --- API keys ------------------------------------------------------------

OPEN_API_KEY = get("OPEN_API_KEY")
orgnId = get("orgnId")
RAPIDAPI_KEY = get("RAPIDAPI_KEY")              # job search
TAVILY_API_KEY = get("TAVILY_API_KEY")          # web search
FIREBASE_ADMIN_KEY = get("FIREBASE_ADMIN_KEY")  # service-account JSON string

DISABLE_AUTH = get_bool("DISABLE_AUTH")  # local development only


# --- LLM -----------------------------------------------------------------

LLM_PROVIDER = get("LLM_PROVIDER", "openai")  # for Now only Open AI is Supported
LLM_TEMPERATURE = get_float("LLM_TEMPERATURE", 0.0)


# --- Embeddings ----------------------------------------------------------
# The model is recorded on the Qdrant collection. Changing it later stops
# startup: point QDRANT_COLLECTION_NAME at a new collection and re-ingest.

EMBEDDING_MODEL_NAME = get("EMBEDDING_MODEL_NAME", "text-embedding-3-small")
SPARSE_EMBEDDING_MODEL = get("SPARSE_EMBEDDING_MODEL", "Qdrant/bm25")  # keyword search


# --- Qdrant (vector database) --------------------------------------------

QDRANT_URL = get("QDRANT_URL")
QDRANT_API_KEY = get("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = get("QDRANT_COLLECTION_NAME", "enterprise_docs")
REDIS_URL = get("REDIS_URL")


# --- Chunking (sizes in tokens) ------------------------------------------

CHUNK_ENCODING_NAME = get("CHUNK_ENCODING_NAME", "cl100k_base")
CHUNK_SIZE_TOKENS = get_int("CHUNK_SIZE_TOKENS", 400)
CHUNK_OVERLAP_TOKENS = get_int("CHUNK_OVERLAP_TOKENS", 60)


# --- Retrieval and reranking ---------------------------------------------
# Search returns RETRIEVAL_TOP_K candidates, an LLM scores them, and the best
# RERANK_TOP_N go into the prompt. Use a non-reasoning model for reranking.

RETRIEVAL_TOP_K = get_int("RETRIEVAL_TOP_K", 40)
RERANK_MODEL_NAME = get("RERANK_MODEL_NAME", "gpt-3.5-turbo")
RERANK_TOP_N = get_int("RERANK_TOP_N", 5)


# --- LangSmith tracing ---------------------------------------------------

for _key in ("LANGSMITH_API_KEY", "LANGSMITH_PROJECT"):
    if _value := get(_key):
        os.environ[_key] = _value
os.environ["LANGSMITH_TRACING"] = "true" if get("LANGSMITH_API_KEY") else "false"
