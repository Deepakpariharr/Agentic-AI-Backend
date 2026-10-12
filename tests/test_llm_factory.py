"""services/llm_factory.py: which models get built, with what settings, and
the guards against unsupported providers and embedding models. No network —
building a LangChain client doesn't call OpenAI."""
from types import SimpleNamespace

import pytest

from services import llm_factory


@pytest.fixture(autouse=True)
def _fresh_factory(monkeypatch):
    """Every factory function is lru_cached; start and end each test uncached."""
    cached = (
        llm_factory.get_chat_model,
        llm_factory.get_rerank_model,
        llm_factory.get_embeddings,
        llm_factory.get_embedding_dimension,
    )
    for fn in cached:
        fn.cache_clear()
    monkeypatch.setattr(llm_factory, "OPEN_API_KEY", "sk-test")
    yield
    for fn in cached:
        fn.cache_clear()


def test_chat_model_uses_configured_temperature():
    llm = llm_factory.get_chat_model(temperature=0.3)
    assert llm.model_name == "gpt-3.5-turbo"
    assert llm.temperature == 0.3


def test_chat_model_is_built_once():
    assert llm_factory.get_chat_model() is llm_factory.get_chat_model()


def test_rerank_model_is_deterministic_and_returns_json(monkeypatch):
    monkeypatch.setattr(llm_factory, "RERANK_MODEL_NAME", "gpt-4o-mini")

    llm = llm_factory.get_rerank_model()

    assert llm.model_name == "gpt-4o-mini"
    assert llm.temperature == 0
    assert llm.max_tokens == 300
    assert llm.model_kwargs == {"response_format": {"type": "json_object"}}


def test_embeddings_use_configured_model(monkeypatch):
    monkeypatch.setattr(llm_factory, "EMBEDDING_MODEL_NAME", "text-embedding-3-large")
    assert llm_factory.get_embeddings().model == "text-embedding-3-large"


@pytest.mark.parametrize("factory", ["get_chat_model", "get_embeddings"])
def test_unknown_provider_is_rejected(monkeypatch, factory):
    monkeypatch.setattr(llm_factory, "LLM_PROVIDER", "anthropic")
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        getattr(llm_factory, factory)()


def test_non_openai_embedding_model_is_rejected(monkeypatch):
    monkeypatch.setattr(llm_factory, "EMBEDDING_MODEL_NAME", "bge-small-en")
    with pytest.raises(ValueError, match="Unsupported EMBEDDING_MODEL_NAME"):
        llm_factory.get_embeddings()


@pytest.mark.parametrize(
    ("model", "dimension"),
    [("text-embedding-3-small", 1536), ("text-embedding-3-large", 3072)],
)
def test_known_embedding_dimension_needs_no_api_call(monkeypatch, model, dimension):
    monkeypatch.setattr(llm_factory, "get_embeddings", pytest.fail)
    assert llm_factory.get_embedding_dimension(model) == dimension


def test_unknown_embedding_dimension_is_measured(monkeypatch):
    probe = SimpleNamespace(embed_query=lambda text: [0.0] * 7)
    monkeypatch.setattr(llm_factory, "get_embeddings", lambda: probe)

    assert llm_factory.get_embedding_dimension("text-embedding-future") == 7
