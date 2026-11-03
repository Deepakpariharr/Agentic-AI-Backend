import config


def test_get_falls_back_to_environment(monkeypatch):
    monkeypatch.setenv("RAPIDAPI_KEY", "from-env")
    assert config.get("RAPIDAPI_KEY") == "from-env"


def test_get_returns_default_when_unset(monkeypatch):
    monkeypatch.delenv("RAPIDAPI_KEY", raising=False)
    assert config.get("RAPIDAPI_KEY", "fallback") == "fallback"
