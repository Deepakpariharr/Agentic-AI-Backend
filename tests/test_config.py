import config


def test_get_prefers_ssm_over_environment(monkeypatch):
    monkeypatch.setattr(config, "_ssm_values", {"RAPIDAPI_KEY": "from-ssm"})
    monkeypatch.setenv("RAPIDAPI_KEY", "from-env")
    assert config.get("RAPIDAPI_KEY") == "from-ssm"


def test_get_falls_back_to_environment(monkeypatch):
    monkeypatch.setattr(config, "_ssm_values", {})
    monkeypatch.setenv("RAPIDAPI_KEY", "from-env")
    assert config.get("RAPIDAPI_KEY") == "from-env"


def test_get_returns_default_when_unset(monkeypatch):
    monkeypatch.setattr(config, "_ssm_values", {})
    monkeypatch.delenv("RAPIDAPI_KEY", raising=False)
    assert config.get("RAPIDAPI_KEY", "fallback") == "fallback"


def test_load_from_ssm_strips_prefix_from_names(monkeypatch):
    pages = [
        {"Parameters": [{"Name": "/mcpbe/prod/OPEN_API_KEY", "Value": "sk-1"}]},
        {"Parameters": [{"Name": "/mcpbe/prod/nested/orgnId", "Value": "org-1"}]},
    ]

    class FakeClient:
        def get_paginator(self, _name):
            return self

        def paginate(self, **kwargs):
            assert kwargs == {"Path": "/mcpbe/prod", "Recursive": True, "WithDecryption": True}
            return pages

    import boto3

    monkeypatch.setattr(boto3, "client", lambda _service: FakeClient())
    assert config._load_from_ssm("/mcpbe/prod") == {"OPEN_API_KEY": "sk-1", "orgnId": "org-1"}
