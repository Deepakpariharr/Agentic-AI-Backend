import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from middleware import firebase_middleware
from middleware.firebase_middleware import firebase_auth_middleware
from middleware.rate_limiter import custom_key_func


@pytest.fixture
def auth_client():
    """A bare app with only the Firebase middleware, so it can be tested even
    though the main app runs with DISABLE_AUTH in tests."""
    app = FastAPI()
    app.middleware("http")(firebase_auth_middleware)

    @app.get("/")
    async def health():
        return {"ok": True}

    @app.get("/private")
    async def private(request: Request):
        return {"user": request.state.user}

    @app.get("/chat")
    async def chat(request: Request):
        return {"user": request.state.user}

    return TestClient(app)


def test_public_paths_skip_auth(auth_client):
    assert auth_client.get("/").status_code == 200


def test_missing_token_is_rejected(auth_client):
    response = auth_client.get("/private")
    assert response.status_code == 401
    assert response.json()["message"] == "Missing or invalid Authorization header"


def test_non_bearer_token_is_rejected(auth_client):
    response = auth_client.get("/private", headers={"Authorization": "Basic abc"})
    assert response.status_code == 401


def test_invalid_token_is_rejected(auth_client, monkeypatch):
    def verify(_token):
        raise ValueError("expired")

    monkeypatch.setattr(firebase_middleware.auth, "verify_id_token", verify)
    response = auth_client.get("/private", headers={"Authorization": "Bearer bad"})
    assert response.status_code == 401
    assert response.json()["message"] == "Invalid or expired Firebase token"


def test_valid_token_sets_user(auth_client, monkeypatch):
    monkeypatch.setattr(
        firebase_middleware.auth,
        "verify_id_token",
        lambda token: {"uid": f"uid-for-{token}"},
    )
    response = auth_client.get("/private", headers={"Authorization": "Bearer good"})
    assert response.status_code == 200
    assert response.json() == {"user": "uid-for-good"}


def test_preflight_requests_skip_auth(auth_client):
    assert auth_client.options("/private").status_code != 401


# /chat routes spend OpenAI / RapidAPI / Tavily credit and key chat history by
# user, so they get no exemption: anonymous callers are turned away.

def test_chat_route_requires_a_token(auth_client):
    assert auth_client.get("/chat").status_code == 401


def test_chat_route_rejects_an_invalid_token(auth_client, monkeypatch):
    def verify(_token):
        raise ValueError("expired")

    monkeypatch.setattr(firebase_middleware.auth, "verify_id_token", verify)
    response = auth_client.get("/chat", headers={"Authorization": "Bearer bad"})
    assert response.status_code == 401


def test_chat_route_with_valid_token_sets_user(auth_client, monkeypatch):
    monkeypatch.setattr(
        firebase_middleware.auth, "verify_id_token", lambda token: {"uid": f"uid-for-{token}"}
    )
    response = auth_client.get("/chat", headers={"Authorization": "Bearer good"})
    assert response.json() == {"user": "uid-for-good"}


# ── Firebase credentials ─────────────────────────────────────────────────────

def test_credentials_come_from_ssm_json_when_set(monkeypatch):
    loaded = []
    monkeypatch.setattr(firebase_middleware.credentials, "Certificate", loaded.append)
    monkeypatch.setattr(firebase_middleware, "FIREBASE_ADMIN_KEY", '{"project_id": "demo"}')

    firebase_middleware._load_credentials()

    assert loaded == [{"project_id": "demo"}]


def test_credentials_fall_back_to_local_key_file(monkeypatch):
    loaded = []
    monkeypatch.setattr(firebase_middleware.credentials, "Certificate", loaded.append)
    monkeypatch.setattr(firebase_middleware, "FIREBASE_ADMIN_KEY", None)

    firebase_middleware._load_credentials()

    assert loaded == ["Secret/firebaseAdminKey.json"]


class _FakeRequest:
    def __init__(self, user=None):
        self.state = type("State", (), {})()
        if user:
            self.state.user = user
        self.client = type("Client", (), {"host": "10.0.0.1"})()
        self.headers = {}


def test_rate_limit_key_uses_firebase_uid():
    assert custom_key_func(_FakeRequest(user="uid-123")) == "uid-123"


def test_rate_limit_key_falls_back_to_ip():
    assert custom_key_func(_FakeRequest()) == "10.0.0.1"


def test_responses_carry_request_id(client):
    first = client.get("/").headers["X-Request-ID"]
    second = client.get("/").headers["X-Request-ID"]
    assert first and second and first != second
