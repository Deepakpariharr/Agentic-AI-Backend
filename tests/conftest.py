"""Shared test setup.

Several app modules do real work at import time (config reads .env/SSM,
firebase_middleware loads a service-account key, agent_service builds an
OpenAI client). The module-level code below runs before any test imports the
app, so the suite is hermetic: no .env, no SSM, no Firebase key, no network.
"""
import os

import dotenv
import firebase_admin

# Ignore the developer's local .env — tests must behave the same everywhere.
dotenv.load_dotenv = lambda *args, **kwargs: False

for _key in (
    "SSM_PREFIX",
    "OPEN_API_KEY",
    "orgnId",
    "RAPIDAPI_KEY",
    "TAVILY_API_KEY",
    "FIREBASE_ADMIN_KEY",
    "LANGSMITH_API_KEY",
    "LANGSMITH_PROJECT",
):
    os.environ.pop(_key, None)

os.environ["DISABLE_AUTH"] = "true"
# agent_service builds a ChatOpenAI client at import; with OPEN_API_KEY unset it
# falls back to this. Never used to make a real call — the agent is mocked.
os.environ["OPENAI_API_KEY"] = "sk-test"

# firebase_middleware initialises the Admin SDK at import from a key file that
# only exists on developer machines. Stub both calls before it gets imported.
firebase_admin.credentials.Certificate = lambda *args, **kwargs: object()
firebase_admin.initialize_app = lambda *args, **kwargs: None

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from middleware.rate_limiter import limiter  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_rate_limits():
    """Chat endpoints allow 5 requests/minute; start every test with a clean bucket."""
    limiter.reset()
    yield


@pytest.fixture
def client():
    from main import app

    # No `with` block: that would run the lifespan and try to close MCP sessions.
    return TestClient(app)
