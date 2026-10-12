"""Evals hit the real agent, the real MCP tools and a real judge model.

Unlike tests/conftest.py nothing is stubbed here: keys come from .env (or SSM
when SSM_PREFIX is set), exactly as the running app would load them.
"""
import pytest

import harness
from config import OPEN_API_KEY


def pytest_configure(config):
    if not OPEN_API_KEY:
        raise pytest.UsageError(
            "OPEN_API_KEY is not set — evals call the real agent and judge model. "
            "Add it to .env first."
        )


@pytest.fixture(scope="session", autouse=True)
def _close_mcp():
    """Shut the MCP subprocess/sessions down once, after the last eval."""
    yield
    harness.shutdown()
