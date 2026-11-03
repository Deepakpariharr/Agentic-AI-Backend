"""Chat memory store: Postgres when DATABASE_URL is set, in-process memory otherwise."""
from unittest.mock import AsyncMock, MagicMock

import pytest
from langgraph.checkpoint.memory import MemorySaver

from services import checkpointer


@pytest.fixture(autouse=True)
def _fresh_checkpointer(monkeypatch):
    monkeypatch.setattr(checkpointer, "_saver", None)
    monkeypatch.setattr(checkpointer, "_pool", None)


async def test_without_a_database_it_falls_back_to_memory(monkeypatch):
    monkeypatch.setattr(checkpointer, "DATABASE_URL", None)

    saver = await checkpointer.get_checkpointer()

    assert isinstance(saver, MemorySaver)
    assert await checkpointer.get_checkpointer() is saver  # one shared store, built once


async def test_with_a_database_it_uses_a_small_pool_with_connection_health_checks(monkeypatch):
    pools = []

    class FakePool:
        check_connection = staticmethod(lambda conn: None)

        def __init__(self, url, **kwargs):
            self.url, self.kwargs, self.open = url, kwargs, AsyncMock()
            pools.append(self)

        async def close(self):
            self.closed = True

    class FakeSaver:
        def __init__(self, pool):
            self.pool = pool

    import langgraph.checkpoint.postgres.aio as pg_aio
    import psycopg_pool

    monkeypatch.setattr(checkpointer, "DATABASE_URL", "postgresql://u:p@host/db")
    monkeypatch.setattr(checkpointer, "DB_POOL_MAX", 3)
    monkeypatch.setattr(psycopg_pool, "AsyncConnectionPool", FakePool)
    monkeypatch.setattr(pg_aio, "AsyncPostgresSaver", FakeSaver)

    saver = await checkpointer.get_checkpointer()
    assert await checkpointer.get_checkpointer() is saver

    assert len(pools) == 1  # not one pool per call
    pool = pools[0]
    pool.open.assert_awaited_once()
    assert pool.kwargs["max_size"] == 3
    assert pool.kwargs["check"] is FakePool.check_connection  # replaces connections that died while frozen
    assert pool.kwargs["kwargs"]["autocommit"] is True  # LangGraph needs this to write
    assert pool.kwargs["kwargs"]["prepare_threshold"] is None  # safe behind a connection pooler

    await checkpointer.close_checkpointer()
    assert pool.closed
    assert checkpointer._saver is None


async def test_close_without_a_pool_is_harmless():
    await checkpointer.close_checkpointer()
