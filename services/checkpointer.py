import asyncio
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from config import DATABASE_URL, DB_POOL_MAX

_saver = None
_lock = asyncio.Lock()

async def get_checkpointer():
    """Postgres-backed LangGraph memory, or MemorySaver when DATABASE_URL is unset."""
    global _saver
    if _saver is not None:
        return _saver
    async with _lock:
        if _saver is not None:
            return _saver
        if not DATABASE_URL:
            _saver = MemorySaver()
            return _saver
        pool = AsyncConnectionPool(
            DATABASE_URL,
            min_size=1,
            max_size=DB_POOL_MAX,
            max_idle=60,
            check=AsyncConnectionPool.check_connection,
            open=False,
            kwargs={"autocommit": True, "row_factory": dict_row, "prepare_threshold": None},
        )
        await pool.open()
        _saver = AsyncPostgresSaver(pool)
        return _saver