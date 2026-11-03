"""Async SQLAlchemy engine, created lazily so importing this module needs no database."""
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from config import DATABASE_URL, DB_POOL_MAX

_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


def _sqlalchemy_url(url: str) -> str:
    """Accept a plain postgresql:// URL (what Neon gives you) and select psycopg 3."""
    for prefix in ("postgresql://", "postgres://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix):]
    return url


def get_engine() -> AsyncEngine:
    global _engine, _sessionmaker
    if _engine is None:
        if not DATABASE_URL:
            raise RuntimeError("DATABASE_URL must be set to use the database.")
        _engine = create_async_engine(
            _sqlalchemy_url(DATABASE_URL),
            pool_size=DB_POOL_MAX,
            max_overflow=0,
            pool_pre_ping=True,
            pool_recycle=300,
            connect_args={"prepare_threshold": None},  # safe behind Neon's PgBouncer pooler
        )
        _sessionmaker = async_sessionmaker(_engine, expire_on_commit=False)
    return _engine


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency: one session per request, committed on success."""
    get_engine()
    assert _sessionmaker is not None
    async with _sessionmaker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


def db_enabled() -> bool:
    return bool(DATABASE_URL)


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """Same as get_session, for code that isn't a FastAPI dependency."""
    get_engine()
    assert _sessionmaker is not None
    async with _sessionmaker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def dispose_engine() -> None:
    """Close the pool on app shutdown."""
    global _engine, _sessionmaker
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _sessionmaker = None
