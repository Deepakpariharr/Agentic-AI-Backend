"""Create/upgrade the database schema. Run manually before starting the app.

    python -m scripts.db_migrate
"""
import asyncio

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg import AsyncConnection
from psycopg.rows import dict_row

from config import DATABASE_URL
from db.engine import dispose_engine, get_engine
from db.models import Base


async def main() -> None:
    if not DATABASE_URL:
        raise SystemExit("DATABASE_URL is not set.")

    # Our tables (idempotent). Swap for Alembic once the schema starts changing.
    async with get_engine().begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await dispose_engine()

    # LangGraph's checkpoint tables.
    async with await AsyncConnection.connect(DATABASE_URL, autocommit=True, row_factory=dict_row) as conn:
        await AsyncPostgresSaver(conn).setup()

    print("Database schema is up to date.")


if __name__ == "__main__":
    asyncio.run(main())
