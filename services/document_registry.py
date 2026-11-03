"""Bookkeeping for uploaded files in Postgres (name, size, status), next to the chunks in Qdrant.

Every function is a no-op when DATABASE_URL is unset, and fails open when the
database errors: a Postgres outage must not stop people uploading or chatting,
it only costs us the duplicate check and the filename listing.
"""
import logging

from db.engine import db_enabled, session_scope
from db.repositories import documents as repo

logger = logging.getLogger(__name__)


async def register(user_id: str, file_hash: str, file_name: str, size_bytes: int) -> bool:
    """True if the file should be ingested; False if this user already has it."""
    if not db_enabled():
        return True
    try:
        async with session_scope() as session:
            return await repo.start_ingest(session, user_id, file_hash, file_name, size_bytes) is not None
    except Exception:
        logger.exception("Document registry unavailable; ingesting without the duplicate check.")
        return True


async def mark_ready(user_id: str, file_hash: str, chunk_count: int | None) -> None:
    if not db_enabled():
        return
    try:
        async with session_scope() as session:
            await repo.mark_ready(session, user_id, file_hash, chunk_count or 0)
    except Exception:
        logger.exception("Could not mark document %s ready.", file_hash)


async def mark_failed(user_id: str, file_hash: str, error: str) -> None:
    if not db_enabled():
        return
    try:
        async with session_scope() as session:
            await repo.mark_failed(session, user_id, file_hash, error)
    except Exception:
        logger.exception("Could not mark document %s failed.", file_hash)


async def list_documents(user_id: str) -> list[dict] | None:
    """The user's documents with filenames, or None when the registry can't answer
    (caller falls back to listing straight from Qdrant)."""
    if not db_enabled():
        return None
    try:
        async with session_scope() as session:
            rows = await repo.list_for_user(session, user_id)
    except Exception:
        logger.exception("Document registry unavailable; falling back to the vector store listing.")
        return None
    return [
        {
            "source_id": row.file_hash,
            "file_name": row.file_name,
            "chunk_count": row.chunk_count,
            "size_bytes": row.size_bytes,
            "status": row.status,
            "created_at": row.created_at.isoformat(),
        }
        for row in rows
    ]


async def forget(user_id: str, file_hash: str) -> bool:
    """Remove the registry row. True if one existed."""
    if not db_enabled():
        return False
    try:
        async with session_scope() as session:
            return await repo.delete_owned(session, user_id, file_hash)
    except Exception:
        logger.exception("Could not remove document %s from the registry.", file_hash)
        return False
