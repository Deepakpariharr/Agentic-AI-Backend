from datetime import timedelta

from sqlalchemy import and_, delete, func, or_, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Document


STALE_PROCESSING = timedelta(minutes=10)


async def start_ingest(
    session: AsyncSession, user_id: str, file_hash: str, file_name: str, size_bytes: int
) -> Document | None:
    """Reserve a row for this upload.

    Returns the row when the caller should ingest, or None when the user already
    has this file (ready, or still being processed). A previous failed attempt, or
    one stuck in 'processing' for 10+ minutes (crashed worker), is retried. The
    unique constraint makes the duplicate check race-free.
    """
    stmt = insert(Document).values(
        user_id=user_id, file_hash=file_hash, file_name=file_name, size_bytes=size_bytes, status="processing"
    )
    stmt = stmt.on_conflict_do_update(
        constraint="uq_documents_user_hash",
        set_={
            "status": "processing",
            "error": None,
            "file_name": stmt.excluded.file_name,
            "size_bytes": stmt.excluded.size_bytes,
            "created_at": func.now(),
        },
        where=or_(
            Document.status == "failed",
            and_(Document.status == "processing", Document.created_at < func.now() - STALE_PROCESSING),
        ),
    ).returning(Document)
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def mark_ready(session: AsyncSession, user_id: str, file_hash: str, chunk_count: int) -> None:
    await session.execute(
        update(Document)
        .where(Document.user_id == user_id, Document.file_hash == file_hash)
        .values(status="ready", chunk_count=chunk_count, error=None)
    )


async def mark_failed(session: AsyncSession, user_id: str, file_hash: str, error: str) -> None:
    await session.execute(
        update(Document)
        .where(Document.user_id == user_id, Document.file_hash == file_hash)
        .values(status="failed", error=error[:500])
    )


async def list_for_user(session: AsyncSession, user_id: str) -> list[Document]:
    result = await session.execute(
        select(Document).where(Document.user_id == user_id).order_by(Document.created_at.desc())
    )
    return list(result.scalars())


async def get_owned(session: AsyncSession, user_id: str, file_hash: str) -> Document | None:
    result = await session.execute(
        select(Document).where(Document.user_id == user_id, Document.file_hash == file_hash)
    )
    return result.scalar_one_or_none()


async def delete_owned(session: AsyncSession, user_id: str, file_hash: str) -> bool:
    result = await session.execute(
        delete(Document).where(Document.user_id == user_id, Document.file_hash == file_hash)
    )
    return result.rowcount > 0
