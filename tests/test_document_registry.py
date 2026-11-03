from contextlib import asynccontextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from services import document_registry as registry


@pytest.fixture
def db(monkeypatch):
    @asynccontextmanager
    async def scope():
        yield "session"

    monkeypatch.setattr(registry, "db_enabled", lambda: True)
    monkeypatch.setattr(registry, "session_scope", scope)
    for name in ("start_ingest", "mark_ready", "mark_failed", "list_for_user", "delete_owned"):
        monkeypatch.setattr(registry.repo, name, AsyncMock())
    return registry.repo


async def test_everything_is_a_noop_without_a_database(monkeypatch):
    monkeypatch.setattr(registry, "db_enabled", lambda: False)

    assert await registry.register("u1", "h", "cv.pdf", 10) is True  # ingest as before
    await registry.mark_ready("u1", "h", 3)
    await registry.mark_failed("u1", "h", "boom")
    assert await registry.list_documents("u1") is None
    assert await registry.forget("u1", "h") is False


async def test_a_new_file_is_registered(db):
    db.start_ingest.return_value = SimpleNamespace(id=1)
    assert await registry.register("u1", "h", "cv.pdf", 10) is True
    db.start_ingest.assert_awaited_once_with("session", "u1", "h", "cv.pdf", 10)


async def test_a_duplicate_upload_is_refused(db):
    db.start_ingest.return_value = None
    assert await registry.register("u1", "h", "cv.pdf", 10) is False


async def test_register_fails_open_when_the_database_errors(db):
    db.start_ingest.side_effect = RuntimeError("db down")
    assert await registry.register("u1", "h", "cv.pdf", 10) is True  # uploads keep working


async def test_status_updates_never_raise(db):
    db.mark_ready.side_effect = RuntimeError("db down")
    db.mark_failed.side_effect = RuntimeError("db down")
    await registry.mark_ready("u1", "h", 3)
    await registry.mark_failed("u1", "h", "boom")


async def test_mark_ready_stores_zero_when_the_count_is_unknown(db):
    await registry.mark_ready("u1", "h", None)
    db.mark_ready.assert_awaited_once_with("session", "u1", "h", 0)


async def test_listing_includes_file_names(db):
    created = datetime(2026, 11, 2, tzinfo=timezone.utc)
    db.list_for_user.return_value = [
        SimpleNamespace(file_hash="h", file_name="cv.pdf", chunk_count=4, size_bytes=99, status="ready", created_at=created)
    ]

    assert await registry.list_documents("u1") == [
        {
            "source_id": "h",
            "file_name": "cv.pdf",
            "chunk_count": 4,
            "size_bytes": 99,
            "status": "ready",
            "created_at": created.isoformat(),
        }
    ]


async def test_listing_falls_back_to_none_on_error(db):
    db.list_for_user.side_effect = RuntimeError("db down")
    assert await registry.list_documents("u1") is None  # caller lists from Qdrant instead


async def test_forget_reports_whether_a_row_existed(db):
    db.delete_owned.return_value = True
    assert await registry.forget("u1", "h") is True
    db.delete_owned.side_effect = RuntimeError("db down")
    assert await registry.forget("u1", "h") is False
