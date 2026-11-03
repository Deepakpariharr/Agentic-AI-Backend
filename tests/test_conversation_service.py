import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from services import conversation_service as cs


@pytest.fixture
def db(monkeypatch):
    """Pretend a database is configured; repositories are replaced by mocks."""

    @asynccontextmanager
    async def scope():
        yield "session"

    monkeypatch.setattr(cs, "db_enabled", lambda: True)
    monkeypatch.setattr(cs, "session_scope", scope)
    conversation_id = uuid.uuid4()
    conversation = SimpleNamespace(id=conversation_id, agent="rag", title=None)
    monkeypatch.setattr(cs.repo, "get_owned", AsyncMock(return_value=conversation))
    monkeypatch.setattr(cs.repo, "touch", AsyncMock())
    return SimpleNamespace(id=conversation_id, conversation=conversation)


def test_default_threads_are_per_agent_and_user():
    assert cs.legacy_thread_id("rag", "u1") == "rag:u1"
    assert cs.legacy_thread_id("mcp", "u1") != cs.legacy_thread_id("rag", "u1")


async def test_no_conversation_id_means_no_conversation(monkeypatch):
    monkeypatch.setattr(cs, "db_enabled", lambda: False)
    assert await cs.resolve_thread("u1", "rag", None) is None
    assert await cs.resolve_thread("u1", "rag", "") is None


async def test_conversation_id_needs_a_database(monkeypatch):
    monkeypatch.setattr(cs, "db_enabled", lambda: False)
    with pytest.raises(cs.ConversationsDisabled):
        await cs.resolve_thread("u1", "rag", str(uuid.uuid4()))


async def test_owned_conversation_resolves_to_its_thread(db):
    assert await cs.resolve_thread("u1", "rag", str(db.id)) == str(db.id)
    cs.repo.get_owned.assert_awaited_once_with("session", db.id, "u1")  # ownership is checked


async def test_garbage_id_is_not_found(db):
    with pytest.raises(cs.ConversationNotFound):
        await cs.resolve_thread("u1", "rag", "not-a-uuid")


async def test_someone_elses_conversation_is_not_found(db):
    cs.repo.get_owned.return_value = None  # the repository only returns the caller's own
    with pytest.raises(cs.ConversationNotFound):
        await cs.resolve_thread("u2", "rag", str(db.id))


async def test_conversation_for_the_other_agent_is_not_found(db):
    with pytest.raises(cs.ConversationNotFound):
        await cs.resolve_thread("u1", "mcp", str(db.id))


async def test_touch_names_an_untitled_conversation_after_its_first_message(db):
    await cs.touch("u1", str(db.id), "  What are my skills?  ")
    cs.repo.touch.assert_awaited_once_with("session", db.id, "u1", "What are my skills?")


async def test_touch_keeps_an_existing_title(db):
    db.conversation.title = "Resume review"
    await cs.touch("u1", str(db.id), "next question")
    assert cs.repo.touch.await_args.args[3] is None


async def test_touch_truncates_long_titles(db):
    await cs.touch("u1", str(db.id), "x" * 500)
    assert len(cs.repo.touch.await_args.args[3]) == cs.TITLE_MAX_CHARS


async def test_touch_ignores_default_threads(db):
    await cs.touch("u1", None, "hi")
    cs.repo.touch.assert_not_awaited()


async def test_a_failed_touch_never_breaks_the_chat(db):
    cs.repo.get_owned.side_effect = RuntimeError("db down")
    await cs.touch("u1", str(db.id), "hi")  # must not raise
