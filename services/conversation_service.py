"""Chat endpoints' view of conversations: turn an optional conversation_id into a LangGraph thread id.

No conversation_id keeps the old behaviour (one thread per user per agent), so
existing clients keep working. A conversation_id selects that conversation's
own thread, after checking it belongs to the caller.
"""
import logging
import uuid

from db.engine import db_enabled, session_scope
from db.repositories import conversations as repo

logger = logging.getLogger(__name__)

TITLE_MAX_CHARS = 60


class ConversationsDisabled(Exception):
    """A conversation_id was sent but no database is configured."""


class ConversationNotFound(Exception):
    """Missing, someone else's, or created for the other agent. Callers answer 404 for all three."""


def legacy_thread_id(agent: str, user_id: str) -> str:
    """Per-user thread when no conversation is given. Namespaced by agent because
    both agents share one checkpoint store."""
    return f"{agent}:{user_id}"


async def resolve_thread(user_id: str, agent: str, conversation_id: str | None) -> str | None:
    """The conversation's thread id, or None when the caller didn't ask for one."""
    if not conversation_id:
        return None
    if not db_enabled():
        raise ConversationsDisabled()
    try:
        parsed = uuid.UUID(conversation_id)
    except ValueError:
        raise ConversationNotFound() from None

    async with session_scope() as session:
        conversation = await repo.get_owned(session, parsed, user_id)
    if conversation is None or conversation.agent != agent:
        raise ConversationNotFound()
    return str(conversation.id)


async def touch(user_id: str, thread_id: str | None, first_message: str) -> None:
    """Bump updated_at after a turn, naming the conversation after its first message."""
    if not thread_id or not db_enabled():
        return
    try:
        async with session_scope() as session:
            conversation = await repo.get_owned(session, uuid.UUID(thread_id), user_id)
            title = None
            if conversation is not None and not conversation.title:
                title = first_message.strip()[:TITLE_MAX_CHARS] or None
            await repo.touch(session, uuid.UUID(thread_id), user_id, title)
    except Exception:
        logger.exception("Could not update conversation %s.", thread_id)
