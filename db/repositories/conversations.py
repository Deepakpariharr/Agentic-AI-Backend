import uuid

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Conversation


async def create(session: AsyncSession, user_id: str, agent: str, title: str | None = None) -> Conversation:
    conversation = Conversation(user_id=user_id, agent=agent, title=title)
    session.add(conversation)
    await session.flush()  # populate id / timestamps
    await session.refresh(conversation)
    return conversation


async def list_for_user(session: AsyncSession, user_id: str, limit: int = 50) -> list[Conversation]:
    result = await session.execute(
        select(Conversation)
        .where(Conversation.user_id == user_id)
        .order_by(Conversation.updated_at.desc())
        .limit(limit)
    )
    return list(result.scalars())


async def get_owned(session: AsyncSession, conversation_id: uuid.UUID, user_id: str) -> Conversation | None:
    """None when it doesn't exist OR belongs to someone else (callers return 404 for both)."""
    result = await session.execute(
        select(Conversation).where(Conversation.id == conversation_id, Conversation.user_id == user_id)
    )
    return result.scalar_one_or_none()


async def touch(session: AsyncSession, conversation_id: uuid.UUID, user_id: str, title: str | None = None) -> None:
    """Bump updated_at (and set the title, e.g. from the first message)."""
    values: dict = {"updated_at": func.now()}
    if title:
        values["title"] = title
    await session.execute(
        update(Conversation).where(Conversation.id == conversation_id, Conversation.user_id == user_id).values(**values)
    )


async def delete_owned(session: AsyncSession, conversation_id: uuid.UUID, user_id: str) -> bool:
    result = await session.execute(
        delete(Conversation).where(Conversation.id == conversation_id, Conversation.user_id == user_id)
    )
    return result.rowcount > 0
