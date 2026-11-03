"""Conversation endpoints: one conversation = one LangGraph thread (thread_id = conversation id)."""
import logging
import uuid
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from db.engine import get_session
from db.repositories import conversations as repo
from middleware.firebase_middleware import _extract_uid_from_state
from middleware.rate_limiter import limiter
from model.response_models import success_response
from services.checkpointer import get_checkpointer

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/conversations", tags=["conversations"])


class CreateConversation(BaseModel):
    agent: Literal["rag", "mcp"] = "rag"
    title: str | None = None


def _require_uid(request: Request) -> str:
    uid = _extract_uid_from_state(request)
    if not uid:
        raise HTTPException(status_code=401, detail="Unauthorized: No valid session.")
    return uid


def _serialize(c) -> dict:
    return {
        "id": str(c.id),
        "agent": c.agent,
        "title": c.title,
        "created_at": c.created_at.isoformat(),
        "updated_at": c.updated_at.isoformat(),
    }


async def _owned_or_404(session: AsyncSession, conversation_id: uuid.UUID, uid: str):
    # Missing and "someone else's" both give 404, so ids can't be probed.
    conversation = await repo.get_owned(session, conversation_id, uid)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return conversation


@router.post("")
@limiter.limit("30/minute")
async def create_conversation(
    request: Request, body: CreateConversation, session: AsyncSession = Depends(get_session)
):
    uid = _require_uid(request)
    conversation = await repo.create(session, uid, body.agent, body.title)
    return success_response(data=_serialize(conversation), message="Conversation created")


@router.get("")
@limiter.limit("60/minute")
async def list_conversations(request: Request, session: AsyncSession = Depends(get_session)):
    uid = _require_uid(request)
    items = await repo.list_for_user(session, uid)
    return success_response(data={"conversations": [_serialize(c) for c in items]}, message="Conversations retrieved")


@router.get("/{conversation_id}/messages")
@limiter.limit("60/minute")
async def get_messages(
    request: Request, conversation_id: uuid.UUID, session: AsyncSession = Depends(get_session)
):
    uid = _require_uid(request)
    await _owned_or_404(session, conversation_id, uid)

    checkpointer = await get_checkpointer()
    checkpoint_tuple = await checkpointer.aget_tuple({"configurable": {"thread_id": str(conversation_id)}})
    raw = (checkpoint_tuple.checkpoint["channel_values"].get("messages", []) if checkpoint_tuple else [])

    messages = [
        {"role": getattr(m, "type", "unknown"), "content": m.content if isinstance(m.content, str) else str(m.content)}
        for m in raw
        if getattr(m, "type", None) in ("human", "ai") and m.content  # hide tool plumbing
    ]
    return success_response(data={"id": str(conversation_id), "messages": messages}, message="Messages retrieved")


@router.delete("/{conversation_id}")
@limiter.limit("30/minute")
async def delete_conversation(
    request: Request, conversation_id: uuid.UUID, session: AsyncSession = Depends(get_session)
):
    uid = _require_uid(request)
    await _owned_or_404(session, conversation_id, uid)

    checkpointer = await get_checkpointer()
    await checkpointer.adelete_thread(str(conversation_id))  # memory first; row last so a failure can be retried
    await repo.delete_owned(session, conversation_id, uid)
    return success_response(data={"id": str(conversation_id)}, message="Conversation deleted")
