"""/conversations endpoints. The database layer and the checkpointer are faked."""
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from controllers import conversations_controller as ctrl
from db.engine import get_session

NOW = datetime(2026, 11, 2, 12, 0, tzinfo=timezone.utc)


def _conversation(agent="rag", title="Resume review"):
    return SimpleNamespace(id=uuid.uuid4(), agent=agent, title=title, created_at=NOW, updated_at=NOW)


@pytest.fixture
def api(monkeypatch):
    """Returns client_for(user): requests carry `user` the way the auth middleware sets it."""
    from main import app

    async def fake_session():
        yield "session"

    app.dependency_overrides[get_session] = fake_session

    def client_for(user):
        async def with_user(scope, receive, send):
            if scope["type"] == "http" and user is not None:
                scope = {**scope, "state": {"user": user}}
            await app(scope, receive, send)

        return TestClient(with_user)

    yield client_for
    app.dependency_overrides.pop(get_session)


def test_every_endpoint_requires_a_user(api):
    client = api(None)
    cid = uuid.uuid4()
    assert client.post("/conversations", json={}).status_code == 401
    assert client.get("/conversations").status_code == 401
    assert client.get(f"/conversations/{cid}/messages").status_code == 401
    assert client.delete(f"/conversations/{cid}").status_code == 401


def test_create_a_conversation(api, monkeypatch):
    conversation = _conversation(title=None)
    create = AsyncMock(return_value=conversation)
    monkeypatch.setattr(ctrl.repo, "create", create)

    body = api("uid-1").post("/conversations", json={"agent": "mcp"}).json()

    assert body["statusCode"] == 200
    assert body["data"]["id"] == str(conversation.id)
    create.assert_awaited_once_with("session", "uid-1", "mcp", None)


def test_create_defaults_to_the_rag_agent_and_rejects_unknown_agents(api, monkeypatch):
    create = AsyncMock(return_value=_conversation())
    monkeypatch.setattr(ctrl.repo, "create", create)
    client = api("uid-1")

    client.post("/conversations", json={})
    assert create.await_args.args[2] == "rag"
    assert client.post("/conversations", json={"agent": "admin"}).status_code == 422


def test_list_only_asks_for_the_callers_conversations(api, monkeypatch):
    items = [_conversation(), _conversation(agent="mcp")]
    list_for_user = AsyncMock(return_value=items)
    monkeypatch.setattr(ctrl.repo, "list_for_user", list_for_user)

    body = api("uid-1").get("/conversations").json()

    assert [c["id"] for c in body["data"]["conversations"]] == [str(c.id) for c in items]
    list_for_user.assert_awaited_once_with("session", "uid-1")


def test_messages_show_only_the_human_and_ai_text(api, monkeypatch):
    conversation = _conversation()
    monkeypatch.setattr(ctrl.repo, "get_owned", AsyncMock(return_value=conversation))
    saved = [
        HumanMessage(content="find jobs"),
        AIMessage(content="", tool_calls=[{"name": "search_jobs", "args": {}, "id": "c1"}]),
        ToolMessage(content="[raw tool output]", tool_call_id="c1"),
        AIMessage(content="Here are 3 jobs"),
    ]
    saver = SimpleNamespace(aget_tuple=AsyncMock(return_value=SimpleNamespace(checkpoint={"channel_values": {"messages": saved}})))
    monkeypatch.setattr(ctrl, "get_checkpointer", AsyncMock(return_value=saver))

    body = api("uid-1").get(f"/conversations/{conversation.id}/messages").json()

    assert body["data"]["messages"] == [
        {"role": "human", "content": "find jobs"},
        {"role": "ai", "content": "Here are 3 jobs"},
    ]
    config = saver.aget_tuple.await_args.args[0]
    assert config == {"configurable": {"thread_id": str(conversation.id)}}  # thread = conversation id


def test_messages_of_a_new_conversation_are_empty(api, monkeypatch):
    conversation = _conversation()
    monkeypatch.setattr(ctrl.repo, "get_owned", AsyncMock(return_value=conversation))
    saver = SimpleNamespace(aget_tuple=AsyncMock(return_value=None))
    monkeypatch.setattr(ctrl, "get_checkpointer", AsyncMock(return_value=saver))

    body = api("uid-1").get(f"/conversations/{conversation.id}/messages").json()

    assert body["data"]["messages"] == []


def test_someone_elses_conversation_looks_like_it_does_not_exist(api, monkeypatch):
    get_owned = AsyncMock(return_value=None)  # the repository filters by user id
    monkeypatch.setattr(ctrl.repo, "get_owned", get_owned)
    saver = SimpleNamespace(aget_tuple=AsyncMock(), adelete_thread=AsyncMock())
    monkeypatch.setattr(ctrl, "get_checkpointer", AsyncMock(return_value=saver))
    cid = uuid.uuid4()
    client = api("intruder")

    assert client.get(f"/conversations/{cid}/messages").status_code == 404
    assert client.delete(f"/conversations/{cid}").status_code == 404
    get_owned.assert_awaited_with("session", cid, "intruder")
    saver.aget_tuple.assert_not_awaited()  # never touched the other user's memory
    saver.adelete_thread.assert_not_awaited()


def test_delete_removes_the_memory_and_the_row(api, monkeypatch):
    conversation = _conversation()
    monkeypatch.setattr(ctrl.repo, "get_owned", AsyncMock(return_value=conversation))
    delete_owned = AsyncMock(return_value=True)
    monkeypatch.setattr(ctrl.repo, "delete_owned", delete_owned)
    saver = SimpleNamespace(adelete_thread=AsyncMock())
    monkeypatch.setattr(ctrl, "get_checkpointer", AsyncMock(return_value=saver))

    body = api("uid-1").delete(f"/conversations/{conversation.id}").json()

    assert body["statusCode"] == 200
    saver.adelete_thread.assert_awaited_once_with(str(conversation.id))
    delete_owned.assert_awaited_once_with("session", conversation.id, "uid-1")


def test_malformed_ids_are_rejected(api):
    assert api("uid-1").get("/conversations/not-a-uuid/messages").status_code == 422
