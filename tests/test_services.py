import io
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import UploadFile
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage

from services import agent_service, mcp_client, openai_service

# ── agent_service ────────────────────────────────────────────────────────────

@pytest.fixture
def fake_agent(monkeypatch):
    agent = MagicMock()
    agent.ainvoke = AsyncMock(return_value={"messages": [AIMessage(content="  answer  ")]})
    monkeypatch.setattr(agent_service, "_get_agent", AsyncMock(return_value=agent))
    return agent


async def test_agent_response_scopes_history_to_user(fake_agent):
    result = await agent_service.agent_response("hello", user_id="uid-42")

    assert result == "answer"
    config = fake_agent.ainvoke.call_args.kwargs["config"]
    assert config["configurable"]["thread_id"] == "uid-42"


async def test_agent_response_folds_pdf_text_into_prompt(fake_agent, monkeypatch):
    monkeypatch.setattr(agent_service, "pdf_reader_fn", lambda file_content: "PDF BODY")
    upload = UploadFile(file=io.BytesIO(b"%PDF"), filename="cv.pdf")

    await agent_service.agent_response("review my cv", upload, user_id="u")

    prompt = fake_agent.ainvoke.call_args.args[0]["messages"][0].content
    assert prompt.startswith("review my cv")
    assert "cv.pdf" in prompt
    assert "PDF BODY" in prompt


async def test_agent_response_handles_empty_answer(fake_agent):
    fake_agent.ainvoke.return_value = {"messages": [AIMessage(content="")]}
    result = await agent_service.agent_response("hello", user_id="u")
    assert result.startswith("🤔")


async def test_agent_response_swallows_errors(fake_agent):
    fake_agent.ainvoke.side_effect = RuntimeError("boom")
    result = await agent_service.agent_response("hello", user_id="u")
    assert result == "❌ Error Occured Please try after sometime"


async def test_agent_is_built_once_with_mcp_tools(monkeypatch):
    monkeypatch.setattr(agent_service, "_agent", None)
    monkeypatch.setattr(agent_service, "get_mcp_tools", AsyncMock(return_value=["tool"]))
    create_agent = MagicMock(return_value="agent")
    monkeypatch.setattr(agent_service, "create_agent", create_agent)

    assert await agent_service._get_agent() == "agent"
    assert await agent_service._get_agent() == "agent"

    create_agent.assert_called_once()
    kwargs = create_agent.call_args.kwargs
    assert kwargs["tools"] == ["tool"]
    assert kwargs["system_prompt"] == agent_service.MCP_AGENT_SYSTEM_PROMPT
    assert kwargs["checkpointer"] is agent_service.memory


def _conversation(turns):
    messages = []
    for i in range(turns):
        messages += [HumanMessage(content=f"q{i}"), AIMessage(content=f"a{i}")]
    return messages


def test_trim_history_keeps_short_conversations():
    state = {"messages": _conversation(2)}
    assert agent_service._trim_history.before_model(state, None) is None


def test_trim_history_drops_old_messages():
    messages = _conversation(6)

    update = agent_service._trim_history.before_model({"messages": messages}, None)

    remove_all, *kept = update["messages"]
    assert isinstance(remove_all, RemoveMessage)
    assert len(kept) <= agent_service.MAX_HISTORY_MESSAGES
    assert isinstance(kept[0], HumanMessage)  # never starts mid-exchange
    assert kept[-1] == messages[-1]


# ── openai_service ───────────────────────────────────────────────────────────

def _completion(content):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


@pytest.fixture
def openai_client(monkeypatch):
    client = MagicMock()
    monkeypatch.setattr(openai_service, "client", client)
    monkeypatch.setattr(openai_service, "OPEN_API_KEY", "sk-test")
    return client


def test_openai_response_without_client(monkeypatch):
    monkeypatch.setattr(openai_service, "client", None)
    assert "not configured" in openai_service.openai_response("hi")


def test_openai_response_rejects_blank_message(openai_client):
    assert openai_service.openai_response("   ").startswith("❓")
    openai_client.chat.completions.create.assert_not_called()


def test_openai_response_returns_model_output(openai_client):
    openai_client.chat.completions.create.return_value = _completion("Hello there")
    assert openai_service.openai_response("hi") == "Hello there"


@pytest.mark.parametrize(
    "error, expected_prefix",
    [
        ("Error 429: quota exceeded", "🚫"),
        ("401 Unauthorized", "🔑"),
        ("rate_limit hit", "⏱️"),
        ("connection reset", "❌"),
    ],
)
def test_openai_response_maps_errors(openai_client, error, expected_prefix):
    openai_client.chat.completions.create.side_effect = Exception(error)
    assert openai_service.openai_response("hi").startswith(expected_prefix)


# ── mcp_client ───────────────────────────────────────────────────────────────

@pytest.fixture
def mcp(monkeypatch):
    """Isolate mcp_client's module-level cache and fake out the transports."""
    monkeypatch.setattr(mcp_client, "_exit_stack", None)
    monkeypatch.setattr(mcp_client, "_tools", None)
    monkeypatch.setattr(mcp_client, "_sessions", {})

    connect = AsyncMock(side_effect=lambda stack, cfg: f"session-{cfg['transport']}")
    monkeypatch.setattr(mcp_client, "_connect", connect)

    async def load_tools(session):
        return [SimpleNamespace(name=f"tool-from-{session}")]

    monkeypatch.setattr(mcp_client, "load_mcp_tools", load_tools)
    return connect


def _servers(**overrides):
    servers = {
        "local": {"transport": "stdio", "enabled": True, "required": True},
        "remote": {"transport": "http", "enabled": True, "required": False},
    }
    for name, cfg in overrides.items():
        servers[name].update(cfg)
    return servers


async def test_mcp_tools_merge_all_servers_and_are_cached(mcp, monkeypatch):
    monkeypatch.setattr(mcp_client, "_MCP_SERVERS", _servers())

    first = await mcp_client.get_mcp_tools()
    second = await mcp_client.get_mcp_tools()

    assert [t.name for t in first] == ["tool-from-session-stdio", "tool-from-session-http"]
    assert second is first
    assert mcp.await_count == 2  # connected once per server, not once per call


async def test_mcp_disabled_server_is_skipped(mcp, monkeypatch):
    monkeypatch.setattr(mcp_client, "_MCP_SERVERS", _servers(remote={"enabled": False}))

    tools = await mcp_client.get_mcp_tools()

    assert [t.name for t in tools] == ["tool-from-session-stdio"]


async def test_mcp_optional_server_failure_is_tolerated(mcp, monkeypatch):
    monkeypatch.setattr(mcp_client, "_MCP_SERVERS", _servers())

    async def connect(stack, cfg):
        if cfg["transport"] == "http":
            raise ConnectionError("tavily down")
        return "session-stdio"

    mcp.side_effect = connect

    tools = await mcp_client.get_mcp_tools()

    assert [t.name for t in tools] == ["tool-from-session-stdio"]
    assert list(mcp_client._sessions) == ["local"]


async def test_mcp_required_server_failure_raises(mcp, monkeypatch):
    monkeypatch.setattr(mcp_client, "_MCP_SERVERS", _servers())
    mcp.side_effect = ConnectionError("subprocess died")

    with pytest.raises(ConnectionError):
        await mcp_client.get_mcp_tools()

    assert mcp_client._tools is None
    assert mcp_client._sessions == {}


async def test_mcp_unknown_transport_is_rejected():
    from contextlib import AsyncExitStack

    async with AsyncExitStack() as stack:
        with pytest.raises(ValueError, match="Unknown MCP transport"):
            await mcp_client._connect(stack, {"transport": "carrier-pigeon"})


class _FakeSession:
    def __init__(self, read, write):
        self.streams = (read, write)
        self.initialize = AsyncMock()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


async def test_mcp_stdio_launch_passes_command_cwd_and_env(monkeypatch):
    from contextlib import AsyncExitStack, asynccontextmanager

    launched = []

    @asynccontextmanager
    async def fake_stdio_client(params):
        launched.append(params)
        yield "read", "write"

    monkeypatch.setattr(mcp_client, "stdio_client", fake_stdio_client)
    monkeypatch.setattr(mcp_client, "ClientSession", _FakeSession)

    cfg = mcp_client._MCP_SERVERS["career-tools"]
    async with AsyncExitStack() as stack:
        session = await mcp_client._connect(stack, cfg)

    params, = launched
    assert params.args == ["-m", "mcp_server"]
    assert params.cwd == str(mcp_client._PROJECT_ROOT)
    assert params.env == cfg["env"]
    assert session.streams == ("read", "write")
    session.initialize.assert_awaited_once()


def test_career_tools_env_only_carries_set_keys():
    # conftest clears RAPIDAPI_KEY / TAVILY_API_KEY, so nothing is passed on.
    assert mcp_client._CAREER_TOOLS_ENV == {}


async def test_mcp_http_server_connects_with_url_and_headers(monkeypatch):
    from contextlib import AsyncExitStack, asynccontextmanager

    opened = []

    @asynccontextmanager
    async def fake_http_client(url, headers=None):
        opened.append((url, headers))
        yield "read", "write", lambda: "session-id"

    monkeypatch.setattr(mcp_client, "streamablehttp_client", fake_http_client)
    monkeypatch.setattr(mcp_client, "ClientSession", _FakeSession)

    cfg = {"transport": "http", "url": "https://mcp.example.com/mcp", "headers": {"X-Key": "k"}}
    async with AsyncExitStack() as stack:
        session = await mcp_client._connect(stack, cfg)

    assert opened == [("https://mcp.example.com/mcp", {"X-Key": "k"})]
    assert session.streams == ("read", "write")
    session.initialize.assert_awaited_once()


async def test_close_mcp_client_tears_down_and_resets_cache(monkeypatch):
    stack = SimpleNamespace(aclose=AsyncMock())
    monkeypatch.setattr(mcp_client, "_exit_stack", stack)
    monkeypatch.setattr(mcp_client, "_tools", ["tool"])
    monkeypatch.setattr(mcp_client, "_sessions", {"local": "session"})

    await mcp_client.close_mcp_client()

    stack.aclose.assert_awaited_once()
    assert (mcp_client._exit_stack, mcp_client._tools, mcp_client._sessions) == (None, None, {})


async def test_close_mcp_client_survives_a_dead_transport(monkeypatch):
    stack = SimpleNamespace(aclose=AsyncMock(side_effect=BrokenPipeError))
    monkeypatch.setattr(mcp_client, "_exit_stack", stack)
    monkeypatch.setattr(mcp_client, "_tools", ["tool"])
    monkeypatch.setattr(mcp_client, "_sessions", {})

    await mcp_client.close_mcp_client()  # must not raise

    assert mcp_client._tools is None
