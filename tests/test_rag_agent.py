"""RAG agent (services/rag_agent/) and its POST /chat/documents endpoint. The LLM, the
MCP tools and the vector store are all faked."""
import io
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import fitz
import pytest
from fastapi import UploadFile
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver

from dependencies import get_rag_agent
from services.rag.service import NO_CONTEXT_MESSAGE
from services.rag_agent import graph as graph_module
from services.rag_agent import service as service_module
from services.rag_agent.service import RagAgentService
from services.rag_agent.tools import search_uploaded_documents
from services.rag_agent.uploads import ingest_pdfs
from utility.request_limits import MAX_FILE_SIZE_BYTES, MAX_FILES, MAX_QUERY_CHARS


def _pdf(text: str | None) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    if text:
        page.insert_text((72, 72), text)
    return doc.tobytes()


def _upload(content: bytes, filename: str = "resume.pdf") -> UploadFile:
    return UploadFile(file=io.BytesIO(content), filename=filename)


# ── graph + search_uploaded_documents ───────────────────────────────────────

async def test_graph_runs_the_document_tool_then_answers(monkeypatch):
    replies = [
        AIMessage(
            content="",
            tool_calls=[{"name": "search_uploaded_documents", "args": {"query": "skills"}, "id": "call-1"}],
        ),
        AIMessage(content="You know Python [resume.pdf#1]"),
    ]
    seen = []

    async def ainvoke(messages, config=None):
        seen.append(messages)
        return replies.pop(0)

    llm = MagicMock()
    llm.bind_tools.return_value = SimpleNamespace(ainvoke=ainvoke)
    monkeypatch.setattr(graph_module, "get_chat_model", lambda: llm)
    rag = SimpleNamespace(search_context=AsyncMock(return_value="[resume.pdf#1] Python"))

    graph = graph_module.build_graph([search_uploaded_documents], MemorySaver())
    state = await graph.ainvoke(
        {"session_id": "u1", "messages": [("user", "what are my skills?")]},
        config={"configurable": {"thread_id": "u1", "session_id": "u1", "rag_service": rag}},
    )

    assert state["messages"][-1].content == "You know Python [resume.pdf#1]"
    rag.search_context.assert_awaited_once_with(query="skills", session_id="u1")
    assert seen[0][0].content == graph_module.RAG_SYSTEM_PROMPT  # system prompt goes first
    assert seen[1][-1].content == "[resume.pdf#1] Python"  # tool result fed back


async def _search(config, context=None, error=None):
    rag = SimpleNamespace(search_context=AsyncMock(return_value=context, side_effect=error))
    configurable = {"rag_service": rag, "session_id": "u1", **config}
    return await search_uploaded_documents.ainvoke(
        {"query": "q"}, config={"configurable": configurable}
    )


async def test_document_search_needs_session_context():
    assert (await _search({"session_id": None, "rag_service": None})).startswith(
        "Document service unavailable"
    )


async def test_document_search_reports_no_matches():
    assert (await _search({}, context=NO_CONTEXT_MESSAGE)).startswith("No matching content")


async def test_document_search_reports_errors():
    assert await _search({}, error=RuntimeError("qdrant down")) == (
        "Error reading documents from vector storage."
    )


# ── uploads ─────────────────────────────────────────────────────────────────

async def test_ingest_pdfs_indexes_readable_pdfs_only():
    rag = SimpleNamespace(ingest_document=AsyncMock(return_value="Ingested 1 chunks"))
    files = [
        _upload(_pdf("Senior Python Developer")),
        _upload(b"notes", filename="notes.txt"),
        _upload(b"", filename="empty.pdf"),
        _upload(_pdf(None), filename="blank.pdf"),
    ]

    assert await ingest_pdfs(rag, files, "u1") is True

    rag.ingest_document.assert_awaited_once()
    kwargs = rag.ingest_document.await_args.kwargs
    assert "Senior Python Developer" in kwargs["raw_text"]
    assert (kwargs["session_id"], kwargs["source_id"]) == ("u1", "resume.pdf")


@pytest.mark.parametrize(
    "outcome",
    [{"return_value": "Ingestion failed"}, {"side_effect": RuntimeError("qdrant down")}],
)
async def test_ingest_pdfs_reports_nothing_ingested(outcome):
    rag = SimpleNamespace(ingest_document=AsyncMock(**outcome))
    assert await ingest_pdfs(rag, [_upload(_pdf("text"))], "u1") is False


# ── RagAgentService ─────────────────────────────────────────────────────────

@pytest.fixture
def graph(monkeypatch):
    """Fake compiled graph + tools; records what build_graph was given."""
    compiled = SimpleNamespace(
        ainvoke=AsyncMock(return_value={"messages": [AIMessage(content="final answer")]})
    )
    builds = []

    def build(tools, checkpointer):
        builds.append(tools)
        return compiled

    monkeypatch.setattr(service_module, "build_graph", build)
    monkeypatch.setattr(service_module, "get_mcp_tools", AsyncMock(return_value=["mcp-tool"]))
    compiled.builds = builds
    return compiled


async def test_handle_runs_graph_with_mcp_and_document_tools(graph):
    agent = RagAgentService(rag_service="rag")

    assert await agent.handle("hello", session_id="u1") == "final answer"
    assert await agent.handle("again", session_id="u1") == "final answer"

    assert graph.builds == [["mcp-tool", search_uploaded_documents]]  # built once, cached
    state, = graph.ainvoke.await_args.args
    assert state["messages"][0].content == "again"
    configurable = graph.ainvoke.await_args.kwargs["config"]["configurable"]
    assert configurable == {"thread_id": "u1", "session_id": "u1", "rag_service": "rag"}


async def test_handle_without_mcp_uses_document_search_and_retries_later(graph, monkeypatch):
    monkeypatch.setattr(service_module, "get_mcp_tools", AsyncMock(side_effect=ConnectionError))
    agent = RagAgentService()

    await agent.handle("hi", session_id="u1")
    await agent.handle("hi", session_id="u1")

    assert graph.builds == [[search_uploaded_documents], [search_uploaded_documents]]


async def test_handle_ingests_files_and_defaults_the_query(graph, monkeypatch):
    ingest = AsyncMock(return_value=True)
    monkeypatch.setattr(service_module, "ingest_pdfs", ingest)
    files = [_upload(b"%PDF")]

    await RagAgentService(rag_service="rag").handle("  ", files=files, session_id="u1")

    ingest.assert_awaited_once_with("rag", files, "u1")
    assert graph.ainvoke.await_args.args[0]["messages"][0].content.startswith("Summarize the uploaded")


async def test_handle_reports_ingestion_failure(graph, monkeypatch):
    monkeypatch.setattr(service_module, "ingest_pdfs", AsyncMock(side_effect=RuntimeError))
    result = await RagAgentService(rag_service="rag").handle("q", files=[_upload(b"x")], session_id="u1")
    assert result == "I encountered an error processing your uploaded files."
    graph.ainvoke.assert_not_awaited()


async def test_handle_rejects_empty_request(graph):
    assert await RagAgentService().handle("") == "Please enter a message or upload a document."


async def test_handle_with_no_messages_apologises(graph):
    graph.ainvoke.return_value = {"messages": []}
    assert (await RagAgentService().handle("hi")).startswith("I'm sorry")


async def test_handle_falls_back_to_plain_llm(graph, monkeypatch):
    graph.ainvoke.side_effect = RuntimeError("graph broke")
    llm = SimpleNamespace(ainvoke=AsyncMock(return_value=AIMessage(content="plain answer")))
    monkeypatch.setattr(service_module, "get_chat_model", lambda: llm)

    assert await RagAgentService().handle("hi") == "plain answer"
    system, _ = llm.ainvoke.await_args.args[0]
    assert system.content == service_module.FALLBACK_SYSTEM_PROMPT


async def test_handle_fallback_failure_apologises(graph, monkeypatch):
    graph.ainvoke.side_effect = RuntimeError("graph broke")
    monkeypatch.setattr(service_module, "get_chat_model", MagicMock(side_effect=RuntimeError))
    assert (await RagAgentService().handle("hi")).startswith("I'm sorry")


# ── POST /chat/documents ────────────────────────────────────────────────────

@pytest.fixture
def rag_chat():
    """TestClient whose requests carry `user` the way firebase_auth_middleware
    would set it, with the RAG agent swapped for a mock."""
    from main import app

    agent = SimpleNamespace(handle=AsyncMock(return_value="rag answer"))
    app.dependency_overrides[get_rag_agent] = lambda: agent

    def client_for(user):
        async def with_user(scope, receive, send):
            if scope["type"] == "http" and user is not None:
                scope = {**scope, "state": {"user": user}}
            await app(scope, receive, send)

        return TestClient(with_user)

    yield client_for, agent
    app.dependency_overrides.pop(get_rag_agent)


@pytest.mark.parametrize("user", ["uid-1", {"uid": "uid-1"}, SimpleNamespace(uid="uid-1")])
def test_chat_answers_for_the_signed_in_user(rag_chat, user):
    client_for, agent = rag_chat
    files = {"files": ("resume.pdf", b"%PDF-1.4", "application/pdf")}

    response = client_for(user).post("/chat/documents", data={"query": "my skills?"}, files=files)

    assert response.json() == {"response": "rag answer"}
    kwargs = agent.handle.await_args.kwargs
    assert (kwargs["query"], kwargs["session_id"]) == ("my skills?", "uid-1")
    assert kwargs["files"][0].filename == "resume.pdf"


def test_chat_requires_a_user(rag_chat):
    client_for, agent = rag_chat
    assert client_for(None).post("/chat/documents", data={"query": "hi"}).status_code == 401
    agent.handle.assert_not_awaited()


def test_chat_rejects_overlong_query(rag_chat):
    client_for, agent = rag_chat

    response = client_for("uid-1").post(
        "/chat/documents", data={"query": "x" * (MAX_QUERY_CHARS + 1)}
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Message must be at most 4000 characters"
    agent.handle.assert_not_awaited()


@pytest.mark.parametrize(
    ("files", "expected"),
    [
        (
            [("files", ("big.pdf", b"0" * (MAX_FILE_SIZE_BYTES + 1), "application/pdf"))],
            "'big.pdf' is larger than 1 MB",
        ),
        (
            [("files", ("notes.docx", b"data", "application/octet-stream"))],
            "Only PDF files are allowed",
        ),
        (
            [("files", (f"cv{i}.pdf", b"%PDF-1.4", "application/pdf")) for i in range(MAX_FILES + 1)],
            f"Upload at most {MAX_FILES} files at a time",
        ),
    ],
)
def test_chat_rejects_uploads_over_the_limits(rag_chat, files, expected):
    client_for, agent = rag_chat

    response = client_for("uid-1").post("/chat/documents", data={"query": "read these"}, files=files)

    assert response.status_code == 400
    assert response.json()["detail"] == expected
    agent.handle.assert_not_awaited()
