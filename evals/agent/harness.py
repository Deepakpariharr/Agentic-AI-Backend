"""Drives the real agent for evals and records what it did.

agent_response() only returns the final text, but tool-use metrics also need
to know which tools ran. After each call the turn is read back out of the
agent's checkpointed state (same thread_id), so the production function's
signature stays untouched and the evals exercise exactly what the API does.

All agent calls run inside one long-lived asyncio task on a background
thread. The MCP sessions opened on the first call are bound to the loop *and*
the task that opened them (anyio cancel scopes must be exited in the task they
were entered in), so a fresh loop per test, or even a new task per call on the
same loop, would break the later calls or the final close.
"""
import asyncio
import concurrent.futures
import io
import os
import threading
import uuid
from dataclasses import dataclass, field

import fitz  # PyMuPDF
from deepeval.test_case import ToolCall, Turn
from fastapi import UploadFile
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from config import OPEN_API_KEY, orgnId

# deepeval's judge reads the standard OpenAI variables; this app keeps its key
# under OPEN_API_KEY. Set before anything builds a judge model.
if OPEN_API_KEY:
    os.environ.setdefault("OPENAI_API_KEY", OPEN_API_KEY)
if orgnId:
    os.environ.setdefault("OPENAI_ORG_ID", orgnId)

from services import agent_service  # noqa: E402
from services.mcp_client import close_mcp_client  # noqa: E402

# The model that grades answers. Should be stronger than the agent's own
# gpt-3.5-turbo. gpt-5-nano is the only other model this OpenAI project can
# use today; override per run with DEEPEVAL_JUDGE_MODEL=gpt-4.1 etc.
JUDGE_MODEL = os.getenv("DEEPEVAL_JUDGE_MODEL", "gpt-5-nano")

_ERROR_PREFIX = "❌ Error"

_loop = asyncio.new_event_loop()
_queue: asyncio.Queue = asyncio.Queue()


async def _worker() -> None:
    """Await every submitted coroutine directly in this one task."""
    while True:
        coro, result = await _queue.get()
        try:
            result.set_result(await coro)
        except BaseException as exc:  # handed back to the calling test
            result.set_exception(exc)
        if coro.__name__ == "close_mcp_client":
            return


threading.Thread(target=_loop.run_until_complete, args=(_worker(),), daemon=True).start()


def _call(coro):
    result = concurrent.futures.Future()
    _loop.call_soon_threadsafe(_queue.put_nowait, (coro, result))
    return result.result()


@dataclass
class AgentRun:
    output: str
    tools_called: list[ToolCall] = field(default_factory=list)

    @property
    def tool_names(self) -> list[str]:
        return [t.name for t in self.tools_called]


def new_user_id() -> str:
    """A fresh thread_id, so no test sees another test's chat history."""
    return f"eval-{uuid.uuid4().hex[:8]}"


async def _run(prompt: str, user_id: str, file: UploadFile | None) -> AgentRun:
    output = await agent_service.agent_response(prompt, file, user_id=user_id)
    if output.startswith(_ERROR_PREFIX):
        # agent_response swallows exceptions; grading that string would just
        # produce a confusing low score. Fail loudly instead (see app logs).
        raise RuntimeError(f"Agent errored on {prompt!r}: {output}")

    agent = await agent_service._get_agent()
    state = await agent.aget_state({"configurable": {"thread_id": user_id}})
    messages = state.values.get("messages", [])

    # Only this turn: everything after the most recent human message.
    last_human = max(
        (i for i, m in enumerate(messages) if isinstance(m, HumanMessage)),
        default=-1,
    )
    turn = messages[last_human + 1 :]
    results = {m.tool_call_id: m.content for m in turn if isinstance(m, ToolMessage)}
    tools = [
        ToolCall(name=call["name"], input_parameters=call["args"], output=results.get(call["id"]))
        for m in turn
        if isinstance(m, AIMessage)
        for call in m.tool_calls
    ]
    return AgentRun(output=output, tools_called=tools)


def run_agent(prompt: str, *, user_id: str, pdf: bytes | None = None, pdf_name: str = "upload.pdf") -> AgentRun:
    """One user turn through agent_response, optionally with a PDF upload."""
    file = UploadFile(io.BytesIO(pdf), filename=pdf_name) if pdf else None
    return _call(_run(prompt, user_id, file))


def run_conversation(user_messages: list[str], *, user_id: str) -> list[Turn]:
    """Play several user turns in one thread and return the full transcript."""
    turns: list[Turn] = []
    for message in user_messages:
        run = run_agent(message, user_id=user_id)
        turns.append(Turn(role="user", content=message))
        turns.append(Turn(role="assistant", content=run.output, tools_called=run.tools_called or None))
    return turns


def make_pdf(text: str) -> bytes:
    """Build a one-page PDF in memory, so PDF evals need no fixture files."""
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), text, fontsize=11)
    return doc.tobytes()


def shutdown() -> None:
    _call(close_mcp_client())  # also ends the worker, and with it the loop thread
