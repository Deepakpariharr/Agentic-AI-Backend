"""Owns the MCP client connections that back the agent's toolset.

Servers are declared in `_MCP_SERVERS` below — adding one is a dict entry,
not new code. Two transports are supported:

* **stdio** — a local command spawned as a subprocess (our own `mcp_server`
  package, or any `npx`-style community server).
* **http**  — somebody else's hosted MCP server, reached over the network
  (most third-party servers).

Every session is opened once and kept alive for the lifetime of the process
— repeated chat requests reuse them instead of paying spawn + handshake cost
every time (see services/agent_service.py, which caches the agent the same
way). Tools from all servers are merged into one flat list, which is all the
agent ever sees.

Servers marked `required` bring the app down if they fail; everything else
degrades to "that server's tools are missing" and logs why.

Safe to reuse across warm AWS Lambda invocations too: Mangum drives the ASGI
app with `asyncio.get_event_loop()` and does not close it between
invocations within the same warm container, so this module-level cache
stays valid across requests there as well.
"""
import asyncio
import logging
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client

from config import RAPIDAPI_KEY, TAVILY_API_KEY

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# The MCP SDK starts stdio servers with only HOME/LOGNAME/PATH/SHELL/TERM/USER
# from our environment, so SSM_PREFIX never reaches the child and its config.py
# can't load from SSM. Pass on just the keys its tools need, as config.py in
# this process already resolved them (SSM on Lambda, .env locally).
_CAREER_TOOLS_ENV = {
    key: value
    for key, value in {"RAPIDAPI_KEY": RAPIDAPI_KEY, "TAVILY_API_KEY": TAVILY_API_KEY}.items()
    if value
}


# ─────────────────────────────────────────────────────────────────────────────
# Server registry — the only thing you edit to add/remove an MCP server.
#
#   transport : "stdio" (spawn a subprocess) or "http" (call a hosted server)
#   enabled   : usually a truthiness check on whatever credential it needs
#   required  : True  -> a failure here is fatal (app can't work without it)
#               False -> log and carry on without that server's tools
#
#   stdio keys : command, args, cwd, env (all but command optional)
#   http  keys : url, headers (optional)
# ─────────────────────────────────────────────────────────────────────────────
_MCP_SERVERS: dict[str, dict[str, Any]] = {
    # Our own tools (mcp_server/): calculate / search_jobs / search_web.
    # `sys.executable` (not a bare "python") so the subprocess runs the same
    # virtualenv interpreter the API is running under.
    "career-tools": {
        "transport": "stdio",
        "command": sys.executable,
        "args": ["-m", "mcp_server"],
        "cwd": str(_PROJECT_ROOT),
        "env": _CAREER_TOOLS_ENV,
        "enabled": True,
        "required": True,
    },
}


_exit_stack: AsyncExitStack | None = None
_sessions: dict[str, ClientSession] = {}
_tools: list[BaseTool] | None = None
_init_lock = asyncio.Lock()


async def _connect(stack: AsyncExitStack, cfg: dict[str, Any]) -> ClientSession:
    """Open one MCP session from a registry entry, whatever its transport.

    Both transports end up in the same place — a pair of streams wrapped in a
    ClientSession — which is why everything above this function is unaware of
    whether a tool runs in a child process or in someone else's datacentre.
    """
    transport = cfg["transport"]

    if transport == "stdio":
        params = StdioServerParameters(
            command=cfg["command"],
            args=cfg.get("args", []),
            cwd=cfg.get("cwd"),
            env=cfg.get("env"),
        )
        read, write = await stack.enter_async_context(stdio_client(params))
    elif transport == "http":
        # streamablehttp_client yields a third value (a session-id getter)
        # that the stdio transport has no equivalent for; we don't need it.
        read, write, _ = await stack.enter_async_context(
            streamablehttp_client(cfg["url"], headers=cfg.get("headers"))
        )
    else:
        raise ValueError(f"Unknown MCP transport: {transport!r}")

    session = await stack.enter_async_context(ClientSession(read, write))
    await session.initialize()
    return session


async def get_mcp_tools() -> list[BaseTool]:
    """Return LangChain-compatible tools backed by every enabled MCP server,
    opening the connections on first call and reusing them after."""
    global _exit_stack, _tools

    if _tools is not None:
        return _tools

    async with _init_lock:
        if _tools is not None:  # another request may have won the race
            return _tools

        stack = AsyncExitStack()
        tools: list[BaseTool] = []
        try:
            for name, cfg in _MCP_SERVERS.items():
                if not cfg.get("enabled", True):
                    logger.warning(
                        "MCP server %r disabled — %s",
                        name,
                        cfg.get("disabled_reason", "not enabled"),
                    )
                    continue

                try:
                    logger.info(
                        "Connecting to MCP server %r (%s)", name, cfg["transport"]
                    )
                    _sessions[name] = await _connect(stack, cfg)
                    tools += await load_mcp_tools(_sessions[name])
                except Exception:
                    _sessions.pop(name, None)
                    if cfg.get("required"):
                        raise
                    logger.exception(
                        "MCP server %r unavailable — continuing without its tools",
                        name,
                    )

            _exit_stack = stack
            _tools = tools
            logger.info(
                "MCP ready with %d tool(s) from %s: %s",
                len(tools),
                list(_sessions),
                [t.name for t in tools],
            )
            return _tools
        except Exception:
            logger.exception("Failed to start MCP client")
            _sessions.clear()
            await stack.aclose()
            raise


#this one is used for closing the work mcp is doing while working with this
async def close_mcp_client() -> None:
    """Tear down every MCP session/subprocess. Call on app shutdown."""
    global _exit_stack, _tools
    if _exit_stack is not None:
        try:
            await _exit_stack.aclose()
        except Exception:
            # Shutdown is best-effort: a transport that already died must not
            # turn into a crash on the way out.
            logger.exception("Error while closing MCP connections")
    _exit_stack = None
    _sessions.clear()
    _tools = None
