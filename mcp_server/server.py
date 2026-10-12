"""The MCP server this app exposes: career tools for the chat agent.

Spawned over stdio by services/mcp_client.py — the agent reaches these tools
as an MCP client, never through direct Python imports. Each tool lives in its
own module under tools/; this file only decides what gets exposed and how it
is described to the client.

Run it standalone from the project root:
    python -m mcp_server
    npx @modelcontextprotocol/inspector python -m mcp_server   # browse/call tools
"""
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from mcp_server.tools.calculator import calculate
from mcp_server.tools.jobs import search_jobs
from mcp_server.tools.web_search import search_web

mcp = FastMCP("career-tools")

mcp.add_tool(
    calculate,
    title="Calculator",
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False),
)
mcp.add_tool(
    search_jobs,
    title="Job search",
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True),
)
mcp.add_tool(
    search_web,
    title="Web search",
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True),
)
