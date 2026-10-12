"""`python -m mcp_server` — serve the tools over stdio."""
from mcp_server.server import mcp

mcp.run(transport="stdio")
