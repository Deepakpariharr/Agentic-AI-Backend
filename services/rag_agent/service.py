import asyncio
import logging
from typing import List, Optional

from fastapi import UploadFile
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver

from prompts import FALLBACK_SYSTEM_PROMPT
from services.llm_factory import get_chat_model
from services.mcp_client import get_mcp_tools
from services.rag import RAGService
from services.rag_agent.graph import build_graph
from services.rag_agent.tools import search_uploaded_documents
from services.rag_agent.uploads import ingest_pdfs

logger = logging.getLogger(__name__)


class RagAgentService:
    """Chat over the user's uploaded documents plus the MCP tools (POST /chat/documents)."""

    def __init__(self, rag_service: Optional[RAGService] = None):
        self.rag_service = rag_service
        # Shared by every graph this service builds, so a conversation's
        # history survives a rebuild after the MCP tools come back.
        self._memory = MemorySaver()
        self._graph = None
        self._graph_lock = asyncio.Lock()

    async def _get_graph(self):
        """Compile the graph once the MCP tools are available.

        If the tool server is down the graph still runs with document search
        only, but isn't cached — the next request tries MCP again.
        """
        if self._graph is not None:
            return self._graph

        async with self._graph_lock:
            if self._graph is not None:  # another request may have won the race
                return self._graph
            try:
                mcp_tools = await get_mcp_tools()
            except Exception as exc:
                logger.warning("MCP tools unavailable; answering with document search only: %s", exc)
                return build_graph([search_uploaded_documents], self._memory)

            self._graph = build_graph([*mcp_tools, search_uploaded_documents], self._memory)
            return self._graph

    async def _fallback_response(self, query: str) -> str:
        try:
            llm = get_chat_model()
            messages = [
                SystemMessage(content=FALLBACK_SYSTEM_PROMPT),
                HumanMessage(content=query),
            ]
            res = await llm.ainvoke(messages)
            return str(res.content)
        except Exception as exc:
            logger.error("Direct LLM fallback execution failed: %s", exc)
            return "I'm sorry, I could not process your request at this time."

    async def handle(
        self,
        query: str,
        files: Optional[List[UploadFile]] = None,
        session_id: str = "default_session",
    ) -> str:
        # 1. INGEST FILES FIRST: Process upload before validating text query
        if files and self.rag_service:
            logger.info("Ingesting %d files for session %s", len(files), session_id)
            try:
                await ingest_pdfs(self.rag_service, files, session_id)
            except Exception as e:
                logger.error("File ingestion failed: %s", e, exc_info=True)
                return "I encountered an error processing your uploaded files."

        # 2. DEFAULT QUERY FOR FILE-ONLY UPLOADS
        clean_query = query.strip() if query else ""
        if not clean_query:
            if files:
                clean_query = "Summarize the uploaded document and extract key highlights."
            else:
                return "Please enter a message or upload a document."

        # 3. PREPARE AGENT STATE
        initial_state = {
            "session_id": session_id,
            "messages": [HumanMessage(content=clean_query)],
        }

        config = {
            "configurable": {
                "thread_id": session_id,
                "session_id": session_id,
                "rag_service": self.rag_service,
            }
        }

        # 4. EXECUTE GRAPH (FALLBACK: plain LLM if the graph can't be built or run)
        try:
            graph = await self._get_graph()
            final_state = await graph.ainvoke(initial_state, config=config)

            messages = final_state.get("messages", [])
            if messages:
                return str(messages[-1].content)
            return "I'm sorry, I could not process your request at this time."

        except Exception as exc:
            logger.error("Graph execution failed for session %s: %s", session_id, exc, exc_info=True)
            return await self._fallback_response(clean_query)
