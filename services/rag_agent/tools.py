import logging
from langchain_core.tools import tool
from langchain_core.runnables.config import RunnableConfig

from services.rag.service import NO_CONTEXT_MESSAGE

logger = logging.getLogger(__name__)

@tool
async def search_uploaded_documents(query: str, config: RunnableConfig) -> str:
    """
    Searches the current user's uploaded documents, resume, or background data
    in the vector database. ALWAYS invoke this tool first when the user asks
    questions about their uploaded files. Returns labelled excerpts such as
    [resume.pdf#3]; cite those labels in the answer.
    """
    configurable = config.get("configurable", {})
    rag_service = configurable.get("rag_service")
    session_id = configurable.get("session_id") or configurable.get("thread_id")

    if not rag_service or not session_id:
        logger.warning("Missing rag_service or session_id in RunnableConfig.")
        return "Document service unavailable: missing session context."

    try:
        context = await rag_service.search_context(query=query, session_id=session_id)
        if not context or context == NO_CONTEXT_MESSAGE:
            return "No matching content found in uploaded documents for this session."
        return context
    except Exception as e:
        logger.error(f"Document tool execution failed: {e}", exc_info=True)
        return "Error reading documents from vector storage."