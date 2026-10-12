"""Process-wide singletons handed to routes through FastAPI's Depends()."""
from services.rag import RAGService
from services.rag_agent import RagAgentService

_rag_service_instance: RAGService | None = None
_rag_agent_instance: RagAgentService | None = None


def get_rag_service() -> RAGService:
    global _rag_service_instance
    if _rag_service_instance is None:
        _rag_service_instance = RAGService()
    return _rag_service_instance


def get_rag_agent() -> RagAgentService:
    global _rag_agent_instance
    if _rag_agent_instance is None:
        _rag_agent_instance = RagAgentService(rag_service=get_rag_service())
    return _rag_agent_instance
