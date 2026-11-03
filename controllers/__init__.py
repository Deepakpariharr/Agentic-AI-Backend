# Controllers package
"""
Controllers module for FastAPI MCP Backend
Handles all API endpoints and routing
"""

from .chat_controller import router as chat_router
from .documents_controller import router as documents_router

# Export all routers for easy import
__all__ = [
    'chat_router',
    'documents_router',
]