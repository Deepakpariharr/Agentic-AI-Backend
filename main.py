from contextlib import asynccontextmanager

from aws_lambda_powertools import Logger
from aws_lambda_powertools.logging.utils import copy_config_to_registered_loggers
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded

from config import DISABLE_AUTH, SSM_PREFIX
from controllers.chat_controller import router as chat_router
from controllers.documents_controller import router as documents_router
from dependencies import get_rag_service
from middleware.firebase_middleware import firebase_auth_middleware
from middleware.rate_limiter import limiter
from middleware.request_id import request_id_middleware
from model.response_models import ResponseModel, success_response
from services.mcp_client import close_mcp_client
from services.rag.retrieval import CollectionSchemaError

logger = Logger(service="mcpbe")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: connect the shared RAG service (the same instance /chat and
    # /documents use). MCP tool connections are opened lazily on first use.
    try:
        get_rag_service()
        logger.info("✅ RAG service initialized successfully at startup")
    except CollectionSchemaError:
        # Incompatible existing collection: fail fast rather than run degraded
        # (and never touch the stored documents).
        logger.critical("❌ Qdrant collection schema is incompatible; refusing to start.")
        raise
    except Exception as exc:
        logger.warning("⚠️ RAG service unavailable at startup: %s", exc)

    yield

    # Shutdown: close every MCP session and the tool-server subprocess.
    await close_mcp_client()


app = FastAPI(
    title="DBAI Chat Learn and Grow",
    description="Two-tier architecture with Database Manager and Controllers",
    version="1.0.1",
    lifespan=lifespan,
)

# ------------------ CORS ------------------
# Keep these production origins in sync with the frontend deployments.
origins = [
    "https://dbaichat.space",
    "https://mcpfrontend.web.app",
]
# The frontend's dev server, allowed only when running locally (SSM_PREFIX is
# set on Lambda, unset locally).
if not SSM_PREFIX:
    origins.append("http://localhost:3000")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)

# ------------------ Middleware ------------------
if not DISABLE_AUTH:
    app.middleware("http")(firebase_auth_middleware)
else:
    logger.warning("DISABLE_AUTH is set — running without Firebase authentication.")
app.state.limiter = limiter

app.middleware("http")(request_id_middleware)
copy_config_to_registered_loggers(source_logger=logger)


# ------------------ Rate limit handler ------------------
async def custom_exception_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={
            "success": False,
            "message": "You have reached your limit. Please try again after 1 minute.",
            "data": None,
        },
        headers={"Retry-After": "60"},
    )

app.add_exception_handler(RateLimitExceeded, custom_exception_handler)

# ------------------ Routers ------------------
app.include_router(chat_router)
app.include_router(documents_router)


# ------------------ Health ------------------
# Health Checking Of Service :)
@app.get("/", response_model=ResponseModel)
async def root():
    return success_response(
        data={
            "status": "healthy",
            "version": "1.0.0",
            "timestamp": "2025-10-26T06:30:00Z"
        },
        message="FastAPI MCP Backend is running!"
    )


if __name__ == "__main__":
    import uvicorn
    # Pass the app object (not "main:app") so main.py isn't imported twice;
    # access_log off because request_id_middleware already logs each request.
    uvicorn.run(app, host="0.0.0.0", port=8000, access_log=False)
