import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from firebase_admin import auth

from dependencies import get_rag_service
from middleware.firebase_middleware import _extract_uid_from_state
from middleware.rate_limiter import limiter
from model.response_models import error_response, not_found_response, success_response
from services import document_registry
from services.rag import RAGService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])


def _require_session_id(request: Request) -> str:
    session_id = _extract_uid_from_state(request)

    if not session_id:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.lower().startswith("bearer "):
            token = auth_header.split(None, 1)[1].strip()
            try:
                decoded = auth.verify_id_token(token)
            except Exception:
                raise HTTPException(status_code=401, detail="Invalid or expired Firebase token")
            session_id = decoded.get("uid")
            if session_id:
                request.state.user = session_id
                request.state.uid = session_id

    if not session_id:
        raise HTTPException(status_code=401, detail="Unauthorized: No valid session.")
    return session_id


@router.get("")
@limiter.limit("30/minute")
async def list_documents(
    request: Request,
    rag: RAGService = Depends(get_rag_service),
):
    session_id = _require_session_id(request)

    # Postgres knows filenames, sizes and status; fall back to counting chunks in
    # Qdrant when there is no database (or it is unreachable).
    documents = await document_registry.list_documents(session_id)
    if documents is not None:
        return success_response(data={"documents": documents}, message="Documents retrieved")

    try:
        documents = await rag.list_documents(session_id=session_id)
    except Exception as e:
        logger.error(f"Failed to list documents for session {session_id}: {e}", exc_info=True)
        return error_response(message="Failed to list documents.")

    return success_response(data={"documents": documents}, message="Documents retrieved")


# Source Id is Related to the Unique of Document in Vector
@router.delete("/{source_id:path}")
@limiter.limit("20/minute")
async def delete_document(
    request: Request,
    source_id: str,
    rag: RAGService = Depends(get_rag_service),
):
    session_id = _require_session_id(request)

    try:
        deleted_chunks = await rag.delete_document(session_id=session_id, source_id=source_id)
    except Exception as e:
        logger.error(f"Failed to delete document {source_id!r} for session {session_id}: {e}", exc_info=True)
        return error_response(message="Failed to delete document.")

    # Chunks first, registry row last: if the Qdrant delete failed we returned above and
    # the row remains, so the user can retry. A leftover row (e.g. a failed upload with
    # no chunks) is cleaned up here too.
    had_row = await document_registry.forget(session_id, source_id)

    if deleted_chunks == 0 and not had_row:
        return not_found_response(message=f"No document named '{source_id}' found for this user.")

    return success_response(
        data={"source_id": source_id, "deleted_chunks": deleted_chunks},
        message=f"Deleted '{source_id}' ({deleted_chunks} chunks).",
    )
