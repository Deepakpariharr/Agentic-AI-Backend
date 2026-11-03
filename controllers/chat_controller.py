import logging
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langsmith import traceable

from config import DISABLE_AUTH
from dependencies import get_rag_agent
from middleware.rate_limiter import custom_key_func, limiter
from model.chatModels import ChatRequest
from model.response_models import (
    ResponseModel,
    error_response,
    server_error_response,
    success_response,
    validation_error_response,
)
from services.agent_service import agent_response
from services.conversation_service import (
    ConversationNotFound,
    ConversationsDisabled,
    resolve_thread,
    touch,
)
from services.openai_service import openai_response
from services.rag_agent import RagAgentService
from utility.request_limits import (
    limits,
    validate_query,
    validate_upload,
    validate_uploads,
)

security = HTTPBearer(auto_error=not DISABLE_AUTH)
logger = logging.getLogger(__name__)


router = APIRouter(prefix="/chat", tags=["chat"])


# The query/upload limits the chat endpoints enforce (utility/request_limits.py),
# so the frontend can reject oversized input before uploading it.
@router.get("/limits", response_model=ResponseModel)
async def chat_limits():
    return success_response(data=limits(), message="Chat limits")


# This Router is Used for Directly Connecting the MCP and Open AI
# Will be Depriciated in Upcoming Advancements
# This one Is Depreciated 
@router.post("/openAIChat", response_model=ResponseModel)
@limiter.limit("5/minute")
@traceable()
async def openAIChat(
    request: Request,
    body: ChatRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    try:
        reply = openai_response(body.message)
        return success_response(
            data={
                "response": reply,
            },
            message="Chat response generated successfully",
        )

    except Exception as e:
        return server_error_response(
            message=f"Failed to generate chat response: {str(e)}"
        )



# Traceable will be used for the Tracing the Request and It will go to Langsmith
@router.post("/mcpAgentChat", response_model=ResponseModel)
@limiter.limit("5/minute")
@traceable()
async def mcpAgentChat(
    request: Request,
    message: str = Form(...),
    file: UploadFile = File(None),
    conversation_id: Optional[str] = Form(None),
):
    try:
        message = message.strip()
        if error := validate_query(message):
            return validation_error_response(message=error)
        if file and (error := await validate_upload(file)):
            return validation_error_response(message=error)

        user_id = custom_key_func(request)

        # Optional: continue a stored conversation (POST /conversations) instead of
        # the user's single default thread.
        try:
            thread_id = await resolve_thread(user_id, "mcp", conversation_id)
        except ConversationsDisabled:
            return validation_error_response(message="Conversations are not enabled on this server")
        except ConversationNotFound:
            return error_response(status_code=404, message="Conversation not found.")

        agent_kwargs = {"thread_id": thread_id} if thread_id else {}
        result = await agent_response(
            message, file, user_id=user_id, **agent_kwargs
        )

        if not result:
            return error_response(status_code=500, message="Error Occured", data=None)
        await touch(user_id, thread_id, message)
        data = {"response": result}
        if thread_id:
            data["conversation_id"] = thread_id
        return success_response(
            data=data,
            message="Chat response generated successfully",
        )
    except Exception as ex:
        logger.exception(f"mcpAgentChat failed {ex}")
        return server_error_response(
            message="Failed to Generate chat response Please try after sometime"
        )


# RAG chat: answers from the user's uploaded documents (indexed per user) plus
# the MCP tools. Uploaded PDFs are ingested first, so /documents can list them.
@router.post("/documents")
@limiter.limit("10/minute")
async def chat(
    request: Request,
    query: str = Form(...),
    files: Optional[list[UploadFile]] = File(None),
    conversation_id: Optional[str] = Form(None),
    agent: RagAgentService = Depends(get_rag_agent),
):
    user = getattr(request.state, "user", None)

    if isinstance(user, dict):
        session_id = user.get("uid")
    elif isinstance(user, str):
        session_id = user
    else:
        session_id = getattr(user, "uid", None)

    if not session_id:
        raise HTTPException(status_code=401, detail="Unauthorized: No valid session.")

    # A blank query is fine with files: the agent summarises them instead.
    if error := validate_query(query):
        raise HTTPException(status_code=400, detail=error)
    if files and (error := await validate_uploads(files)):
        raise HTTPException(status_code=400, detail=error)

    # Optional: continue a stored conversation instead of the user's default thread.
    try:
        thread_id = await resolve_thread(session_id, "rag", conversation_id)
    except ConversationsDisabled:
        raise HTTPException(status_code=400, detail="Conversations are not enabled on this server")
    except ConversationNotFound:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    agent_kwargs = {"thread_id": thread_id} if thread_id else {}
    response_text = await agent.handle(
        query=query, files=files, session_id=session_id, **agent_kwargs
    )
    await touch(session_id, thread_id, query)
    body = {"response": response_text}
    if thread_id:
        body["conversation_id"] = thread_id
    return body
