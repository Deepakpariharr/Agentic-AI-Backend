import asyncio
import base64
import logging

from fastapi import UploadFile
from langchain.agents import create_agent
from langchain.agents.middleware import AgentState, Runtime, before_model
from langchain_core.messages import HumanMessage, RemoveMessage, trim_messages
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langsmith import traceable

from config import OPEN_API_KEY, orgnId
from prompts import MCP_AGENT_SYSTEM_PROMPT
from services.mcp_client import get_mcp_tools
from services.pdf_reader import read_pdf as pdf_reader_fn

logger = logging.getLogger(__name__)


memory=MemorySaver()

# How much chat history to keep per user. Older messages are dropped from the
# checkpointed state entirely, so a long conversation can't keep growing the
# prompt — and the token bill — on every turn.
MAX_HISTORY_MESSAGES = 5


@before_model
def _trim_history(state: AgentState, runtime: Runtime) -> dict | None:
    """Keep only the most recent messages in the conversation state.

    trim_messages handles the bookkeeping that naive slicing gets wrong: it
    never leaves a ToolMessage stranded without the AI message that asked for
    it, which the OpenAI API rejects outright.
    """
    messages = state["messages"]
    kept = trim_messages(
        messages,
        max_tokens=MAX_HISTORY_MESSAGES,
        token_counter=len,  # count messages, not tokens
        strategy="last",
        start_on="human",
        include_system=True,
    )

    # Nothing to drop, or trimming would leave the model with nothing to answer
    # (possible mid tool-loop, when the window holds no human message).
    if not kept or len(kept) == len(messages):
        return None

    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *kept]}


# Built once and reused across requests — this doesn't depend on anything
# request-scoped, unlike the tools (see _get_agent below).

# this is we are sending the Model for creating the Agents
_llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    api_key=OPEN_API_KEY,
    organization=orgnId,
)

_agent = None
_agent_lock = asyncio.Lock()


#it will be used for getting the agent and used Locking so agent is creating only one Time
async def _get_agent():
    """Build the LangChain agent once, wiring in tools served by the MCP
    server (services/mcp_client.py), and reuse it across requests instead of
    recompiling the graph on every call."""
    global _agent
    if _agent is not None:
        return _agent

    async with _agent_lock:
        if _agent is not None:  # another request may have won the race
            return _agent

        tools = await get_mcp_tools()
        _agent = create_agent(
            model=_llm,
            tools=tools,
            system_prompt=MCP_AGENT_SYSTEM_PROMPT,
            middleware=[_trim_history],
            checkpointer=memory
        )
        return _agent



@traceable()
async def agent_response(
    user_prompt: str, file: UploadFile = None, *, user_id: str
) -> str:
    """
    AI agent that automatically decides when and which tools to call.
    Tools are served by a real MCP server (mcp_server/server.py) over stdio.
    Supports multi-step reasoning (chaining tools) via a LangChain tool-calling agent.

    user_id is keyword-only on purpose: it becomes the checkpointer's thread_id,
    so getting it wrong would hand one user another user's chat history.
    """
    try:
        # 🧠 Step 1: Build the user input, folding in PDF content if uploaded.
        # Done as a direct local call (not an MCP round-trip) since this is
        # deterministic pre-processing, not something the agent is deciding to do
        input_text = user_prompt
        
        if file:
            file_bytes = await file.read()
            pdf_base64 = base64.b64encode(file_bytes).decode("utf-8")
            pdf_text = pdf_reader_fn(file_content=pdf_base64)

            input_text += (
                f"\n\nThe user uploaded a PDF named '{file.filename}'. "
                f"Here is the extracted content:\n\n{pdf_text}"
            )

        # 🧩 Step 2: Get the cached agent (MCP tools fetched once, lazily)
        agent = await _get_agent()

        # 🧩 Step 3: Run the agent (LangChain handles the multi-tool reasoning loop)
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=input_text)]},
            # thread_id scopes the checkpointed history to one user, so each
            # caller resumes their own conversation and nobody else's.
            config={
                "recursion_limit": 12,  # roughly mirrors the old 5-step cap
                "configurable": {"thread_id": user_id},
            },
        )

        final_message = result["messages"][-1]
        final_answer = (final_message.content or "").strip()
        logger.info("✅ Final model response (LangChain agent via MCP).")
        return final_answer or "🤔 I couldn’t find a result please try after sometime"

    except Exception as e:
        logger.exception(f"❌ Error Occured Please try after sometime {e}")
        return "❌ Error Occured Please try after sometime"
