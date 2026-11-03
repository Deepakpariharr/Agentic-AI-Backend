"""The RAG agent's LangGraph: agent node ⇄ tool node until the model answers."""
from typing import Annotated, TypedDict

from langchain_core.messages import SystemMessage, trim_messages
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from prompts import RAG_SYSTEM_PROMPT
from services.llm_factory import get_chat_model


# Messages sent to the model per turn. The stored conversation is never cut, only
# what the model sees, so a long chat can't keep growing the prompt and the bill.
MAX_HISTORY_MESSAGES = 20


def _recent(messages: list) -> list:
    """Last MAX_HISTORY_MESSAGES, starting on a human turn so a tool result is
    never separated from the AI message that requested it (OpenAI rejects that)."""
    kept = trim_messages(
        messages,
        max_tokens=MAX_HISTORY_MESSAGES,
        token_counter=len,  # count messages, not tokens
        strategy="last",
        start_on="human",
        include_system=False,
    )
    return kept or messages


class AgentState(TypedDict):
    session_id: str

    # Annotated with add_messages is REQUIRED for tool calling agents
    # This ensures ToolMessages and AIMessages append chronologically
    messages: Annotated[list, add_messages]


def build_graph(tools: list[BaseTool], checkpointer: BaseCheckpointSaver):
    """Assembles and compiles the autonomous Tool-Calling Agent graph."""
    llm_with_tools = get_chat_model().bind_tools(tools)
    system_prompt = SystemMessage(content=RAG_SYSTEM_PROMPT)

    async def agent_node(state: AgentState, config: RunnableConfig) -> dict:
        """General-purpose assistant node with bound tools."""
        response = await llm_with_tools.ainvoke(
            [system_prompt, *_recent(state.get("messages", []))], config=config
        )
        return {"messages": [response]}

    graph = StateGraph(AgentState)

    # 1. Add the agent node (file ingestion happens in RagAgentService.handle,
    #    before the graph runs)
    graph.add_node("agent", agent_node)

    # 2. Add LangGraph's prebuilt ToolNode
    # This executes the tool calls the model asks for (MCP tools + document search)
    graph.add_node("tools", ToolNode(tools))

    # 3. Define the start flow
    graph.add_edge(START, "agent")

    # 4. The Magic Conditional Routing
    # If the LLM output contains a tool call, this routes to 'tools'.
    # If it's a normal text response, this routes to END.
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})

    # 5. After tools finish executing, return the results to the agent
    graph.add_edge("tools", "agent")

    return graph.compile(checkpointer=checkpointer)
