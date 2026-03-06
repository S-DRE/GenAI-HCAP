import os
from typing import Annotated

import structlog
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from src.guardrails.validators import validate_response
from src.tools.escalation import escalate
from src.tools.rag import retrieve_care_info

logger = structlog.get_logger()

SYSTEM_PROMPT = """You are a home care assistant helping elderly patients and their caregivers.
You have access to the patient's care plans and medical guidelines.

Rules you must always follow:
- Never provide a medical diagnosis.
- Never recommend changing medication dosages.
- Always escalate to a caregiver or medical staff if the patient reports an emergency or severe symptoms.
- Keep responses clear, calm, and simple — your users may be elderly or under stress.
- Always base your answers on the retrieved care information when available."""

TOOLS = [retrieve_care_info, escalate]


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def build_graph() -> StateGraph:
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=os.environ.get("GROQ_API_KEY"),
    ).bind_tools(TOOLS)

    tool_node = ToolNode(TOOLS)

    def call_llm(state: AgentState) -> AgentState:
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        response = llm.invoke(messages)
        logger.info("llm_response", tool_calls=bool(response.tool_calls))
        return {"messages": [response]}

    def should_continue(state: AgentState) -> str:
        last = state["messages"][-1]
        return "tools" if last.tool_calls else END

    graph = StateGraph(AgentState)
    graph.add_node("llm", call_llm)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("llm")
    graph.add_conditional_edges("llm", should_continue)
    graph.add_edge("tools", "llm")

    return graph.compile()


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


async def run_agent(user_message: str) -> str:
    result = await get_graph().ainvoke({"messages": [HumanMessage(content=user_message)]})
    raw_response = result["messages"][-1].content
    return validate_response(raw_response)
