import os
from abc import ABC, abstractmethod
from typing import Annotated

import structlog
from groq import BadRequestError
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from src.guardrails.validators import ResponseValidator, default_validator
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


class LLMProvider(ABC):
    @abstractmethod
    def get_llm_with_tools(self, tools: list): ...

    @abstractmethod
    def get_llm(self): ...


class GroqLLMProvider(LLMProvider):
    def __init__(self) -> None:
        self._model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
        self._api_key = os.environ.get("GROQ_API_KEY")
        logger.info("groq_llm_provider_init", model=self._model)

    def get_llm_with_tools(self, tools: list):
        return ChatGroq(
            model=self._model,
            temperature=0,
            api_key=self._api_key,
        ).bind_tools(tools)

    def get_llm(self):
        return ChatGroq(
            model=self._model,
            temperature=0,
            api_key=self._api_key,
        )


class GraphBuilder:
    def __init__(
        self,
        llm_provider: LLMProvider | None = None,
        tools: list | None = None,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        self._llm_provider = llm_provider or GroqLLMProvider()
        self._tools = tools if tools is not None else TOOLS
        self._system_prompt = system_prompt

    def build(self) -> StateGraph:
        llm = self._llm_provider.get_llm_with_tools(self._tools)
        llm_no_tools = self._llm_provider.get_llm()
        tool_node = ToolNode(self._tools)

        def call_llm(state: AgentState) -> AgentState:
            messages = [SystemMessage(content=self._system_prompt)] + state["messages"]
            try:
                response = llm.invoke(messages)
            except BadRequestError as exc:
                # Smaller models (e.g. 8B) sometimes emit malformed tool-call JSON;
                # retry without tools so the request still completes.
                if "tool_use_failed" in str(exc):
                    logger.warning(
                        "llm_tool_call_failed_retrying_without_tools",
                        error=str(exc)[:200],
                    )
                    response = llm_no_tools.invoke(messages)
                else:
                    raise
            logger.info("llm_response", tool_calls=bool(getattr(response, "tool_calls", [])))
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
        _graph = GraphBuilder().build()
    return _graph


async def run_agent(
    user_message: str,
    *,
    validator: ResponseValidator | None = None,
) -> str:
    _validator = validator or default_validator
    result = await get_graph().ainvoke({"messages": [HumanMessage(content=user_message)]})
    raw_response = result["messages"][-1].content
    return _validator.validate(raw_response)
