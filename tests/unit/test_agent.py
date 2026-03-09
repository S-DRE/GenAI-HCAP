"""Tests for the LangGraph agent.

After the SOLID refactor:
- LLMProvider is an abstract interface; GroqLLMProvider is the concrete impl.
- GraphBuilder assembles the graph given an LLMProvider.
- run_agent() accepts an optional ResponseValidator via keyword argument.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.guardrails.validators import GuardrailRule, ResponseValidator


# ---------------------------------------------------------------------------
# run_agent
# ---------------------------------------------------------------------------

class TestRunAgent:
    @pytest.mark.asyncio
    async def test_run_agent_returns_string(self):
        mock_result = {
            "messages": [AIMessage(content="Your care plan says to rest.")]
        }
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(return_value=mock_result)

        with patch("src.agent.graph.get_graph", return_value=mock_graph):
            from src.agent.graph import run_agent
            result = await run_agent("What should I do today?")

        assert isinstance(result, str)
        assert result == "Your care plan says to rest."

    @pytest.mark.asyncio
    async def test_run_agent_passes_human_message(self):
        mock_result = {
            "messages": [AIMessage(content="Response")]
        }
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(return_value=mock_result)

        with patch("src.agent.graph.get_graph", return_value=mock_graph):
            from src.agent.graph import run_agent
            await run_agent("My question")

        call_args = mock_graph.ainvoke.call_args[0][0]
        assert len(call_args["messages"]) == 1
        assert isinstance(call_args["messages"][0], HumanMessage)
        assert call_args["messages"][0].content == "My question"

    @pytest.mark.asyncio
    async def test_run_agent_uses_injected_validator(self):
        mock_result = {
            "messages": [AIMessage(content="raw llm output")]
        }
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(return_value=mock_result)

        class CapturingRule(GuardrailRule):
            def __init__(self):
                self.received = None

            def check(self, response: str) -> str | None:
                self.received = response
                return None

        rule = CapturingRule()
        validator = ResponseValidator(rules=[rule])

        with patch("src.agent.graph.get_graph", return_value=mock_graph):
            from src.agent.graph import run_agent
            await run_agent("test", validator=validator)

        assert rule.received == "raw llm output"

    @pytest.mark.asyncio
    async def test_run_agent_validator_can_transform_response(self):
        mock_result = {
            "messages": [AIMessage(content="raw llm output")]
        }
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(return_value=mock_result)

        class AlwaysBlockRule(GuardrailRule):
            def check(self, response: str) -> str | None:
                return "validated output"

        validator = ResponseValidator(rules=[AlwaysBlockRule()])

        with patch("src.agent.graph.get_graph", return_value=mock_graph):
            from src.agent.graph import run_agent
            result = await run_agent("test", validator=validator)

        assert result == "validated output"

    @pytest.mark.asyncio
    async def test_run_agent_uses_default_validator_when_none_given(self):
        """When no validator is passed, the default_validator is used and
        a blocked phrase in the LLM output is intercepted."""
        mock_result = {
            "messages": [AIMessage(content="You are diagnosed with high blood pressure.")]
        }
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(return_value=mock_result)

        with patch("src.agent.graph.get_graph", return_value=mock_graph):
            from src.agent.graph import run_agent
            result = await run_agent("test")

        assert "medical advice" in result


# ---------------------------------------------------------------------------
# GraphBuilder
# ---------------------------------------------------------------------------

class TestGraphBuilder:
    def test_graph_builds_without_error(self):
        with patch("src.agent.graph.ChatGroq") as mock_llm_cls:
            mock_llm = MagicMock()
            mock_llm.bind_tools.return_value = mock_llm
            mock_llm_cls.return_value = mock_llm

            from src.agent.graph import GraphBuilder
            graph = GraphBuilder().build()

        assert graph is not None

    def test_graph_builder_accepts_custom_llm_provider(self):
        from src.agent.graph import GraphBuilder, LLMProvider

        class FakeLLMProvider(LLMProvider):
            def get_llm_with_tools(self, tools):
                mock = MagicMock()
                mock.bind_tools = MagicMock(return_value=mock)
                return mock

            def get_llm(self):
                return MagicMock()

        builder = GraphBuilder(llm_provider=FakeLLMProvider())
        graph = builder.build()
        assert graph is not None

    def test_groq_llm_provider_is_default(self):
        from src.agent.graph import GraphBuilder, GroqLLMProvider
        builder = GraphBuilder()
        assert isinstance(builder._llm_provider, GroqLLMProvider)
