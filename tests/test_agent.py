from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage


class TestRunAgent:
    @pytest.mark.asyncio
    async def test_run_agent_returns_string(self):
        mock_result = {
            "messages": [AIMessage(content="Your care plan says to rest.")]
        }
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(return_value=mock_result)

        with patch("src.agent.graph.get_graph", return_value=mock_graph):
            with patch("src.guardrails.validators.validate_response", side_effect=lambda x: x):
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
            with patch("src.guardrails.validators.validate_response", side_effect=lambda x: x):
                from src.agent.graph import run_agent
                await run_agent("My question")

        call_args = mock_graph.ainvoke.call_args[0][0]
        assert len(call_args["messages"]) == 1
        assert isinstance(call_args["messages"][0], HumanMessage)
        assert call_args["messages"][0].content == "My question"

    @pytest.mark.asyncio
    async def test_run_agent_applies_guardrails(self):
        mock_result = {
            "messages": [AIMessage(content="raw llm output")]
        }
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(return_value=mock_result)

        with patch("src.agent.graph.get_graph", return_value=mock_graph):
            with patch("src.agent.graph.validate_response", return_value="validated output") as mock_validate:
                from src.agent.graph import run_agent
                result = await run_agent("test")

        mock_validate.assert_called_once_with("raw llm output")
        assert result == "validated output"


class TestBuildGraph:
    def test_graph_builds_without_error(self):
        with patch("src.agent.graph.ChatGroq") as mock_llm_cls:
            mock_llm = MagicMock()
            mock_llm.bind_tools.return_value = mock_llm
            mock_llm_cls.return_value = mock_llm

            from src.agent.graph import build_graph
            graph = build_graph()

        assert graph is not None
