"""Integration tests.

Tests the full request pipeline from HTTP request through FastAPI → agent →
RAG/escalation tools → guardrails → HTTP response.

The LLM (Groq) is mocked so these tests run offline without any API keys.
All other components — FastAPI routing, agent graph wiring, tool execution,
and guardrail validation — run with real code.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage


def _make_llm_response(content: str, tool_calls: list | None = None) -> AIMessage:
    """Helper to build a realistic AIMessage from the LLM."""
    msg = AIMessage(content=content)
    msg.tool_calls = tool_calls or []
    return msg


def _make_app_client(llm_responses: list[AIMessage]) -> TestClient:
    """Build a TestClient with a mock LLM that returns the given responses in order."""
    mock_llm = MagicMock()
    mock_llm.bind_tools.return_value = mock_llm
    mock_llm.invoke.side_effect = llm_responses

    with patch("src.agent.graph.ChatGroq", return_value=mock_llm):
        # Reset the cached graph so it rebuilds with the mock LLM
        import src.agent.graph as graph_module
        graph_module._graph = None

        from src.api.main import app
        client = TestClient(app)
        # Trigger graph build now while patch is still active
        client.get("/health")
        return client


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

class TestHealthIntegration:
    def test_health_endpoint_returns_ok(self):
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        with patch("src.agent.graph.ChatGroq", return_value=mock_llm):
            import src.agent.graph as graph_module
            graph_module._graph = None
            from src.api.main import app
            client = TestClient(app)
            response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# Chat — direct LLM answer (no tool use)
# ---------------------------------------------------------------------------

class TestChatDirectAnswer:
    def test_routine_question_returns_llm_response(self):
        llm_answer = "Your morning routine starts at 7:00 AM with breakfast."
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm.invoke.return_value = _make_llm_response(llm_answer)

        with patch("src.agent.graph.ChatGroq", return_value=mock_llm):
            import src.agent.graph as graph_module
            graph_module._graph = None
            from src.api.main import app
            client = TestClient(app)
            response = client.post("/chat", json={"message": "What time is breakfast?"})

        assert response.status_code == 200
        assert response.json()["response"] == llm_answer

    def test_response_contains_string(self):
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm.invoke.return_value = _make_llm_response("Please drink 6 glasses of water daily.")

        with patch("src.agent.graph.ChatGroq", return_value=mock_llm):
            import src.agent.graph as graph_module
            graph_module._graph = None
            from src.api.main import app
            client = TestClient(app)
            response = client.post("/chat", json={"message": "How much water?"})

        assert isinstance(response.json()["response"], str)
        assert len(response.json()["response"]) > 0


# ---------------------------------------------------------------------------
# Chat — guardrails intercept unsafe LLM responses
# ---------------------------------------------------------------------------

class TestChatGuardrailsIntegration:
    def test_diagnosis_phrase_is_blocked_by_guardrails(self):
        unsafe_response = "Based on your symptoms, you have hypertension."
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm.invoke.return_value = _make_llm_response(unsafe_response)

        with patch("src.agent.graph.ChatGroq", return_value=mock_llm):
            import src.agent.graph as graph_module
            graph_module._graph = None
            from src.api.main import app
            client = TestClient(app)
            response = client.post("/chat", json={"message": "Do I have high blood pressure?"})

        body = response.json()["response"]
        assert "medical advice" in body
        assert unsafe_response not in body

    def test_escalation_keyword_in_response_is_intercepted(self):
        unsafe_response = "The patient may be having a heart attack."
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm.invoke.return_value = _make_llm_response(unsafe_response)

        with patch("src.agent.graph.ChatGroq", return_value=mock_llm):
            import src.agent.graph as graph_module
            graph_module._graph = None
            from src.api.main import app
            client = TestClient(app)
            response = client.post("/chat", json={"message": "My chest hurts."})

        body = response.json()["response"]
        assert "emergency services" in body
        assert unsafe_response not in body

    def test_safe_response_passes_through_unchanged(self):
        safe_response = "Remember to take your morning walk after breakfast."
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm.invoke.return_value = _make_llm_response(safe_response)

        with patch("src.agent.graph.ChatGroq", return_value=mock_llm):
            import src.agent.graph as graph_module
            graph_module._graph = None
            from src.api.main import app
            client = TestClient(app)
            response = client.post("/chat", json={"message": "What should I do after breakfast?"})

        assert response.json()["response"] == safe_response


# ---------------------------------------------------------------------------
# Chat — RAG tool is called and its output feeds back into the LLM
# ---------------------------------------------------------------------------

class TestChatRAGToolIntegration:
    def test_llm_calls_rag_tool_then_answers(self):
        """Simulate the agent calling the RAG tool then producing a final answer."""
        rag_tool_call = _make_llm_response(
            content="",
            tool_calls=[{
                "id": "call_1",
                "name": "retrieve_care_info",
                "args": {"query": "blood glucose targets"},
            }],
        )
        final_answer = _make_llm_response(
            "Your fasting blood glucose target is between 4.0 and 7.0 mmol/L."
        )

        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm.invoke.side_effect = [rag_tool_call, final_answer]

        with patch("src.agent.graph.ChatGroq", return_value=mock_llm), \
             patch("langchain_chroma.Chroma") as mock_chroma:
            mock_doc = MagicMock()
            mock_doc.page_content = "Fasting target: 4.0–7.0 mmol/L"
            mock_chroma.return_value.similarity_search.return_value = [mock_doc]

            import src.agent.graph as graph_module
            graph_module._graph = None
            from src.api.main import app
            client = TestClient(app)
            response = client.post("/chat", json={"message": "What are my blood glucose targets?"})

        assert response.status_code == 200
        assert mock_llm.invoke.call_count == 2

    def test_rag_tool_result_is_passed_back_to_llm(self):
        """Verify the tool result message is included in the second LLM call."""
        rag_tool_call = _make_llm_response(
            content="",
            tool_calls=[{
                "id": "call_2",
                "name": "retrieve_care_info",
                "args": {"query": "medication schedule"},
            }],
        )
        final_answer = _make_llm_response("Take Metformin at 08:00 and 19:00 with food.")

        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm.invoke.side_effect = [rag_tool_call, final_answer]

        with patch("src.agent.graph.ChatGroq", return_value=mock_llm), \
             patch("langchain_chroma.Chroma") as mock_chroma:
            mock_doc = MagicMock()
            mock_doc.page_content = "Morning: Metformin 500mg. Evening: Metformin 500mg."
            mock_chroma.return_value.similarity_search.return_value = [mock_doc]

            import src.agent.graph as graph_module
            graph_module._graph = None
            from src.api.main import app
            client = TestClient(app)
            response = client.post("/chat", json={"message": "What medications does the patient take?"})

        # Second LLM call should have more messages than the first (tool result included)
        first_call_messages = mock_llm.invoke.call_args_list[0][0][0]
        second_call_messages = mock_llm.invoke.call_args_list[1][0][0]
        assert len(second_call_messages) > len(first_call_messages)


# ---------------------------------------------------------------------------
# Chat — escalation tool is called
# ---------------------------------------------------------------------------

class TestChatEscalationToolIntegration:
    def test_escalation_tool_is_called_for_emergency(self):
        """Simulate the agent calling the escalation tool."""
        escalation_tool_call = _make_llm_response(
            content="",
            tool_calls=[{
                "id": "call_3",
                "name": "escalate",
                "args": {"reason": "Patient reports chest pain"},
            }],
        )
        final_answer = _make_llm_response(
            "I have alerted your care team. Please stay calm."
        )

        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm.invoke.side_effect = [escalation_tool_call, final_answer]

        with patch("src.agent.graph.ChatGroq", return_value=mock_llm):
            import src.agent.graph as graph_module
            graph_module._graph = None
            from src.api.main import app
            client = TestClient(app)
            response = client.post("/chat", json={"message": "I have chest pain right now."})

        assert response.status_code == 200
        assert mock_llm.invoke.call_count == 2


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidationIntegration:
    def test_empty_message_field_rejected(self):
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        with patch("src.agent.graph.ChatGroq", return_value=mock_llm):
            import src.agent.graph as graph_module
            graph_module._graph = None
            from src.api.main import app
            client = TestClient(app)
            response = client.post("/chat", json={})
        assert response.status_code == 422

    def test_missing_body_rejected(self):
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        with patch("src.agent.graph.ChatGroq", return_value=mock_llm):
            import src.agent.graph as graph_module
            graph_module._graph = None
            from src.api.main import app
            client = TestClient(app)
            response = client.post("/chat")
        assert response.status_code == 422
