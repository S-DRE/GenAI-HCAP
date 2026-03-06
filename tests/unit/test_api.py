"""Unit tests for the FastAPI endpoints.

After the SOLID refactor:
- run_agent is imported inside the /chat route function.
- ServiceFactory owns voice pipeline construction; routes use Depends(get_factory).
- get_voice_pipeline() is no longer a module-level function.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(return_value={"messages": []})
    with patch("src.agent.graph.get_graph", return_value=mock_graph):
        from src.api.main import app
        return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_ok_status(self, client):
        response = client.get("/health")
        assert response.json() == {"status": "ok"}


class TestChatEndpoint:
    def test_chat_returns_200(self, client):
        with patch("src.agent.graph.run_agent", new=AsyncMock(return_value="Here is your care info.")):
            response = client.post("/chat", json={"message": "What are my medications?"})
        assert response.status_code == 200

    def test_chat_returns_response_field(self, client):
        with patch("src.agent.graph.run_agent", new=AsyncMock(return_value="Drink plenty of water.")):
            response = client.post("/chat", json={"message": "How much water should I drink?"})
        assert "response" in response.json()
        assert response.json()["response"] == "Drink plenty of water."

    def test_chat_passes_message_to_agent(self, client):
        mock_agent = AsyncMock(return_value="Response text")
        with patch("src.agent.graph.run_agent", new=mock_agent):
            client.post("/chat", json={"message": "Hello"})
        mock_agent.assert_called_once_with("Hello")

    def test_chat_requires_message_field(self, client):
        response = client.post("/chat", json={})
        assert response.status_code == 422

    def test_chat_rejects_empty_body(self, client):
        response = client.post("/chat")
        assert response.status_code == 422


class TestChatInputValidation:
    def test_message_above_max_length_rejected(self, client):
        long_message = "a" * 2_001
        response = client.post("/chat", json={"message": long_message})
        assert response.status_code == 422

    def test_message_at_max_length_accepted(self, client):
        with patch("src.agent.graph.run_agent", new=AsyncMock(return_value="ok")):
            response = client.post("/chat", json={"message": "a" * 2_000})
        assert response.status_code == 200

    def test_empty_string_message_rejected(self, client):
        response = client.post("/chat", json={"message": ""})
        assert response.status_code == 422


class TestChatErrorHandling:
    def test_agent_error_returns_503(self, client):
        with patch("src.agent.graph.run_agent", new=AsyncMock(side_effect=RuntimeError("boom"))):
            response = client.post("/chat", json={"message": "Hello"})
        assert response.status_code == 503

    def test_agent_error_response_does_not_leak_detail(self, client):
        secret = "org_supersecretorgid_and_api_key_data"
        with patch("src.agent.graph.run_agent", new=AsyncMock(side_effect=RuntimeError(secret))):
            response = client.post("/chat", json={"message": "Hello"})
        body = response.text
        assert secret not in body

    def test_agent_error_returns_generic_message(self, client):
        with patch("src.agent.graph.run_agent", new=AsyncMock(side_effect=Exception("rate limit 429"))):
            response = client.post("/chat", json={"message": "Hello"})
        assert "temporarily unavailable" in response.json()["detail"]


class TestServiceFactory:
    def test_factory_is_injected_into_voice_endpoint(self):
        """ServiceFactory is supplied via Depends(get_factory) in the voice route's
        function signature. Verify that the endpoint function declares this parameter."""
        import inspect
        from src.api.main import voice, get_factory
        from fastapi import params as fastapi_params

        sig = inspect.signature(voice)
        factory_param = sig.parameters.get("factory")
        assert factory_param is not None, "voice() has no 'factory' parameter"
        default = factory_param.default
        # FastAPI wraps Depends() in a params.Depends instance
        assert isinstance(default, fastapi_params.Depends), "factory parameter is not a Depends()"
        assert default.dependency is get_factory
