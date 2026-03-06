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
        with patch("src.api.main.run_agent", new=AsyncMock(return_value="Here is your care info.")):
            response = client.post("/chat", json={"message": "What are my medications?"})
        assert response.status_code == 200

    def test_chat_returns_response_field(self, client):
        with patch("src.api.main.run_agent", new=AsyncMock(return_value="Drink plenty of water.")):
            response = client.post("/chat", json={"message": "How much water should I drink?"})
        assert "response" in response.json()
        assert response.json()["response"] == "Drink plenty of water."

    def test_chat_passes_message_to_agent(self, client):
        mock_agent = AsyncMock(return_value="Response text")
        with patch("src.api.main.run_agent", new=mock_agent):
            client.post("/chat", json={"message": "Hello"})
        mock_agent.assert_called_once_with("Hello")

    def test_chat_requires_message_field(self, client):
        response = client.post("/chat", json={})
        assert response.status_code == 422

    def test_chat_rejects_empty_body(self, client):
        response = client.post("/chat")
        assert response.status_code == 422
