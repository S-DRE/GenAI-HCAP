from unittest.mock import MagicMock, patch

from src.tools.escalation import escalate


class TestEscalateTool:
    def test_returns_confirmation_message(self):
        result = escalate.invoke({"reason": "Patient reported chest pain"})
        assert "care team" in result
        assert "emergency services" in result

    def test_includes_reason_in_response(self):
        reason = "Patient fell and cannot get up"
        result = escalate.invoke({"reason": reason})
        assert reason in result

    def test_returns_string(self):
        result = escalate.invoke({"reason": "Test reason"})
        assert isinstance(result, str)

    def test_logs_warning_on_escalation(self):
        with patch("src.tools.escalation.logger") as mock_logger:
            escalate.invoke({"reason": "Emergency situation"})
            mock_logger.warning.assert_called_once()
            call_kwargs = mock_logger.warning.call_args
            assert "escalation_triggered" in call_kwargs[0]
