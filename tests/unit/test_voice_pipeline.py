"""Tests for VoicePipeline.

After the SOLID refactor, VoicePipeline accepts an AgentRunner parameter, so
the agent is injected rather than imported. This makes all tests simpler —
no patching of run_agent needed.
"""

import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.voice.pipeline import DefaultAgentRunner, VoicePipeline
from src.voice.protocols import AgentRunner, STTProvider, TTSProvider


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

class MockSTT(STTProvider):
    def __init__(self, text: str = "Hello, I need help."):
        self._text = text

    def transcribe(self, audio_path: str) -> str:
        return self._text


class MockTTS(TTSProvider):
    def __init__(self, output_path: str = "response.wav"):
        self._output_path = output_path
        self.last_text: str | None = None

    def speak(self, text: str, output_path: str = "response.wav") -> str:
        self.last_text = text
        return output_path


class MockAgentRunner(AgentRunner):
    def __init__(self, response: str = "Take your medication."):
        self._response = response
        self.received_message: str | None = None

    async def run(self, message: str) -> str:
        self.received_message = message
        return self._response


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def agent():
    return MockAgentRunner()


@pytest.fixture
def pipeline(agent):
    return VoicePipeline(stt=MockSTT(), tts=MockTTS(), agent_runner=agent)


# ---------------------------------------------------------------------------
# AgentRunner protocol
# ---------------------------------------------------------------------------

class TestAgentRunnerProtocol:
    def test_default_agent_runner_implements_protocol(self):
        assert issubclass(DefaultAgentRunner, AgentRunner)

    def test_mock_agent_runner_implements_protocol(self):
        assert issubclass(MockAgentRunner, AgentRunner)


# ---------------------------------------------------------------------------
# VoicePipeline.process
# ---------------------------------------------------------------------------

class TestVoicePipelineProcess:
    async def test_returns_output_path(self, pipeline):
        result = await pipeline.process("input.wav")
        assert isinstance(result, str)
        assert result.endswith(".wav")

    async def test_transcribed_text_passed_to_agent(self, agent):
        pipeline = VoicePipeline(stt=MockSTT("Hello, I need help."), tts=MockTTS(), agent_runner=agent)
        await pipeline.process("input.wav")
        assert agent.received_message == "Hello, I need help."

    async def test_agent_response_passed_to_tts(self):
        mock_tts = MockTTS()
        agent = MockAgentRunner("Your appointment is at 10am.")
        pipeline = VoicePipeline(stt=MockSTT("What time is my appointment?"), tts=mock_tts, agent_runner=agent)
        await pipeline.process("input.wav")
        assert mock_tts.last_text == "Your appointment is at 10am."

    async def test_raises_on_empty_transcription(self):
        pipeline = VoicePipeline(stt=MockSTT(""), tts=MockTTS(), agent_runner=MockAgentRunner())
        with pytest.raises(ValueError, match="no text"):
            await pipeline.process("silent.wav")

    async def test_stt_called_with_correct_path(self):
        mock_stt = MagicMock(spec=STTProvider)
        mock_stt.transcribe.return_value = "Hello"
        pipeline = VoicePipeline(stt=mock_stt, tts=MockTTS(), agent_runner=MockAgentRunner())
        await pipeline.process("/tmp/audio.wav")
        mock_stt.transcribe.assert_called_once_with("/tmp/audio.wav")


# ---------------------------------------------------------------------------
# Dependency inversion
# ---------------------------------------------------------------------------

class TestVoicePipelineDependencyInversion:
    def test_accepts_any_stt_provider(self):
        custom_stt = MockSTT("custom transcription")
        pipeline = VoicePipeline(stt=custom_stt, tts=MockTTS(), agent_runner=MockAgentRunner())
        assert pipeline._stt is custom_stt

    def test_accepts_any_tts_provider(self):
        custom_tts = MockTTS("custom_output.wav")
        pipeline = VoicePipeline(stt=MockSTT(), tts=custom_tts, agent_runner=MockAgentRunner())
        assert pipeline._tts is custom_tts

    def test_accepts_any_agent_runner(self):
        agent = MockAgentRunner()
        pipeline = VoicePipeline(stt=MockSTT(), tts=MockTTS(), agent_runner=agent)
        assert pipeline._agent is agent

    def test_default_output_dir_is_temp(self):
        pipeline = VoicePipeline(stt=MockSTT(), tts=MockTTS(), agent_runner=MockAgentRunner())
        assert pipeline._output_dir == tempfile.gettempdir()

    def test_custom_output_dir_is_respected(self):
        pipeline = VoicePipeline(stt=MockSTT(), tts=MockTTS(), agent_runner=MockAgentRunner(), output_dir="/custom/path")
        assert pipeline._output_dir == "/custom/path"

    def test_default_agent_runner_used_when_none_given(self):
        pipeline = VoicePipeline(stt=MockSTT(), tts=MockTTS())
        assert isinstance(pipeline._agent, DefaultAgentRunner)


# ---------------------------------------------------------------------------
# POST /voice API endpoint
# ---------------------------------------------------------------------------

class TestVoiceAPIEndpoint:
    def _make_mock_pipeline(self, wav_file):
        mock_pipeline = MagicMock()
        mock_pipeline.process = AsyncMock(return_value=str(wav_file))
        return mock_pipeline

    def test_voice_endpoint_returns_200(self, tmp_path):
        wav_file = tmp_path / "response.wav"
        wav_file.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt ")
        mock_pipeline = self._make_mock_pipeline(wav_file)

        mock_factory = MagicMock()
        mock_factory.get_voice_pipeline.return_value = mock_pipeline

        from src.api.main import app, get_factory
        from fastapi.testclient import TestClient
        app.dependency_overrides[get_factory] = lambda: mock_factory
        try:
            with patch("src.agent.graph.get_graph"):
                client = TestClient(app)
                response = client.post(
                    "/voice",
                    files={"audio": ("test.wav", b"RIFF$\x00\x00\x00WAVEfmt ", "audio/wav")},
                )
        finally:
            app.dependency_overrides.clear()

        assert response.status_code == 200

    def test_voice_endpoint_returns_wav_content_type(self, tmp_path):
        wav_file = tmp_path / "response.wav"
        wav_file.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt ")
        mock_pipeline = self._make_mock_pipeline(wav_file)

        mock_factory = MagicMock()
        mock_factory.get_voice_pipeline.return_value = mock_pipeline

        from src.api.main import app, get_factory
        from fastapi.testclient import TestClient
        app.dependency_overrides[get_factory] = lambda: mock_factory
        try:
            with patch("src.agent.graph.get_graph"):
                client = TestClient(app)
                response = client.post(
                    "/voice",
                    files={"audio": ("test.wav", b"fake", "audio/wav")},
                )
        finally:
            app.dependency_overrides.clear()

        assert "audio" in response.headers.get("content-type", "")

    def test_voice_endpoint_returns_422_on_empty_audio(self):
        mock_pipeline = MagicMock()
        mock_pipeline.process = AsyncMock(side_effect=ValueError("Audio transcription produced no text."))

        mock_factory = MagicMock()
        mock_factory.get_voice_pipeline.return_value = mock_pipeline

        from src.api.main import app, get_factory
        from fastapi.testclient import TestClient
        app.dependency_overrides[get_factory] = lambda: mock_factory
        try:
            with patch("src.agent.graph.get_graph"):
                client = TestClient(app)
                response = client.post(
                    "/voice",
                    files={"audio": ("silent.wav", b"fake_silent_audio", "audio/wav")},
                )
        finally:
            app.dependency_overrides.clear()

        assert response.status_code == 422

    def test_voice_endpoint_returns_503_on_pipeline_error(self):
        mock_pipeline = MagicMock()
        mock_pipeline.process = AsyncMock(side_effect=RuntimeError("model crashed"))

        mock_factory = MagicMock()
        mock_factory.get_voice_pipeline.return_value = mock_pipeline

        from src.api.main import app, get_factory
        from fastapi.testclient import TestClient
        app.dependency_overrides[get_factory] = lambda: mock_factory
        try:
            with patch("src.agent.graph.get_graph"):
                client = TestClient(app)
                response = client.post(
                    "/voice",
                    files={"audio": ("test.wav", b"fake", "audio/wav")},
                )
        finally:
            app.dependency_overrides.clear()

        assert response.status_code == 503

    def test_voice_endpoint_503_does_not_leak_error_detail(self):
        secret = "internal_crash_detail_with_secret"
        mock_pipeline = MagicMock()
        mock_pipeline.process = AsyncMock(side_effect=RuntimeError(secret))

        mock_factory = MagicMock()
        mock_factory.get_voice_pipeline.return_value = mock_pipeline

        from src.api.main import app, get_factory
        from fastapi.testclient import TestClient
        app.dependency_overrides[get_factory] = lambda: mock_factory
        try:
            with patch("src.agent.graph.get_graph"):
                client = TestClient(app)
                response = client.post(
                    "/voice",
                    files={"audio": ("test.wav", b"fake", "audio/wav")},
                )
        finally:
            app.dependency_overrides.clear()

        assert secret not in response.text
