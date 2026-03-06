"""Tests for VoicePipeline.

All tests inject mock STT and TTS implementations via the STTProvider /
TTSProvider interfaces. No real models are loaded, no files are written to
disk, and the agent is mocked — making these fast, offline, and dependency-free.
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.voice.pipeline import VoicePipeline
from src.voice.protocols import STTProvider, TTSProvider


class MockSTT(STTProvider):
    """Minimal STT implementation for testing."""

    def __init__(self, text: str = "Hello, I need help."):
        self._text = text

    def transcribe(self, audio_path: str) -> str:
        return self._text


class MockTTS(TTSProvider):
    """Minimal TTS implementation for testing."""

    def __init__(self, output_path: str = "response.wav"):
        self._output_path = output_path
        self.last_text: str | None = None

    def speak(self, text: str, output_path: str = "response.wav") -> str:
        self.last_text = text
        return output_path


@pytest.fixture
def pipeline():
    return VoicePipeline(stt=MockSTT(), tts=MockTTS())


class TestVoicePipelineProcess:
    async def test_returns_output_path(self, pipeline):
        with patch("src.voice.pipeline.run_agent", new=AsyncMock(return_value="Take your medication.")):
            result = await pipeline.process("input.wav")
        assert isinstance(result, str)
        assert result.endswith(".wav")

    async def test_transcribed_text_passed_to_agent(self, pipeline):
        mock_agent = AsyncMock(return_value="ok")
        with patch("src.voice.pipeline.run_agent", new=mock_agent):
            await pipeline.process("input.wav")
        mock_agent.assert_called_once_with("Hello, I need help.")

    async def test_agent_response_passed_to_tts(self):
        mock_tts = MockTTS()
        pipeline = VoicePipeline(stt=MockSTT("What time is my appointment?"), tts=mock_tts)
        with patch("src.voice.pipeline.run_agent", new=AsyncMock(return_value="Your appointment is at 10am.")):
            await pipeline.process("input.wav")
        assert mock_tts.last_text == "Your appointment is at 10am."

    async def test_raises_on_empty_transcription(self):
        pipeline = VoicePipeline(stt=MockSTT(""), tts=MockTTS())
        with patch("src.voice.pipeline.run_agent", new=AsyncMock(return_value="ok")):
            with pytest.raises(ValueError, match="no text"):
                await pipeline.process("silent.wav")

    async def test_stt_called_with_correct_path(self):
        mock_stt = MagicMock(spec=STTProvider)
        mock_stt.transcribe.return_value = "Hello"
        pipeline = VoicePipeline(stt=mock_stt, tts=MockTTS())
        with patch("src.voice.pipeline.run_agent", new=AsyncMock(return_value="ok")):
            await pipeline.process("/tmp/audio.wav")
        mock_stt.transcribe.assert_called_once_with("/tmp/audio.wav")


class TestVoicePipelineDependencyInversion:
    def test_accepts_any_stt_provider(self):
        """VoicePipeline should accept any STTProvider, not just WhisperSTT."""
        custom_stt = MockSTT("custom transcription")
        pipeline = VoicePipeline(stt=custom_stt, tts=MockTTS())
        assert pipeline._stt is custom_stt

    def test_accepts_any_tts_provider(self):
        """VoicePipeline should accept any TTSProvider, not just CoquiTTS."""
        custom_tts = MockTTS("custom_output.wav")
        pipeline = VoicePipeline(stt=MockSTT(), tts=custom_tts)
        assert pipeline._tts is custom_tts

    def test_default_output_dir_is_temp(self):
        import tempfile
        pipeline = VoicePipeline(stt=MockSTT(), tts=MockTTS())
        assert pipeline._output_dir == tempfile.gettempdir()

    def test_custom_output_dir_is_respected(self):
        pipeline = VoicePipeline(stt=MockSTT(), tts=MockTTS(), output_dir="/custom/path")
        assert pipeline._output_dir == "/custom/path"


class TestVoiceAPIEndpoint:
    def test_voice_endpoint_returns_200(self, tmp_path):
        # Create a real wav file so FileResponse can serve it
        wav_file = tmp_path / "response.wav"
        wav_file.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt ")

        mock_pipeline = MagicMock()
        mock_pipeline.process = AsyncMock(return_value=str(wav_file))

        with patch("src.api.main.get_voice_pipeline", return_value=mock_pipeline), \
             patch("src.agent.graph.get_graph"):
            from src.api.main import app
            from fastapi.testclient import TestClient
            client = TestClient(app)

            audio_bytes = b"RIFF$\x00\x00\x00WAVEfmt "
            response = client.post(
                "/voice",
                files={"audio": ("test.wav", audio_bytes, "audio/wav")},
            )

        assert response.status_code == 200

    def test_voice_endpoint_returns_wav_content_type(self, tmp_path):
        wav_file = tmp_path / "response.wav"
        wav_file.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt ")

        mock_pipeline = MagicMock()
        mock_pipeline.process = AsyncMock(return_value=str(wav_file))

        with patch("src.api.main.get_voice_pipeline", return_value=mock_pipeline), \
             patch("src.agent.graph.get_graph"):
            from src.api.main import app
            from fastapi.testclient import TestClient
            client = TestClient(app)
            response = client.post(
                "/voice",
                files={"audio": ("test.wav", b"fake", "audio/wav")},
            )

        assert "audio" in response.headers.get("content-type", "")

    def test_voice_endpoint_returns_422_on_empty_audio(self):
        mock_pipeline = MagicMock()
        mock_pipeline.process = AsyncMock(side_effect=ValueError("Audio transcription produced no text."))

        with patch("src.api.main.get_voice_pipeline", return_value=mock_pipeline), \
             patch("src.agent.graph.get_graph"):
            from src.api.main import app
            from fastapi.testclient import TestClient
            client = TestClient(app)
            response = client.post(
                "/voice",
                files={"audio": ("silent.wav", b"fake_silent_audio", "audio/wav")},
            )

        assert response.status_code == 422

    def test_voice_endpoint_returns_503_on_pipeline_error(self):
        mock_pipeline = MagicMock()
        mock_pipeline.process = AsyncMock(side_effect=RuntimeError("model crashed"))

        with patch("src.api.main.get_voice_pipeline", return_value=mock_pipeline), \
             patch("src.agent.graph.get_graph"):
            from src.api.main import app
            from fastapi.testclient import TestClient
            client = TestClient(app)
            response = client.post(
                "/voice",
                files={"audio": ("test.wav", b"fake", "audio/wav")},
            )

        assert response.status_code == 503

    def test_voice_endpoint_503_does_not_leak_error_detail(self):
        secret = "internal_crash_detail_with_secret"
        mock_pipeline = MagicMock()
        mock_pipeline.process = AsyncMock(side_effect=RuntimeError(secret))

        with patch("src.api.main.get_voice_pipeline", return_value=mock_pipeline), \
             patch("src.agent.graph.get_graph"):
            from src.api.main import app
            from fastapi.testclient import TestClient
            client = TestClient(app)
            response = client.post(
                "/voice",
                files={"audio": ("test.wav", b"fake", "audio/wav")},
            )

        assert secret not in response.text
