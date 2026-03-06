"""Tests for WhisperSTT.

whisper is not installed, so each test patches sys.modules for the full
duration of the test (module import AND lazy _load_model call).
"""

import sys
from unittest.mock import MagicMock, patch

from src.voice.protocols import STTProvider


class TestWhisperSTT:
    def _get_fresh_whisper_stt(self, mock_whisper):
        """Import WhisperSTT fresh while the whisper mock is active."""
        sys.modules.pop("src.voice.stt", None)
        import importlib
        import src.voice.stt as stt_module
        importlib.invalidate_caches()
        return stt_module.WhisperSTT

    def test_implements_stt_provider(self):
        mock_whisper = MagicMock()
        with patch.dict(sys.modules, {"whisper": mock_whisper}):
            WhisperSTT = self._get_fresh_whisper_stt(mock_whisper)
            stt = WhisperSTT()
        assert isinstance(stt, STTProvider)

    def test_transcribe_returns_text(self):
        mock_whisper = MagicMock()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "  Hello, I need help.  "}
        mock_whisper.load_model.return_value = mock_model

        with patch.dict(sys.modules, {"whisper": mock_whisper}):
            WhisperSTT = self._get_fresh_whisper_stt(mock_whisper)
            stt = WhisperSTT(model_size="base")
            result = stt.transcribe("fake_audio.wav")

        assert result == "Hello, I need help."

    def test_model_loaded_lazily(self):
        mock_whisper = MagicMock()
        mock_whisper.load_model.return_value = MagicMock()
        mock_whisper.load_model.return_value.transcribe.return_value = {"text": "test"}

        with patch.dict(sys.modules, {"whisper": mock_whisper}):
            WhisperSTT = self._get_fresh_whisper_stt(mock_whisper)
            stt = WhisperSTT()
            mock_whisper.load_model.assert_not_called()
            stt.transcribe("audio.wav")
            mock_whisper.load_model.assert_called_once()

    def test_model_loaded_only_once(self):
        mock_whisper = MagicMock()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "hello"}
        mock_whisper.load_model.return_value = mock_model

        with patch.dict(sys.modules, {"whisper": mock_whisper}):
            WhisperSTT = self._get_fresh_whisper_stt(mock_whisper)
            stt = WhisperSTT()
            stt.transcribe("a.wav")
            stt.transcribe("b.wav")
            mock_whisper.load_model.assert_called_once()

    def test_default_model_size_is_base(self):
        mock_whisper = MagicMock()
        with patch.dict(sys.modules, {"whisper": mock_whisper}):
            WhisperSTT = self._get_fresh_whisper_stt(mock_whisper)
            stt = WhisperSTT()
            assert stt._model_size == "base"
