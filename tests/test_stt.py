import sys
from unittest.mock import MagicMock, patch

from src.voice.protocols import STTProvider


class TestWhisperSTT:
    def test_implements_stt_provider(self):
        mock_whisper = MagicMock()
        with patch.dict(sys.modules, {"whisper": mock_whisper}):
            import importlib
            from src.voice import stt as stt_module
            importlib.reload(stt_module)
            stt = stt_module.WhisperSTT()
        assert isinstance(stt, STTProvider)

    def test_transcribe_returns_text(self):
        mock_whisper = MagicMock()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "  Hello, I need help.  "}
        mock_whisper.load_model.return_value = mock_model

        with patch.dict(sys.modules, {"whisper": mock_whisper}):
            import importlib
            from src.voice import stt as stt_module
            importlib.reload(stt_module)
            stt = stt_module.WhisperSTT(model_size="base")
            result = stt.transcribe("fake_audio.wav")

        assert result == "Hello, I need help."

    def test_model_loaded_lazily(self):
        mock_whisper = MagicMock()
        mock_whisper.load_model.return_value = MagicMock()
        mock_whisper.load_model.return_value.transcribe.return_value = {"text": "test"}

        with patch.dict(sys.modules, {"whisper": mock_whisper}):
            import importlib
            from src.voice import stt as stt_module
            importlib.reload(stt_module)
            stt = stt_module.WhisperSTT()
            mock_whisper.load_model.assert_not_called()
            stt.transcribe("audio.wav")
            mock_whisper.load_model.assert_called_once()

    def test_model_loaded_only_once(self):
        mock_whisper = MagicMock()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "hello"}
        mock_whisper.load_model.return_value = mock_model

        with patch.dict(sys.modules, {"whisper": mock_whisper}):
            import importlib
            from src.voice import stt as stt_module
            importlib.reload(stt_module)
            stt = stt_module.WhisperSTT()
            stt.transcribe("a.wav")
            stt.transcribe("b.wav")
            mock_whisper.load_model.assert_called_once()

    def test_default_model_size_is_base(self):
        mock_whisper = MagicMock()
        with patch.dict(sys.modules, {"whisper": mock_whisper}):
            import importlib
            from src.voice import stt as stt_module
            importlib.reload(stt_module)
            stt = stt_module.WhisperSTT()
            assert stt._model_size == "base"
