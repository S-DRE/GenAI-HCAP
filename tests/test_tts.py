import sys
from unittest.mock import MagicMock, patch


def _make_tts_mock():
    mock_tts_instance = MagicMock()
    mock_tts_cls = MagicMock(return_value=mock_tts_instance)
    mock_tts_api = MagicMock()
    mock_tts_api.TTS = mock_tts_cls
    mock_tts_module = MagicMock()
    mock_tts_module.api = mock_tts_api
    return mock_tts_module, mock_tts_cls, mock_tts_instance


class TestTTS:
    def test_speak_returns_output_path(self):
        mock_tts_module, _, _ = _make_tts_mock()
        with patch.dict(sys.modules, {"TTS": mock_tts_module, "TTS.api": mock_tts_module.api}):
            import importlib
            from src.voice import tts as tts_module
            importlib.reload(tts_module)
            tts = tts_module.TTS()
            result = tts.speak("Hello there.", output_path="test_output.wav")
        assert result == "test_output.wav"

    def test_speak_calls_tts_to_file(self):
        mock_tts_module, mock_tts_cls, mock_tts_instance = _make_tts_mock()
        with patch.dict(sys.modules, {"TTS": mock_tts_module, "TTS.api": mock_tts_module.api}):
            import importlib
            from src.voice import tts as tts_module
            importlib.reload(tts_module)
            tts = tts_module.TTS()
            tts.speak("Take your medicine.", output_path="out.wav")
        mock_tts_instance.tts_to_file.assert_called_once_with(
            text="Take your medicine.", file_path="out.wav"
        )

    def test_model_loaded_lazily(self):
        mock_tts_module, mock_tts_cls, _ = _make_tts_mock()
        with patch.dict(sys.modules, {"TTS": mock_tts_module, "TTS.api": mock_tts_module.api}):
            import importlib
            from src.voice import tts as tts_module
            importlib.reload(tts_module)
            tts = tts_module.TTS()
            mock_tts_cls.assert_not_called()
            tts.speak("test")
            mock_tts_cls.assert_called_once()

    def test_model_loaded_only_once(self):
        mock_tts_module, mock_tts_cls, _ = _make_tts_mock()
        with patch.dict(sys.modules, {"TTS": mock_tts_module, "TTS.api": mock_tts_module.api}):
            import importlib
            from src.voice import tts as tts_module
            importlib.reload(tts_module)
            tts = tts_module.TTS()
            tts.speak("first")
            tts.speak("second")
        mock_tts_cls.assert_called_once()

    def test_default_output_path(self):
        mock_tts_module, _, _ = _make_tts_mock()
        with patch.dict(sys.modules, {"TTS": mock_tts_module, "TTS.api": mock_tts_module.api}):
            import importlib
            from src.voice import tts as tts_module
            importlib.reload(tts_module)
            tts = tts_module.TTS()
            result = tts.speak("hello")
        assert result == "output.wav"
