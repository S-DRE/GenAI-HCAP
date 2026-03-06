from unittest.mock import MagicMock, patch


class TestTTS:
    def test_speak_returns_output_path(self):
        mock_tts = MagicMock()

        with patch("TTS.api.TTS", return_value=mock_tts):
            from src.voice.tts import TTS
            tts = TTS()
            result = tts.speak("Hello there.", output_path="test_output.wav")

        assert result == "test_output.wav"

    def test_speak_calls_tts_to_file(self):
        mock_tts = MagicMock()

        with patch("TTS.api.TTS", return_value=mock_tts):
            from src.voice.tts import TTS
            tts = TTS()
            tts.speak("Take your medicine.", output_path="out.wav")

        mock_tts.tts_to_file.assert_called_once_with(
            text="Take your medicine.", file_path="out.wav"
        )

    def test_model_loaded_lazily(self):
        with patch("TTS.api.TTS") as mock_tts_cls:
            mock_tts_cls.return_value = MagicMock()

            from src.voice.tts import TTS
            tts = TTS()
            mock_tts_cls.assert_not_called()

            tts.speak("test")
            mock_tts_cls.assert_called_once()

    def test_model_loaded_only_once(self):
        with patch("TTS.api.TTS") as mock_tts_cls:
            mock_tts_cls.return_value = MagicMock()

            from src.voice.tts import TTS
            tts = TTS()
            tts.speak("first")
            tts.speak("second")

        mock_tts_cls.assert_called_once()

    def test_default_output_path(self):
        mock_tts = MagicMock()

        with patch("TTS.api.TTS", return_value=mock_tts):
            from src.voice.tts import TTS
            tts = TTS()
            result = tts.speak("hello")

        assert result == "output.wav"
