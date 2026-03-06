from unittest.mock import MagicMock, patch


class TestSTT:
    def test_transcribe_returns_text(self):
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "  Hello, I need help.  "}

        with patch("whisper.load_model", return_value=mock_model):
            from src.voice.stt import STT
            stt = STT(model_size="base")
            result = stt.transcribe("fake_audio.wav")

        assert result == "Hello, I need help."

    def test_model_loaded_lazily(self):
        with patch("whisper.load_model") as mock_load:
            mock_load.return_value = MagicMock()
            mock_load.return_value.transcribe.return_value = {"text": "test"}

            from src.voice.stt import STT
            stt = STT()
            mock_load.assert_not_called()

            stt.transcribe("audio.wav")
            mock_load.assert_called_once()

    def test_model_loaded_only_once(self):
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "hello"}

        with patch("whisper.load_model", return_value=mock_model) as mock_load:
            from src.voice.stt import STT
            stt = STT()
            stt.transcribe("a.wav")
            stt.transcribe("b.wav")

        mock_load.assert_called_once()

    def test_default_model_size_is_base(self):
        from src.voice.stt import STT
        stt = STT()
        assert stt._model_size == "base"
