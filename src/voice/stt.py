"""Whisper-based Speech-to-Text implementation.

WhisperSTT is a concrete STTProvider that uses the openai-whisper package to
transcribe audio locally. No API key or network connection is required after
the model is downloaded on first use.
"""

import structlog

from src.voice.protocols import STTProvider

logger = structlog.get_logger()


class WhisperSTT(STTProvider):
    """Local Whisper speech-to-text.

    Model loading is deferred until the first call to transcribe() so that
    importing this module does not trigger a multi-second model download.
    """

    def __init__(self, model_size: str = "base"):
        self._model_size = model_size
        self._model = None

    def _load_model(self) -> None:
        if self._model is None:
            import whisper
            logger.info("loading_whisper_model", size=self._model_size)
            self._model = whisper.load_model(self._model_size)

    def transcribe(self, audio_path: str) -> str:
        """Transcribe an audio file and return the text.

        Args:
            audio_path: Path to the audio file (wav, mp3, m4a, etc.)

        Returns:
            Transcribed text string.
        """
        self._load_model()
        logger.info("transcribing_audio", path=audio_path)
        result = self._model.transcribe(audio_path)
        text = result["text"].strip()
        logger.info("transcription_complete", text_length=len(text))
        return text
