import structlog

from src.voice.protocols import STTProvider

logger = structlog.get_logger()


class WhisperSTT(STTProvider):
    # Model is loaded lazily on first transcribe() call to avoid slow startup.
    def __init__(self, model_size: str = "base"):
        self._model_size = model_size
        self._model = None

    def _load_model(self) -> None:
        if self._model is None:
            import whisper
            logger.info("loading_whisper_model", size=self._model_size)
            self._model = whisper.load_model(self._model_size)

    def transcribe(self, audio_path: str) -> str:
        self._load_model()
        logger.info("transcribing_audio", path=audio_path)
        result = self._model.transcribe(audio_path)
        text = result["text"].strip()
        logger.info("transcription_complete", text_length=len(text))
        return text
