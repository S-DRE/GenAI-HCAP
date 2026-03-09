import structlog

from src.voice.protocols import TTSProvider

logger = structlog.get_logger()


class CoquiTTS(TTSProvider):
    # Model is loaded lazily on first speak() call to avoid slow startup.
    DEFAULT_MODEL = "tts_models/en/ljspeech/tacotron2-DDC"

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self._model_name = model_name
        self._tts = None

    def _load_model(self) -> None:
        if self._tts is None:
            from TTS.api import TTS as CoquiTTSModel
            logger.info("loading_tts_model", model=self._model_name)
            self._tts = CoquiTTSModel(self._model_name)

    def speak(self, text: str, output_path: str = "output.wav") -> str:
        self._load_model()
        logger.info("synthesising_speech", text_length=len(text), output=output_path)
        self._tts.tts_to_file(text=text, file_path=output_path)
        logger.info("speech_synthesis_complete", output=output_path)
        return output_path
