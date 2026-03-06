import structlog

logger = structlog.get_logger()


class TTS:
    """Coqui TTS text-to-speech wrapper.

    Uses the TTS package to synthesise speech locally.
    No API key or network connection required after the model is downloaded.
    """

    DEFAULT_MODEL = "tts_models/en/ljspeech/tacotron2-DDC"

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self._model_name = model_name
        self._tts = None

    def _load_model(self):
        if self._tts is None:
            from TTS.api import TTS as CoquiTTS
            logger.info("loading_tts_model", model=self._model_name)
            self._tts = CoquiTTS(self._model_name)

    def speak(self, text: str, output_path: str = "output.wav") -> str:
        """Synthesise speech from text and save to a wav file.

        Args:
            text: The text to convert to speech.
            output_path: File path where the wav will be saved.

        Returns:
            Path to the generated audio file.
        """
        self._load_model()
        logger.info("synthesising_speech", text_length=len(text), output=output_path)
        self._tts.tts_to_file(text=text, file_path=output_path)
        logger.info("speech_synthesis_complete", output=output_path)
        return output_path
