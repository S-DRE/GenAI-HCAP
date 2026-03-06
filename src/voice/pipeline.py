"""Voice pipeline — sequences STT → agent → TTS.

VoicePipeline has a single responsibility: coordinate the three steps of a
voice interaction. It depends on the STTProvider and TTSProvider abstractions,
not on any concrete implementation, so it can be used with real models in
production or with lightweight mocks in tests.
"""

import os
import tempfile

import structlog

from src.agent.graph import run_agent
from src.voice.protocols import STTProvider, TTSProvider

logger = structlog.get_logger()


class VoicePipeline:
    """Orchestrates a full voice round-trip.

    1. Transcribes incoming audio to text (STT).
    2. Passes the text to the conversational agent.
    3. Synthesises the agent's response back to audio (TTS).

    Args:
        stt: Any STTProvider implementation.
        tts: Any TTSProvider implementation.
        output_dir: Directory where response audio files are written.
                    Defaults to the system temp directory.
    """

    def __init__(
        self,
        stt: STTProvider,
        tts: TTSProvider,
        output_dir: str | None = None,
    ) -> None:
        self._stt = stt
        self._tts = tts
        self._output_dir = output_dir or tempfile.gettempdir()

    async def process(self, audio_path: str) -> str:
        """Run a full voice interaction cycle.

        Args:
            audio_path: Path to the incoming audio file from the user.

        Returns:
            Path to the generated response audio file.

        Raises:
            ValueError: If the transcribed text is empty.
        """
        logger.info("voice_pipeline_start", input=audio_path)

        text = self._stt.transcribe(audio_path)
        if not text:
            raise ValueError("Audio transcription produced no text.")

        logger.info("voice_pipeline_transcribed", text_length=len(text))

        response_text = await run_agent(text)

        output_path = os.path.join(self._output_dir, "response.wav")
        self._tts.speak(response_text, output_path)

        logger.info("voice_pipeline_complete", output=output_path)
        return output_path
