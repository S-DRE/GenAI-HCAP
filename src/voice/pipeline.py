"""Voice pipeline — sequences STT → agent → TTS.

Design:
- VoicePipeline has a single responsibility: coordinate the three steps of a
  voice interaction (SRP).
- It depends on STTProvider, TTSProvider, and AgentRunner abstractions, not
  on any concrete implementation (DIP).
- The DefaultAgentRunner wraps the real run_agent function so the pipeline
  can be tested with a lightweight mock without importing the agent (OCP).
"""

import os
import tempfile

import structlog

from src.voice.protocols import AgentRunner, STTProvider, TTSProvider

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Default agent runner — thin wrapper so the pipeline stays decoupled
# ---------------------------------------------------------------------------

class DefaultAgentRunner(AgentRunner):
    """Delegates to the real LangGraph agent."""

    async def run(self, message: str) -> str:
        from src.agent.graph import run_agent
        return await run_agent(message)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class VoicePipeline:
    """Orchestrates a full voice round-trip.

    1. Transcribes incoming audio to text (STT).
    2. Passes the text to the conversational agent.
    3. Synthesises the agent's response back to audio (TTS).

    Args:
        stt:          Any STTProvider implementation.
        tts:          Any TTSProvider implementation.
        agent_runner: Any AgentRunner implementation.
        output_dir:   Directory where response audio files are written.
                      Defaults to the system temp directory.
    """

    def __init__(
        self,
        stt: STTProvider,
        tts: TTSProvider,
        agent_runner: AgentRunner | None = None,
        output_dir: str | None = None,
    ) -> None:
        self._stt = stt
        self._tts = tts
        self._agent = agent_runner or DefaultAgentRunner()
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

        response_text = await self._agent.run(text)

        output_path = os.path.join(self._output_dir, "response.wav")
        self._tts.speak(response_text, output_path)

        logger.info("voice_pipeline_complete", output=output_path)
        return output_path
