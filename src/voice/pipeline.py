import os
import tempfile
import uuid

import structlog

from src.voice.protocols import AgentRunner, STTProvider, TTSProvider

logger = structlog.get_logger()


class DefaultAgentRunner(AgentRunner):
    async def run(self, message: str) -> str:
        from src.agent.graph import run_agent
        return await run_agent(message)


class VoicePipeline:
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
        logger.info("voice_pipeline_start", input=audio_path)

        text = self._stt.transcribe(audio_path)
        if not text:
            raise ValueError("Audio transcription produced no text.")

        logger.info("voice_pipeline_transcribed", text_length=len(text))

        response_text = await self._agent.run(text)

        output_path = os.path.join(self._output_dir, f"response_{uuid.uuid4().hex}.wav")
        self._tts.speak(response_text, output_path)

        logger.info("voice_pipeline_complete", output=output_path)
        return output_path
