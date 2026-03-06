"""Abstract interfaces for speech and agent providers.

Defining these as ABCs/protocols means:

- Open/Closed: new backends (e.g. cloud Whisper, ElevenLabs) can be added by
  implementing these interfaces without touching any existing code.
- Liskov Substitution: any concrete implementation can be dropped in wherever
  the abstract type is expected — including test mocks.
- Dependency Inversion: VoicePipeline and the API depend on these abstractions,
  not on Whisper, Coqui, or the agent directly.
"""

from abc import ABC, abstractmethod


class STTProvider(ABC):
    """Converts audio files to text."""

    @abstractmethod
    def transcribe(self, audio_path: str) -> str:
        """Transcribe an audio file and return the text.

        Args:
            audio_path: Path to the audio file on disk.

        Returns:
            Transcribed text string.
        """


class TTSProvider(ABC):
    """Converts text to audio files."""

    @abstractmethod
    def speak(self, text: str, output_path: str) -> str:
        """Synthesise speech from text and save to an audio file.

        Args:
            text: The text to convert to speech.
            output_path: Destination file path for the generated audio.

        Returns:
            Path to the generated audio file.
        """


class AgentRunner(ABC):
    """Runs the conversational agent and returns a text response."""

    @abstractmethod
    async def run(self, message: str) -> str:
        """Send a text message to the agent and return its response.

        Args:
            message: The user's input text.

        Returns:
            The agent's text response.
        """
