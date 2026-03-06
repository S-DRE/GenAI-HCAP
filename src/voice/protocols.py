"""Abstract interfaces for speech providers.

Defining STTProvider and TTSProvider as ABCs (Abstract Base Classes) means:

- Open/Closed: new backends (e.g. cloud Whisper, ElevenLabs) can be added by
  implementing these interfaces without touching any existing code.
- Liskov Substitution: any concrete implementation can be dropped in wherever
  the abstract type is expected — including test mocks.
- Dependency Inversion: VoicePipeline and the API depend on these abstractions,
  not on Whisper or Coqui directly.
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
