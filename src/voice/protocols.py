from abc import ABC, abstractmethod


class STTProvider(ABC):
    @abstractmethod
    def transcribe(self, audio_path: str) -> str: ...


class TTSProvider(ABC):
    @abstractmethod
    def speak(self, text: str, output_path: str) -> str: ...


class AgentRunner(ABC):
    @abstractmethod
    async def run(self, message: str) -> str: ...
