from abc import ABC, abstractmethod


class TranscriptionServiceInterface(ABC):
    @abstractmethod
    async def initialize_client(self):
        """Initialize the transcription client."""
        pass

    @abstractmethod
    async def transcribe(self, file_path: str) -> str:
        """Transcribe audio from a file."""
        pass

    @abstractmethod
    async def close(self):
        """Close the transcription client."""
        pass
