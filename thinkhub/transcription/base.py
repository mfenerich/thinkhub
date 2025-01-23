from abc import ABC, abstractmethod


class TranscriptionServiceInterface(ABC):
    """Interface for transcription services."""

    @abstractmethod
    async def initialize_client(self):
        """Initializes the client."""
        pass

    @abstractmethod
    async def transcribe(self, file_path: str) -> str:
        """Transcribes the audio/video file."""
        pass

    @abstractmethod
    async def close(self):
        """Closes the client and releases resources."""
        pass
