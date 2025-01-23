"""
Define the abstract base class for transcription services.

It provides an interface (`TranscriptionServiceInterface`) that must be implemented
by all transcription service classes. The interface enforces the implementation
of methods for initializing a client, transcribing audio/video files, and
closing the client to release resources.
"""

from abc import ABC, abstractmethod


class TranscriptionServiceInterface(ABC):
    """Interface for transcription services."""

    @abstractmethod
    async def initialize_client(self):
        """Initialize the client."""
        pass

    @abstractmethod
    async def transcribe(self, file_path: str) -> str:
        """Transcribes the audio/video file."""
        pass

    @abstractmethod
    async def close(self):
        """Close the client and releases resources."""
        pass
