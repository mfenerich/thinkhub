"""
Module for OpenAI Whisper transcription service.

Provides asynchronous transcription functionality using OpenAI APIs.
"""

import io
import os

import aiofiles

from thinkhub.transcription.base import TranscriptionServiceInterface
from thinkhub.transcription.exceptions import (
    AudioFileNotFoundError,
    ClientInitializationError,
    MissingAPIKeyError,
    TranscriptionJobError,
)


class OpenAITranscriptionService(TranscriptionServiceInterface):
    """Transcribing audio using OpenAI Whisper asynchronously."""

    def __init__(self, model: str = "whisper-1") -> None:
        """
        Initialize the OpenAITranscriptionService with the given parameters.

        Args:
            model (str): The Whisper model to use for transcription. Default is "whisper-1".
        """
        self.api_key = os.getenv("CHATGPT_API_KEY")
        if not self.api_key:
            raise MissingAPIKeyError("CHATGPT_API_KEY environment variable is missing.")
        self.model = model
        self.client = None

    async def initialize_client(self) -> None:
        """
        Initialize the AsyncOpenAI client.

        Raises:
            ClientInitializationError: If the client fails to initialize.
        """
        try:
            from openai import AsyncOpenAI

            self.client = AsyncOpenAI(api_key=self.api_key)
        except ImportError as e:
            raise ClientInitializationError(
                "Failed to initialize AsyncOpenAI client. Ensure the `openai` library is installed."
            ) from e

    async def transcribe(self, file_path: str) -> str:
        """
        Asynchronously transcribe an audio file using OpenAI Whisper.

        Args:
            file_path (str): The path to the audio file to transcribe.

        Returns:
            str: The transcribed text.

        Raises:
            AudioFileNotFoundError: If the specified audio file does not exist.
            TranscriptionJobError: If the transcription process fails.
        """
        if self.client is None:
            await self.initialize_client()

        if not os.path.exists(file_path):
            raise AudioFileNotFoundError(f"Audio file not found: {file_path}")

        try:
            async with aiofiles.open(file_path, "rb") as af:
                audio_data = await af.read()

                # Convert the bytes into a file-like object
                audio_file = io.BytesIO(audio_data)
                audio_file.name = os.path.basename(file_path)

                # Use OpenAI Whisper API for transcription
                response = await self.client.audio.transcriptions.create(
                    model=self.model, file=audio_file
                )

            transcription = response.text

            # If there's no transcription at all, return a more explicit message
            return transcription if transcription else "No transcription available."

        except Exception as e:
            raise TranscriptionJobError(f"Transcription failed: {e}") from e

    async def close(self) -> None:
        """
        Close the client and release resources.

        No specific close action is required for the AsyncOpenAI client,
        but this method exists for consistency with the interface.
        """
        self.client = None
