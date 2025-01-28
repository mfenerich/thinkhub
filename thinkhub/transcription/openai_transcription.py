"""
Module for OpenAI Whisper transcription service.

Provides asynchronous transcription functionality using OpenAI APIs.
"""

import io
import logging
import os
from typing import Optional

import aiofiles
from openai import APIConnectionError, AsyncOpenAI, RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential

from thinkhub.transcription.base import TranscriptionServiceInterface
from thinkhub.transcription.exceptions import (
    AudioFileNotFoundError,
    ClientInitializationError,
    MissingAPIKeyError,
    TranscriptionJobError,
)


class OpenAITranscriptionService(TranscriptionServiceInterface):
    """Transcribing audio using OpenAI Whisper asynchronously."""

    def __init__(
        self,
        model: str = "whisper-1",
        api_key: Optional[str] = None,
        logging_level: int = logging.INFO,
    ) -> None:
        """
        Initialize the OpenAITranscriptionService with the given parameters.

        Args:
            model (str): The Whisper model to use for transcription. Default is "whisper-1".
            api_key (Optional[str]): Explicit API key for flexible configuration.
            logging_level (int): Logging configuration.
        """
        # Flexible API key retrieval
        self.api_key = api_key or os.getenv("CHATGPT_API_KEY")
        if not self.api_key:
            raise MissingAPIKeyError("No OpenAI API key found.")

        # Logging setup
        logging.basicConfig(level=logging_level)
        self.logger = logging.getLogger(__name__)

        # Model and client configuration
        self.model = model
        self.client: Optional[AsyncOpenAI] = None

    async def initialize_client(self) -> None:
        """
        Initialize the AsyncOpenAI client.

        Raises:
            ClientInitializationError: If the client fails to initialize.
        """
        try:
            self.client = AsyncOpenAI(api_key=self.api_key)
        except ImportError as e:
            self.logger.error("Failed to initialize AsyncOpenAI client.")
            raise ClientInitializationError(
                "Failed to initialize AsyncOpenAI client. Ensure the `openai` library is installed."
            ) from e

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def _safe_transcription_call(self, audio_file: io.BytesIO) -> str:
        """
        Safely call the OpenAI Whisper API with retry and logging.

        Args:
            audio_file (io.BytesIO): The audio file to transcribe.

        Returns:
            str: The transcribed text.

        Raises:
            TranscriptionJobError: If the transcription process fails.
        """
        try:
            response = await self.client.audio.transcriptions.create(
                model=self.model, file=audio_file
            )
            return response.text
        except (RateLimitError, APIConnectionError) as e:
            self.logger.error(f"Transcription API call failed: {e}")
            raise TranscriptionJobError(f"Transcription failed: {e}") from e

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
            self.logger.error(f"Audio file not found: {file_path}")
            raise AudioFileNotFoundError(f"Audio file not found: {file_path}")

        try:
            async with aiofiles.open(file_path, "rb") as af:
                audio_data = await af.read()

                # Convert the bytes into a file-like object
                audio_file = io.BytesIO(audio_data)
                audio_file.name = os.path.basename(file_path)

                # Use OpenAI Whisper API for transcription
                transcription = await self._safe_transcription_call(audio_file)

                # If there's no transcription at all, return a more explicit message
                return transcription if transcription else "No transcription available."

        except Exception as e:
            self.logger.error(f"Transcription failed: {e}")
            raise TranscriptionJobError(f"Transcription failed: {e}") from e

    async def close(self) -> None:
        """
        Close the client and release resources.

        No specific close action is required for the AsyncOpenAI client,
        but this method exists for consistency with the interface.
        """
        self.client = None
        self.logger.info("Client closed and resources released.")
