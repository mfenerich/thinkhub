"""
Module for Google Cloud Speech-to-Text transcription service.

Provides asynchronous transcription functionality using Google APIs.
"""

import os
from typing import Optional

import aiofiles
from google.cloud import speech_v1

from thinkhub.transcription.base import TranscriptionServiceInterface
from thinkhub.transcription.exceptions import (
    AudioFileNotFoundError,
    ClientInitializationError,
    InvalidGoogleCredentialsPathError,
    MissingGoogleCredentialsError,
    TranscriptionJobError,
)


class GoogleTranscriptionService(TranscriptionServiceInterface):
    """Transcribing audio using Google Cloud Speech-to-Text asynchronously."""

    def __init__(self, sample_rate: int = 24000, bucket_name: str = "") -> None:
        """
        Initialize the GoogleTranscriptionService with the given parameters.

        Args:
            sample_rate (int): The sampling rate of the input audio. Default is 24000.
            (Optional) The name of a Google Cloud Storage bucket if needed.
        """
        self.client: Optional[speech_v1.SpeechAsyncClient] = None
        self.bucket_name = bucket_name
        self.sample_rate = sample_rate

        # Ensure Google credentials environment variable is set
        self._load_google_credentials()

        # Initialize the client (asynchronously).
        # If you prefer to explicitly initialize later, remove this line
        # and call `await self.initialize_client()` manually.
        # But be aware that `__init__` cannot be truly async.
        # Instead, you could do lazy initialization on first use in `transcribe`.
        # For demonstration, we show how to handle it separately.
        # In practice, you might leave it to be called in `transcribe`
        # if you need real async initialization.
        #
        # Example if you want lazy initialization:
        #     pass
        #
        # Otherwise, to do it here (blocking call), see the comment
        # inside initialize_client.
        #
        # However, because `initialize_client` is async, we typically
        # won't call it directly in __init__.
        # We'll rely on the check in `transcribe` to do it for us.

    def _load_google_credentials(self) -> None:
        """
        Load and validate the GOOGLE_APPLICATION_CREDENTIALS environment variable.

        Raises:
            MissingGoogleCredentialsError: If the environment variable is not set.
            InvalidGoogleCredentialsPathError: If the file path provided does not exist.
        """
        google_creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if not google_creds_path:
            raise MissingGoogleCredentialsError(
                "GOOGLE_APPLICATION_CREDENTIALS environment variable is not set."
            )

        if not os.path.exists(google_creds_path):
            raise InvalidGoogleCredentialsPathError(
                f"GOOGLE_APPLICATION_CREDENTIALS file not found: {google_creds_path}"
            )

        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_creds_path

    async def initialize_client(self) -> None:
        """
        Asynchronously initialize the Google Speech client.

        Raises:
            ClientInitializationError: If the client fails to initialize.
        """
        # Because this is an async method, if you call it from __init__, you need
        # an async context. Typically, we do lazy initialization in `transcribe`.
        try:
            self.client = speech_v1.SpeechAsyncClient()
        except Exception as e:
            raise ClientInitializationError(
                f"Failed to initialize Google Speech client: {e}"
            ) from e

    async def transcribe(self, file_path: str) -> str:
        """
        Asynchronously transcribe an audio file using Google Cloud Speech-to-Text.

        Args:
            file_path (str): The path to the audio file to transcribe.

        Returns:
            str: The transcribed text.

        Raises:
            AudioFileNotFoundError: If the specified audio file does not exist.
            ClientInitializationError: If the client cannot be initialized.
            TranscriptionJobError: If the transcription process fails.
        """
        # Ensure client is initialized
        if self.client is None:
            await self.initialize_client()

        if not os.path.exists(file_path):
            raise AudioFileNotFoundError(f"Audio file not found: {file_path}")

        try:
            # Use aiofiles for non-blocking file operations
            async with aiofiles.open(file_path, "rb") as f:
                audio_content = await f.read()

            audio = speech_v1.RecognitionAudio(content=audio_content)
            config = speech_v1.RecognitionConfig(
                encoding=speech_v1.RecognitionConfig.AudioEncoding.FLAC,
                sample_rate_hertz=self.sample_rate,
                language_code="en-US",
            )

            # Perform asynchronous recognition
            response = await self.client.recognize(config=config, audio=audio)

            # Extract transcripts from response
            transcription = "".join(
                result.alternatives[0].transcript for result in response.results
            )

            # If there's no transcription at all, return a more explicit message
            return transcription if transcription else "No transcription available."

        except Exception as e:
            raise TranscriptionJobError(f"Transcription failed: {e}") from e

    async def close(self) -> None:
        """Close the gRPC client connection gracefully."""
        if self.client:
            await self.client.close()
            self.client = None
