"""
Module for Google Cloud Speech-to-Text transcription service.

Provides asynchronous transcription functionality using Google APIs.
"""

import logging
import os
from typing import Optional

import aiofiles
from google.api_core.exceptions import GoogleAPICallError, RetryError
from google.cloud import speech_v1, storage
from pydub import AudioSegment
from tenacity import retry, stop_after_attempt, wait_exponential

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

    def __init__(
        self,
        sample_rate: int = 24000,
        bucket_name: Optional[str] = None,
        language_code: str = "en-US",
        logging_level: int = logging.INFO,
    ) -> None:
        """
        Initialize the GoogleTranscriptionService with the given parameters.

        Args:
            sample_rate (int): The sampling rate of the input audio. Default is 24000.
            bucket_name (Optional[str]): The name of a Google Cloud Storage bucket if needed.
            language_code (str): The language code for transcription. Default is "en-US".
            logging_level (int): Logging configuration.
        """
        # Logging setup
        logging.basicConfig(level=logging_level)
        self.logger = logging.getLogger(__name__)

        # Client and configuration
        self.client: Optional[speech_v1.SpeechAsyncClient] = None
        self.bucket_name = bucket_name
        self.sample_rate = sample_rate
        self.language_code = language_code

        # Validate Google credentials
        self._load_google_credentials()

        if not bucket_name:
            self.logger.warning(
                "Bucket name not provided. Audios longer than 1 minute cannot be transcribed."
            )

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
        try:
            self.client = speech_v1.SpeechAsyncClient()
            self.logger.info("Google Speech client initialized successfully.")
        except Exception as e:
            self.logger.error(f"Failed to initialize Google Speech client: {e}")
            raise ClientInitializationError(
                f"Failed to initialize Google Speech client: {e}"
            ) from e

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def _safe_upload_to_gcs(
        self, file_path: str, destination_blob_name: str
    ) -> str:
        """
        Safely upload a file to Google Cloud Storage with retry and logging.

        Args:
            file_path (str): The path to the file to upload.
            destination_blob_name (str): The name of the blob in GCS.

        Returns:
            str: The GCS URI of the uploaded file.

        Raises:
            TranscriptionJobError: If the upload fails.
        """
        if not self.bucket_name:
            raise TranscriptionJobError(
                "Bucket name is not set. Cannot upload files to GCS."
            )

        try:
            storage_client = storage.Client()
            bucket = storage_client.bucket(self.bucket_name)
            blob = bucket.blob(destination_blob_name)

            blob.upload_from_filename(file_path)
            self.logger.info(f"File uploaded to GCS: {destination_blob_name}")

            return f"gs://{self.bucket_name}/{destination_blob_name}"
        except Exception as e:
            self.logger.error(f"Failed to upload file to GCS: {e}")
            raise TranscriptionJobError(f"Failed to upload file to GCS: {e}") from e

    def _create_recognition_config(
        self, audio_content: Optional[bytes] = None, gcs_uri: Optional[str] = None
    ) -> tuple[speech_v1.RecognitionConfig, speech_v1.RecognitionAudio]:
        """
        Create a recognition config and audio object for the Google Speech API.

        Args:
            audio_content (Optional[bytes]): The audio content as bytes.
            gcs_uri (Optional[str]): The GCS URI of the audio file.

        Returns:
            tuple: A tuple containing the recognition config and audio objects.

        Raises:
            ValueError: If neither audio_content nor gcs_uri is provided.
        """
        if gcs_uri:
            audio = speech_v1.RecognitionAudio(uri=gcs_uri)
        elif audio_content:
            audio = speech_v1.RecognitionAudio(content=audio_content)
        else:
            raise ValueError("Either audio_content or gcs_uri must be provided.")

        config = speech_v1.RecognitionConfig(
            encoding=speech_v1.RecognitionConfig.AudioEncoding.FLAC,
            sample_rate_hertz=self.sample_rate,
            language_code=self.language_code,
        )

        return config, audio

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def _safe_transcription_call(
        self, config: speech_v1.RecognitionConfig, audio: speech_v1.RecognitionAudio
    ) -> speech_v1.RecognizeResponse:
        """
        Safely call the Google Speech API with retry and logging.

        Args:
            config: The recognition config.
            audio: The recognition audio.

        Returns:
            speech_v1.RecognizeResponse: The response from the Google Speech API.

        Raises:
            TranscriptionJobError: If the transcription process fails.
        """
        try:
            if audio.uri:
                operation = await self.client.long_running_recognize(
                    config=config, audio=audio
                )
                response = await operation.result(timeout=300)
            else:
                response = await self.client.recognize(config=config, audio=audio)

            return response
        except (GoogleAPICallError, RetryError) as e:
            self.logger.error(f"Transcription API call failed: {e}")
            raise TranscriptionJobError(f"Transcription failed: {e}") from e

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
        if self.client is None:
            await self.initialize_client()

        if not os.path.exists(file_path):
            self.logger.error(f"Audio file not found: {file_path}")
            raise AudioFileNotFoundError(f"Audio file not found: {file_path}")

        try:
            audio_segment = AudioSegment.from_file(file_path)
            duration_seconds = len(audio_segment) / 1000

            if duration_seconds > 60:
                if not self.bucket_name:
                    raise TranscriptionJobError(
                        "Bucket name is required to transcribe audio files longer than 1 minute."
                    )

                temp_audio_path = "temp_audio.flac"
                audio_segment.export(temp_audio_path, format="flac")
                gcs_uri = await self._safe_upload_to_gcs(
                    temp_audio_path, "temp_audio.flac"
                )

                config, audio = self._create_recognition_config(gcs_uri=gcs_uri)
            else:
                async with aiofiles.open(file_path, "rb") as f:
                    audio_content = await f.read()

                config, audio = self._create_recognition_config(
                    audio_content=audio_content
                )

            response = await self._safe_transcription_call(config, audio)

            transcription = "".join(
                result.alternatives[0].transcript for result in response.results
            )

            return transcription if transcription else "No transcription available."

        except Exception as e:
            self.logger.error(f"Transcription failed: {e}")
            raise TranscriptionJobError(f"Transcription failed: {e}") from e

    async def close(self) -> None:
        """Close the gRPC client connection gracefully."""
        if self.client:
            await self.client.close()
            self.client = None
            self.logger.info("Google Speech client closed.")
