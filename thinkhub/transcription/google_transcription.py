import os

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
    def __init__(self, sample_rate=24000, bucket_name=""):
        """
        Initialize GoogleTranscriptionService with sample rate and bucket name.

        Sets up Google Application Credentials from the .env file.
        """
        self.client = None
        self.bucket_name = bucket_name
        self.rate = sample_rate

        # Load GOOGLE_APPLICATION_CREDENTIALS from .env
        self._load_google_credentials()

    def _load_google_credentials(self):
        """Load GOOGLE_APPLICATION_CREDENTIALS from .env."""
        google_creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if not google_creds:
            # Raise custom exception for missing credentials
            raise MissingGoogleCredentialsError(
                "GOOGLE_APPLICATION_CREDENTIALS environment variable is not set."
            )

        if not os.path.exists(google_creds):
            # Raise custom exception for an invalid file path
            raise InvalidGoogleCredentialsPathError(
                f"GOOGLE_APPLICATION_CREDENTIALS file not found: {google_creds}"
            )

        # Set the environment variable for Google Cloud
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_creds

    async def initialize_client(self):
        """Initialize SpeechAsyncClient."""
        try:
            self.client = speech_v1.SpeechAsyncClient()
        except Exception as e:
            # Raise a custom initialization error
            raise ClientInitializationError(
                f"Failed to initialize Google Speech client: {e}"
            ) from e

    async def transcribe(self, file_path: str) -> str:
        """Asynchronously transcribe audio."""
        if self.client is None:
            await self.initialize_client()

        if not os.path.exists(file_path):
            # Raise a custom exception when the audio file is not found
            raise AudioFileNotFoundError(f"File {file_path} not found.")

        try:
            with open(file_path, "rb") as f:
                audio_content = f.read()

            audio = speech_v1.RecognitionAudio(content=audio_content)
            config = speech_v1.RecognitionConfig(
                encoding=speech_v1.RecognitionConfig.AudioEncoding.FLAC,
                sample_rate_hertz=self.rate,
                language_code="en-US",
            )

            # Perform async recognition
            response = await self.client.recognize(config=config, audio=audio)

            # Extract transcripts
            transcription = "".join(
                result.alternatives[0].transcript for result in response.results
            )
            return transcription or "No transcription available."

        except Exception as e:
            # Raise a custom transcription job error
            raise TranscriptionJobError(f"Transcription failed: {e}") from e

    async def close(self):
        """Close the gRPC client."""
        if self.client:
            await self.client.close()
            self.client = None  # Important to avoid resource warnings
