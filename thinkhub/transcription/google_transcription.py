import os
from google.cloud import speech_v1
from thinkhub.transcription.base import TranscriptionServiceInterface


class GoogleTranscriptionService(TranscriptionServiceInterface):
    def __init__(self, sample_rate=24000, bucket_name="speechmarcel"):
        """
        Initialize GoogleTranscriptionService with sample rate and bucket name.
        Sets up Google Application Credentials from the .env file.
        """
        self.client = None
        self.bucket_name = bucket_name
        self.rate = sample_rate

        # Load GOOGLE_APPLICATION_CREDENTIALS from .env
        self._load_google_credentials()

    def ls_load_google_credentials(self):
        """Load GOOGLE_APPLICATION_CREDENTIALS from .env."""
        google_creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if not google_creds:
            raise ValueError("GOOGLE_APPLICATION_CREDENTIALS environment variable is not set.")
        if not os.path.exists(google_creds):
            raise FileNotFoundError(f"GOOGLE_APPLICATION_CREDENTIALS file not found: {google_creds}")
        
        # Set the environment variable for Google Cloud
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_creds

    async def initialize_client(self):
        """Initialize SpeechAsyncClient within the correct event loop."""
        self.client = speech_v1.SpeechAsyncClient()

    async def transcribe(self, file_path: str) -> str:
        """Asynchronously transcribe audio."""
        if self.client is None:
            await self.initialize_client()

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found.")

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
                result.alternatives[0].transcript
                for result in response.results
            )
            return transcription or "No transcription available."
        except Exception as e:
            return f"Transcription failed due to an error: {e}"

    async def close(self):
        """Close the gRPC client."""
        if self.client:
            await self.client.close()
