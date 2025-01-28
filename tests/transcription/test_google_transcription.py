"""
Unit tests for the GoogleTranscriptionService.

These tests cover various scenarios, including:
- Missing or invalid credentials.
- Initialization errors.
- File not found errors.
- Successful and failed transcription jobs.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from thinkhub.transcription.exceptions import (
    AudioFileNotFoundError,
    ClientInitializationError,
    InvalidGoogleCredentialsPathError,
    MissingGoogleCredentialsError,
    TranscriptionJobError,
)
from thinkhub.transcription.google_transcription import GoogleTranscriptionService


@pytest.fixture
def mock_env_creds(monkeypatch):
    """Mock the GOOGLE_APPLICATION_CREDENTIALS env var to a fake path."""
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/fake/path/to/creds.json")


@pytest.fixture
def mock_credentials_file():
    """
    Patches os.path.exists so that the Google creds file is always treated as existing.

    By default, we return True for all paths so that tests pass initialization.
    Tests can override or patch again if they want to simulate a missing audio file, etc.
    """
    with patch("os.path.exists") as mock_exists:
        mock_exists.return_value = True
        yield


@pytest.mark.asyncio
class TestGoogleTranscriptionService:
    """Test cases for the GoogleTranscriptionService class."""

    @pytest.mark.usefixtures("mock_credentials_file")
    async def test_missing_credentials_raises_error(self, monkeypatch):
        """Test MissingGoogleCredentialsError if GOOGLE_APPLICATION_CREDENTIALS is not set."""
        monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
        with pytest.raises(MissingGoogleCredentialsError):
            GoogleTranscriptionService()

    @pytest.mark.usefixtures("mock_env_creds")
    async def test_invalid_credentials_path_raises_error(self):
        """
        Test InvalidGoogleCredentialsPathError if the file doesn't exist.

        Here, we override the default fixture's behavior to force os.path.exists=False.
        """
        with patch("os.path.exists", return_value=False):
            with pytest.raises(InvalidGoogleCredentialsPathError):
                GoogleTranscriptionService()

    @pytest.mark.usefixtures("mock_env_creds", "mock_credentials_file")
    @patch(
        "thinkhub.transcription.google_transcription.speech_v1.SpeechAsyncClient",
        autospec=True,
    )
    async def test_initialize_client_failure(self, mock_client_class):
        """
        Test that ClientInitializationError is raised if SpeechAsyncClient() fails.

        We patch the constructor so it raises an exception.
        """
        mock_client_class.side_effect = Exception("Init fail")

        service = GoogleTranscriptionService()
        with pytest.raises(ClientInitializationError) as excinfo:
            await service.initialize_client()

        assert "Failed to initialize Google Speech client: Init fail" in str(
            excinfo.value
        )

    @pytest.mark.usefixtures("mock_env_creds", "mock_credentials_file")
    @patch(
        "thinkhub.transcription.google_transcription.speech_v1.SpeechAsyncClient",
        autospec=True,
    )
    async def test_transcribe_file_not_found(self, mock_client_class):
        """
        Test AudioFileNotFoundError is raised for a non-existent local audio file.

        We override os.path.exists so the creds file is found, but the audio file is not.
        """

        def fake_exists(path):
            # The creds file exists, but the audio file does not
            return path == "/fake/path/to/creds.json"

        with patch("os.path.exists", side_effect=fake_exists):
            # Constructor returns a mock instance (so init succeeds)
            mock_client_instance = MagicMock()
            mock_client_class.return_value = mock_client_instance

            service = GoogleTranscriptionService()
            with pytest.raises(AudioFileNotFoundError) as excinfo:
                await service.transcribe("dummy_file.flac")
            assert "Audio file not found: dummy_file.flac" in str(excinfo.value)

    @pytest.mark.usefixtures("mock_env_creds", "mock_credentials_file")
    @patch("aiofiles.open", new_callable=MagicMock)
    @patch("os.path.exists", return_value=True)
    @patch(
        "thinkhub.transcription.google_transcription.speech_v1.SpeechAsyncClient",
        autospec=True,
    )
    async def test_transcription_job_error(
        self, mock_client_class, mock_exists, mock_aiofiles_open
    ):
        """Test that a generic exception during recognition raises TranscriptionJobError."""
        # Set up the mock instance
        mock_client_instance = MagicMock()
        # Mock the async recognize call to raise an exception
        mock_client_instance.recognize = AsyncMock(
            side_effect=Exception("Transcription failure")
        )
        mock_client_class.return_value = mock_client_instance

        # Mock aiofiles.open to simulate a valid file read
        mock_aiofiles_open.return_value.__aenter__.return_value.read.return_value = (
            b"audio data"
        )

        service = GoogleTranscriptionService()

        with pytest.raises(TranscriptionJobError) as excinfo:
            await service.transcribe("dummy_file.flac")

        assert "Transcription failed: [Errno 2] No such file or directory" in str(
            excinfo.value
        )

    @pytest.mark.usefixtures("mock_env_creds", "mock_credentials_file")
    @patch("pydub.AudioSegment.from_file")
    @patch("aiofiles.open", new_callable=MagicMock)
    @patch("os.path.exists", return_value=True)
    @patch(
        "thinkhub.transcription.google_transcription.speech_v1.SpeechAsyncClient",
        autospec=True,
    )
    async def test_transcription_success(
        self, mock_client_class, mock_exists, mock_aiofiles_open, mock_audio_segment
    ):
        """Test a successful transcription scenario."""
        # Mock audio segment
        mock_audio_segment.return_value = MagicMock()
        mock_audio_segment.return_value.__len__.return_value = 30 * 1000  # 30 seconds

        # Mock aiofiles.open to simulate reading the audio file
        mock_aiofiles_open.return_value.__aenter__.return_value.read.return_value = (
            b"audio data"
        )

        # Set up the mock instance with a normal return for recognize()
        mock_client_instance = MagicMock()
        mock_client_instance.recognize = AsyncMock()

        # Mock a valid response
        mock_result = MagicMock()
        mock_result.alternatives = [MagicMock(transcript="Hello, World!")]
        mock_response = MagicMock(results=[mock_result])
        mock_client_instance.recognize.return_value = mock_response

        # The constructor returns this mock instance
        mock_client_class.return_value = mock_client_instance

        service = GoogleTranscriptionService()

        transcript = await service.transcribe("dummy_file.flac")
        assert transcript == "Hello, World!"

    @pytest.mark.usefixtures("mock_env_creds", "mock_credentials_file")
    async def test_warn_on_missing_bucket_name(self):
        """Test that a warning is issued if bucket_name is not provided."""
        with patch("logging.Logger.warning") as mock_warning:
            GoogleTranscriptionService()
            mock_warning.assert_called_once_with(
                "Bucket name not provided. Audios longer than 1 minute cannot be transcribed."
            )

    @pytest.mark.usefixtures("mock_env_creds", "mock_credentials_file")
    @patch("pydub.AudioSegment.from_file")
    @patch("os.path.exists", return_value=True)
    @patch(
        "thinkhub.transcription.google_transcription.speech_v1.SpeechAsyncClient",
        autospec=True,
    )
    async def test_error_on_long_audio_without_bucket_name(
        self, mock_client_class, mock_exists, mock_audio_segment
    ):
        """Test that an error is raised for long audio without a bucket name."""
        # Mock audio segment
        mock_audio_segment.return_value = MagicMock()
        mock_audio_segment.return_value.__len__.return_value = 61 * 1000  # 61 seconds

        service = GoogleTranscriptionService()

        with pytest.raises(TranscriptionJobError) as excinfo:
            await service.transcribe("long_audio.flac")

        assert (
            "Bucket name is required to transcribe audio files longer than 1 minute."
            in str(excinfo.value)
        )


# Ensure pytest-asyncio deprecation warning is resolved
pytestmark = pytest.mark.asyncio("function")
