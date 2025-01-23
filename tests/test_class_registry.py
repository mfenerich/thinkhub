import pytest

from thinkhub.exceptions import ProviderNotFoundError
from thinkhub.transcription import (
    TranscriptionServiceError,
    get_available_providers,
    get_transcription_service,
    register_transcription_service,
)
from thinkhub.transcription.base import TranscriptionServiceInterface


@register_transcription_service("fake")
class FakeTranscriptionService(TranscriptionServiceInterface):
    """A simple fake transcription service for testing registration and retrieval."""

    def __init__(self, test_value=None):
        self.test_value = test_value

    async def initialize_client(self):
        pass

    async def transcribe(self, file_path: str) -> str:
        return f"Fake transcription for {file_path} with test_value={self.test_value}"

    async def close(self):
        pass


def test_register_transcription_service():
    """
    Verify the 'fake' service is registered.
    """
    providers = get_available_providers()
    assert "fake" in providers, "Expected 'fake' to be in available providers."


def test_re_register_service_with_warning(caplog):
    """
    Re-register 'fake' with another class, ensuring a warning is logged
    and the service is replaced.
    """

    @register_transcription_service("fake")
    class AnotherFakeService(TranscriptionServiceInterface):
        def __init__(self, test_value=None):
            self.test_value = test_value

        async def initialize_client(self):
            pass

        async def transcribe(self, file_path: str) -> str:
            return f"Another fake transcription with test_value={self.test_value}"

        async def close(self):
            pass

    assert (
        "Overriding transcription service: fake" in caplog.text
    ), "Expected a warning about overriding 'fake' service."

    instance = get_transcription_service("fake")
    assert isinstance(
        instance, AnotherFakeService
    ), "Expected to get the new AnotherFakeService, not the original one."


def test_get_transcription_service_with_params():
    """
    Test passing custom parameters to AnotherFakeService.
    """
    service = get_transcription_service("fake", test_value=123)
    assert service.test_value == 123


def test_unregistered_provider():
    """
    Test requesting an unregistered provider raises ProviderNotFoundError.
    """
    with pytest.raises(ProviderNotFoundError) as excinfo:
        get_transcription_service("non_existent")
    assert "Unsupported provider: non_existent" in str(excinfo.value)


def test_transcription_service_init_failure():
    """
    Test that an exception in the constructor raises TranscriptionServiceError.
    """

    @register_transcription_service("failing")
    class FailingService(TranscriptionServiceInterface):
        def __init__(self, *args, **kwargs):
            raise ValueError("Constructor failed")

        async def initialize_client(self):
            pass

        async def transcribe(self, file_path: str) -> str:
            return "Should never get here."

        async def close(self):
            pass

    with pytest.raises(TranscriptionServiceError) as excinfo:
        get_transcription_service("failing")
    assert "Constructor failed" in str(excinfo.value)
