"""Unit tests for testing the transcription service class registry."""

import pytest

from thinkhub.exceptions import ProviderNotFoundError
from thinkhub.transcription import (
    TranscriptionServiceError,
    get_available_providers,
    get_transcription_service,
    register_transcription_service,
    _REQUIRED_DEPENDENCIES
)
from thinkhub.transcription.base import TranscriptionServiceInterface


# Define test classes at module level
class FakeTranscriptionService(TranscriptionServiceInterface):
    """A simple fake transcription service for testing registration and retrieval."""

    def __init__(self, test_value=None):
        """Initialize the fake transcription service."""
        self.test_value = test_value

    async def initialize_client(self):
        """Initialize the fake client (noop for the fake service)."""
        pass

    async def transcribe(self, file_path: str) -> str:
        """Simulate transcription of an audio file."""
        return f"Fake transcription for {file_path} with test_value={self.test_value}"

    async def close(self):
        """Close the fake transcription service (noop for the fake service)."""
        pass


class AnotherFakeService(TranscriptionServiceInterface):
    """Another fake service for testing service replacement."""
    
    async def initialize_client(self):
        pass

    async def transcribe(self, file_path: str) -> str:
        return "Another fake transcription"

    async def close(self):
        pass


class FailingService(TranscriptionServiceInterface):
    """A service that fails during initialization for testing error handling."""
    
    def __init__(self, *args, **kwargs):
        raise ValueError("Constructor failed")

    async def initialize_client(self):
        pass

    async def transcribe(self, file_path: str) -> str:
        return "Should never get here"

    async def close(self):
        pass


def test_register_transcription_service():
    """Verify the 'fake' service is registered."""
    # Register the fake service
    register_transcription_service("fake")("tests.test_class_registry.FakeTranscriptionService")
    
    providers = get_available_providers()
    assert "fake" in providers, "Expected 'fake' to be in available providers."


def test_re_register_service_with_warning(caplog):
    """Re-register 'fake' with another class, ensuring a warning is logged."""
    # Register the new service
    register_transcription_service("fake")("tests.test_class_registry.AnotherFakeService")

    assert "Overriding transcription service: fake" in caplog.text, (
        "Expected a warning about overriding 'fake' service."
    )

    # Get the service and verify it's the new one
    instance = get_transcription_service("fake")
    assert isinstance(instance, AnotherFakeService), (
        "Expected to get the new AnotherFakeService instance"
    )


def test_get_transcription_service_with_params():
    """Test passing custom parameters to the service."""
    register_transcription_service("fake")("tests.test_class_registry.FakeTranscriptionService")
    service = get_transcription_service("fake", test_value=123)
    assert service.test_value == 123, "Expected test_value to be passed to constructor"


def test_unregistered_provider():
    """Test requesting an unregistered provider raises ProviderNotFoundError."""
    with pytest.raises(ProviderNotFoundError) as excinfo:
        get_transcription_service("non_existent")
    assert "Unsupported provider: non_existent" in str(excinfo.value)


def test_transcription_service_init_failure():
    """Test that an exception in the constructor raises TranscriptionServiceError."""
    register_transcription_service("failing")("tests.test_class_registry.FailingService")

    with pytest.raises(TranscriptionServiceError) as excinfo:
        get_transcription_service("failing")
    assert "Constructor failed" in str(excinfo.value)


def test_get_transcription_service_missing_dependencies(monkeypatch):
    """Test that missing dependencies are properly handled."""
    register_transcription_service("fake")("tests.test_class_registry.FakeTranscriptionService")
    
    # Simulate missing dependencies for the fake provider
    monkeypatch.setitem(_REQUIRED_DEPENDENCIES, "fake", ["non_existent_package"])
    
    with pytest.raises(ImportError) as excinfo:
        get_transcription_service("fake")
    assert "Missing dependencies for provider" in str(excinfo.value)