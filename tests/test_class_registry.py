"""Unit tests for testing the transcription service class registry."""

from unittest.mock import patch

import pytest

from thinkhub.exceptions import ProviderNotFoundError
from thinkhub.transcription import (
    _REQUIRED_DEPENDENCIES,
    TranscriptionServiceError,
    get_available_providers,
    get_transcription_service,
    register_transcription_service,
)
from thinkhub.transcription.base import TranscriptionServiceInterface

# Constants for test values and messages
FAKE_PROVIDER = "fake"
FAILING_PROVIDER = "failing"
NON_EXISTENT_PROVIDER = "non_existent"
TEST_VALUE = 123
VALID_TEST_VALUE = "valid_test"
NON_EXISTENT_PACKAGE = "non_existent_package"

TEST_MODULE_PATH = "tests.test_class_registry"
FAKE_SERVICE_PATH = f"{TEST_MODULE_PATH}.FakeTranscriptionService"
ANOTHER_FAKE_SERVICE_PATH = f"{TEST_MODULE_PATH}.AnotherFakeService"
FAILING_SERVICE_PATH = f"{TEST_MODULE_PATH}.FailingService"

ERROR_MESSAGES = {
    "constructor_failed": "Constructor failed",
    "missing_dependencies": "Missing dependencies for provider",
    "unsupported_provider": f"Unsupported provider: {NON_EXISTENT_PROVIDER}",
    "override_warning": f"Overriding transcription service: {FAKE_PROVIDER}",
    "expected_fake": "Expected 'fake' to be in available providers.",
    "expected_test_value": "Expected test_value to be passed to constructor",
}


def fake_transcription_text(file_path: str, test_value) -> str:
    """Generate fake transcription text for testing.

    Args:
        file_path: Path to the file being transcribed
        test_value: Test value to include in the transcription

    Returns:
        Formatted transcription text string
    """
    return f"Fake transcription for {file_path} with test_value={test_value}"


class FakeTranscriptionService(TranscriptionServiceInterface):
    """A simple fake transcription service for testing registration and retrieval."""

    def __init__(self, test_value=None):
        """Initialize the fake transcription service.

        Args:
            test_value: Optional test value to store
        """
        self.test_value = test_value

    async def initialize_client(self):
        """Initialize the fake client (noop for testing)."""
        pass

    async def transcribe(self, file_path: str) -> str:
        """Simulate transcription of an audio file.

        Args:
            file_path: Path to the file to transcribe

        Returns:
            Fake transcription text
        """
        return fake_transcription_text(file_path, self.test_value)

    async def close(self):
        """Close the fake transcription service (noop for testing)."""
        pass


class AnotherFakeService(TranscriptionServiceInterface):
    """Another fake service for testing service replacement."""

    async def initialize_client(self):
        """Initialize the client (noop for testing)."""
        pass

    async def transcribe(self, file_path: str) -> str:
        """Simulate transcription with a different implementation.

        Args:
            file_path: Path to the file to transcribe

        Returns:
            Static fake transcription text
        """
        return "Another fake transcription"

    async def close(self):
        """Close the service (noop for testing)."""
        pass


class FailingService(TranscriptionServiceInterface):
    """A service that fails during initialization for testing error handling."""

    def __init__(self, *args, **kwargs):
        """Initialize the failing service.

        Raises:
            ValueError: Always raises to simulate initialization failure
        """
        raise ValueError(ERROR_MESSAGES["constructor_failed"])

    async def initialize_client(self):
        """Initialize the client (never called due to __init__ failure)."""
        pass

    async def transcribe(self, file_path: str) -> str:
        """Simulate transcription (never called due to __init__ failure).

        Args:
            file_path: Path to the file to transcribe

        Returns:
            Unreachable error message
        """
        return "Should never get here"

    async def close(self):
        """Close the service (never called due to __init__ failure)."""
        pass


def register_and_get_provider(provider_name, service_class):
    """Register a provider and get its instance in one step.

    Args:
        provider_name: Name of the provider to register
        service_class: Class to register for the provider

    Returns:
        Instance of the registered service
    """
    register_transcription_service(provider_name)(service_class)
    return get_transcription_service(provider_name)


def test_register_transcription_service():
    """Verify the 'fake' service is registered."""
    register_transcription_service(FAKE_PROVIDER)(FAKE_SERVICE_PATH)
    providers = get_available_providers()
    assert FAKE_PROVIDER in providers, ERROR_MESSAGES["expected_fake"]


def test_re_register_service_with_warning(caplog):
    """Re-register 'fake' with another class, ensuring a warning is logged."""
    register_transcription_service(FAKE_PROVIDER)(ANOTHER_FAKE_SERVICE_PATH)
    assert ERROR_MESSAGES["override_warning"] in caplog.text

    with patch("thinkhub.transcription.validate_dependencies", return_value=None):
        instance = get_transcription_service(FAKE_PROVIDER)
    assert isinstance(instance, AnotherFakeService)


def test_get_transcription_service_with_params():
    """Test passing custom parameters to the service."""
    register_transcription_service(FAKE_PROVIDER)(FAKE_SERVICE_PATH)
    with patch("thinkhub.transcription.validate_dependencies", return_value=None):
        service = get_transcription_service(FAKE_PROVIDER, test_value=TEST_VALUE)
    assert service.test_value == TEST_VALUE, ERROR_MESSAGES["expected_test_value"]


def test_unregistered_provider():
    """Test requesting an unregistered provider raises ProviderNotFoundError."""
    with pytest.raises(ProviderNotFoundError) as excinfo:
        get_transcription_service(NON_EXISTENT_PROVIDER)
    assert ERROR_MESSAGES["unsupported_provider"] in str(excinfo.value)


def test_transcription_service_init_failure():
    """Test that an exception in the constructor raises TranscriptionServiceError."""
    register_transcription_service(FAILING_PROVIDER)(FAILING_SERVICE_PATH)

    with patch("thinkhub.transcription.validate_dependencies", return_value=None):
        with pytest.raises(TranscriptionServiceError) as excinfo:
            get_transcription_service(FAILING_PROVIDER)
    assert ERROR_MESSAGES["constructor_failed"] in str(excinfo.value)


def test_validate_dependencies_failure(monkeypatch):
    """Test that missing dependencies raise an ImportError."""
    register_transcription_service(FAKE_PROVIDER)(FAKE_SERVICE_PATH)
    monkeypatch.setitem(_REQUIRED_DEPENDENCIES, FAKE_PROVIDER, [NON_EXISTENT_PACKAGE])

    with pytest.raises(ImportError) as excinfo:
        get_transcription_service(FAKE_PROVIDER)
    assert ERROR_MESSAGES["missing_dependencies"] in str(excinfo.value)


def test_validate_dependencies_success(monkeypatch):
    """Test that no exception is raised when dependencies are valid."""
    monkeypatch.setattr(
        "thinkhub.utils.validate_dependencies", lambda provider, deps: None
    )
    service = get_transcription_service(FAKE_PROVIDER, test_value=VALID_TEST_VALUE)
    assert service.test_value == VALID_TEST_VALUE
