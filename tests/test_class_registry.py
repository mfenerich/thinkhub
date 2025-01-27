"""
Unit tests for testing the transcription service class registry.
"""

import pytest
from unittest.mock import patch

from thinkhub.exceptions import ProviderNotFoundError
from thinkhub.transcription import (
    TranscriptionServiceError,
    get_available_providers,
    get_transcription_service,
    register_transcription_service,
    _REQUIRED_DEPENDENCIES,
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
    return f"Fake transcription for {file_path} with test_value={test_value}"

class FakeTranscriptionService(TranscriptionServiceInterface):
    """A simple fake transcription service for testing registration and retrieval."""

    def __init__(self, test_value=None):
        self.test_value = test_value

    async def initialize_client(self):
        pass

    async def transcribe(self, file_path: str) -> str:
        return fake_transcription_text(file_path, self.test_value)

    async def close(self):
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
        raise ValueError(ERROR_MESSAGES["constructor_failed"])

    async def initialize_client(self):
        pass

    async def transcribe(self, file_path: str) -> str:
        return "Should never get here"

    async def close(self):
        pass

def register_and_get_provider(provider_name, service_class):
    """Helper function to register a provider and retrieve its instance."""
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
    monkeypatch.setattr("thinkhub.utils.validate_dependencies", lambda provider, deps: None)
    service = get_transcription_service(FAKE_PROVIDER, test_value=VALID_TEST_VALUE)
    assert service.test_value == VALID_TEST_VALUE