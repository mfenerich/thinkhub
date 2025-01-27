"""
This package provides transcription services for handling audio-to-text operations.

It supports multiple providers and includes functionality for registering, retrieving,
and managing transcription services.
"""

import logging

from thinkhub.exceptions import ProviderNotFoundError
from thinkhub.utils import validate_dependencies

from .base import TranscriptionServiceInterface
from .exceptions import TranscriptionServiceError

logger = logging.getLogger(__name__)

_TRANSCRIPTION_SERVICES: dict[str, str] = {
    "google": "thinkhub.transcription.google_transcription.GoogleTranscriptionService",
    "openai": "thinkhub.transcription.openai_transcription.OpenAITranscriptionService",
}

_REQUIRED_DEPENDENCIES: dict[str, list[str]] = {
    "google": ["google.cloud.speech"],
    "openai": ["openai", "tiktoken"],
}


def register_transcription_service(name: str):
    """Decorate to register a transcription service."""

    def decorator(service_class: type[TranscriptionServiceInterface]):
        name_lower = name.lower()
        if name_lower in _TRANSCRIPTION_SERVICES:
            logger.warning(
                "Overriding transcription service: %s. "
                "Previous service will be replaced.",
                name,
            )
        _TRANSCRIPTION_SERVICES[name_lower] = service_class
        logger.info(f"Registered transcription service: {name}")
        return service_class

    return decorator


def get_transcription_service(provider: str, **kwargs) -> TranscriptionServiceInterface:
    """
    Retrieve a transcription service instance dynamically based on the provider name.

    This function lazily loads and initializes the requested transcription service to optimize memory usage
    and reduce unnecessary imports at the module level. The transcription services are registered with their
    full module paths in the `_TRANSCRIPTION_SERVICES` dictionary and loaded only when needed.

    Args:
        provider (str):
            The name of the transcription service provider to retrieve.
            Example values include "google" and "openai".
        **kwargs:
            Additional keyword arguments to pass to the service's constructor when instantiated.

    Returns:
        TranscriptionServiceInterface:
            An instance of the transcription service corresponding to the requested provider.

    Raises:
        ProviderNotFoundError:
            Raised if the requested provider is not registered in the `_TRANSCRIPTION_SERVICES` dictionary.
        TranscriptionServiceError:
            Raised if there is an issue importing the service class or initializing the provider.

    Example:
        >>> service = get_transcription_service("google", language="en-US")
        >>> transcription = service.transcribe_audio("path/to/audio.wav")
    """
    provider_lower = provider.lower()
    service_class_path = _TRANSCRIPTION_SERVICES.get(provider_lower)
    if not service_class_path:
        raise ProviderNotFoundError(f"Unsupported provider: {provider}")

    # Validate required dependencies
    validate_dependencies(provider_lower, _REQUIRED_DEPENDENCIES)

    try:
        # Dynamically import the service class
        module_name, class_name = service_class_path.rsplit(".", 1)
        module = __import__(module_name, fromlist=[class_name])
        service_class = getattr(module, class_name)
        return service_class(**kwargs)
    except Exception as e:
        raise TranscriptionServiceError(
            f"Failed to initialize provider {provider}: {e}"
        ) from e


def get_available_providers() -> list[str]:
    """Return a list of available transcription providers."""
    return list(_TRANSCRIPTION_SERVICES.keys())
