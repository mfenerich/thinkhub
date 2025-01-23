import logging
from typing import Dict, Type

from thinkhub.exceptions import ProviderNotFoundError
from thinkhub.transcription.google_transcription import GoogleTranscriptionService

from .base import TranscriptionServiceInterface
from .exceptions import TranscriptionServiceError

logger = logging.getLogger(__name__)

_TRANSCRIPTION_SERVICES: Dict[str, Type[TranscriptionServiceInterface]] = {
    "google": GoogleTranscriptionService,  # Pre-register Google service
}


def register_transcription_service(name: str):
    """Decorator to register a transcription service."""

    def decorator(service_class: Type[TranscriptionServiceInterface]):
        name_lower = name.lower()
        if name_lower in _TRANSCRIPTION_SERVICES:
            logger.warning(
                f"Overriding transcription service: {name}. Previous service will be replaced."
            )
        _TRANSCRIPTION_SERVICES[name_lower] = service_class
        logger.info(f"Registered transcription service: {name}")
        return service_class

    return decorator


def get_transcription_service(provider: str, **kwargs) -> TranscriptionServiceInterface:
    """Returns the appropriate transcription service.

    Args:
        provider: Name of the transcription service provider.
        **kwargs: Arguments passed to the service constructor.

    Raises:
        ProviderNotFoundError: If the provider is not registered.
    """
    provider_lower = provider.lower()
    service_class = _TRANSCRIPTION_SERVICES.get(provider_lower)
    if service_class is None:
        raise ProviderNotFoundError(f"Unsupported provider: {provider}")
    try:
        return service_class(**kwargs)
    except Exception as e:
        raise TranscriptionServiceError(
            f"Failed to initialize provider {provider}: {e}"
        ) from e


def get_available_providers() -> list[str]:
    """Returns a list of available transcription providers."""
    return list(_TRANSCRIPTION_SERVICES.keys())
