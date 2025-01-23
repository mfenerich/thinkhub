"""
This package provides utilities for managing and registering chat services.

It includes:
- A decorator for registering chat services dynamically.
- Methods for retrieving registered services.
- Support for multiple chat service providers.
"""

import logging

from thinkhub.exceptions import ProviderNotFoundError

from .base import ChatServiceInterface
from .exceptions import ChatServiceError
from .openai_chat import OpenAIChatService

logger = logging.getLogger(__name__)

_CHAT_SERVICES: dict[str, type[ChatServiceInterface]] = {
    "openai": OpenAIChatService,  # Pre-register OpenAI service
}


def register_chat_service(name: str):
    """Register a chat service dynamically."""

    def decorator(service_class: type[ChatServiceInterface]):
        name_lower = name.lower()
        if name_lower in _CHAT_SERVICES:
            logger.warning(
                f"Overriding chat service: {name}. Previous service will be replaced."
            )
        _CHAT_SERVICES[name_lower] = service_class
        logger.info(f"Registered chat service: {name}")
        return service_class

    return decorator


def get_chat_service(provider: str, **kwargs) -> ChatServiceInterface:
    """Get the appropriate chat service.

    Args:
        provider: Name of the chat service provider.
        **kwargs: Arguments passed to the service constructor.

    Raises:
        ProviderNotFoundError: If the provider is not registered.
        ChatServiceError: If there is an issue initializing the provider.
    """
    provider_lower = provider.lower()
    service_class = _CHAT_SERVICES.get(provider_lower)
    if service_class is None:
        raise ProviderNotFoundError(f"Unsupported provider: {provider}")
    try:
        return service_class(**kwargs)
    except Exception as e:
        raise ChatServiceError(f"Failed to initialize provider {provider}: {e}") from e


def get_available_chat_providers() -> list[str]:
    """Get a list of available chat providers."""
    return list(_CHAT_SERVICES.keys())
