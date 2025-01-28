"""
This package provides utilities for managing and registering chat services.

It includes:
- A decorator for registering chat services dynamically.
- Methods for retrieving registered services.
- Support for multiple chat service providers.
"""

import logging

from thinkhub.exceptions import ProviderNotFoundError
from thinkhub.utils import validate_dependencies

from .base import ChatServiceInterface
from .exceptions import ChatServiceError

logger = logging.getLogger(__name__)

_CHAT_SERVICES: dict[str, str] = {
    "openai": "thinkhub.chat.openai_chat.OpenAIChatService",
    "anthropic": "thinkhub.chat.anthropic_chat.AnthropicChatService",
    "google-generativeai": "thinkhub.chat.gemini_chat.GeminiChatService",
}

_REQUIRED_DEPENDENCIES: dict[str, list[str]] = {
    "openai": ["openai", "tiktoken"],
    "anthropic": ["anthropic"],
    "google-generativeai": ["google.generativeai", "PIL"],
}


def get_chat_service(provider: str, **kwargs) -> ChatServiceInterface:
    """
    Retrieve a chat service instance dynamically based on the provider name.

    This function lazily loads and initializes the requested chat service to optimize memory usage
    and reduce unnecessary imports at the module level. The chat services are registered with their
    full module paths in the `_CHAT_SERVICES` dictionary and loaded only when needed.

    Args:
        provider (str):
            The name of the chat service provider to retrieve.
            Example values include "openai" and "anthropic".
        **kwargs:
            Additional keyword arguments to pass to the service's constructor when instantiated.

    Returns:
        ChatServiceInterface:
            An instance of the chat service corresponding to the requested provider.

    Raises:
        ProviderNotFoundError:
            Raised if the requested provider is not registered in the `_CHAT_SERVICES` dictionary.
        ChatServiceError:
            Raised if there is an issue importing the service class or initializing the provider.

    Example:
        >>> service = get_chat_service("openai", model="gpt-4")
    """
    provider_lower = provider.lower()
    service_class_path = _CHAT_SERVICES.get(provider_lower)
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
        raise ChatServiceError(f"Failed to initialize provider {provider}: {e}") from e


def get_available_chat_providers() -> list[str]:
    """Get a list of available chat providers."""
    return list(_CHAT_SERVICES.keys())
