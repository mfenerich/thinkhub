"""
This module defines custom exceptions for the chat service.

The exceptions handle specific error cases like missing API keys,
exceeding token limits, and invalid input data, inheriting from a
common base exception `BaseServiceError`.
"""

from thinkhub.exceptions import BaseServiceError


class ChatServiceError(BaseServiceError):
    """Base exception for chat service related errors."""

    pass


class MissingAPIKeyError(ChatServiceError):
    """Raised when the API key is missing."""

    pass


class TokenLimitExceededError(ChatServiceError):
    """Raised when token limit is exceeded."""

    pass


class InvalidInputDataError(ChatServiceError):
    """Raised when input data to the chat service is invalid."""

    pass
