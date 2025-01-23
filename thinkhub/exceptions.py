"""
Ddefine custom exception classes for handling errors in the ThinkHub service.

It includes a base exception class and specific exceptions for different
error scenarios, such as when a requested provider is not found.
"""


class BaseServiceError(Exception):
    """Base exception for all service-related errors."""

    pass


class ProviderNotFoundError(BaseServiceError):
    """Raised when a requested provider is not found."""

    pass
