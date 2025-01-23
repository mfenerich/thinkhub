class BaseServiceError(Exception):
    """Base exception for all service-related errors."""
    pass


class ProviderNotFoundError(BaseServiceError):
    """Raised when a requested provider is not found."""
    pass
