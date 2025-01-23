class ChatServiceError(Exception):
    """Base exception for transcription service related errors."""
    pass

class ProviderNotFoundError(ChatServiceError):
    """Raised when a requested provider is not found."""
    pass