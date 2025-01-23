class TranscriptionServiceError(Exception):
    """Base exception for transcription service related errors."""
    pass

class ProviderNotFoundError(TranscriptionServiceError):
    """Raised when a requested provider is not found."""
    pass