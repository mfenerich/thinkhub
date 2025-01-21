from .base import TranscriptionServiceInterface
from .google_transcription import GoogleTranscriptionService

def get_transcriptions_service(
    provider: str = "google", 
    **kwargs
) -> TranscriptionServiceInterface:
    """
    Returns the appropriate chat service based on the 'provider' argument.
    kwargs are passed through to the service constructor.
    """
    if provider.lower() == "google":
        return GoogleTranscriptionService(**kwargs)
    # ... other providers ...
    else:
        raise ValueError(f"Unsupported provider: {provider}")
