from .base import ChatServiceInterface  # Optionally re-export
from .openai_chat import OpenAIChatService  # If you want direct import usage

def get_chat_service(
    provider: str = "openai", 
    **kwargs
) -> ChatServiceInterface:
    """
    Returns the appropriate chat service based on the 'provider' argument.
    kwargs are passed through to the service constructor.
    """
    if provider.lower() == "openai":
        return OpenAIChatService(**kwargs)
    # ... other providers ...
    else:
        raise ValueError(f"Unsupported provider: {provider}")
