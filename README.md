# ThinkHub

ThinkHub is a Python-based framework that provides a unified interface for interacting with multiple AI services. Designed for extensibility, users can integrate new services by creating and registering their own plugins or classes. The project simplifies configurations, supports multiple providers, and prioritizes user-friendly customization.

## Key Features

- **Multi-Service Integration**: Interact seamlessly with multiple AI services (e.g., chat, transcription).
- **Plugin System**: Register and use custom classes to extend functionality.
- **Dynamic Configuration**: Load and manage configurations with environment variable overrides.
- **Error Handling**: Robust exception system for identifying and managing provider-related issues.
- **Poetry Support**: Modern dependency and environment management with Poetry.
- **Python 3.11+**: Leverages the latest features of Python for performance and simplicity.

---

## Supported Services

### **Audio Transcriptions**
- **OpenAI**: Using the `whisper-1` model.
- **Google Speech-to-Text**

### **Chat**
- **OpenAI**: Supporting models like `gpt-4` and `gpt-3.5`.
- **Anthropic**: Supporting `Claude.ai`.

### **TODO**
#### Transcription
- **AWS**: Add support for [Amazon Transcribe](https://aws.amazon.com/transcribe/).
- **Local service**: Integrate with [OpenAI Whisper](https://github.com/openai/whisper) for local transcription.
- **TBD**: Other services under consideration.

#### Chat
- **Gemini**: Add support for Google's Gemini models.
- **TBD**: Other services under consideration.

### **Future Plans**
- More features to be planned and implemented.

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/mfenerich/thinkhub.git
   cd thinkhub
   ```

2. **Install dependencies with Poetry:**
   Ensure Poetry is installed on your system. Then run:
   ```bash
   poetry install
   ```

3. **Activate the virtual environment:**
   ```bash
   poetry shell
   ```

---

## Usage

> **Note**: This library currently supports asynchronous calls only because it was originally designed for an internal project. Synchronous calls will be added in the near future.

### **Chat Services**
To use a chat service like OpenAI:
```python
from thinkhub.chat import get_chat_service

chat_service = get_chat_service("openai", model="gpt-4o")
async for response in chat_service.stream_chat_response("Hello, ThinkHub!"):
    print(response)
```

### **Transcription Services**
To use a transcription service like Google:
```python
from thinkhub.transcription import get_transcription_service

transcription_service = get_transcription_service("google")
result = await transcription_service.transcribe("path/to/audio.flac")
print(result)
```

### **Registering Custom Plugins**

ThinkHub allows users to create and register their own services by extending the base classes and utilizing the factory functions (`get_chat_service` and `get_transcription_service`).

#### Example: Registering a Custom Chat Service

To register a custom chat service, extend the base class `ChatServiceInterface` and implement its methods:

```python
from thinkhub.chat.base import ChatServiceInterface
from thinkhub.chat import register_chat_service

class CustomChatService(ChatServiceInterface):
    """A custom implementation of a chat service."""
    
    def __init__(self, **kwargs):
        self.custom_param = kwargs.get("custom_param", "default_value")
    
    async def stream_chat_response(self, input_data, system_prompt=""):
        yield f"Custom response to: {input_data}"

# Register the service
register_chat_service("custom", CustomChatService)
```

#### Usage
Once registered, the custom service can be retrieved via `get_chat_service`:

```python
from thinkhub.chat import get_chat_service

chat_service = get_chat_service("custom", custom_param="example")
async for response in chat_service.stream_chat_response("Hello!"):
    print(response)
```

#### Example: Registering a Custom Transcription Service

Similarly, transcription services can be registered by extending the `TranscriptionServiceInterface`:

```python
from thinkhub.transcription.base import TranscriptionServiceInterface
from thinkhub.transcription import register_transcription_service

class CustomTranscriptionService(TranscriptionServiceInterface):
    """A custom implementation of a transcription service."""
    
    async def transcribe(self, file_path):
        return f"Transcription for {file_path}"

# Register the service
register_transcription_service("custom", CustomTranscriptionService)
```

#### Usage
The custom transcription service can then be retrieved via `get_transcription_service`:

```python
from thinkhub.transcription import get_transcription_service

transcription_service = get_transcription_service("custom")
result = await transcription_service.transcribe("path/to/file")
print(result)
```

---

## Error Handling

Custom exceptions are provided to make debugging easier:

- **BaseServiceError**: Base class for all service-related errors.
- **ProviderNotFoundError**: Raised when a requested provider is not found.

Example:
```python
from thinkhub.exceptions import ProviderNotFoundError

try:
    raise ProviderNotFoundError("Provider not found!")
except ProviderNotFoundError as e:
    print(e)
```

---

## Development

1. **Run Tests:**
   Add your tests in the appropriate directories and run:
   ```bash
   poetry run pytest
   ```

2. **Code Linting:**
   Ensure code quality with:
   ```bash
   poetry run flake8
   ```

3. **Build the Project:**
   ```bash
   poetry build
   ```

---

## Contributing

Contributions are welcome! Please fork the repository and create a pull request for any changes.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgments

Special thanks to the open-source community for providing the tools and libraries that made this project possible.

