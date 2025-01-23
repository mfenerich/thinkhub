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

### **Configuration**
ThinkHub uses YAML configuration files, merged and overridden by environment variables.

Example usage of `ConfigLoader`:
```python
from thinkhub.config_loader import ConfigLoader

config = ConfigLoader().load()
print(config)
```

### **Chat Plugins**
To use the OpenAI Chat API:
```python
from thinkhub.chat.openai_chat import OpenAIChat

chat = OpenAIChat(api_key="your_openai_api_key")
response = chat.send_message("Hello, ThinkHub!")
print(response)
```

### **Transcription Plugins**
To use the Google Transcription API:
```python
from thinkhub.transcription.google_transcription import GoogleTranscription

transcriber = GoogleTranscription(api_key="your_google_api_key")
result = transcriber.transcribe("path/to/audio.flac")
print(result)
```

### **Registering Custom Plugins**
Users can create their own services by extending the base classes:

Example for a custom chat plugin:
```python
from thinkhub.chat.base import BaseChat

class CustomChat(BaseChat):
    def send_message(self, message: str) -> str:
        return f"Custom response to: {message}"

# Register and use
chat = CustomChat()
print(chat.send_message("Hello!"))
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