# ThinkHub
[![PyPI Downloads](https://static.pepy.tech/badge/thinkhub)](https://pepy.tech/projects/thinkhub)

![ThinkHub Logo](assets/logo.png)

ThinkHub is a Python-based framework that provides a unified interface for interacting with multiple AI services. Designed for extensibility, users can integrate new services by creating and registering their own plugins or classes. The project simplifies configurations, supports multiple providers, and prioritizes user-friendly customization.

## Key Features

- **Lazy Loading**: Only loads the dependencies required for the selected service, reducing memory and installation overhead.
- **Multi-Service Integration**: Interact seamlessly with multiple AI services (e.g., chat, transcription, image processing).
- **Plugin System**: Register and use custom classes to extend functionality.
- **Dynamic Configuration**: Load and manage configurations with environment variable overrides.
- **Error Handling**: Robust exception system for identifying and managing provider-related issues.
- **Poetry and pip Support**: Flexible dependency and environment management.
- **Python 3.11+**: Leverages the latest features of Python for performance and simplicity.

---

## Supported Services

### **Audio Transcriptions**
- **OpenAI**: Using the `whisper-1` model.
- **Google Speech-to-Text**

### **Chat**
- **OpenAI**: Supporting models like `gpt-4` and `gpt-3.5`.
- **Anthropic**: Supporting `Claude.ai`.
- **Gemini**: Supporting `Google Gemini`.

### **Image Processing**
- **OpenAI**: Analyze and process images with AI models.
- **Anthropic**: Analyze and process images with AI models.
- **Gemini**: Analyze and process images with AI models.

---

## Installation

ThinkHub uses a lazy-loading strategy to optimize memory usage and avoid installing unused dependencies. You can install ThinkHub using either **Poetry** or **pip**, as shown below:

### 1. **Install the Base Library**
   - **Poetry**:
     ```bash
     poetry add thinkhub
     ```
   - **pip**:
     ```bash
     pip install thinkhub
     ```

### 2. **Install with Specific Extras**
   Install only the required dependencies based on the service(s) you plan to use:

   - **OpenAI Chat**:
     - **Poetry**:
       ```bash
       poetry add thinkhub --extras openai
       ```
     - **pip**:
       ```bash
       pip install thinkhub[openai]
       ```

   - **Google Transcription**:
     - **Poetry**:
       ```bash
       poetry add thinkhub --extras google
       ```
     - **pip**:
       ```bash
       pip install thinkhub[google]
       ```

   - **Anthropic Chat**:
     - **Poetry**:
       ```bash
       poetry add thinkhub --extras anthropic
       ```
     - **pip**:
       ```bash
       pip install thinkhub[anthropic]
       ```

  - **Gemini Chat**:
     - **Poetry**:
       ```bash
       poetry add thinkhub --extras google-generativeai
       ```
     - **pip**:
       ```bash
       pip install thinkhub[google-generativeai]
       ```

   - **Multiple Services** (e.g., OpenAI and Anthropic):
     - **Poetry**:
       ```bash
       poetry add thinkhub --extras openai --extras anthropic
       ```
     - **pip**:
       ```bash
       pip install thinkhub[openai,anthropic]
       ```

### 3. **Install All Services**
   If you want to install all available services:
   - **Poetry**:
     ```bash
     poetry add thinkhub --extras all
     ```
   - **pip**:
     ```bash
     pip install thinkhub[all]
     ```

### 4. **Activate the Virtual Environment**
   - **Poetry**:
     ```bash
     poetry shell
     ```

---

## Usage

### **Lazy Loading**

ThinkHub uses lazy loading to dynamically import the dependencies required for a specific provider. This means that:

1. **Dependencies are only loaded when needed.**
2. **Missing dependencies are flagged with clear error messages.**

Example:
If you attempt to use OpenAI services without the `openai` extra installed, ThinkHub will raise an error like this:
```plaintext
ImportError: Missing dependencies for provider 'openai': tiktoken. 
Install them using 'poetry install --extras openai' or 'pip install thinkhub[openai]'.
```

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

### **Image Processing with OpenAI**
ThinkHub supports image processing with OpenAI. Here’s an example of how to process multiple images asynchronously:

```python
import asyncio
from thinkhub.chat import get_chat_service

async def process_image_with_openai(image_payloads):
    chat_service = get_chat_service("openai", model="gpt-4")
    async for response in chat_service.stream_chat_response(
        input_data=image_payloads, 
        system_prompt="Analyze these images."
    ):
        print(response)

# Prepare image payloads
image_payloads = [{"image_path": "path/to/image1.jpg"}, {"image_path": "path/to/image2.jpg"}]

# Process images with OpenAI
asyncio.run(process_image_with_openai(image_payloads))
```

---

## Error Handling

ThinkHub includes robust error handling to simplify debugging and configuration:

- **Dependency Validation**:
  ThinkHub will check for required dependencies dynamically and provide clear installation instructions.

- **Custom Exceptions**:
  - `ProviderNotFoundError`: Raised when a requested provider is not found.
  - `ImportError`: Raised when dependencies for a provider are missing.

Example:
```python
from thinkhub.exceptions import ProviderNotFoundError

try:
    service = get_chat_service("unsupported_provider")
except ProviderNotFoundError as e:
    print(e)
```

---

## Development

1. **Run Tests:**
   ```bash
   poetry run pytest
   ```

2. **Code Linting:**
   ```bash
   poetry run ruff check .
   ```

3. **Build the Project:**
   ```bash
   poetry build
   ```

---

## Supported Providers

| Provider                                                                 | Completion | Streaming | Async Completion | Async Streaming | Async Embedding | Async Image Generation |
|--------------------------------------------------------------------------|------------|------------|------------------|-----------------|-----------------|---------------------|
| [OpenAI](https://platform.openai.com/docs/overview)                      | ❌         | ❌         | ❌               | ✅               | ❌               | ❌                  |
| [Google Gemini](https://ai.google.dev/)                                  | ❌         | ❌         | ❌               | ✅               | ❌               | ❌                  |
| [Anthropic - Claude.ai](https://www.anthropic.com/api)                   | ❌         | ❌         | ❌               | ✅               | ❌               | ❌                  |

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgments

Special thanks to the open-source community for providing the tools and libraries that made this project possible.
