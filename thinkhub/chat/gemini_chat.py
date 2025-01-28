"""
This module defines the GeminiChatService class, which implements the ChatServiceInterface for interacting with the Google Gemini API.

The GeminiChatService supports streaming chat responses for both text and
image inputs, leveraging the 'start_chat' and 'send_message' methods
of the Google Generative AI library.
"""

import logging
import os
from collections.abc import AsyncGenerator
from typing import Optional, Union

import google.generativeai as genai
from PIL import Image

from thinkhub.chat.base import ChatServiceInterface
from thinkhub.chat.exceptions import InvalidInputDataError, TokenLimitExceededError
from thinkhub.transcription.exceptions import MissingAPIKeyError


class GeminiChatService(ChatServiceInterface):
    """
    A ChatServiceInterface implementation.

    Streams responses from the Google Gemini API using 'start_chat' + 'send_message(stream=True)'.
    Can handle plain text or images as input.
    """

    def __init__(
        self,
        model: str = "gemini-1.5-flash",
        api_key: Optional[str] = None,
        logging_level: int = logging.INFO,
    ):
        """
        Initialize the GeminiChatService with the Gemini API client.

        Args:
            model (str): Model name to use for the chat session.
            api_key (Optional[str]): Explicit API key; falls back to env var if not provided.
            logging_level (int): Logging configuration level.
        """
        # Flexible API key retrieval
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise MissingAPIKeyError("No Gemini API key found.")

        # Configure logging
        logging.basicConfig(level=logging_level)
        self.logger = logging.getLogger(__name__)

        # Configure the Generative AI library
        genai.configure(api_key=self.api_key)

        # Store model settings
        self.model_name = model

        # We'll store a ChatSession object once created
        self.chat_session = None

    def _ensure_chat_session(self, system_prompt: str):
        """
        Create or reuse the ChatSession if it doesn't exist yet.

        Optionally set a system prompt as the first message.
        """
        if self.chat_session is None:
            model = genai.GenerativeModel(model_name=self.model_name)
            history = []

            if system_prompt:
                history.append({"role": "model", "parts": system_prompt})

            self.chat_session = model.start_chat(history=history)
            self.logger.debug("Created a new chat session with system prompt.")

    def _validate_image_input(self, input_data: list[dict[str, str]]) -> bool:
        """Validate that the input is a list of dicts containing 'image_path'."""
        return isinstance(input_data, list) and all(
            isinstance(item, dict) and "image_path" in item for item in input_data
        )

    def _prepare_image_input_list(
        self, image_data: list[dict[str, str]], prompt: str
    ) -> list:
        """For the chat approach, we'll pass a list: [prompt, PIL.Image1, PIL.Image2, ...]."""
        parts = [prompt]  # The first element is our prompt
        for item in image_data:
            image_path = item["image_path"]
            # Open the image with PIL
            try:
                pil_image = Image.open(image_path)
            except OSError as e:
                raise InvalidInputDataError(f"Failed to open image: {image_path}\n{e}")

            parts.append(pil_image)

        return parts

    async def stream_chat_response(
        self,
        input_data: Union[str, list[dict[str, str]]],
        system_prompt: str = "You are a helpful assistant.",
    ) -> AsyncGenerator[str, None]:
        """
        Stream response chunks from the Google Generative AI service.

        Args:
            input_data (Union[str, list[dict[str, str]]]): The user input,
                either text or a list of dicts with 'image_path'.
            system_prompt (str): A system-level prompt to guide the assistant's behavior.

        Yields:
            AsyncGenerator[str, None]: Partial response tokens from the chat service.
        """
        if not input_data:
            return  # No input, nothing to yield

        # Ensure we have a ChatSession (with optional system prompt)
        self._ensure_chat_session(system_prompt)

        # Prepare the user input for send_message()
        if isinstance(input_data, str):
            # Plain text scenario, just pass the string
            user_prompt = input_data
        elif self._validate_image_input(input_data):
            # Image scenario:
            # We'll interpret the system_prompt as our question/command
            # Then pass the images themselves in a list.
            user_prompt = self._prepare_image_input_list(input_data, system_prompt)
        else:
            raise InvalidInputDataError(
                "Invalid input format: must be a string or a list of dicts with 'image_path'."
            )

        try:
            response_iter = self.chat_session.send_message(user_prompt, stream=True)

            # Convert the synchronous iterator to an async generator
            for chunk in response_iter:
                if chunk.text:
                    yield chunk.text

        except TokenLimitExceededError as e:
            self.logger.error(f"Token limit exceeded: {e}")
            yield f"[Error: {e}]"
        except Exception as e:
            self.logger.error(f"Chat response generation failed: {e}")
            yield f"[Error: {e}]"
