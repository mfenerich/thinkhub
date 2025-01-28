"""
This module defines the GeminiChatService class, which implements the ChatServiceInterface for interacting with the Google Gemini API using token-aware logic.

The GeminiChatService supports streaming chat responses for both text and
image inputs, leveraging 'start_chat' and 'send_message_async(stream=True)'.
It ensures token limits are not exceeded.
"""

import logging
import os
from collections.abc import AsyncGenerator
from typing import Optional, Union

import google.generativeai as genai
from google.generativeai.types.content_types import to_contents
from PIL import Image

from thinkhub.chat.base import ChatServiceInterface
from thinkhub.chat.exceptions import InvalidInputDataError, TokenLimitExceededError
from thinkhub.transcription.exceptions import MissingAPIKeyError


class GeminiChatService(ChatServiceInterface):
    """
    A ChatServiceInterface implementation that streams responses from the Google Gemini API using token-aware methods.

    Supports both text and image inputs.
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
        model_info = genai.get_model("models/" + self.model_name)
        self.input_token_limit = model_info.input_token_limit
        self.output_token_limit = model_info.output_token_limit

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

            # Treat the system prompt as the first "model" response.
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
        """Prepare a list for multi-modal input, combining text and images."""
        parts = [prompt]
        for item in image_data:
            image_path = item["image_path"]
            try:
                pil_image = Image.open(image_path)
            except OSError as e:
                raise InvalidInputDataError(f"Failed to open image: {image_path}\n{e}")
            parts.append(pil_image)
        return parts

    async def _count_tokens(self, content: Union[str, list]) -> int:
        """
        Count tokens for a given content (text or multi-modal).

        Args:
            content (Union[str, list]): The input content to check for token usage.

        Returns:
            int: Total token count for the input.
        """
        # Ensure the chat session and its model are initialized
        if not self.chat_session:
            raise ValueError("Chat session has not been initialized.")

        # Count tokens using the existing model
        token_count_response = await self.chat_session.model.count_tokens_async(content)
        return token_count_response.total_tokens

    async def stream_chat_response(
        self,
        input_data: Union[str, list[dict[str, str]]],
        system_prompt: str = "You are a helpful assistant.",
    ) -> AsyncGenerator[str, None]:
        """
        Stream response chunks from the Google Generative AI service, ensuring token limits.

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

        # Prepare the user input
        if isinstance(input_data, str):
            user_prompt = input_data
        elif self._validate_image_input(input_data):
            user_prompt = self._prepare_image_input_list(input_data, system_prompt)
        else:
            raise InvalidInputDataError(
                "Invalid input format: must be a string or a list of dicts with 'image_path'."
            )

        # Append the new user input to the history and count tokens
        new_history = self.chat_session.history + to_contents(user_prompt)
        total_tokens_response = await self.chat_session.model.count_tokens_async(
            new_history
        )
        total_tokens = total_tokens_response.total_tokens

        if total_tokens > self.input_token_limit:
            raise TokenLimitExceededError(
                f"Token limit exceeded: {total_tokens}/{self.input_token_limit}."
            )

        try:
            # Send the message with streaming
            response_async = await self.chat_session.send_message_async(
                user_prompt, stream=True
            )

            # If the library's streaming is implemented as an async iterator:
            async for chunk in response_async:
                if chunk.text:
                    yield chunk.text

        except TokenLimitExceededError as e:
            self.logger.error(f"Token limit exceeded: {e}")
            yield f"[Error: {e}]"
        except Exception as e:
            self.logger.error(f"Chat response generation failed: {e}")
            yield f"[Error: {e}]"
