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

    Streams responses from the Google Gemini API using 'start_chat' + 'send_message_async(stream=True)'.
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
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise MissingAPIKeyError("No Gemini API key found.")

        logging.basicConfig(level=logging_level)
        self.logger = logging.getLogger(__name__)

        genai.configure(api_key=self.api_key)
        self.model_name = model
        self.chat_session = None

    async def _ensure_chat_session(self, system_prompt: str):
        """
        Create or reuse the ChatSession if it doesn't exist yet.

        Optionally set a system prompt as the first message.
        Ensure the token count does not exceed the model's limit.
        """
        if self.chat_session is None:
            model = genai.GenerativeModel(model_name=self.model_name)
            history = []

            if system_prompt:
                history.append({"role": "model", "parts": system_prompt})

            self.chat_session = model.start_chat(history=history)
            self.logger.debug("Created a new chat session with system prompt.")
        else:
            # Check token count and remove old messages if necessary
            token_count = await self._count_tokens(self.chat_session.history)
            model_info = genai.get_model(f"models/{self.model_name}")
            max_tokens = model_info.input_token_limit

            while token_count > max_tokens and len(self.chat_session.history) > 1:
                # Remove the oldest message (excluding the system prompt)
                self.chat_session.history.pop(1)
                token_count = await self._count_tokens(self.chat_session.history)
                self.logger.debug(
                    f"Removed old message to reduce token count to {token_count}"
                )

    def _validate_image_input(self, input_data: list[dict[str, str]]) -> bool:
        """Validate that the input is a list of dicts containing 'image_path'."""
        return isinstance(input_data, list) and all(
            isinstance(item, dict) and "image_path" in item for item in input_data
        )

    async def _count_tokens(self, contents) -> int:
        """
        Count the number of tokens in the given contents.

        Args:
            contents: The contents to count tokens for.

        Returns:
            int: The number of tokens.
        """
        model = genai.GenerativeModel(model_name=self.model_name)
        response = await model.count_tokens_async(contents)
        return response.total_tokens

    async def _prepare_image_input_list(
        self, image_data: list[dict[str, str]], prompt: str
    ) -> list:
        """For the chat approach, we'll pass a list: [prompt, PIL.Image1, PIL.Image2, ...]."""
        parts = [prompt]
        for item in image_data:
            image_path = item["image_path"]
            try:
                pil_image = Image.open(image_path)
                parts.append(pil_image)
            except OSError as e:
                raise InvalidInputDataError(f"Failed to open image: {image_path}\n{e}")
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
            return

        await self._ensure_chat_session(system_prompt)

        if isinstance(input_data, str):
            user_prompt = input_data
        elif self._validate_image_input(input_data):
            user_prompt = await self._prepare_image_input_list(
                input_data, system_prompt
            )
        else:
            raise InvalidInputDataError(
                "Invalid input format: must be a string or a list of dicts with 'image_path'."
            )

        # Check token count before sending the message
        token_count = await self._count_tokens(
            [*self.chat_session.history, {"role": "user", "parts": user_prompt}]
        )
        model_info = genai.get_model(f"models/{self.model_name}")
        max_tokens = model_info.input_token_limit

        if token_count > max_tokens:
            raise TokenLimitExceededError(
                f"Token limit exceeded: {token_count} > {max_tokens}"
            )

        try:
            response_iter = await self.chat_session.send_message_async(
                user_prompt, stream=True
            )
            async for chunk in response_iter:
                if chunk.text:
                    yield chunk.text
        except TokenLimitExceededError as e:
            self.logger.error(f"Token limit exceeded: {e}")
            yield f"[Error: {e}]"
        except Exception as e:
            self.logger.error(f"Chat response generation failed: {e}")
            yield f"[Error: {e}]"
