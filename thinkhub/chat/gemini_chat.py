"""
Provides an implementation of ChatServiceInterface for Google's Gemini API.

This module defines the GeminiChatService class, which offers advanced
chat interaction capabilities with Google's Gemini models, supporting
features like multi-modal input, token management, and robust error handling.
"""

import logging
from collections.abc import AsyncGenerator
from typing import Optional, Union

import google.generativeai as genai
from PIL import Image

from thinkhub.chat.base import ChatServiceInterface
from thinkhub.chat.exceptions import InvalidInputDataError, TokenLimitExceededError
from thinkhub.chat.utils import (
    api_retry,
    get_api_key,
    setup_logging,
    validate_image_input,
)


class GeminiChatService(ChatServiceInterface):
    """Enhanced Gemini Chat Service with advanced features."""

    def __init__(
        self,
        model: str = "gemini-1.5-flash",
        api_key: Optional[str] = None,
        logging_level: int = logging.INFO,
    ):
        """
        Initialize the enhanced GeminiChatService.

        Args:
            model (str): Gemini model to use.
            api_key (Optional[str]): Explicit API key for flexible configuration.
            logging_level (int): Logging configuration.
        """
        self.api_key = get_api_key(api_key, "GEMINI_API_KEY")
        self.logger = setup_logging(logging_level)

        # Client and model configuration
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
        return validate_image_input(input_data)

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
        """Prepare multi-modal messages with image processing."""
        parts = [prompt]
        for item in image_data:
            image_path = item["image_path"]
            try:
                pil_image = Image.open(image_path)
                parts.append(pil_image)
            except OSError as e:
                self.logger.error(f"Failed to open image: {image_path}\n{e}")
                raise InvalidInputDataError(f"Failed to open image: {image_path}\n{e}")
        return parts

    @api_retry()
    async def _safe_api_call(self, prompt: Union[str, list]):
        """Safe API call with retry and logging."""
        try:
            return await self.chat_session.send_message_async(prompt, stream=True)
        except Exception as e:
            self.logger.error(f"API call failed: {e}")
            raise

    async def stream_chat_response(
        self,
        input_data: Union[str, list[dict[str, str]]],
        system_prompt: str = "You are a helpful assistant.",
    ) -> AsyncGenerator[str, None]:
        """
        Advanced streaming chat response with multi-modal support.

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

        # Prepare user prompt
        user_prompt = (
            input_data
            if isinstance(input_data, str)
            else await self._prepare_image_input_list(input_data, system_prompt)
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
            async for chunk in await self._safe_api_call(user_prompt):
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            self.logger.error(f"Chat response generation failed: {e}")
            yield f"[Error: {e}]"
