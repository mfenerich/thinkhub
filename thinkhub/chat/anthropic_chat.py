"""
Provides an implementation of ChatServiceInterface for Anthropic's Claude API.

This module defines the AnthropicChatService class, which offers advanced
chat interaction capabilities with Anthropic's language models, supporting
features like multi-modal input, token management, and robust error handling.
"""

import logging
from collections.abc import AsyncGenerator
from typing import Optional, Union

from anthropic import APIConnectionError, AsyncAnthropic, RateLimitError

from thinkhub.chat.base import ChatServiceInterface
from thinkhub.chat.exceptions import InvalidInputDataError
from thinkhub.chat.utils import (
    api_retry,
    get_api_key,
    setup_logging,
    validate_image_input,
)
from thinkhub.chat.utils import encode_image as utils_encode_image


class AnthropicChatService(ChatServiceInterface):
    """Chat Service with advanced features and conversation history."""

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20240620",
        max_tokens: int = 4096,
        api_key: Optional[str] = None,
        logging_level: int = logging.INFO,
    ):
        """
        Initialize the enhanced AnthropicChatService.

        Args:
            model (str): Claude model to use.
            max_tokens (int): Maximum tokens for context management.
            api_key (Optional[str]): Explicit API key for flexible configuration.
            logging_level (int): Logging configuration.
        """
        self.api_key = get_api_key(api_key, "ANTHROPIC_API_KEY")

        # Logging setup
        self.logger = setup_logging(logging_level)

        # Client and model configuration
        self.anthropic = AsyncAnthropic(api_key=self.api_key)
        self.model = model
        self.MAX_TOKENS = max_tokens

        # Message context management
        self.conversation_history: list[dict[str, any]] = []
        self.system_prompt: Optional[str] = None

    def _count_tokens(self, content: any) -> int:
        """Token counting using Anthropic's message token estimation."""
        try:
            # Sanitize content
            if isinstance(content, list):
                content = " ".join(
                    str(item.get("text", ""))
                    for item in content
                    if isinstance(item, dict)
                )

            # Estimate tokens for the content
            return self.anthropic.messages.estimate_tokens(
                model=self.model, messages=[{"role": "user", "content": str(content)}]
            )
        except Exception as e:
            self.logger.warning(
                f"Token counting failed: {e}. Falling back to word-based estimation."
            )
            return len(str(content).split())

    def _manage_context_window(self):
        """
        Intelligent context window management.

        Removes messages strategically to maintain context while preserving recent interactions.
        """
        while (
            self._total_tokens() > self.MAX_TOKENS
            and len(self.conversation_history) > 2
        ):
            # Always keep the most recent system and user message
            self.conversation_history.pop(1)
            self.logger.info("Removed an older message to manage token limit")

    def _total_tokens(self) -> int:
        """Calculate total tokens across all messages."""
        return sum(
            self._count_tokens(msg.get("content", ""))
            for msg in self.conversation_history
        )

    def add_user_message(self, message: str):
        """
        Add a user message to the conversation history.

        Args:
            message (str): The user's input message.
        """
        self.conversation_history.append({"role": "user", "content": message})

    def add_assistant_message(self, message: str):
        """
        Add an assistant message to the conversation history.

        Args:
            message (str): The assistant's response message.
        """
        self.conversation_history.append({"role": "assistant", "content": message})

    @api_retry()
    async def _safe_api_call(self, **kwargs):
        """Safe API call with retry and logging."""
        try:
            return await self.anthropic.messages.create(**kwargs)
        except (RateLimitError, APIConnectionError) as e:
            self.logger.error(f"API call failed: {e}")
            raise

    def encode_image(self, image_path: str) -> str:
        """
        Encode an image into a base64 string.

        Args:
            image_path (str): Path to the image file.

        Returns:
            str: Base64-encoded string of the image.

        Raises:
            InvalidInputDataError: If the image cannot be encoded.
        """
        try:
            return utils_encode_image(image_path)
        except OSError as e:
            self.logger.error(f"Image encoding failed: {e}")
            raise InvalidInputDataError(f"Failed to encode image: {e}")

    async def stream_chat_response(
        self,
        input_data: Union[str, list[dict[str, str]]],
        system_prompt: Optional[str] = "You are a helpful assistant.",
    ) -> AsyncGenerator[str, None]:
        """Advanced streaming chat response with multi-modal support and conversation history."""
        # Validate and prepare input
        if not input_data:
            return

        # Manage system prompt
        self.system_prompt = system_prompt

        # Prepare API payload
        api_payload = {
            "model": self.model,
            "max_tokens": self.MAX_TOKENS,
            "system": self.system_prompt,
            "messages": self.conversation_history.copy(),
            "stream": True,
        }

        # Process input data
        try:
            if isinstance(input_data, str):
                self.add_user_message(input_data)
                api_payload["messages"].append({"role": "user", "content": input_data})
            elif self._validate_image_input(input_data):
                image_messages = self._prepare_image_messages(input_data)
                self.conversation_history.extend(image_messages)
                api_payload["messages"].extend(image_messages)
            else:
                raise InvalidInputDataError("Invalid input format")

            # Manage context window
            self._manage_context_window()

            # Stream response
            full_response_chunks = []
            stream = await self._safe_api_call(**api_payload)
            async for event in stream:
                if event.type == "content_block_delta" and event.delta.text:
                    chunk = event.delta.text
                    full_response_chunks.append(chunk)
                    yield chunk

            # Update context with full response
            full_response = "".join(full_response_chunks)
            self.add_assistant_message(full_response)

        except Exception as e:
            self.logger.error(f"Chat response generation failed: {e}")
            yield f"[Error: {e!s}]"

    def _validate_image_input(self, input_data: list[dict[str, str]]) -> bool:
        return validate_image_input(input_data)

    def _prepare_image_messages(
        self, input_data: list[dict[str, str]]
    ) -> list[dict[str, any]]:
        """Prepare multi-modal messages with image processing."""
        image_contents = []
        for item in input_data:
            image_path = item["image_path"]
            base64_image = self.encode_image(image_path)
            image_contents.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64_image,
                    },
                }
            )

        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze the following images."},
                    *image_contents,
                ],
            }
        ]
