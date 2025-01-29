"""
Provides an implementation of ChatServiceInterface for Anthropic's Claude API.

This module defines the AnthropicChatService class, which offers advanced
chat interaction capabilities with Anthropic's language models, supporting
features like multi-modal input, token management, and robust error handling.
"""

import base64
import logging
import os
from collections.abc import AsyncGenerator
from typing import Optional, Union

from anthropic import APIConnectionError, AsyncAnthropic, RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential

from thinkhub.chat.base import ChatServiceInterface
from thinkhub.chat.exceptions import InvalidInputDataError, MissingAPIKeyError


class AnthropicChatService(ChatServiceInterface):
    """Enhanced Anthropic Chat Service with advanced features."""

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
        # Flexible API key retrieval
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise MissingAPIKeyError("No Anthropic API key found.")

        # Logging setup
        logging.basicConfig(level=logging_level)
        self.logger = logging.getLogger(__name__)

        # Client and model configuration
        self.anthropic = AsyncAnthropic(api_key=self.api_key)
        self.model = model
        self.MAX_TOKENS = max_tokens

        # Message context management
        self.messages: list[dict[str, any]] = []
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

        Removes messages strategically to maintain context.
        """
        while self._total_tokens() > self.MAX_TOKENS and len(self.messages) > 1:
            # Prioritize removing older user messages first
            removed = self.messages.pop(1)
            self.logger.info(f"Removed message to manage token limit: {removed}")

    def _total_tokens(self) -> int:
        """Calculate total tokens across all messages."""
        return sum(self._count_tokens(msg.get("content", "")) for msg in self.messages)

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def _safe_api_call(self, **kwargs):
        """Safe API call with retry and logging."""
        try:
            return await self.anthropic.messages.create(**kwargs)
        except (RateLimitError, APIConnectionError) as e:
            self.logger.error(f"API call failed: {e}")
            raise

    def encode_image(self, image_path: str) -> str:
        """Robust image encoding with enhanced error handling."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except OSError as e:
            self.logger.error(f"Image encoding failed: {e}")
            raise InvalidInputDataError(f"Failed to encode image: {e}")

    async def stream_chat_response(
        self,
        input_data: Union[str, list[dict[str, str]]],
        system_prompt: Optional[str] = "You are a helpful assistant.",
    ) -> AsyncGenerator[str, None]:
        """Advanced streaming chat response with multi-modal support."""
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
            "messages": [],
            "stream": True,
        }

        # Process input data
        try:
            if isinstance(input_data, str):
                api_payload["messages"].append({"role": "user", "content": input_data})
            elif self._validate_image_input(input_data):
                api_payload["messages"] = self._prepare_image_messages(input_data)
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

            # Update context
            full_response = "".join(full_response_chunks)
            self.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            self.logger.error(f"Chat response generation failed: {e}")
            yield f"[Error: {e!s}]"

    def _validate_image_input(self, input_data: list[dict[str, str]]) -> bool:
        """Validate multi-modal input structure."""
        return isinstance(input_data, list) and all(
            isinstance(item, dict) and "image_path" in item for item in input_data
        )

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
