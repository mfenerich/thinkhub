"""
Define the `OpenAIChatService` class, which interacts with the OpenAI API.

to generate chat responses asynchronously. It supports both text and image inputs
and streams responses back to the user in chunks.
"""

import logging
from collections.abc import AsyncGenerator
from typing import Optional, Union

import tiktoken
from openai import APIConnectionError, AsyncOpenAI, RateLimitError

from thinkhub.chat.base import ChatServiceInterface
from thinkhub.chat.exceptions import (
    InvalidInputDataError,
    TokenLimitExceededError,
)
from thinkhub.utils import api_retry, get_api_key, setup_logging, validate_image_input
from thinkhub.utils import encode_image as utils_encode_image


class OpenAIChatService(ChatServiceInterface):
    """
    An implementation of the `ChatServiceInterface` for interacting with the OpenAI API.

    This service provides functionality to:
    - Stream chat responses from OpenAI.
    - Handle both text and image inputs.
    - Manage token limits for the model's context.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        max_tokens: int = 4096,
        api_key: Optional[str] = None,
        logging_level: int = logging.INFO,
    ):
        """
        Initialize the OpenAIChatService with a hypothetical AsyncOpenAI client.

        Args:
            model (str): Model name to use for chat.
            max_tokens (int): Maximum tokens for context management.
            api_key (Optional[str]): Explicit API key for flexible configuration.
            logging_level (int): Logging configuration.
        """
        self.api_key = get_api_key(api_key, "CHATGPT_API_KEY")
        self.logger = setup_logging(logging_level)

        # Client and model configuration
        self.openai = AsyncOpenAI(api_key=self.api_key)
        self.model = model
        self.MAX_TOKENS = max_tokens

        # Initialize the message context
        self.messages: list[dict[str, str]] = []
        self.system_prompt: Optional[str] = None

        # Token management
        self.model_encoding = tiktoken.encoding_for_model(model)

    def _count_tokens(self, content: any) -> int:
        """Token counting using OpenAI's token estimation."""
        try:
            # Sanitize content
            if isinstance(content, list):
                content = " ".join(
                    str(item.get("text", ""))
                    for item in content
                    if isinstance(item, dict)
                )

            # Estimate tokens for the content
            return len(self.model_encoding.encode(str(content)))
        except Exception as e:
            self.logger.warning(
                f"Token counting failed: {e}. Falling back to word-based estimation."
            )
            return len(str(content).split())

    def _manage_context_window(self):
        """
        Intelligent context window management.

        Removes messages strategically to maintain context.
        Raises TokenLimitExceededError if the context cannot be reduced further.
        """
        while self._total_tokens() > self.MAX_TOKENS and len(self.messages) > 1:
            # Prioritize removing older user messages first
            removed = self.messages.pop(1)
            self.logger.info(f"Removed message to manage token limit: {removed}")

        # If still over the limit after removing messages, raise an error
        if self._total_tokens() > self.MAX_TOKENS:
            raise TokenLimitExceededError(
                "Cannot reduce the message context further; token limit exceeded."
            )

    def _total_tokens(self) -> int:
        """Calculate total tokens across all messages."""
        return sum(self._count_tokens(msg.get("content", "")) for msg in self.messages)

    @api_retry()
    async def _safe_api_call(self, **kwargs):
        """Safe API call with retry and logging."""
        try:
            return await self.openai.chat.completions.create(**kwargs)
        except (RateLimitError, APIConnectionError) as e:
            self.logger.error(f"API call failed: {e}")
            raise

    def encode_image(self, image_path: str) -> str:
        """Image encoding using base64 encoding."""
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
            async with await self._safe_api_call(**api_payload) as stream:
                async for event in stream:
                    if event.choices and event.choices[0].delta.content:
                        chunk = event.choices[0].delta.content
                        full_response_chunks.append(chunk)
                        yield chunk

            # Update context
            full_response = "".join(full_response_chunks)
            self.messages.append({"role": "assistant", "content": full_response})

        except TokenLimitExceededError as e:
            self.logger.error(f"Token limit exceeded: {e}")
            yield f"[Error: {e!s}]"
        except Exception as e:
            self.logger.error(f"Chat response generation failed: {e}")
            yield f"[Error: {e!s}]"

    def _validate_image_input(self, input_data: list[dict[str, str]]) -> bool:
        """Validate multi-modal input structure."""
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
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
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
