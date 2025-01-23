"""This module defines the `ChatServiceInterface` that provides a blueprint for chat services."""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Union


class ChatServiceInterface(ABC):
    """
    Abstract base class for chat service interfaces.

    Provides a method for streaming chat responses based on input data and system prompts.
    """

    @abstractmethod
    async def stream_chat_response(
        self,
        input_data: Union[str, list[dict[str, str]]],
        system_prompt: str = "You are a helpful assistant.",
    ) -> AsyncGenerator[str, None]:
        """
        Stream responses from a chat service.

        Args:
            input_data (Union[str, list[dict[str, str]]]): The user input,
            either as plain text or a list of dictionaries containing image paths.
            system_prompt (str): A system-level
            prompt to guide the assistant's behavior.

        Yields:
            AsyncGenerator[str, None]: Partial response tokens from the chat service.
        """
        pass
