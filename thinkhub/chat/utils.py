"""Central utilities module for shared functionality across ThinkHub components."""

import base64
import logging
import os
from typing import Optional

from tenacity import retry, stop_after_attempt, wait_exponential

from thinkhub.chat.exceptions import MissingAPIKeyError


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def api_retry():
    """Return a decorator for API call retry logic."""
    return retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )


def encode_image(image_path: str) -> str:
    """Encode image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def validate_image_input(input_data: list) -> bool:
    """Validate multi-modal input structure."""
    return isinstance(input_data, list) and all(
        isinstance(item, dict) and "image_path" in item for item in input_data
    )


def get_api_key(api_key_arg: Optional[str], env_var_name: str) -> str:
    """Retrieve API key from argument or environment variable."""
    api_key = api_key_arg or os.getenv(env_var_name)
    if not api_key:
        raise MissingAPIKeyError(
            f"API key not found in environment variable {env_var_name}"
        )
    return api_key


def setup_logging(logging_level: int = logging.INFO) -> logging.Logger:
    """Configure logging and return logger."""
    logging.basicConfig(level=logging_level)
    return logging.getLogger(__name__)
