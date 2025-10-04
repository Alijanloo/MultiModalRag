"""Base Google GenAI service with common exception handling and token switching."""

import re
import time
from typing import Union, List, TypeVar
from google import genai
from google.genai.types import HttpOptions

from multimodal_rag.frameworks.logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar('T')


def parse_retry_delay_from_error(error: Exception) -> float:
    """Extract retry delay from error message."""
    error_str = str(error)
    # Look for patterns like "retry after 15 seconds" or "15s"
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:second|sec|s)", error_str, re.IGNORECASE)
    if match:
        return float(match.group(1))

    # Look for patterns like "retry_delay: 15"
    match = re.search(
        r"retry[_\s]*delay[:\s]*(\d+(?:\.\d+)?)", error_str, re.IGNORECASE
    )
    if match:
        return float(match.group(1))

    return 60.0  # Default to 60 seconds if no retry delay found


class GoogleGenAIBaseService:
    """Base class for Google GenAI services with common exception handling."""

    def __init__(
        self,
        api_keys: Union[str, List[str]],
        max_retries: int = 7,
    ):
        """
        Initialize the base Google GenAI service.

        Args:
            api_keys: Single API key string or list of API keys for token switching
            max_retries: Maximum number of retries for failed requests
        """
        if isinstance(api_keys, str):
            self._api_keys = [api_keys]
        else:
            self._api_keys = api_keys
        
        self._token_index = 0
        self._max_retries = max_retries

        # Initialize client with first API key
        self._client = genai.Client(
            api_key=self._api_keys[0], http_options=HttpOptions(timeout=60000)
        )

    def _switch_to_next_api_key(self) -> bool:
        """
        Switch to the next available API key.

        Returns:
            True if switched to a new key, False if no more keys available
        """
        if len(self._api_keys) <= 1:
            return False

        self._token_index = (self._token_index + 1) % len(self._api_keys)
        new_api_key = self._api_keys[self._token_index]

        self._client = genai.Client(
            api_key=new_api_key, http_options=HttpOptions(timeout=60000)
        )

        logger.info(f"Switched to API key index {self._token_index}")

        # If we've cycled back to token 0, we've tried all tokens, delay 60 seconds
        if self._token_index == 0:
            logger.warning(
                "Cycled back to token 0, waiting 60 seconds before continuing"
            )
            time.sleep(60)

        return True

    def _execute_with_retry_and_token_switching(
        self,
        operation_name: str,
        *args,
        **kwargs
    ) -> T:
        """
        Execute an operation with retry logic and token switching.

        Args:
            operation_name: Name of the operation for logging
            *args: Arguments to pass to the operation method
            **kwargs: Keyword arguments to pass to the operation method

        Returns:
            Result of the operation

        Raises:
            RuntimeError: If all API keys fail
        """
        operation_method_name = f"_execute_{operation_name.replace(' ', '_').replace(':', '').lower()}"
        
        while True:
            current_key_index = self._token_index

            logger.info(f"Trying {operation_name} with API key index {current_key_index}")

            for attempt in range(self._max_retries + 1):
                try:
                    logger.debug(f"Executing {operation_name} (attempt {attempt + 1})")
                    operation_method = getattr(self, operation_method_name)
                    result = operation_method(*args, **kwargs)
                    logger.info(f"Successfully executed {operation_name} with API key index {current_key_index}")
                    return result

                except Exception as e:
                    error_str = str(e).lower()
                    is_rate_limit_error = any(
                        term in error_str
                        for term in ["rate limit", "quota", "429", "too many requests"]
                    )

                    if is_rate_limit_error:
                        # Immediately switch to next token on rate limit error
                        logger.warning(
                            f"Rate limit error with API key index {current_key_index}, switching immediately"
                        )
                        break
                    elif attempt == self._max_retries:
                        logger.error(
                            f"Error executing {operation_name} after {self._max_retries + 1} attempts with API key index {current_key_index}: {str(e)}"
                        )
                        break
                    else:
                        # Try to parse retry delay from error message, fallback to exponential backoff
                        retry_delay = parse_retry_delay_from_error(e)
                        if (
                            retry_delay == 60.0
                        ):  # Default value means no specific delay found
                            retry_delay = min(
                                2**attempt, 60
                            )  # Exponential backoff with max 60s

                        logger.warning(
                            f"Error executing {operation_name} (attempt {attempt + 1}/{self._max_retries + 1}), retrying in {retry_delay}s: {str(e)}"
                        )
                        time.sleep(retry_delay)

            if not self._switch_to_next_api_key():
                break

        raise RuntimeError(
            f"Failed to execute {operation_name} after trying all {len(self._api_keys)} available API keys"
        )
