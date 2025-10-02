"""Google GenAI LLM service implementation."""

import json
import re
import time
from typing import Optional, Dict, Any, Union, List
from google import genai
from google.genai.types import HttpOptions
from google.genai import types

from multimodal_rag.usecases.interfaces.llm_service import LLMServiceInterface
from multimodal_rag.frameworks.logging_config import get_logger

logger = get_logger(__name__)


def parse_retry_delay_from_error(error: Exception) -> float:
    """Extract retry delay from error message."""
    error_str = str(error)

    # Try to parse delay from error message
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


class GoogleGenAILLMService(LLMServiceInterface):
    """Google GenAI implementation of the LLM service."""

    def __init__(
        self,
        api_keys: Union[str, List[str]],
        default_model: str = "gemini-2.5-flash",
        max_retries: int = 3,
    ):
        """
        Initialize the Google GenAI LLM service.

        Args:
            api_keys: Single API key string or list of API keys for token switching
            default_model: Default model name for content generation
            max_retries: Maximum number of retries for failed requests
        """
        self._api_keys = api_keys
        self._token_index = 0
        self._default_model = default_model
        self._max_retries = max_retries

        self._client = genai.Client(
            api_key=self._api_keys[0], http_options=HttpOptions(timeout=60000)
        )
        self._default_model = default_model
        self._max_retries = max_retries

        self._available_models = [
            "gemini-2.5-flash",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
        ]

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
        return True

    async def generate_content(
        self, prompt: str, model: Optional[str] = None, **kwargs: Any
    ) -> str:
        """
        Generate content using the LLM.

        Args:
            prompt: The input prompt for content generation
            model: Optional model name to use for generation
            **kwargs: Additional model-specific parameters

        Returns:
            Generated text content
        """
        model_name = model or self._default_model

        tried_keys = set()

        while len(tried_keys) < len(self._api_keys):
            current_key_index = self._token_index
            tried_keys.add(current_key_index)

            logger.info(f"Trying with API key index {current_key_index}")

            for attempt in range(self._max_retries + 1):
                try:
                    logger.debug(
                        f"Generating content with model: {model_name} (attempt {attempt + 1})"
                    )

                    response = self._client.models.generate_content(
                        model=model_name, contents=prompt, **kwargs
                    )

                    generated_text = response.text
                    logger.info(
                        f"Generated content of length {len(generated_text)} with API key index {current_key_index}"
                    )

                    return generated_text

                except Exception as e:
                    error_str = str(e).lower()
                    is_rate_limit_error = any(
                        term in error_str
                        for term in ["rate limit", "quota", "429", "too many requests"]
                    )

                    if attempt == self._max_retries:
                        if is_rate_limit_error and len(tried_keys) < len(
                            self._api_keys
                        ):
                            logger.warning(
                                f"Exhausted retries with API key index {current_key_index} due to rate limiting, switching to next key"
                            )
                            break
                        else:
                            logger.error(
                                f"Error generating content after {self._max_retries + 1} attempts with API key index {current_key_index}: {str(e)}"
                            )
                            if len(tried_keys) >= len(self._api_keys):
                                raise RuntimeError(
                                    f"Failed to generate content after trying all {len(self._api_keys)} available API keys"
                                )
                            break
                    else:
                        retry_delay = parse_retry_delay_from_error(e)

                        logger.warning(
                            f"Error generating content (attempt {attempt + 1}/{self._max_retries + 1}), retrying in {retry_delay}s: {str(e)}"
                        )
                        time.sleep(retry_delay)

            if not self._switch_to_next_api_key():
                break

        # If we've tried all API keys and none worked
        raise RuntimeError(
            f"Failed to generate content after trying all {len(self._api_keys)} available API keys"
        )

    async def generate_structured_content(
        self,
        prompt: str,
        response_schema: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Generate structured content using the LLM.

        Args:
            prompt: The input prompt for content generation
            response_schema: Optional schema for structured response
            model: Optional model name to use for generation
            **kwargs: Additional model-specific parameters

        Returns:
            Structured response as dictionary
        """
        model_name = model or self._default_model

        structured_prompt = prompt
        if response_schema:
            schema_str = json.dumps(response_schema, indent=2)
            structured_prompt = f"""{prompt}

Ensure to respond with a JSON object that follows this schema:
{schema_str}

Ensure your response is valid JSON and follows the schema exactly."""

        tried_keys = set()

        while len(tried_keys) < len(self._api_keys):
            current_key_index = self._token_index
            tried_keys.add(current_key_index)

            logger.info(
                f"Trying structured content with API key index {current_key_index}"
            )

            for attempt in range(self._max_retries + 1):
                try:
                    logger.debug(
                        f"Generating structured content with model: {model_name} (attempt {attempt + 1})"
                    )

                    response = self._client.models.generate_content(
                        model=model_name, contents=structured_prompt, **kwargs
                    )

                    generated_text = response.text.strip()

                    try:
                        if "```json" in generated_text:
                            json_start = generated_text.find("```json") + 7
                            json_end = generated_text.find("```", json_start)
                            generated_text = generated_text[json_start:json_end].strip()
                        elif "```" in generated_text:
                            json_start = generated_text.find("```") + 3
                            json_end = generated_text.rfind("```")
                            generated_text = generated_text[json_start:json_end].strip()

                        structured_response = json.loads(generated_text)
                        logger.info(
                            f"Successfully generated structured content with API key index {current_key_index}"
                        )
                        return structured_response
                    except json.JSONDecodeError:
                        logger.warning(
                            "Response was not valid JSON, returning as text field"
                        )
                        return {"text": generated_text}

                except Exception as e:
                    error_str = str(e).lower()
                    is_rate_limit_error = any(
                        term in error_str
                        for term in ["rate limit", "quota", "429", "too many requests"]
                    )

                    if attempt == self._max_retries:
                        if is_rate_limit_error and len(tried_keys) < len(
                            self._api_keys
                        ):
                            logger.warning(
                                f"Exhausted retries with API key index {current_key_index} due to rate limiting, switching to next key"
                            )
                            break
                        else:
                            logger.error(
                                f"Error generating structured content after {self._max_retries + 1} attempts with API key index {current_key_index}: {str(e)}"
                            )
                            if len(tried_keys) >= len(self._api_keys):
                                raise RuntimeError(
                                    f"Failed to generate structured content after trying all {len(self._api_keys)} available API keys"
                                )
                            break
                    else:
                        retry_delay = parse_retry_delay_from_error(e)

                        logger.warning(
                            f"Error generating structured content (attempt {attempt + 1}/{self._max_retries + 1}), retrying in {retry_delay}s: {str(e)}"
                        )
                        time.sleep(retry_delay)

            if not self._switch_to_next_api_key():
                break

        # If we've tried all API keys and none worked
        raise RuntimeError(
            f"Failed to generate structured content after trying all {len(self._api_keys)} available API keys"
        )

    async def generate_content_with_tools(
        self,
        prompt: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Generate content with tool calling support.

        Args:
            prompt: The input prompt for content generation
            tools: List of tool declarations for function calling
            model: Optional model name to use for generation
            **kwargs: Additional model-specific parameters

        Returns:
            Dictionary containing response and potential function calls
        """
        model_name = model or self._default_model

        tried_keys = set()

        while len(tried_keys) < len(self._api_keys):
            current_key_index = self._token_index
            tried_keys.add(current_key_index)

            logger.info(
                f"Trying content generation with tools using API key index {current_key_index}"
            )

            for attempt in range(self._max_retries + 1):
                try:
                    logger.debug(
                        f"Generating content with tools using model: {model_name} (attempt {attempt + 1})"
                    )

                    config = None
                    if tools:
                        # Convert tools to Google GenAI format
                        function_declarations = []
                        for tool in tools:
                            if "function" in tool:
                                function_declarations.append(tool["function"])
                            else:
                                function_declarations.append(tool)

                        tools_config = types.Tool(
                            function_declarations=function_declarations
                        )
                        config = types.GenerateContentConfig(tools=[tools_config])

                    response = self._client.models.generate_content(
                        model=model_name, contents=prompt, config=config, **kwargs
                    )

                    # Parse the response
                    result = {
                        "text": response.text if response.text else "",
                        "function_calls": [],
                        "has_function_call": False,
                    }

                    # Check for function calls
                    if response.candidates and len(response.candidates) > 0:
                        candidate = response.candidates[0]
                        if candidate.content and candidate.content.parts:
                            for part in candidate.content.parts:
                                if (
                                    hasattr(part, "function_call")
                                    and part.function_call
                                ):
                                    function_call = part.function_call
                                    result["function_calls"].append(
                                        {
                                            "name": function_call.name,
                                            "args": dict(function_call.args)
                                            if function_call.args
                                            else {},
                                        }
                                    )
                                    result["has_function_call"] = True

                    logger.info(
                        f"Successfully generated content with tools using API key index {current_key_index}"
                    )
                    return result

                except Exception as e:
                    error_str = str(e).lower()
                    is_rate_limit_error = any(
                        term in error_str
                        for term in ["rate limit", "quota", "429", "too many requests"]
                    )

                    if attempt == self._max_retries:
                        if is_rate_limit_error and len(tried_keys) < len(
                            self._api_keys
                        ):
                            logger.warning(
                                f"Exhausted retries with API key index {current_key_index} due to rate limiting, switching to next key"
                            )
                            break
                        else:
                            logger.error(
                                f"Error generating content with tools after {self._max_retries + 1} attempts with API key index {current_key_index}: {str(e)}"
                            )
                            if len(tried_keys) >= len(self._api_keys):
                                raise RuntimeError(
                                    f"Failed to generate content with tools after trying all {len(self._api_keys)} available API keys"
                                )
                            break
                    else:
                        retry_delay = parse_retry_delay_from_error(e)

                        logger.warning(
                            f"Error generating content with tools (attempt {attempt + 1}/{self._max_retries + 1}), retrying in {retry_delay}s: {str(e)}"
                        )
                        time.sleep(retry_delay)

            if not self._switch_to_next_api_key():
                break

        # If we've tried all API keys and none worked
        raise RuntimeError(
            f"Failed to generate content with tools after trying all {len(self._api_keys)} available API keys"
        )

    def get_available_models(self) -> list[str]:
        """
        Get list of available models.

        Returns:
            List of available model names
        """
        return self._available_models.copy()
