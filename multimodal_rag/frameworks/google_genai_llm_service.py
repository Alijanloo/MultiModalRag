"""Google GenAI LLM service implementation."""

import json
import re
import time
from typing import Optional, Dict, Any
from google import genai
from google.genai.types import HttpOptions

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
        api_key: str,
        default_model: str = "gemini-2.5-flash",
        max_retries: int = 3,
    ):
        """
        Initialize the Google GenAI LLM service.

        Args:
            api_key: Google GenAI API key
            default_model: Default model name for content generation
            max_retries: Maximum number of retries for failed requests
        """
        self._client = genai.Client(
            api_key=api_key, http_options=HttpOptions(timeout=5000)
        )
        self._default_model = default_model
        self._max_retries = max_retries

        # Available models (this could be fetched dynamically if API supports it)
        self._available_models = [
            "gemini-2.5-flash",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
        ]

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

        for attempt in range(self._max_retries + 1):
            try:
                logger.debug(
                    f"Generating content with model: {model_name} (attempt {attempt + 1})"
                )

                response = self._client.models.generate_content(
                    model=model_name, contents=prompt, **kwargs
                )

                generated_text = response.text
                logger.info(f"Generated content of length {len(generated_text)}")

                return generated_text

            except Exception as e:
                if attempt == self._max_retries:
                    logger.error(
                        f"Error generating content after {self._max_retries + 1} attempts: {str(e)}"
                    )
                    raise
                else:
                    retry_delay = parse_retry_delay_from_error(e)
                    if (
                        retry_delay == 60.0
                    ):  # Default value means no specific delay found
                        retry_delay = min(
                            2**attempt, 60
                        )  # Exponential backoff with max 60s

                    logger.warning(
                        f"Error generating content (attempt {attempt + 1}/{self._max_retries + 1}), retrying in {retry_delay}s: {str(e)}"
                    )
                    time.sleep(retry_delay)

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

        # Enhance prompt for structured output
        structured_prompt = prompt
        if response_schema:
            schema_str = json.dumps(response_schema, indent=2)
            structured_prompt = f"""{prompt}

Ensure to respond with a JSON object that follows this schema:
{schema_str}

Ensure your response is valid JSON and follows the schema exactly."""

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
                    return structured_response
                except json.JSONDecodeError:
                    logger.warning(
                        "Response was not valid JSON, returning as text field"
                    )
                    return {"text": generated_text}

            except Exception as e:
                if attempt == self._max_retries:
                    logger.error(
                        f"Error generating structured content after {self._max_retries + 1} attempts: {str(e)}"
                    )
                    raise
                else:
                    retry_delay = parse_retry_delay_from_error(e)
                    if (
                        retry_delay == 60.0
                    ):  # Default value means no specific delay found
                        retry_delay = min(
                            2**attempt, 60
                        )  # Exponential backoff with max 60s

                    logger.warning(
                        f"Error generating structured content (attempt {attempt + 1}/{self._max_retries + 1}), retrying in {retry_delay}s: {str(e)}"
                    )
                    time.sleep(retry_delay)

    def get_available_models(self) -> list[str]:
        """
        Get list of available models.

        Returns:
            List of available model names
        """
        return self._available_models.copy()
