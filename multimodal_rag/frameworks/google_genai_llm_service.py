"""Google GenAI LLM service implementation."""

import json
from typing import Optional, Dict, Any, Union, List
from google.genai import types

from multimodal_rag.usecases.interfaces.llm_service import LLMServiceInterface
from multimodal_rag.frameworks.google_genai_base_service import GoogleGenAIBaseService
from multimodal_rag.frameworks.logging_config import get_logger

logger = get_logger(__name__)


class GoogleGenAILLMService(GoogleGenAIBaseService, LLMServiceInterface):
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
        super().__init__(api_keys, max_retries)
        self._default_model = default_model

        self._available_models = [
            "gemini-2.5-flash",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
        ]

    def _execute_content_generation(
        self, model_name: str, prompt: str, **kwargs
    ) -> str:
        """Execute content generation operation."""
        response = self._client.models.generate_content(
            model=model_name, contents=prompt, **kwargs
        )
        return response.text

    def _execute_structured_content_generation(
        self, model_name: str, structured_prompt: str, **kwargs
    ) -> Dict[str, Any]:
        """Execute structured content generation operation."""
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
            logger.warning("Response was not valid JSON, returning as text field")
            return {"text": generated_text}

    def _execute_content_generation_with_tools(
        self,
        model_name: str,
        prompt: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Execute content generation with tools operation."""
        config = None
        if tools:
            # Convert tools to Google GenAI format
            function_declarations = []
            for tool in tools:
                if "function" in tool:
                    function_declarations.append(tool["function"])
                else:
                    function_declarations.append(tool)

            tools_config = types.Tool(function_declarations=function_declarations)
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
                    if hasattr(part, "function_call") and part.function_call:
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

        return result

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

        return self._execute_with_retry_and_token_switching(
            "content_generation", model_name, prompt, **kwargs
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

        return self._execute_with_retry_and_token_switching(
            "structured_content_generation", model_name, structured_prompt, **kwargs
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

        return self._execute_with_retry_and_token_switching(
            "content_generation_with_tools", model_name, prompt, tools, **kwargs
        )

    def get_available_models(self) -> list[str]:
        """
        Get list of available models.

        Returns:
            List of available model names
        """
        return self._available_models.copy()
