"""Google GenAI LLM service implementation."""

import json
from typing import Optional, Dict, Any
from google import genai

from multimodal_rag.usecases.interfaces.llm_service import LLMServiceInterface
from multimodal_rag import logger


class GoogleGenAILLMService(LLMServiceInterface):
    """Google GenAI implementation of the LLM service."""

    def __init__(
        self, 
        api_key: str, 
        default_model: str = "gemini-2.5-flash"
    ):
        """
        Initialize the Google GenAI LLM service.
        
        Args:
            api_key: Google GenAI API key
            default_model: Default model name for content generation
        """
        self._client = genai.Client(api_key=api_key)
        self._default_model = default_model
        
        # Available models (this could be fetched dynamically if API supports it)
        self._available_models = [
            "gemini-2.5-flash",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
        ]

    async def generate_content(
        self, 
        prompt: str, 
        model: Optional[str] = None,
        **kwargs: Any
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
        try:
            model_name = model or self._default_model
            
            logger.debug(f"Generating content with model: {model_name}")
            
            response = self._client.models.generate_content(
                model=model_name,
                contents=prompt,
                **kwargs
            )
            
            generated_text = response.text
            logger.info(f"Generated content of length {len(generated_text)}")
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Error generating content: {str(e)}")
            raise

    async def generate_structured_content(
        self, 
        prompt: str, 
        response_schema: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        **kwargs: Any
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
        try:
            model_name = model or self._default_model
            
            # Enhance prompt for structured output
            structured_prompt = prompt
            if response_schema:
                schema_str = json.dumps(response_schema, indent=2)
                structured_prompt = f"""{prompt}

Please respond with a JSON object that follows this schema:
{schema_str}

Ensure your response is valid JSON and follows the schema exactly."""
            
            logger.debug(f"Generating structured content with model: {model_name}")
            
            response = self._client.models.generate_content(
                model=model_name,
                contents=structured_prompt,
                **kwargs
            )
            
            generated_text = response.text
            
            # Try to parse JSON response
            try:
                structured_response = json.loads(generated_text)
                logger.info("Successfully generated structured content")
                return structured_response
            except json.JSONDecodeError:
                # If JSON parsing fails, return the text in a structured format
                logger.warning("Response was not valid JSON, returning as text field")
                return {"text": generated_text}
            
        except Exception as e:
            logger.error(f"Error generating structured content: {str(e)}")
            raise

    def get_available_models(self) -> list[str]:
        """
        Get list of available models.
        
        Returns:
            List of available model names
        """
        return self._available_models.copy()
