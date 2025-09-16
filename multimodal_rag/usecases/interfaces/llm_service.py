"""LLM service interface for the multimodal RAG application."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class LLMServiceInterface(ABC):
    """Interface for Large Language Model services."""

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def get_available_models(self) -> list[str]:
        """
        Get list of available models.
        
        Returns:
            List of available model names
        """
        pass
