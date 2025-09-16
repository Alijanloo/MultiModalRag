"""Embedding service interface for the multimodal RAG application."""

from abc import ABC, abstractmethod
from typing import List, Union


class EmbeddingServiceInterface(ABC):
    """Interface for embedding generation services."""

    @abstractmethod
    async def embed_content(self, content: Union[str, List[str]]) -> List[List[float]]:
        """
        Generate embeddings for the given content.
        
        Args:
            content: Single text string or list of text strings to embed
            
        Returns:
            List of embedding vectors (list of floats)
        """
        pass

    @abstractmethod
    async def embed_single(self, text: str) -> List[float]:
        """
        Generate embedding for a single text string.
        
        Args:
            text: Text string to embed
            
        Returns:
            Embedding vector as list of floats
        """
        pass

    @abstractmethod
    def get_embedding_dimensions(self) -> int:
        """
        Get the dimensions of the embedding vectors.
        
        Returns:
            Number of dimensions in the embedding vectors
        """
        pass
