"""Google GenAI embedding service implementation."""

from typing import List, Union
from google import genai

from multimodal_rag.usecases.interfaces.embedding_service import EmbeddingServiceInterface
from multimodal_rag import logger


class GoogleGenAIEmbeddingService(EmbeddingServiceInterface):
    """Google GenAI implementation of the embedding service."""

    def __init__(
        self, 
        api_key: str, 
        model: str = "gemini-embedding-001"
    ):
        """
        Initialize the Google GenAI embedding service.
        
        Args:
            api_key: Google GenAI API key
            model: Model name for embeddings
        """
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._embedding_dimensions = 768  # Default for gemini-embedding-001

    async def embed_content(self, content: Union[str, List[str]]) -> List[List[float]]:
        """
        Generate embeddings for the given content.
        
        Args:
            content: Single text string or list of text strings to embed
            
        Returns:
            List of embedding vectors (list of floats)
        """
        try:
            if isinstance(content, str):
                content = [content]
            
            embeddings = []
            for text in content:
                result = self._client.models.embed_content(
                    model=self._model,
                    contents=text
                )
                embeddings.append(result.embeddings[0])
            
            logger.info(f"Generated {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise

    async def embed_single(self, text: str) -> List[float]:
        """
        Generate embedding for a single text string.
        
        Args:
            text: Text string to embed
            
        Returns:
            Embedding vector as list of floats
        """
        try:
            result = self._client.models.embed_content(
                model=self._model,
                contents=text
            )
            
            embedding = result.embeddings[0]
            logger.debug(f"Generated embedding for text of length {len(text)}")
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating single embedding: {str(e)}")
            raise

    def get_embedding_dimensions(self) -> int:
        """
        Get the dimensions of the embedding vectors.
        
        Returns:
            Number of dimensions in the embedding vectors
        """
        return self._embedding_dimensions
