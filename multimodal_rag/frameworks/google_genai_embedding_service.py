"""Google GenAI embedding service implementation."""

from typing import List, Union

from multimodal_rag.usecases.interfaces.embedding_service import (
    EmbeddingServiceInterface,
)
from multimodal_rag.frameworks.google_genai_base_service import GoogleGenAIBaseService
from multimodal_rag.frameworks.logging_config import get_logger

logger = get_logger(__name__)


class GoogleGenAIEmbeddingService(GoogleGenAIBaseService, EmbeddingServiceInterface):
    """Google GenAI implementation of the embedding service."""

    def __init__(
        self,
        api_keys: Union[str, List[str]],
        model: str = "gemini-embedding-001",
        embedding_dimensions: int = 768,
        max_retries: int = 7,
    ):
        """
        Initialize the Google GenAI embedding service.

        Args:
            api_keys: Single API key string or list of API keys for token switching
            model: Model name for embeddings
            embedding_dimensions: Number of dimensions for embeddings
            max_retries: Maximum number of retries for failed requests
        """
        super().__init__(api_keys, max_retries)
        self._model = model
        self._embedding_dimensions = embedding_dimensions

    def _execute_embeddings_for_content(self, content: List[str]) -> List[List[float]]:
        """Execute embedding operation for multiple content items."""
        result = self._client.models.embed_content(
            model=self._model,
            contents=content,
            config={"output_dimensionality": self._embedding_dimensions},
        )
        embeddings = []
        for embedding in result.embeddings:
            embeddings.append(embedding.values)
        return embeddings

    def _execute_single_embedding_for_text(self, text: str) -> List[float]:
        """Execute embedding operation for single text."""
        result = self._client.models.embed_content(
            model=self._model,
            contents=text,
            config={"output_dimensionality": self._embedding_dimensions},
        )
        return result.embeddings[0].values

    async def embed_content(self, content: Union[str, List[str]]) -> List[List[float]]:
        """
        Generate embeddings for the given content.

        Args:
            content: Single text string or list of text strings to embed

        Returns:
            List of embedding vectors (list of floats)
        """
        if isinstance(content, str):
            content = [content]

        return self._execute_with_retry_and_token_switching(
            "embeddings_for_content", content
        )

    async def embed_single(self, text: str) -> List[float]:
        """
        Generate embedding for a single text string.

        Args:
            text: Text string to embed

        Returns:
            Embedding vector as list of floats
        """
        return self._execute_with_retry_and_token_switching(
            "single_embedding_for_text", text
        )

    def get_embedding_dimensions(self) -> int:
        """
        Get the dimensions of the embedding vectors.

        Returns:
            Number of dimensions in the embedding vectors
        """
        return self._embedding_dimensions
