"""Google GenAI embedding service implementation."""

import re
import time
from typing import List, Union
from google import genai
from google.genai.types import HttpOptions

from multimodal_rag.usecases.interfaces.embedding_service import (
    EmbeddingServiceInterface,
)
from multimodal_rag.frameworks.logging_config import get_logger

logger = get_logger(__name__)


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


class GoogleGenAIEmbeddingService(EmbeddingServiceInterface):
    """Google GenAI implementation of the embedding service."""

    def __init__(
        self,
        api_keys: Union[str, List[str]],
        model: str = "gemini-embedding-001",
        embedding_dimensions: int = 768,
        max_retries: int = 3,
    ):
        """
        Initialize the Google GenAI embedding service.

        Args:
            api_keys: Single API key string or list of API keys for token switching
            model: Model name for embeddings
            embedding_dimensions: Number of dimensions for embeddings
            max_retries: Maximum number of retries for failed requests
        """
        self._api_keys = api_keys
        self._token_index = 0
        self._model = model
        self._embedding_dimensions = embedding_dimensions
        self._max_retries = max_retries
        
        self._client = genai.Client(
            api_key=self._api_keys[0], http_options=HttpOptions(timeout=5000)
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
            api_key=new_api_key, http_options=HttpOptions(timeout=5000)
        )
        
        logger.info(f"Switched to API key index {self._token_index}")
        return True

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

        tried_keys = set()
        
        while len(tried_keys) < len(self._api_keys):
            current_key_index = self._token_index
            tried_keys.add(current_key_index)
            
            logger.info(f"Trying embeddings with API key index {current_key_index}")
            
            for attempt in range(self._max_retries + 1):
                try:
                    logger.debug(
                        f"Generating embeddings for {len(content)} items (attempt {attempt + 1})"
                    )

                    result = self._client.models.embed_content(
                        model=self._model,
                        contents=content,
                        config={"output_dimensionality": self._embedding_dimensions},
                    )
                    embeddings = []
                    for embedding in result.embeddings:
                        embeddings.append(embedding.values)

                    logger.info(f"Generated {len(embeddings)} embeddings with API key index {current_key_index}")
                    return embeddings

                except Exception as e:
                    error_str = str(e).lower()
                    is_rate_limit_error = any(term in error_str for term in ['rate limit', 'quota', '429', 'too many requests'])
                    
                    if attempt == self._max_retries:
                        if is_rate_limit_error and len(tried_keys) < len(self._api_keys):
                            logger.warning(f"Exhausted retries with API key index {current_key_index} due to rate limiting, switching to next key")
                            break
                        else:
                            logger.error(f"Error generating embeddings after {self._max_retries + 1} attempts with API key index {current_key_index}: {str(e)}")
                            if len(tried_keys) >= len(self._api_keys):
                                raise RuntimeError(f"Failed to generate embeddings after trying all {len(self._api_keys)} available API keys")
                            break
                    else:
                        retry_delay = parse_retry_delay_from_error(e)

                        logger.warning(
                            f"Error generating embeddings (attempt {attempt + 1}/{self._max_retries + 1}), retrying in {retry_delay}s: {str(e)}"
                        )
                        time.sleep(retry_delay)
            
            if not self._switch_to_next_api_key():
                break
        
        raise RuntimeError(f"Failed to generate embeddings after trying all {len(self._api_keys)} available API keys")

    async def embed_single(self, text: str) -> List[float]:
        """
        Generate embedding for a single text string.

        Args:
            text: Text string to embed

        Returns:
            Embedding vector as list of floats
        """
        tried_keys = set()
        
        while len(tried_keys) < len(self._api_keys):
            current_key_index = self._token_index
            tried_keys.add(current_key_index)
            
            logger.info(f"Trying single embedding with API key index {current_key_index}")
            
            for attempt in range(self._max_retries + 1):
                try:
                    logger.debug(
                        f"Generating single embedding for text of length {len(text)} (attempt {attempt + 1})"
                    )

                    result = self._client.models.embed_content(
                        model=self._model,
                        contents=text,
                        config={"output_dimensionality": self._embedding_dimensions},
                    )

                    embedding = result.embeddings[0]
                    logger.debug(f"Generated embedding for text of length {len(text)} with API key index {current_key_index}")
                    return embedding.values

                except Exception as e:
                    error_str = str(e).lower()
                    is_rate_limit_error = any(term in error_str for term in ['rate limit', 'quota', '429', 'too many requests'])
                    
                    if attempt == self._max_retries:
                        if is_rate_limit_error and len(tried_keys) < len(self._api_keys):
                            logger.warning(f"Exhausted retries with API key index {current_key_index} due to rate limiting, switching to next key")
                            break
                        else:
                            logger.error(f"Error generating single embedding after {self._max_retries + 1} attempts with API key index {current_key_index}: {str(e)}")
                            if len(tried_keys) >= len(self._api_keys):
                                raise RuntimeError(f"Failed to generate single embedding after trying all {len(self._api_keys)} available API keys")
                            break
                    else:
                        retry_delay = parse_retry_delay_from_error(e)

                        logger.warning(
                            f"Error generating single embedding (attempt {attempt + 1}/{self._max_retries + 1}), retrying in {retry_delay}s: {str(e)}"
                        )
                        time.sleep(retry_delay)
            
            if not self._switch_to_next_api_key():
                break
        
        raise RuntimeError(f"Failed to generate single embedding after trying all {len(self._api_keys)} available API keys")

    def get_embedding_dimensions(self) -> int:
        """
        Get the dimensions of the embedding vectors.

        Returns:
            Number of dimensions in the embedding vectors
        """
        return self._embedding_dimensions
