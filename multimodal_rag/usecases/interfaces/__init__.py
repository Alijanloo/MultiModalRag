"""Interfaces module for use cases."""

from multimodal_rag.usecases.interfaces.document_repository import IDocumentIndexRepository, IEmbeddingService
from multimodal_rag.usecases.interfaces.embedding_service import EmbeddingServiceInterface
from multimodal_rag.usecases.interfaces.llm_service import LLMServiceInterface

__all__ = [
    "IDocumentIndexRepository",
    "IEmbeddingService",
    "EmbeddingServiceInterface",
    "LLMServiceInterface",
]