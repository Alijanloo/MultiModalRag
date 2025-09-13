"""Interfaces module for use cases."""

from .document_repository import IDocumentIndexRepository, IEmbeddingService

__all__ = [
    "IDocumentIndexRepository",
    "IEmbeddingService"
]