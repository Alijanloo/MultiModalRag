"""Entities module for the multimodal RAG system."""

from .document import DoclingDocument, DocChunk, DocMeta, DocumentOrigin

__all__ = [
    "DoclingDocument",
    "DocChunk", 
    "DocMeta",
    "DocumentOrigin"
]