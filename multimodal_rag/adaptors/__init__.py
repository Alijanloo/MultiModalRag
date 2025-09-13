"""Adaptors module for the multimodal RAG system."""

from .elasticsearch_adaptor import ElasticsearchDocumentAdaptor

__all__ = [
    "ElasticsearchDocumentAdaptor"
]