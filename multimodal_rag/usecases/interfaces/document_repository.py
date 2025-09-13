"""Interfaces for document indexing and search operations."""

from abc import ABC, abstractmethod
from typing import List, Optional

from ..dtos import (
    IndexDocumentResponse,
    IndexChunkResponse,
    SearchRequest,
    SearchResponse,
    BulkIndexResponse,
    GetDocumentResponse,
)
from ...entities.document import DoclingDocument, DocChunk


class IDocumentIndexRepository(ABC):
    """Interface for document indexing operations with direct entity passing."""
    
    @abstractmethod
    async def index_document(
        self, 
        document: DoclingDocument,
        document_id: str,
        index_name: Optional[str] = None
    ) -> IndexDocumentResponse:
        """Index a single document."""
        pass
    
    @abstractmethod
    async def index_chunk(
        self, 
        chunk: DocChunk,
        chunk_id: str,
        document_id: Optional[str] = None,
        index_name: Optional[str] = None
    ) -> IndexChunkResponse:
        """Index a single chunk."""
        pass
    
    @abstractmethod
    async def bulk_index_document_with_chunks(
        self,
        document: DoclingDocument,
        chunks: List[DocChunk],
        document_id: str,
        index_name: Optional[str] = None
    ) -> BulkIndexResponse:
        """Bulk index a document with its chunks."""
        pass
    
    @abstractmethod
    async def search_chunks(self, request: SearchRequest) -> SearchResponse:
        """Search chunks using text or vector similarity."""
        pass
    
    @abstractmethod
    async def search_documents(self, request: SearchRequest) -> SearchResponse:
        """Search documents using text query."""
        pass
    
    @abstractmethod
    async def get_document(
        self, 
        document_id: str, 
        index_name: Optional[str] = None
    ) -> GetDocumentResponse:
        """Get a document by ID."""
        pass
    
    @abstractmethod
    async def delete_document(self, document_id: str, index_name: Optional[str] = None) -> bool:
        """Delete a document by ID."""
        pass
    
    @abstractmethod
    async def delete_chunk(self, chunk_id: str, index_name: Optional[str] = None) -> bool:
        """Delete a chunk by ID."""
        pass


class IEmbeddingService(ABC):
    """Interface for embedding generation service."""
    
    @abstractmethod
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for the given text."""
        pass
    
    @abstractmethod
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        pass
