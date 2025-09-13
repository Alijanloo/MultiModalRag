"""Use cases for document indexing and retrieval operations."""

import uuid
from typing import List, Optional

from .interfaces.document_repository import IDocumentIndexRepository, IEmbeddingService
from .dtos import (
    IndexDocumentResponse,
    IndexChunkResponse,
    SearchRequest,
    SearchResponse,
    BulkIndexResponse,
    GetDocumentResponse,
)
from ..entities.document import DoclingDocument, DocChunk


class DocumentIndexingUseCase:
    """Use case for document indexing operations."""
    
    def __init__(
        self,
        document_repository: IDocumentIndexRepository,
        embedding_service: Optional[IEmbeddingService] = None
    ):
        self._document_repository = document_repository
        self._embedding_service = embedding_service
    
    async def index_document(
        self,
        document: DoclingDocument,
        document_id: Optional[str] = None,
        index_name: Optional[str] = None
    ) -> IndexDocumentResponse:
        """Index a complete document."""
        if document_id is None:
            document_id = str(uuid.uuid4())
        
        # Direct entity passing - let the adaptor handle the transformation
        return await self._document_repository.index_document(
            document=document,
            document_id=document_id,
            index_name=index_name
        )
    
    async def index_chunk(
        self,
        chunk: DocChunk,
        chunk_id: Optional[str] = None,
        document_id: Optional[str] = None,
        index_name: Optional[str] = None,
        generate_embedding: bool = True
    ) -> IndexChunkResponse:
        """Index a document chunk with optional vector embedding."""
        if chunk_id is None:
            chunk_id = str(uuid.uuid4())
        
        # Business logic: Generate embedding if needed
        if generate_embedding and self._embedding_service and chunk.vector is None:
            chunk.vector = await self._embedding_service.generate_embedding(chunk.text)
        
        # Direct entity passing
        return await self._document_repository.index_chunk(
            chunk=chunk,
            chunk_id=chunk_id,
            document_id=document_id,
            index_name=index_name
        )
    
    async def bulk_index_document_with_chunks(
        self,
        document: DoclingDocument,
        chunks: List[DocChunk],
        document_id: Optional[str] = None,
        index_name: Optional[str] = None,
        generate_embeddings: bool = True
    ) -> BulkIndexResponse:
        """Index a document along with its chunks."""
        if document_id is None:
            document_id = str(uuid.uuid4())
        
        # Business logic: Generate embeddings for chunks if needed
        if generate_embeddings and self._embedding_service:
            for chunk in chunks:
                if chunk.vector is None:
                    chunk.vector = await self._embedding_service.generate_embedding(chunk.text)
        
        # Direct entity passing
        return await self._document_repository.bulk_index_document_with_chunks(
            document=document,
            chunks=chunks,
            document_id=document_id,
            index_name=index_name
        )


class DocumentSearchUseCase:
    """Use case for document search and retrieval operations."""
    
    def __init__(
        self,
        document_repository: IDocumentIndexRepository,
        embedding_service: Optional[IEmbeddingService] = None
    ):
        self._document_repository = document_repository
        self._embedding_service = embedding_service
    
    async def search_chunks_by_text(
        self,
        query: str,
        size: int = 10,
        filters: Optional[dict] = None,
        index_name: Optional[str] = None
    ) -> SearchResponse:
        """Search chunks using text query."""
        request = SearchRequest(
            query=query,
            size=size,
            filters=filters,
            index_name=index_name
        )
        
        return await self._document_repository.search_chunks(request)
    
    async def search_chunks_by_vector(
        self,
        query_text: str,
        size: int = 10,
        filters: Optional[dict] = None,
        index_name: Optional[str] = None
    ) -> SearchResponse:
        """Search chunks using vector similarity."""
        if not self._embedding_service:
            raise ValueError("Embedding service is required for vector search")
        
        vector = await self._embedding_service.generate_embedding(query_text)
        
        request = SearchRequest(
            vector=vector,
            size=size,
            filters=filters,
            index_name=index_name
        )
        
        return await self._document_repository.search_chunks(request)
    
    async def search_chunks_hybrid(
        self,
        query: str,
        size: int = 10,
        filters: Optional[dict] = None,
        index_name: Optional[str] = None
    ) -> SearchResponse:
        """Search chunks using both text and vector similarity."""
        if not self._embedding_service:
            # Fallback to text search only
            return await self.search_chunks_by_text(query, size, filters, index_name)
        
        vector = await self._embedding_service.generate_embedding(query)
        
        request = SearchRequest(
            query=query,
            vector=vector,
            size=size,
            filters=filters,
            index_name=index_name
        )
        
        return await self._document_repository.search_chunks(request)
    
    async def get_document(
        self,
        document_id: str,
        index_name: Optional[str] = None
    ) -> GetDocumentResponse:
        """Retrieve a document by ID."""
        return await self._document_repository.get_document(
            document_id=document_id,
            index_name=index_name
        )
    
    async def search_documents(
        self,
        query: str,
        size: int = 10,
        filters: Optional[dict] = None,
        index_name: Optional[str] = None
    ) -> SearchResponse:
        """Search documents using text query."""
        request = SearchRequest(
            query=query,
            size=size,
            filters=filters,
            index_name=index_name
        )
        
        return await self._document_repository.search_documents(request)
