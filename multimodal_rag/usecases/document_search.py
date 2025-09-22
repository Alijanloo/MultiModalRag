"""Use cases for document indexing and retrieval operations."""

from typing import Optional, List

from multimodal_rag.frameworks.logging_config import get_logger
from .interfaces.document_repository import IDocumentIndexRepository
from multimodal_rag.usecases.interfaces.embedding_service import (
    EmbeddingServiceInterface,
)
from ..entities.document import DocChunk, DoclingDocument

logger = get_logger(__name__)


class DocumentSearchUseCase:
    """Use case for document search and retrieval operations."""

    def __init__(
        self,
        document_repository: IDocumentIndexRepository,
        embedding_service: Optional[EmbeddingServiceInterface] = None,
    ):
        self._document_repository = document_repository
        self._embedding_service = embedding_service

    async def search_chunks_by_text(
        self,
        query: str,
        size: int = 10,
        filters: Optional[dict] = None,
        index_name: Optional[str] = None,
    ) -> List[DocChunk]:
        """Search chunks using text query."""
        return await self._document_repository.search_chunks(
            query=query, size=size, filters=filters, index_name=index_name
        )

    async def search_chunks_by_vector(
        self,
        query_text: str,
        size: int = 10,
        filters: Optional[dict] = None,
        index_name: Optional[str] = None,
    ) -> List[DocChunk]:
        """Search chunks using vector similarity."""
        if not self._embedding_service:
            raise ValueError("Embedding service is required for vector search")

        vector = await self._embedding_service.embed_single(query_text)

        return await self._document_repository.search_chunks(
            vector=vector, size=size, filters=filters, index_name=index_name
        )

    async def search_chunks_hybrid(
        self,
        query: str,
        size: int = 10,
        filters: Optional[dict] = None,
        index_name: Optional[str] = None,
    ) -> List[DocChunk]:
        """Search chunks using both text and vector similarity."""
        if not self._embedding_service:
            # Fallback to text search only
            return await self.search_chunks_by_text(query, size, filters, index_name)

        vector = await self._embedding_service.embed_single(query)

        return await self._document_repository.search_chunks(
            query=query,
            vector=vector,
            size=size,
            filters=filters,
            index_name=index_name,
        )

    async def get_document(
        self, document_id: str, index_name: Optional[str] = None
    ) -> Optional[DoclingDocument]:
        """Retrieve a document by ID."""
        return await self._document_repository.get_document(
            document_id=document_id, index_name=index_name
        )

    async def search_documents(
        self,
        query: str,
        size: int = 10,
        filters: Optional[dict] = None,
        index_name: Optional[str] = None,
    ) -> List[DoclingDocument]:
        """Search documents using text query."""
        return await self._document_repository.search_documents(
            query=query, size=size, filters=filters, index_name=index_name
        )
