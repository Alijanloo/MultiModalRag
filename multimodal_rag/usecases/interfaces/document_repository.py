"""Interfaces for document indexing and search operations."""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Tuple

from ...entities.document import (
    DoclingDocument,
    DocChunk,
    DocumentText,
    DocumentPicture,
    DocumentTable,
)


class IDocumentIndexRepository(ABC):
    """Interface for document indexing operations with direct entity passing."""

    @abstractmethod
    async def index_document(
        self,
        document: DoclingDocument,
        document_id: str,
        index_name: Optional[str] = None,
    ) -> bool:
        """Index a single document. Returns True if successful."""
        pass

    @abstractmethod
    async def index_chunk(
        self,
        chunk: DocChunk,
        chunk_id: str,
        document_id: Optional[str] = None,
        index_name: Optional[str] = None,
    ) -> bool:
        """Index a single chunk. Returns True if successful."""
        pass

    @abstractmethod
    async def index_text(
        self, text: DocumentText, index_name: Optional[str] = None
    ) -> bool:
        """Index a single text element. Returns True if successful."""
        pass

    @abstractmethod
    async def index_picture(
        self, picture: DocumentPicture, index_name: Optional[str] = None
    ) -> bool:
        """Index a single picture element. Returns True if successful."""
        pass

    @abstractmethod
    async def index_table(
        self, table: DocumentTable, index_name: Optional[str] = None
    ) -> bool:
        """Index a single table element. Returns True if successful."""
        pass

    @abstractmethod
    async def bulk_index_chunks(
        self,
        chunks: List[DocChunk],
        document_id: str,
        index_name: Optional[str] = None,
    ) -> Tuple[int, int, List[str]]:
        """Bulk index multiple chunks. Returns (indexed_count, failed_count, errors)."""
        pass

    @abstractmethod
    async def bulk_index_texts(
        self,
        texts: List[DocumentText],
        index_name: Optional[str] = None,
    ) -> Tuple[int, int, List[str]]:
        """Bulk index multiple text elements. Returns (indexed_count, failed_count, errors)."""
        pass

    @abstractmethod
    async def bulk_index_pictures(
        self,
        pictures: List[DocumentPicture],
        index_name: Optional[str] = None,
    ) -> Tuple[int, int, List[str]]:
        """Bulk index multiple picture elements. Returns (indexed_count, failed_count, errors)."""
        pass

    @abstractmethod
    async def bulk_index_tables(
        self,
        tables: List[DocumentTable],
        index_name: Optional[str] = None,
    ) -> Tuple[int, int, List[str]]:
        """Bulk index multiple table elements. Returns (indexed_count, failed_count, errors)."""
        pass

    @abstractmethod
    async def get_picture(
        self, document_id: str, picture_id: str, index_name: Optional[str] = None
    ) -> Optional[DocumentPicture]:
        """Get a picture by document_id and picture_id."""
        pass

    @abstractmethod
    async def get_text(
        self, document_id: str, text_id: str, index_name: Optional[str] = None
    ) -> Optional[DocumentText]:
        """Get a text by document_id and text_id."""
        pass

    @abstractmethod
    async def get_table(
        self, document_id: str, table_id: str, index_name: Optional[str] = None
    ) -> Optional[DocumentTable]:
        """Get a table by document_id and table_id."""
        pass

    @abstractmethod
    async def search_chunks(
        self,
        query: Optional[str] = None,
        vector: Optional[List[float]] = None,
        filters: Optional[Dict[str, Any]] = None,
        size: int = 10,
        index_name: Optional[str] = None,
    ) -> List[DocChunk]:
        """Search chunks using text or vector similarity."""
        pass

    @abstractmethod
    async def search_documents(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        size: int = 10,
        index_name: Optional[str] = None,
    ) -> List[DoclingDocument]:
        """Search documents using text query."""
        pass

    @abstractmethod
    async def get_document(
        self, document_id: str, index_name: Optional[str] = None
    ) -> Optional[DoclingDocument]:
        """Get a document by ID. Returns document or None if not found."""
        pass

    @abstractmethod
    async def delete_document(
        self, document_id: str, index_name: Optional[str] = None
    ) -> bool:
        """Delete a document by ID."""
        pass

    @abstractmethod
    async def delete_chunk(
        self, chunk_id: str, index_name: Optional[str] = None
    ) -> bool:
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
