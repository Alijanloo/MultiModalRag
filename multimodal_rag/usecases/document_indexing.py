"""Use cases for document indexing and retrieval operations."""

import json
import uuid
from pathlib import Path
from typing import List, Optional

from docling_core.types.doc.document import DoclingDocument as DLDocument
from docling.chunking import HybridChunker

from multimodal_rag.frameworks.logging_config import get_logger
from .interfaces.document_repository import IDocumentIndexRepository
from multimodal_rag.usecases.interfaces.embedding_service import (
    EmbeddingServiceInterface,
)
from .dtos import BulkIndexResult
from ..entities.document import (
    DoclingDocument,
    DocChunk,
    DocumentText,
    DocumentPicture,
    DocumentTable,
    create_document_entities_from_docling,
)

logger = get_logger(__name__)


class DocumentIndexingUseCase:
    """Use case for document indexing operations."""

    def __init__(
        self,
        document_repository: IDocumentIndexRepository,
        embedding_service: Optional[EmbeddingServiceInterface] = None,
    ):
        self._document_repository = document_repository
        self._embedding_service = embedding_service

    async def index_document(
        self,
        document: DoclingDocument,
        document_id: Optional[str] = None,
        index_name: Optional[str] = None,
    ) -> bool:
        """Index a complete document."""
        if document_id is None:
            document_id = str(uuid.uuid4())

        # Direct entity passing - let the adaptor handle the transformation
        return await self._document_repository.index_document(
            document=document, document_id=document_id, index_name=index_name
        )

    async def index_chunk(
        self,
        chunk: DocChunk,
        chunk_id: Optional[str] = None,
        document_id: Optional[str] = None,
        index_name: Optional[str] = None,
        generate_embedding: bool = True,
    ) -> bool:
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
            index_name=index_name,
        )

    async def bulk_index_document_with_elements_and_chunks(
        self,
        document: DoclingDocument,
        texts: List[DocumentText],
        pictures: List[DocumentPicture],
        tables: List[DocumentTable],
        chunks: List[DocChunk],
        document_id: Optional[str] = None,
        index_name: Optional[str] = None,
        generate_embeddings: bool = True,
    ) -> BulkIndexResult:
        """Index a document along with its separated elements and chunks."""
        if document_id is None:
            document_id = str(uuid.uuid4())

        total_indexed = 0
        total_failed = 0
        errors = []

        # Generate embeddings for chunks if needed
        if generate_embeddings and self._embedding_service:
            # Process chunks in batches of 60
            batch_size = 60
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i : i + batch_size]
                texts_for_embedding = [chunk.text for chunk in batch_chunks]

                try:
                    embeddings = await self._embedding_service.embed_content(
                        texts_for_embedding
                    )

                    for chunk, embedding in zip(batch_chunks, embeddings):
                        chunk.vector = embedding
                except Exception as e:
                    error_msg = f"Failed to generate embeddings for batch: {e}"
                    logger.error(error_msg)
                    errors.append(error_msg)

        # Index document
        try:
            if await self._document_repository.index_document(
                document=document, document_id=document_id, index_name=index_name
            ):
                total_indexed += 1
            else:
                total_failed += 1
                errors.append(f"Failed to index document {document_id}")
        except Exception as e:
            total_failed += 1
            error_msg = f"Failed to index document {document_id}: {e}"
            logger.error(error_msg)
            errors.append(error_msg)

        # Use bulk methods for better performance
        # Bulk index texts
        if texts:
            try:
                indexed_count, failed_count, bulk_errors = (
                    await self._document_repository.bulk_index_texts(texts, index_name)
                )
                total_indexed += indexed_count
                total_failed += failed_count
                errors.extend(bulk_errors)
            except Exception as e:
                total_failed += len(texts)
                error_msg = f"Failed to bulk index texts: {e}"
                logger.error(error_msg)
                errors.append(error_msg)

        # Bulk index pictures
        if pictures:
            try:
                indexed_count, failed_count, bulk_errors = (
                    await self._document_repository.bulk_index_pictures(pictures, index_name)
                )
                total_indexed += indexed_count
                total_failed += failed_count
                errors.extend(bulk_errors)
            except Exception as e:
                total_failed += len(pictures)
                error_msg = f"Failed to bulk index pictures: {e}"
                logger.error(error_msg)
                errors.append(error_msg)

        # Bulk index tables
        if tables:
            try:
                indexed_count, failed_count, bulk_errors = (
                    await self._document_repository.bulk_index_tables(tables, index_name)
                )
                total_indexed += indexed_count
                total_failed += failed_count
                errors.extend(bulk_errors)
            except Exception as e:
                total_failed += len(tables)
                error_msg = f"Failed to bulk index tables: {e}"
                logger.error(error_msg)
                errors.append(error_msg)

        # Bulk index chunks
        if chunks:
            try:
                indexed_count, failed_count, bulk_errors = (
                    await self._document_repository.bulk_index_chunks(
                        chunks, document_id, index_name
                    )
                )
                total_indexed += indexed_count
                total_failed += failed_count
                errors.extend(bulk_errors)
            except Exception as e:
                total_failed += len(chunks)
                error_msg = f"Failed to bulk index chunks: {e}"
                logger.error(error_msg)
                errors.append(error_msg)

        return BulkIndexResult(
            total_indexed=total_indexed, total_failed=total_failed, errors=errors
        )

    async def bulk_index_from_directory(
        self,
        indexing_directory: str = "data/indexing",
        index_name: Optional[str] = None,
        max_tokens: int = 512,
        generate_embeddings: bool = True,
    ) -> List[BulkIndexResult]:
        """
        Walk through indexing directories and bulk index all DoclingDocuments with their chunks.

        Args:
            indexing_directory: Path to the directory containing result folders
            index_name: Name of the index to use
            max_tokens: Maximum tokens per chunk for the HybridChunker
            generate_embeddings: Whether to generate embeddings for chunks

        Returns:
            List of BulkIndexResponse for each successfully processed document
        """
        responses = []

        # Create chunker
        chunker = HybridChunker(max_tokens=max_tokens)

        # Walk through all result directories
        indexing_path = Path(indexing_directory)
        if not indexing_path.exists():
            logger.error(f"Indexing directory does not exist: {indexing_directory}")
            return responses

        logger.info(f"Starting bulk indexing from directory: {indexing_directory}")

        for result_dir in indexing_path.iterdir():
            if not result_dir.is_dir():
                continue

            logger.info(f"Processing directory: {result_dir.name}")

            # Look for the single JSON file in the directory
            json_files = list(result_dir.glob("*.json"))

            if not json_files:
                logger.warning(f"No JSON file found in directory: {result_dir.name}")
                continue

            if len(json_files) > 1:
                logger.warning(
                    f"Multiple JSON files found in {result_dir.name}, processing the first one: {json_files[0].name}"
                )

            json_file = json_files[0]  # Take the first (and typically only) JSON file

            try:
                logger.info(f"Processing file: {json_file.name}")

                # Load DoclingDocument from JSON
                with open(json_file, "r", encoding="utf-8") as f:
                    doc_data = json.load(f)

                # Convert to Docling document and then to our entity
                dl_doc = DLDocument.model_validate(doc_data)
                document, texts, pictures, tables = (
                    create_document_entities_from_docling(dl_doc, json_file.stem)
                )

                await self._document_repository.index_document(
                    document=document, document_id=json_file.stem, index_name=index_name
                )

                # Generate chunks using HybridChunker
                chunk_iter = chunker.chunk(dl_doc=dl_doc)
                chunks = []

                for dl_chunk in chunk_iter:
                    # Convert docling chunk to our entity
                    chunk = DocChunk.from_docling_chunk(dl_chunk)
                    chunk.text = chunker.contextualize(chunk=dl_chunk)
                    chunks.append(chunk)

                logger.info(
                    f"Generated {len(chunks)} chunks for document: {json_file.name}"
                )

                # Create document ID from filename (without extension)
                document_id = json_file.stem

                # Index document with all elements and chunks
                response = await self.bulk_index_document_with_elements_and_chunks(
                    document=document,
                    texts=texts,
                    pictures=pictures,
                    tables=tables,
                    chunks=chunks,
                    document_id=document_id,
                    index_name=index_name,
                    generate_embeddings=generate_embeddings,
                )

                responses.append(response)
                logger.info(
                    f"Successfully indexed document {document_id}: {response.total_indexed} items"
                )

                if response.errors:
                    logger.warning(
                        f"Errors during indexing of {document_id}: {response.errors}"
                    )

            except Exception as e:
                logger.error(f"Failed to process {json_file}: {str(e)}")
                continue

        logger.info(f"Bulk indexing completed. Processed {len(responses)} documents.")
        return responses
