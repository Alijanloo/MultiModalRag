"""Example usage of the multimodal RAG system with dependency injection."""

import asyncio
import json
from pathlib import Path
from typing import List

from docling_core.types.doc.document import DoclingDocument as DLDocument

from multimodal_rag.entities.document import (
    DoclingDocument,
    DocChunk,
    DocumentText,
    DocumentPicture,
    DocumentTable,
    create_document_entities_from_docling,
)
from multimodal_rag.container import ApplicationContainer
from multimodal_rag.usecases.document_indexing import DocumentIndexingUseCase
from multimodal_rag.usecases.document_search import DocumentSearchUseCase


async def load_sample_data() -> tuple[
    DoclingDocument,
    List[DocumentText],
    List[DocumentPicture],
    List[DocumentTable],
    List[DocChunk],
]:
    """Load sample data from JSON files."""
    # Load sample DoclingDocument
    doc_path = Path("data/docling_document.json")
    with open(doc_path, "r") as f:
        doc_data = json.load(f)

    # Convert to our entity
    dl_doc = DLDocument.model_validate(doc_data)
    document, texts, pictures, tables = create_document_entities_from_docling(
        dl_doc, "sample_doc_1"
    )

    # Load sample DocChunk
    chunk_path = Path("data/doc_chunk.json")
    with open(chunk_path, "r") as f:
        chunk_data = json.load(f)

    # Create mock DocChunk entity with sample vector
    sample_vector = [0.1] * 768  # Mock embedding vector
    chunk = DocChunk(
        text=chunk_data["text"], meta=chunk_data["meta"], vector=sample_vector
    )

    return document, texts, pictures, tables, [chunk]


async def index_documents(
    indexing_use_case: DocumentIndexingUseCase,
) -> None:
    """Index documents using dependency injection."""
    print("🔄 Indexing documents...")

    # Load sample data
    document, texts, pictures, tables, sample_chunks = await load_sample_data()
    print("✅ Sample data loaded")

    # Index the document with all elements and chunks
    result = await indexing_use_case.bulk_index_document_with_elements_and_chunks(
        document=document,
        texts=texts,
        pictures=pictures,
        tables=tables,
        chunks=sample_chunks,
        document_id="sample_doc_1",
        index_name="example_index",
        generate_embeddings=False,  # Using mock vectors
    )

    print(f"✅ Indexed document and chunks: {result.total_indexed} items")
    if result.errors:
        print(f"❌ Errors: {result.errors}")


async def search_documents(
    search_use_case: DocumentSearchUseCase,
) -> None:
    """Search documents using dependency injection."""
    print("🔍 Searching documents...")

    # Search chunks by text
    search_results = await search_use_case.search_chunks_by_text(
        query="occupational disease", size=5, index_name="example_index"
    )

    print(f"📝 Text search results: {len(search_results)} chunks found")
    for chunk in search_results[:2]:
        print(f"  Text preview: {chunk.text[:200]}...")
        print(f"  Headings: {chunk.meta.headings}")
        print()

    # Get the original document
    doc_result = await search_use_case.get_document(
        document_id="sample_doc_1", index_name="example_index"
    )

    if doc_result:
        print("✅ Retrieved original document")
        print(f"  Document name: {doc_result.name}")
        print(
            f"  Document origin: {doc_result.origin.filename if doc_result.origin else 'N/A'}"
        )
    else:
        print("❌ Could not retrieve original document")

    # Search for documents
    doc_search_results = await search_use_case.search_documents(
        query="comprehensive review", size=5, index_name="example_index"
    )

    print(f"📄 Document search results: {len(doc_search_results)} documents found")
    for hit in doc_search_results.hits:
        document_data = hit.source.get("document", {})
        print(f"  Document: {document_data.get('name', 'N/A')}")
        origin = document_data.get("origin", {})
        print(f"  Origin: {origin.get('filename', 'N/A') if origin else 'N/A'}")


async def initialize_system(
    container: ApplicationContainer,
) -> None:
    """Initialize the system with dependency injection."""
    print("🚀 Initializing MultiModal RAG system with dependency injection...")

    # Get the document repository to initialize indices
    document_repository = await container.document_repository()
    await document_repository.initialize_indices()
    print("✅ Elasticsearch index initialized")


async def main():
    """Main example function demonstrating dependency injection approach."""
    # Create and configure container
    container = ApplicationContainer()

    try:
        # Initialize system
        await initialize_system(container)

        # Get dependencies from container
        indexing_use_case = await container.document_indexing_use_case()
        search_use_case = await container.document_search_use_case()

        # Index documents
        await index_documents(indexing_use_case)

        # Search documents
        await search_documents(search_use_case)

        print("\n🎉 Example completed successfully!")
        print("\nKey benefits demonstrated:")
        print("- Configuration from environment variables")
        print("- Explicit dependency resolution")
        print("- Clean separation of concerns")
        print("- Easy testing with mock overrides")
        print("- Singleton management for expensive resources")
        print("- Direct entity passing between layers for internal operations")
        print("- Still maintains clean architecture principles")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
