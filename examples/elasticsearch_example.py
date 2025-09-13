"""Example usage of the simplified Elasticsearch adaptor."""

import asyncio
import json
from pathlib import Path
from typing import List

from elasticsearch import AsyncElasticsearch
from docling_core.types.doc.document import DoclingDocument as DLDocument

from multimodal_rag.entities.document import DoclingDocument, DocChunk
from multimodal_rag.adaptors.elasticsearch_adaptor import ElasticsearchDocumentAdaptor
from multimodal_rag.usecases.document_indexing import DocumentIndexingUseCase, DocumentSearchUseCase
from multimodal_rag.usecases.dtos import SearchRequest


async def load_sample_data() -> tuple[DoclingDocument, List[DocChunk]]:
    """Load sample data from JSON files."""
    # Load sample DoclingDocument
    doc_path = Path("data/docling_document.json")
    with open(doc_path, "r") as f:
        doc_data = json.load(f)
    
    # Convert to our entity
    dl_doc = DLDocument.model_validate(doc_data)
    document = DoclingDocument.from_docling(dl_doc)
    
    # Load sample DocChunk
    chunk_path = Path("data/doc_chunk.json")
    with open(chunk_path, "r") as f:
        chunk_data = json.load(f)
    
    # Create mock DocChunk entity with sample vector
    sample_vector = [0.1] * 1536  # Mock embedding vector
    chunk = DocChunk(
        text=chunk_data["text"],
        meta=chunk_data["meta"],
        vector=sample_vector
    )
    
    return document, [chunk]


async def main():
    """Main example function demonstrating simplified approach."""
    # Initialize Elasticsearch client
    es_client = AsyncElasticsearch(
        hosts=["http://localhost:9200"],
        # Add authentication if needed
        # http_auth=("username", "password"),
        # verify_certs=False,
    )
    
    try:
        # Initialize the adaptor - now simplified!
        adaptor = ElasticsearchDocumentAdaptor(
            elasticsearch_client=es_client,
            index_name="example_unified_index",
            vector_dimensions=1536
        )
        
        # Initialize index
        await adaptor.initialize_indices()
        print("✅ Elasticsearch unified index initialized")
        
        # Initialize use cases - much simpler now!
        indexing_use_case = DocumentIndexingUseCase(
            document_repository=adaptor,
            embedding_service=None  # You would inject an actual embedding service here
        )
        
        search_use_case = DocumentSearchUseCase(
            document_repository=adaptor,
            embedding_service=None
        )
        
        # Load sample data
        document, sample_chunks = await load_sample_data()
        print("✅ Sample data loaded")
        
        # Index the document with chunks - direct entity passing!
        result = await indexing_use_case.bulk_index_document_with_chunks(
            document=document,
            chunks=sample_chunks,
            document_id="sample_doc_1",
            index_name="example_unified_index",
            generate_embeddings=False  # Using mock vectors
        )
        
        print(f"✅ Indexed document and chunks: {result.total_indexed} items")
        if result.errors:
            print(f"Errors: {result.errors}")
        
        # Search chunks by text - using SearchRequest DTO for external API
        search_request = SearchRequest(
            query="occupational disease",
            size=5,
            index_name="example_unified_index"
        )
        
        search_results = await search_use_case.search_chunks_by_text(search_request)
        
        print(f"\\n📝 Text search results: {len(search_results.hits)} hits")
        for hit in search_results.hits[:2]:
            print(f"  Score: {hit.score:.3f}")
            print(f"  Type: {hit.source.get('type', 'unknown')}")
            chunk_data = hit.source.get('chunk', {})
            print(f"  Text preview: {chunk_data.get('text', 'N/A')[:200]}...")
            print(f"  Document ID: {chunk_data.get('document_id', 'N/A')}")
            print()
        
        # Get the original document - direct entity usage!
        doc_result = await search_use_case.get_document(
            document_id="sample_doc_1",
            index_name="example_unified_index"
        )
        
        if doc_result.found:
            print("✅ Retrieved original document")
            document_data = doc_result.source.get('document', {})
            print(f"  Document type: {doc_result.source.get('type', 'unknown')}")
            print(f"  Document name: {document_data.get('name', 'N/A')}")
            print(f"  Document origin: {document_data.get('origin', {}).get('filename', 'N/A')}")
        else:
            print("❌ Could not retrieve original document")
        
        # Search documents by text
        doc_search_request = SearchRequest(
            query="comprehensive review",
            size=5,
            index_name="example_unified_index"
        )
        
        doc_search_results = await search_use_case.search_documents_by_text(doc_search_request)
        
        print(f"\\n📄 Document search results: {len(doc_search_results.hits)} documents found")
        for hit in doc_search_results.hits:
            document_data = hit.source.get('document', {})
            print(f"  Document: {document_data.get('name', 'N/A')}")
            origin = document_data.get('origin', {})
            print(f"  Origin: {origin.get('filename', 'N/A') if origin else 'N/A'}")
        
        # Vector search example (using same vector as mock)
        vector_search_request = SearchRequest(
            vector=sample_chunks[0].vector,
            size=3,
            index_name="example_unified_index"
        )
        
        vector_results = await search_use_case.search_chunks_by_vector(vector_search_request)
        print(f"\\n🔍 Vector search results: {len(vector_results.hits)} hits")
        for hit in vector_results.hits[:1]:
            chunk_data = hit.source.get('chunk', {})
            print(f"  Score: {hit.score:.3f}")
            print(f"  Text preview: {chunk_data.get('text', 'N/A')[:100]}...")
        
        print("\\n🎉 Example completed successfully!")
        print("\\nKey benefits of the simplified approach:")
        print("- Direct entity passing between layers for internal operations")
        print("- DTOs only used for external API boundaries (SearchRequest/Response)")
        print("- Cleaner, more readable code")
        print("- Less ceremony, more business value")
        print("- Still maintains clean architecture principles")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Close the Elasticsearch client
        await es_client.close()


if __name__ == "__main__":
    asyncio.run(main())
