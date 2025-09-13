# MultiModal RAG Project Structure

## Overview
This project implements a Clean Architecture approach for a Multi-Modal RAG system with Elasticsearch integration for document and chunk indexing.

## Current Implementation

### Core Components

#### Entities (`multimodal_rag/entities/`)
- `document.py`: Core domain entities
  - `DoclingDocument`: Complete document entity from Docling
  - `DocChunk`: Document chunk entity with vector embeddings
  - `DocMeta`: Metadata for document chunks
  - `DocumentOrigin`: Document origin information

#### Use Cases (`multimodal_rag/usecases/`)
- `document_indexing.py`: Business logic for document operations
  - `DocumentIndexingUseCase`: Handles document and chunk indexing
  - `DocumentSearchUseCase`: Handles search and retrieval operations
- `dtos.py`: Data Transfer Objects for external API boundaries
  - `SearchRequest`: External search API request
  - Response DTOs: `SearchResponse`, `IndexDocumentResponse`, etc.
- `interfaces/`: Abstract interfaces for repositories and services
  - `document_repository.py`: Repository and embedding service interfaces

#### Adaptors (`multimodal_rag/adaptors/`)
- `elasticsearch_adaptor.py`: Elasticsearch implementation 
  - Direct entity passing for internal operations
  - Methods accept entities like `DocChunk`, `DoclingDocument` directly
  - Returns DTO responses for consistency with external boundaries
  - Unified index approach storing both documents and chunks
  - Separate field structures for documents (`document.*`) and chunks (`chunk.*`)
  - Vector similarity search support
  - Hybrid text and vector search capabilities

#### Frameworks (`multimodal_rag/frameworks/`)
- `elasticsearch_config.py`: Configuration for Elasticsearch
- `logging_config.py`: Logging configuration

### Index Structure

The Elasticsearch adaptor uses a **unified index** approach where both documents and chunks are stored in the same index with:

- **Common fields**: `type`, `created_at`, `updated_at`
- **Document fields**: Nested under `document.*`
  - `document.name`, `document.origin`, `document.schema_name`, etc.
  - Complete document structure preserved
- **Chunk fields**: Nested under `chunk.*`
  - `chunk.text`, `chunk.meta`, `chunk.vector`, `chunk.document_id`
  - Vector embeddings for semantic search
  - Reference to parent document via `chunk.document_id`

### Key Features

1. **Clean Architecture**: Separation of concerns with dependency inversion
2. **Unified Index**: Single Elasticsearch index for both document types
3. **Type Safety**: Pydantic models throughout for validation
4. **Vector Search**: Dense vector support for semantic similarity
5. **Hybrid Search**: Combined text and vector search capabilities
6. **Document References**: Chunks maintain references to parent documents

### Usage Examples

- `examples/elasticsearch_example.py`: Usage demonstration with simplified internal operations
- `tests/integration/test_elasticsearch_adaptor.py`: Test cases

### Dependencies

- **Elasticsearch**: Document storage and search
- **Docling Core**: Document processing and chunking
- **Pydantic**: Data validation and serialization
- **AsyncIO**: Asynchronous operations

## Next Steps

1. Implement actual embedding service integration
2. Add more sophisticated search ranking
3. Implement document relationship mapping
4. Add monitoring and metrics
5. Optimize index performance for large datasets