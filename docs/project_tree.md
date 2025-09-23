# MultiModal RAG Project Structure

## Overview
This project implements a Clean Architecture approach for a Multi-Modal RAG system with Elasticsearch integration for document and chunk indexing, and Google GenAI services for embeddings and LLM operations.

## Current Implementation

### Core Components

#### Entities (`multimodal_rag/entities/`)
- `document.py`: Core domain entities
  - `DoclingDocument`: Complete document entity from Docling
  - `DocChunk`: Document chunk entity with vector embeddings
  - `DocMeta`: Metadata for document chunks
  - `DocumentOrigin`: Document origin information
  - `DocumentText`, `DocumentPicture`, `DocumentTable`: Specialized document elements

#### Use Cases (`multimodal_rag/usecases/`)
- `document_indexing.py`: Business logic for document indexing
  - `DocumentIndexingUseCase`: Handles document and chunk indexing
- `document_search.py`: Business logic for document search operations
  - `DocumentSearchUseCase`: Handles search and retrieval operations
- `langgraph_agent/`: Agentic RAG implementation using LangGraph
  - `agentic_rag.py`: Main agentic RAG use case with intelligent retrieval
  - `dtos.py`: Data Transfer Objects for agentic RAG operations
- `dtos.py`: Data Transfer Objects for external API boundaries
  - `SearchRequest`: External search API request
  - Response DTOs: `SearchResponse`, `IndexDocumentResponse`, etc.
- `interfaces/`: Abstract interfaces for repositories and services
  - `document_repository.py`: Repository and embedding service interfaces
  - `embedding_service.py`: Interface for embedding generation services
  - `llm_service.py`: Interface for Large Language Model services

#### Adaptors (`multimodal_rag/adaptors/`)
- `elasticsearch_adaptor.py`: Elasticsearch implementation 
  - Direct entity passing for internal operations
  - Methods accept entities like `DocChunk`, `DoclingDocument` directly
  - Returns DTO responses for consistency with external boundaries
  - Separate field structures for documents (`document.*`) and chunks (`chunk.*`)
  - Vector similarity search support
  - Hybrid text and vector search capabilities

#### Frameworks (`multimodal_rag/frameworks/`)
- `elasticsearch_config.py`: Configuration for Elasticsearch
- `logging_config.py`: Logging configuration
- `google_genai_embedding_service.py`: Google GenAI embedding service implementation
- `google_genai_llm_service.py`: Google GenAI LLM service implementation

#### Config (`multimodal_rag/config/`)
- Configuration and settings objects for the application
- Centralized location for application configuration classes
- Environment-specific settings and constants


### Key Features

1. **Clean Architecture**: Separation of concerns with dependency inversion
2. **Type Safety**: Pydantic models throughout for validation
3. **Vector Search**: Dense vector support for semantic similarity with Google GenAI embeddings
4. **Hybrid Search**: Combined text and vector search capabilities
5. **Document References**: Chunks maintain references to parent documents
6. **LLM Integration**: Google GenAI for content generation and Q&A
7. **Agentic RAG**: Intelligent document retrieval with LangGraph workflow
8. **Picture Retrieval**: Automatic retrieval of associated pictures from document chunks
9. **Dependency Injection**: Proper DI container setup for all services
10. **Conversation Management**: Chat session handling with context preservation

### Usage Examples

- `examples/elasticsearch_example.py`: Usage demonstration with simplified internal operations
- `examples/agentic_rag_example.py`: Complete agentic RAG workflow demonstration
- `tests/integration/test_elasticsearch_adaptor.py`: Test cases

### Dependencies

- **Elasticsearch**: Document storage and search
- **Docling Core**: Document processing and chunking
- **Pydantic**: Data validation and serialization
- **AsyncIO**: Asynchronous operations
- **Google GenAI**: Embedding generation and LLM services
- **Dependency Injector**: Dependency injection framework
- **LangGraph**: Workflow orchestration for agentic RAG
- **LangChain**: Tool creation and message handling

## Next Steps

1. ✅ Implement embedding service integration (Google GenAI)
2. ✅ Add LLM service for content generation  
3. ✅ Add agentic RAG pipeline with intelligent retrieval and picture support
4. Add more sophisticated search ranking
5. Implement document relationship mapping
6. Add monitoring and metrics
7. Optimize index performance for large datasets
8. Add conversation persistence and session management