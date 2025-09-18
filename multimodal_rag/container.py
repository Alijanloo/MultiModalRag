"""Application dependency injection container."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from dependency_injector import containers, providers
from elasticsearch import AsyncElasticsearch
from elastic_transport.client_utils import DEFAULT

from multimodal_rag.adaptors.elasticsearch.elasticsearch_config import ElasticsearchConfig
from multimodal_rag.frameworks.logging_config import LoggerFactory
from multimodal_rag.frameworks.google_genai_embedding_service import GoogleGenAIEmbeddingService
from multimodal_rag.frameworks.google_genai_llm_service import GoogleGenAILLMService
from multimodal_rag.adaptors.elasticsearch.elasticsearch_adaptor import ElasticsearchDocumentAdaptor
from multimodal_rag.usecases.document_indexing import DocumentIndexingUseCase
from multimodal_rag.usecases.document_search import DocumentSearchUseCase


@asynccontextmanager
async def elasticsearch_client_resource(
    hosts,
    basic_auth=None,
    verify_certs=True,
    ca_certs=DEFAULT,
) -> AsyncGenerator[AsyncElasticsearch, None]:
    """Create and manage Elasticsearch client lifecycle."""
    client = AsyncElasticsearch(
        hosts=hosts,
        basic_auth=basic_auth,
        verify_certs=verify_certs,
        ca_certs=ca_certs,
    )
    try:
        yield client
    finally:
        await client.close()


class ApplicationContainer(containers.DeclarativeContainer):
    """Main dependency injection container for the multimodal RAG application."""

    # Configuration
    config = providers.Configuration(yaml_files=["config.yaml"])

    # Logging
    logger_factory = providers.Singleton(LoggerFactory)

    # Elasticsearch Configuration
    elasticsearch_config = providers.Singleton(
        ElasticsearchConfig,
        hosts=config.elasticsearch.hosts,
        username=config.elasticsearch.username,
        password=config.elasticsearch.password,
        verify_certs=config.elasticsearch.verify_certs,
        ca_certs=config.elasticsearch.ca_certs,
        index_name=config.elasticsearch.index_name,
        vector_dimensions=config.elasticsearch.vector_dimensions,
        shards=config.elasticsearch.shards,
        replicas=config.elasticsearch.replicas,
        default_search_size=config.elasticsearch.default_search_size,
        max_search_size=config.elasticsearch.max_search_size,
    )

    # Elasticsearch Client
    elasticsearch_client = providers.Resource(
        elasticsearch_client_resource,
        hosts=elasticsearch_config.provided.hosts,
        basic_auth=providers.Callable(
            lambda username, password: (username, password)
            if username and password
            else None,
            elasticsearch_config.provided.username,
            elasticsearch_config.provided.password,
        ),
        verify_certs=elasticsearch_config.provided.verify_certs,
        ca_certs=elasticsearch_config.provided.ca_certs or DEFAULT,
    )

    # Repository Layer
    document_repository = providers.Singleton(
        ElasticsearchDocumentAdaptor,
        elasticsearch_client=elasticsearch_client,
        index_name=elasticsearch_config.provided.index_name,
        vector_dimensions=elasticsearch_config.provided.vector_dimensions,
    )

    # Embedding Service
    embedding_service = providers.Factory(
        GoogleGenAIEmbeddingService,
        api_keys=config.google_genai.api_keys,
        model=config.google_genai.embedding_model,
        embedding_dimensions=config.google_genai.embedding_dimensions,
    )

    # LLM Service for content generation
    llm_service = providers.Factory(
        GoogleGenAILLMService,
        api_keys=config.google_genai.api_keys,
        default_model=config.google_genai.default_llm_model,
    )

    # Use Cases
    document_indexing_use_case = providers.Factory(
        DocumentIndexingUseCase,
        document_repository=document_repository,
        embedding_service=embedding_service,
    )

    document_search_use_case = providers.Factory(
        DocumentSearchUseCase,
        document_repository=document_repository,
        embedding_service=embedding_service,
    )


class TestContainer(containers.DeclarativeContainer):
    """Test-specific dependency injection container with mocked dependencies."""

    # Configuration with test defaults
    config = providers.Configuration()

    # Mock Elasticsearch client
    elasticsearch_client = providers.Factory(
        lambda: None,  # Mock AsyncElasticsearch for testing
    )

    # Mock Elasticsearch config
    elasticsearch_config = providers.Singleton(
        ElasticsearchConfig,
        hosts=["http://localhost:9200"],
        index_name="test_multimodal_index",
        vector_dimensions=384,  # Smaller for testing
    )

    # Mock repository
    document_repository = providers.Factory(
        lambda: None,  # Mock repository for testing
    )

    # Mock embedding service
    embedding_service = providers.Factory(
        lambda: None,  # Mock embedding service for testing
    )

    # Use cases with mocked dependencies
    document_indexing_use_case = providers.Factory(
        DocumentIndexingUseCase,
        document_repository=document_repository,
        embedding_service=embedding_service,
    )

    document_search_use_case = providers.Factory(
        DocumentSearchUseCase,
        document_repository=document_repository,
        embedding_service=embedding_service,
    )
