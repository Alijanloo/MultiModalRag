# Dependency Injection System

This document describes the dependency injection (DI) system implemented in the MultiModal RAG project using the `dependency-injector` library.

## Overview

The dependency injection container provides centralized configuration and dependency management for the entire application, making it easier to:

- Configure services through environment variables or configuration files
- Mock dependencies for testing
- Manage singleton instances of expensive resources
- Maintain clean separation of concerns

## Dependency Tree

```
ApplicationContainer
├── Configuration (from env/files)
├── LoggerFactory (Singleton)
├── ElasticsearchConfig (Singleton)
├── AsyncElasticsearch (Singleton)
├── ElasticsearchDocumentAdaptor (Singleton)
├── EmbeddingService (Factory - placeholder)
├── DocumentIndexingUseCase (Factory)
└── DocumentSearchUseCase (Factory)
```

## Usage Examples

### TestContainer

A separate test container provides mocked dependencies for testing:

```python
from multimodal_rag.containers import TestContainer

test_container = TestContainer()
# All dependencies are mocked for isolated testing
```

### Basic Usage with Injection

```python
from dependency_injector.wiring import Provide, inject
from multimodal_rag.containers import ApplicationContainer
from multimodal_rag.usecases.document_indexing import DocumentIndexingUseCase

@inject
async def index_document(
    indexing_use_case: DocumentIndexingUseCase = Provide[ApplicationContainer.document_indexing_use_case],
    document: DoclingDocument,
    document_id: str
) -> None:
    result = await indexing_use_case.index_document(document, document_id)
    print(f"Indexed: {result.success}")

# Setup container and wire
container = ApplicationContainer()
container.wire(modules=[__name__])

# Function automatically gets dependencies injected
await index_document(my_document, "doc_123")
```

### Testing with Mocks

```python
import unittest.mock as mock
from multimodal_rag.containers import ApplicationContainer

def test_document_indexing():
    container = ApplicationContainer()
    
    # Mock the repository
    mock_repository = mock.AsyncMock()
    mock_repository.index_document.return_value = IndexDocumentResponse(
        document_id="test", success=True, message="Success"
    )
    
    # Override dependency
    with container.document_repository.override(mock_repository):
        container.wire(modules=[__name__])
        
        # Test your code - mock will be injected automatically
        indexing_use_case = container.document_indexing_use_case()
        # ... test logic
```

## Advanced Usage

### Custom Providers

You can extend the container with custom providers:

```python
from multimodal_rag.containers import ApplicationContainer
from dependency_injector import providers

class CustomContainer(ApplicationContainer):
    # Add your custom services
    custom_service = providers.Singleton(
        MyCustomService,
        dependency=super().some_dependency
    )
```

### Runtime Configuration

```python
# Change configuration at runtime
with container.config.elasticsearch.index_name.override("temporary_index"):
    # Code using temporary index
    pass

# Override dependencies for specific operations
with container.document_repository.override(special_repository):
    # Code using special repository
    pass
```

### Async Context Management

```python
async def application_lifecycle():
    container = ApplicationContainer()
    container.wire(modules=[__name__])
    
    try:
        # Application startup
        await initialize_system()
        
        # Application logic
        await run_application()
        
    finally:
        # Cleanup
        es_client = await container.elasticsearch_client()
        await es_client.close()
        container.unwire()
```

## Integration Examples

### FastAPI Integration

```python
from fastapi import FastAPI, Depends
from dependency_injector.wiring import Provide

app = FastAPI()
container = ApplicationContainer()
container.wire(modules=[__name__])

@app.get("/search")
async def search_endpoint(
    query: str,
    search_use_case: DocumentSearchUseCase = Depends(Provide[ApplicationContainer.document_search_use_case])
):
    results = await search_use_case.search_chunks_by_text(query)
    return results

@app.on_event("startup")
async def startup():
    # Initialize resources
    pass

@app.on_event("shutdown")
async def shutdown():
    await container.elasticsearch_client().close()
    container.unwire()
```

### CLI Application Integration

```python
import click
from dependency_injector.wiring import Provide, inject

@click.command()
@click.option('--query', required=True)
@inject
async def search_cli(
    query: str,
    search_use_case: DocumentSearchUseCase = Provide[ApplicationContainer.document_search_use_case]
):
    results = await search_use_case.search_chunks_by_text(query)
    click.echo(f"Found {len(results.hits)} results")

if __name__ == "__main__":
    container = ApplicationContainer()
    container.wire(modules=[__name__])
    search_cli()
```

## Common Provider Types

1. **Factory**: Creates new instance each time
2. **Singleton**: Creates one instance, reuses it
3. **Resource**: Manages lifecycle of expensive objects
4. **Configuration**: Provides configuration values
5. **Callable**: Wraps functions/callables

The key insight is that `Resource` is specifically designed for objects that need both creation AND cleanup, making it perfect for database connections, file handles, network clients, etc.

## Best Practices

1. **Wire Early**: Wire the container at application startup
2. **Use Factories for Stateful Objects**: Use `providers.Factory` for objects that maintain state
3. **Use Singletons for Expensive Resources**: Database connections, clients, etc.
4. **Environment-Based Configuration**: Use environment variables for deployment flexibility
5. **Clean Shutdown**: Always unwire containers and close resources
6. **Test with Mocks**: Use dependency overrides for isolated testing
7. **Configuration Validation**: Validate configuration early in the application lifecycle

## Troubleshooting

### Common Issues

1. **Circular Dependencies**: Ensure no circular imports between modules
2. **Wiring Modules**: Make sure to wire all modules that use injection
3. **Async Dependencies**: Use proper async/await patterns with async providers
4. **Configuration Errors**: Validate configuration values and provide sensible defaults

### Debug Mode

Enable debug logging to see dependency resolution:

```python
import logging
logging.getLogger("dependency_injector").setLevel(logging.DEBUG)
```

This comprehensive dependency injection system provides a robust foundation for managing dependencies throughout the MultiModal RAG application while maintaining testability and flexibility.
