# Agentic RAG with LangGraph

This module implements an agentic RAG (Retrieval-Augmented Generation) system using LangGraph. The system can intelligently decide when to retrieve documents and when to respond directly, providing a more conversational and context-aware experience.

## Features

- **Intelligent Decision Making**: Uses LLM to decide whether to retrieve documents or respond directly
- **Document Grading**: Evaluates retrieved documents for relevance before generating responses
- **Question Rewriting**: Improves queries that don't return relevant results
- **Picture Retrieval**: Automatically retrieves associated pictures when chunks contain picture references
- **Conversation Management**: Maintains conversation state and context
- **Error Handling**: Robust error handling throughout the workflow

## Architecture

The agentic RAG system follows a graph-based workflow:

1. **Agent Node**: Decides whether to retrieve documents or respond directly
2. **Retrieval Tool**: Searches for relevant documents using hybrid search
3. **Document Grading**: Evaluates relevance of retrieved documents
4. **Question Rewriting**: Improves questions that don't return relevant results
5. **Answer Generation**: Generates final response using retrieved context

## Workflow

```
User Query → Agent Decision → [Retrieve?] → Grade Documents → [Relevant?] → Generate Answer
                    ↓                              ↓
              Direct Response              Rewrite Question → Loop Back
```

## Usage

### Basic Usage

```python
from multimodal_rag.container import ApplicationContainer
from multimodal_rag.usecases.langgraph_agent.factory import AgenticRAGFactory

# Initialize container
container = ApplicationContainer()
container.config.from_yaml("config.yaml")
await container.init_resources()

# Create agentic RAG instance
agentic_rag = AgenticRAGFactory.create_agentic_rag(
    document_repository=container.document_repository(),
    embedding_service=container.embedding_service(),
    llm_service=container.llm_service(),
    retrieval_size=5,
)

# Process a message
response = await agentic_rag.process_message(
    message="What is machine learning?",
    chat_id="chat_123"
)

print(f"Response: {response.content}")
print(f"Pictures: {len(response.pictures)}")
print(f"Chunks used: {len(response.chunks_used)}")
```

### With Conversation History

```python
from multimodal_rag.usecases.langgraph_agent.dtos import ChatMessage

conversation_history = [
    ChatMessage(role="user", content="Hello"),
    ChatMessage(role="assistant", content="Hi! How can I help?"),
]

response = await agentic_rag.process_message(
    message="Tell me about neural networks",
    chat_id="chat_123",
    conversation_history=conversation_history
)
```

## Data Transfer Objects (DTOs)

### ChatMessage
Represents a message in the conversation.

### AgentResponse
Contains the agent's response with associated documents and pictures.

### ConversationState
Manages the state of a conversation session.

## Dependencies

- **LangGraph**: For workflow orchestration
- **LangChain**: For tool creation and message handling
- **Elasticsearch**: For document storage and retrieval
- **Google GenAI**: For embeddings and LLM operations

## Example

See `examples/agentic_rag_example.py` for a complete working example.
