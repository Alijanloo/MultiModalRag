"""Unit tests for the Agentic RAG use case."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from multimodal_rag.usecases.langgraph_agent.agentic_rag import AgenticRAGUseCase
from multimodal_rag.usecases.langgraph_agent.dtos import ChatMessage, AgentResponse
from multimodal_rag.entities.document import DocChunk, DocumentPicture, DocMeta


class TestAgenticRAGUseCase:
    """Test cases for the Agentic RAG use case."""

    @pytest.fixture
    def mock_document_repository(self):
        """Mock document repository."""
        repo = AsyncMock()
        repo.search_chunks.return_value = [
            DocChunk(
                text="This is a test document about machine learning.",
                meta=DocMeta(
                    schema_name="test_schema",
                    version="1.0",
                    doc_items=[],
                    headings=["Introduction to ML"]
                ),
                vector=[0.1] * 1536,
                document_id="doc_1",
            )
        ]
        repo.get_picture.return_value = DocumentPicture(
            picture_id="pic_1",
            document_id="doc_1",
            label="ML Architecture Diagram",
            captions=["A diagram showing neural network architecture"],
        )
        return repo

    @pytest.fixture
    def mock_embedding_service(self):
        """Mock embedding service."""
        service = AsyncMock()
        service.generate_embedding.return_value = [0.1] * 1536
        return service

    @pytest.fixture
    def mock_llm_service(self):
        """Mock LLM service."""
        service = AsyncMock()
        service.generate_content.return_value = "This is a generated response about machine learning."
        # Mock the tool calling method to return proper structure
        service.generate_content_with_tools.return_value = {
            "text": "Hello! How can I help you today?",
            "function_calls": [],
            "has_function_call": False
        }
        return service

    @pytest.fixture
    def agentic_rag_use_case(
        self, mock_document_repository, mock_embedding_service, mock_llm_service
    ):
        """Create AgenticRAGUseCase instance with mocked dependencies."""
        return AgenticRAGUseCase(
            document_repository=mock_document_repository,
            embedding_service=mock_embedding_service,
            llm_service=mock_llm_service,
            retrieval_size=5,
        )

    @pytest.mark.asyncio
    async def test_process_simple_greeting(
        self, agentic_rag_use_case, mock_llm_service
    ):
        """Test processing a simple greeting that doesn't require retrieval."""
        mock_llm_service.generate_content.return_value = (
            "Hello! How can I help you today?"
        )

        response = await agentic_rag_use_case.process_message("Hello!")

        assert isinstance(response, AgentResponse)
        assert "Hello" in response.content
        assert len(response.chunks_used) == 0
        assert len(response.pictures) == 0

    @pytest.mark.asyncio
    async def test_process_question_requiring_retrieval(
        self, agentic_rag_use_case, mock_document_repository
    ):
        """Test processing a question that requires document retrieval."""
        response = await agentic_rag_use_case.process_message(
            "What is machine learning?"
        )

        assert isinstance(response, AgentResponse)
        assert response.content is not None
        mock_document_repository.search_chunks.assert_called()

    @pytest.mark.asyncio
    async def test_process_with_chat_id(self, agentic_rag_use_case):
        """Test processing a message with a chat ID."""
        chat_id = "test_chat_123"

        response = await agentic_rag_use_case.process_message(
            "Tell me about neural networks", chat_id=chat_id
        )

        assert response.chat_id == chat_id

    @pytest.mark.asyncio
    async def test_process_with_conversation_history(self, agentic_rag_use_case):
        """Test processing a message with conversation history."""
        history = [
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="assistant", content="Hi! How can I help?"),
        ]

        response = await agentic_rag_use_case.process_message(
            "What is deep learning?", conversation_history=history
        )

        assert isinstance(response, AgentResponse)

    @pytest.mark.asyncio
    async def test_error_handling(
        self, mock_document_repository, mock_embedding_service, mock_llm_service
    ):
        """Test error handling in the agentic RAG system."""
        # Mock an error in the embedding service
        mock_embedding_service.generate_embedding.side_effect = Exception(
            "Embedding service error"
        )

        use_case = AgenticRAGUseCase(
            document_repository=mock_document_repository,
            embedding_service=mock_embedding_service,
            llm_service=mock_llm_service,
        )

        response = await use_case.process_message("What is AI?")

        assert isinstance(response, AgentResponse)
        assert (
            "error" in response.content.lower()
            or "apologize" in response.content.lower()
        )

    @pytest.mark.asyncio
    async def test_conversation_state_methods(self, agentic_rag_use_case):
        """Test conversation state management methods."""
        # These are placeholder implementations, so they should return expected defaults
        state = await agentic_rag_use_case.get_conversation_state("test_chat")
        assert state is None  # Default implementation returns None

        success = await agentic_rag_use_case.save_conversation_state(MagicMock())
        assert success is True  # Default implementation returns True
