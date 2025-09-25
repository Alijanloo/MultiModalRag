"""Data Transfer Objects for agentic RAG operations."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

from multimodal_rag.entities.document import DocChunk, DocumentPicture


class ChatMessage(BaseModel):
    """Chat message representation."""
    
    role: str = Field(description="Role of the message sender (user, assistant, system)")
    content: str = Field(description="Content of the message")
    chat_id: Optional[str] = Field(default=None, description="Chat session identifier")


class ToolCall(BaseModel):
    """Tool call representation."""
    
    id: str = Field(description="Unique identifier for the tool call")
    name: str = Field(description="Name of the tool being called")
    args: Dict[str, Any] = Field(description="Arguments passed to the tool")


class RetrievalResult(BaseModel):
    """Result from document retrieval operation."""
    
    chunks: List[DocChunk] = Field(default_factory=list, description="Retrieved document chunks")
    pictures: List[DocumentPicture] = Field(default_factory=list, description="Associated pictures from chunks")
    total_results: int = Field(default=0, description="Total number of results found")


class GradeDocuments(BaseModel):
    """Binary relevance scoring for retrieved documents."""
    
    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )


class AgentResponse(BaseModel):
    """Response from the agentic RAG system."""
    
    content: str = Field(description="Generated response content")
    retrieved_chunks: List[DocChunk] = Field(default_factory=list, description="Document chunks passed to generator.")
    chunk_ids_used: List[str] = Field(default_factory=list, description="IDs of chunks referenced in the response")
    pictures: List[DocumentPicture] = Field(default_factory=list, description="Pictures associated with the response")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class StructuredAgentResponse(BaseModel):
    """Structured response from the agent with chunk references."""
    
    answer: str = Field(description="The main answer content")
    chunk_ids_used: List[str] = Field(default_factory=list, description="List of chunk IDs used to generate this answer")


class ConversationState(BaseModel):
    """State of the conversation in the agent workflow."""
    
    messages: List[ChatMessage] = Field(default_factory=list, description="Conversation messages")
    chat_id: Optional[str] = Field(default=None, description="Chat session identifier")
    retrieval_context: Optional[RetrievalResult] = Field(default=None, description="Current retrieval context")
    current_chunks: Dict[str, DocChunk] = Field(default_factory=dict, description="Currently available chunks by ID")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional state metadata")
