"""DTOs for document indexing and retrieval operations."""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class IndexDocumentResponse(BaseModel):
    """DTO for indexing a document response."""
    
    document_id: str
    success: bool
    message: Optional[str] = None


class IndexChunkResponse(BaseModel):
    """DTO for indexing a chunk response."""
    
    chunk_id: str
    success: bool
    message: Optional[str] = None


class SearchRequest(BaseModel):
    """DTO for search request."""
    
    query: Optional[str] = None
    vector: Optional[List[float]] = None
    filters: Optional[Dict[str, Any]] = None
    size: int = Field(default=10, ge=1, le=100)
    index_name: Optional[str] = None


class SearchHit(BaseModel):
    """DTO for a single search result."""
    
    id: str
    score: float
    source: Dict[str, Any]
    highlight: Optional[Dict[str, List[str]]] = None


class SearchResponse(BaseModel):
    """DTO for search response."""
    
    hits: List[SearchHit]
    total: int
    max_score: Optional[float] = None


class BulkIndexResponse(BaseModel):
    """DTO for bulk indexing response."""
    
    document_responses: List[IndexDocumentResponse] = Field(default_factory=list)
    chunk_responses: List[IndexChunkResponse] = Field(default_factory=list)
    total_indexed: int
    errors: List[str] = Field(default_factory=list)


class GetDocumentResponse(BaseModel):
    """DTO for getting a document response."""
    
    document_id: str
    found: bool
    source: Optional[Dict[str, Any]] = None
