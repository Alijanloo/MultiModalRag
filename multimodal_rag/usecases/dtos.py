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


class IndexTextResponse(BaseModel):
    """DTO for indexing a text element response."""
    
    text_id: str
    success: bool
    message: Optional[str] = None


class IndexPictureResponse(BaseModel):
    """DTO for indexing a picture element response."""
    
    picture_id: str
    success: bool
    message: Optional[str] = None


class IndexTableResponse(BaseModel):
    """DTO for indexing a table element response."""
    
    table_id: str
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


class GetDocumentResponse(BaseModel):
    """DTO for getting a document response."""
    
    document_id: str
    found: bool
    source: Optional[Dict[str, Any]] = None
