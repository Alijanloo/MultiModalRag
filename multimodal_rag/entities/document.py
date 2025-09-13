"""Document entities for the multimodal RAG system."""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from docling_core.types.doc.document import DoclingDocument as DLDocument
from docling_core.transforms.chunker.hierarchical_chunker import DocChunk as DLDocChunk


class DocumentOrigin(BaseModel):
    """Document origin information."""
    
    mimetype: str
    binary_hash: int
    filename: str


class DoclingDocument(BaseModel):
    """Entity representing a complete document processed by Docling."""
    
    schema_name: str
    version: str
    name: str
    origin: Optional[DocumentOrigin] = None
    furniture: Dict[str, Any]
    body: Dict[str, Any]
    groups: List[Dict[str, Any]] = Field(default_factory=list)
    texts: List[Dict[str, Any]] = Field(default_factory=list)
    pictures: List[Dict[str, Any]] = Field(default_factory=list)
    tables: List[Dict[str, Any]] = Field(default_factory=list)
    key_value_items: List[Dict[str, Any]] = Field(default_factory=list)
    form_items: List[Dict[str, Any]] = Field(default_factory=list)
    pages: Dict[str, Any] = Field(default_factory=dict)
    
    @classmethod
    def from_docling(cls, dl_doc: DLDocument) -> "DoclingDocument":
        """Create entity from Docling document."""
        origin = None
        if dl_doc.origin:
            origin = DocumentOrigin(
                mimetype=dl_doc.origin.mimetype,
                binary_hash=dl_doc.origin.binary_hash,
                filename=dl_doc.origin.filename
            )
        
        return cls(
            schema_name=dl_doc.schema_name,
            version=dl_doc.version,
            name=dl_doc.name,
            origin=origin,
            furniture=dl_doc.furniture.model_dump() if dl_doc.furniture else {},
            body=dl_doc.body.model_dump() if dl_doc.body else {},
            groups=[group.model_dump() for group in dl_doc.groups],
            texts=[text.model_dump() for text in dl_doc.texts],
            pictures=[pic.model_dump() for pic in dl_doc.pictures],
            tables=[table.model_dump() for table in dl_doc.tables],
            key_value_items=[kv.model_dump() for kv in dl_doc.key_value_items],
            form_items=[form.model_dump() for form in dl_doc.form_items],
            pages={str(k): v.model_dump() for k, v in dl_doc.pages.items()}
        )


class DocMeta(BaseModel):
    """Metadata for document chunks."""
    
    schema_name: str
    version: str
    doc_items: List[Dict[str, Any]]
    headings: Optional[List[str]] = None
    origin: Optional[DocumentOrigin] = None


class DocChunk(BaseModel):
    """Entity representing a document chunk with vector embeddings."""
    
    text: str
    meta: DocMeta
    vector: Optional[List[float]] = Field(default=None, description="Vector embedding for semantic search")
    
    @classmethod
    def from_docling_chunk(cls, dl_chunk: DLDocChunk, vector: Optional[List[float]] = None) -> "DocChunk":
        """Create entity from Docling chunk."""
        origin = None
        if dl_chunk.meta.origin:
            origin = DocumentOrigin(
                mimetype=dl_chunk.meta.origin.mimetype,
                binary_hash=dl_chunk.meta.origin.binary_hash,
                filename=dl_chunk.meta.origin.filename
            )
        
        meta = DocMeta(
            schema_name=dl_chunk.meta.schema_name,
            version=dl_chunk.meta.version,
            doc_items=[item.model_dump() for item in dl_chunk.meta.doc_items],
            headings=dl_chunk.meta.headings,
            origin=origin
        )
        
        return cls(
            text=dl_chunk.text,
            meta=meta,
            vector=vector
        )
