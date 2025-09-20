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


class BoundingBox(BaseModel):
    """Bounding box information for document elements."""
    
    left: float
    top: float
    right: float
    bottom: float
    coord_origin: str


class Provenance(BaseModel):
    """Provenance information for document elements."""
    
    page_no: int
    bbox: BoundingBox
    charspan: List[int]


class DocumentText(BaseModel):
    """Entity representing a text element from a document."""
    
    text_id: str
    document_id: str
    text: str
    label: str
    level: Optional[int] = None
    prov: List[Provenance] = Field(default_factory=list)
    orig: Optional[str] = None
    parent_ref: Optional[str] = None
    children_refs: List[str] = Field(default_factory=list)


class ImageData(BaseModel):
    """Image data information."""
    
    mimetype: str
    dpi: int
    size: Dict[str, int]  # width, height
    uri: str  # base64 encoded image data


class DocumentPicture(BaseModel):
    """Entity representing a picture element from a document."""
    
    picture_id: str
    document_id: str
    label: str
    prov: List[Provenance] = Field(default_factory=list)
    image: Optional[ImageData] = None
    captions: List[str] = Field(default_factory=list)
    references: List[str] = Field(default_factory=list)
    footnotes: List[str] = Field(default_factory=list)
    annotations: List[Dict[str, Any]] = Field(default_factory=list)
    parent_ref: Optional[str] = None
    children_refs: List[str] = Field(default_factory=list)


class TableCell(BaseModel):
    """Table cell information."""
    
    bbox: BoundingBox
    row_span: int
    col_span: int
    start_row_offset_idx: int
    end_row_offset_idx: int
    start_col_offset_idx: int
    end_col_offset_idx: int
    text: str
    column_header: bool = False
    row_header: bool = False
    row_section: bool = False


class TableData(BaseModel):
    """Table data structure."""
    
    table_cells: List[TableCell]
    num_rows: int
    num_cols: int
    grid: List[List[Dict[str, Any]]]


class DocumentTable(BaseModel):
    """Entity representing a table element from a document."""
    
    table_id: str
    document_id: str
    label: str
    prov: List[Provenance] = Field(default_factory=list)
    data: Optional[TableData] = None
    captions: List[str] = Field(default_factory=list)
    references: List[str] = Field(default_factory=list)
    footnotes: List[str] = Field(default_factory=list)
    annotations: List[Dict[str, Any]] = Field(default_factory=list)
    parent_ref: Optional[str] = None
    children_refs: List[str] = Field(default_factory=list)


class DoclingDocument(BaseModel):
    """Entity representing the main document metadata (without content elements)."""
    
    schema_name: str
    version: str
    name: str
    origin: Optional[DocumentOrigin] = None
    furniture: Dict[str, Any]
    body: Dict[str, Any]
    groups: List[Dict[str, Any]] = Field(default_factory=list)
    key_value_items: List[Dict[str, Any]] = Field(default_factory=list)
    form_items: List[Dict[str, Any]] = Field(default_factory=list)
    pages: Dict[str, Any] = Field(default_factory=dict)


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
