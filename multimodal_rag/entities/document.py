"""Document entities for the multimodal RAG system."""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
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

    @classmethod
    def from_elastic_data(cls, bbox_data: Dict[str, Any]) -> "BoundingBox":
        """Create BoundingBox from Elasticsearch data."""
        return cls(
            left=bbox_data.get("left", bbox_data.get("l", 0)),
            top=bbox_data.get("top", bbox_data.get("t", 0)),
            right=bbox_data.get("right", bbox_data.get("r", 0)),
            bottom=bbox_data.get("bottom", bbox_data.get("b", 0)),
            coord_origin=bbox_data.get("coord_origin", "TOPLEFT"),
        )


class Provenance(BaseModel):
    """Provenance information for document elements."""

    page_no: int
    bbox: BoundingBox
    charspan: List[int]

    @classmethod
    def from_elastic_data(cls, prov_data: Dict[str, Any]) -> "Provenance":
        """Create Provenance from Elasticsearch data."""
        bbox = BoundingBox.from_elastic_data(prov_data.get("bbox", {}))
        return cls(
            page_no=prov_data.get("page_no", 1),
            bbox=bbox,
            charspan=prov_data.get("charspan", [0, 0]),
        )


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

    def to_elastic_data(self) -> Dict[str, Any]:
        """Convert to Elasticsearch indexing format."""
        return {
            "text": {
                "text_id": self.text_id,
                "document_id": self.document_id,
                "text": self.text,
                "label": self.label,
                "level": self.level,
                "orig": self.orig,
                "parent_ref": self.parent_ref,
                "children_refs": self.children_refs,
                "prov": [prov.model_dump(mode="json") for prov in self.prov],
            }
        }

    @classmethod
    def from_elastic_hit(cls, hit_data: Dict[str, Any]) -> "DocumentText":
        """Create DocumentText from Elasticsearch hit data."""
        text_data = hit_data.get("text", {})

        # Convert provenance data
        prov_list = [
            Provenance.from_elastic_data(prov_data)
            for prov_data in text_data.get("prov", [])
        ]

        return cls(
            text_id=text_data.get("text_id", ""),
            document_id=text_data.get("document_id", ""),
            text=text_data.get("text", ""),
            label=text_data.get("label", "text"),
            level=text_data.get("level"),
            orig=text_data.get("orig"),
            parent_ref=text_data.get("parent_ref"),
            children_refs=text_data.get("children_refs", []),
            prov=prov_list,
        )


class ImageData(BaseModel):
    """Image data information."""

    mimetype: str
    dpi: int
    size: Dict[str, int]  # width, height
    uri: str  # base64 encoded image data

    @classmethod
    def from_elastic_data(cls, image_data: Dict[str, Any]) -> "ImageData":
        """Create ImageData from Elasticsearch data."""
        uri_value = image_data.get("uri", "")
        uri_str = str(uri_value) if uri_value is not None else ""

        return cls(
            mimetype=image_data.get("mimetype", "image/png"),
            dpi=image_data.get("dpi", 72),
            size=image_data.get("size", {"width": 0, "height": 0}),
            uri=uri_str,
        )


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

    def to_elastic_data(self) -> Dict[str, Any]:
        """Convert to Elasticsearch indexing format."""
        return {
            "picture": {
                "picture_id": self.picture_id,
                "document_id": self.document_id,
                "label": self.label,
                "captions": self.captions,
                "references": self.references,
                "footnotes": self.footnotes,
                "parent_ref": self.parent_ref,
                "children_refs": self.children_refs,
                "prov": [prov.model_dump(mode="json") for prov in self.prov],
                "image": self.image.model_dump(mode="json") if self.image else None,
                "annotations": self.annotations,
            }
        }

    @classmethod
    def from_elastic_hit(cls, hit_data: Dict[str, Any]) -> "DocumentPicture":
        """Create DocumentPicture from Elasticsearch hit data."""
        picture_data = hit_data.get("picture", {})

        # Convert provenance data
        prov_list = [
            Provenance.from_elastic_data(prov_data)
            for prov_data in picture_data.get("prov", [])
        ]

        # Convert image data if present
        image_data = None
        if picture_data.get("image"):
            image_data = ImageData.from_elastic_data(picture_data["image"])

        return cls(
            picture_id=picture_data.get("picture_id", ""),
            document_id=picture_data.get("document_id", ""),
            label=picture_data.get("label", "picture"),
            prov=prov_list,
            image=image_data,
            captions=picture_data.get("captions", []),
            references=picture_data.get("references", []),
            footnotes=picture_data.get("footnotes", []),
            annotations=picture_data.get("annotations", []),
            parent_ref=picture_data.get("parent_ref"),
            children_refs=picture_data.get("children_refs", []),
        )


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

    @classmethod
    def from_elastic_data(cls, cell_data: Dict[str, Any]) -> "TableCell":
        """Create TableCell from Elasticsearch data."""
        bbox = BoundingBox.from_elastic_data(cell_data.get("bbox", {}))
        return cls(
            bbox=bbox,
            row_span=cell_data.get("row_span", 1),
            col_span=cell_data.get("col_span", 1),
            start_row_offset_idx=cell_data.get("start_row_offset_idx", 0),
            end_row_offset_idx=cell_data.get("end_row_offset_idx", 1),
            start_col_offset_idx=cell_data.get("start_col_offset_idx", 0),
            end_col_offset_idx=cell_data.get("end_col_offset_idx", 1),
            text=cell_data.get("text", ""),
            column_header=cell_data.get("column_header", False),
            row_header=cell_data.get("row_header", False),
            row_section=cell_data.get("row_section", False),
        )


class TableData(BaseModel):
    """Table data structure."""

    table_cells: List[TableCell]
    num_rows: int
    num_cols: int
    grid: List[List[Dict[str, Any]]]

    @classmethod
    def from_elastic_data(cls, data_info: Dict[str, Any]) -> "TableData":
        """Create TableData from Elasticsearch data."""
        table_cells = [
            TableCell.from_elastic_data(cell_data)
            for cell_data in data_info.get("table_cells", [])
        ]

        return cls(
            table_cells=table_cells,
            num_rows=data_info.get("num_rows", 0),
            num_cols=data_info.get("num_cols", 0),
            grid=data_info.get("grid", []),
        )


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

    def to_elastic_data(self) -> Dict[str, Any]:
        """Convert to Elasticsearch indexing format."""
        return {
            "table": {
                "table_id": self.table_id,
                "document_id": self.document_id,
                "label": self.label,
                "captions": self.captions,
                "references": self.references,
                "footnotes": self.footnotes,
                "parent_ref": self.parent_ref,
                "children_refs": self.children_refs,
                "prov": [prov.model_dump(mode="json") for prov in self.prov],
                "data": self.data.model_dump(mode="json") if self.data else None,
                "annotations": self.annotations,
            }
        }

    @classmethod
    def from_elastic_hit(cls, hit_data: Dict[str, Any]) -> "DocumentTable":
        """Create DocumentTable from Elasticsearch hit data."""
        table_data = hit_data.get("table", {})

        # Convert provenance data
        prov_list = [
            Provenance.from_elastic_data(prov_data)
            for prov_data in table_data.get("prov", [])
        ]

        # Convert table data if present
        table_data_obj = None
        if table_data.get("data"):
            table_data_obj = TableData.from_elastic_data(table_data["data"])

        return cls(
            table_id=table_data.get("table_id", ""),
            document_id=table_data.get("document_id", ""),
            label=table_data.get("label", "table"),
            prov=prov_list,
            data=table_data_obj,
            captions=table_data.get("captions", []),
            references=table_data.get("references", []),
            footnotes=table_data.get("footnotes", []),
            annotations=table_data.get("annotations", []),
            parent_ref=table_data.get("parent_ref"),
            children_refs=table_data.get("children_refs", []),
        )


class DoclingDocument(BaseModel):
    """Entity representing the main document metadata (without content elements)."""

    schema_name: str
    version: str
    name: str
    origin: Optional[DocumentOrigin] = None
    furniture: Dict[str, Any]
    body: Dict[str, Any] = Field(default_factory=dict)
    groups: List[Dict[str, Any]] = Field(default_factory=list)
    key_value_items: List[Dict[str, Any]] = Field(default_factory=list)
    form_items: List[Dict[str, Any]] = Field(default_factory=list)
    pages: Dict[str, Any] = Field(default_factory=dict)

    def to_elastic_data(self) -> Dict[str, Any]:
        """Convert to Elasticsearch indexing format."""
        return {"document": self.model_dump(mode="json")}

    @classmethod
    def from_elastic_hit(cls, hit_data: Dict[str, Any]) -> "DoclingDocument":
        """Create DoclingDocument from Elasticsearch hit data."""
        document_data = hit_data.get("document", {})

        origin = None
        if document_data.get("origin"):
            origin_data = document_data["origin"]
            origin = DocumentOrigin(
                mimetype=origin_data.get("mimetype", ""),
                binary_hash=origin_data.get("binary_hash", 0),
                filename=origin_data.get("filename", ""),
            )

        return cls(
            schema_name=document_data.get("schema_name", ""),
            version=document_data.get("version", ""),
            name=document_data.get("name", ""),
            origin=origin,
            furniture=document_data.get("furniture", {}),
            body=document_data.get("body", {}),
            groups=document_data.get("groups", []),
            key_value_items=document_data.get("key_value_items", []),
            form_items=document_data.get("form_items", []),
            pages=document_data.get("pages", {}),
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

    chunk_id: Optional[str] = None
    text: str
    meta: DocMeta
    document_id: str
    vector: Optional[List[float]] = Field(
        default=None, description="Vector embedding for semantic search"
    )

    def to_elastic_data(self) -> Dict[str, Any]:
        """Convert to Elasticsearch indexing format."""
        chunk_data = {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "document_id": self.document_id,
            "meta": self.meta.model_dump(mode="json"),
        }

        if self.vector:
            chunk_data["vector"] = self.vector

        return {"chunk": chunk_data}

    @classmethod
    def from_docling_chunk(
        cls,
        dl_chunk: DLDocChunk,
        document_id: str,
        chunk_index: int,
        vector: Optional[List[float]] = None,
    ) -> "DocChunk":
        """Create entity from Docling chunk."""
        chunk_id = f"{document_id}_chunk_{chunk_index}"
        origin = None
        if dl_chunk.meta.origin:
            origin = DocumentOrigin(
                mimetype=dl_chunk.meta.origin.mimetype,
                binary_hash=dl_chunk.meta.origin.binary_hash,
                filename=dl_chunk.meta.origin.filename,
            )

        meta = DocMeta(
            schema_name=dl_chunk.meta.schema_name,
            version=dl_chunk.meta.version,
            doc_items=[
                item.model_dump(mode="json") for item in dl_chunk.meta.doc_items
            ],
            headings=dl_chunk.meta.headings,
            origin=origin,
        )

        return cls(
            chunk_id=chunk_id,
            text=dl_chunk.text,
            meta=meta,
            document_id=document_id,
            vector=vector,
        )

    @classmethod
    def from_elastic_hit(cls, hit_data: Dict[str, Any]) -> "DocChunk":
        """Create DocChunk from Elasticsearch hit data."""
        chunk_data = hit_data.get("chunk", {})

        # Convert meta data
        meta_data = chunk_data.get("meta", {})

        origin = None
        if meta_data.get("origin"):
            origin_data = meta_data["origin"]
            origin = DocumentOrigin(
                mimetype=origin_data.get("mimetype", ""),
                binary_hash=origin_data.get("binary_hash", 0),
                filename=origin_data.get("filename", ""),
            )

        meta = DocMeta(
            schema_name=meta_data.get("schema_name", ""),
            version=meta_data.get("version", ""),
            doc_items=meta_data.get("doc_items", []),
            headings=meta_data.get("headings"),
            origin=origin,
        )

        return cls(
            chunk_id=chunk_data.get("chunk_id", ""),
            text=chunk_data.get("text", ""),
            document_id=chunk_data.get("document_id", ""),
            meta=meta,
            vector=chunk_data.get("vector"),
        )
