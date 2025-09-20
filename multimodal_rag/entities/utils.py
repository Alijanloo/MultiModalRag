from typing import List
from docling_core.types.doc.document import DoclingDocument as DLDocument
from .document import (
    DocumentOrigin,
    DoclingDocument,
    BoundingBox,
    Provenance,
    DocumentText,
    ImageData,
    DocumentPicture,
    TableCell,
    TableData,
    DocumentTable,
)


def create_document_entities_from_docling(
    dl_doc: DLDocument, document_id: str
) -> tuple[
    DoclingDocument, List[DocumentText], List[DocumentPicture], List[DocumentTable]
]:
    """Create entities from Docling document, separating content elements."""
    origin = None
    if dl_doc.origin:
        origin = DocumentOrigin(
            mimetype=dl_doc.origin.mimetype,
            binary_hash=dl_doc.origin.binary_hash,
            filename=dl_doc.origin.filename,
        )

    # Create main document without content elements
    document = DoclingDocument(
        schema_name=dl_doc.schema_name,
        version=dl_doc.version,
        name=dl_doc.name,
        origin=origin,
        furniture=dl_doc.furniture.model_dump() if dl_doc.furniture else {},
        body=dl_doc.body.model_dump() if dl_doc.body else {},
        groups=[group.model_dump() for group in dl_doc.groups],
        key_value_items=[kv.model_dump() for kv in dl_doc.key_value_items],
        form_items=[form.model_dump() for form in dl_doc.form_items],
        pages={str(k): v.model_dump() for k, v in dl_doc.pages.items()},
    )

    # Extract text elements
    texts = []
    for i, text_data in enumerate(dl_doc.texts):
        text_dict = (
            text_data.model_dump() if hasattr(text_data, "model_dump") else text_data
        )

        prov_list = []
        if "prov" in text_dict:
            for prov_data in text_dict["prov"]:
                if "bbox" in prov_data:
                    bbox = BoundingBox(
                        left=prov_data["bbox"].get("l", 0),
                        top=prov_data["bbox"].get("t", 0),
                        right=prov_data["bbox"].get("r", 0),
                        bottom=prov_data["bbox"].get("b", 0),
                        coord_origin=prov_data["bbox"].get("coord_origin", "TOPLEFT"),
                    )
                    prov = Provenance(
                        page_no=prov_data.get("page_no", 1),
                        bbox=bbox,
                        charspan=prov_data.get("charspan", [0, 0]),
                    )
                    prov_list.append(prov)

        text_element = DocumentText(
            text_id=f"{document_id}_text_{i}",
            document_id=document_id,
            text=text_dict.get("text", ""),
            label=text_dict.get("label", "text"),
            level=text_dict.get("level"),
            prov=prov_list,
            orig=text_dict.get("orig"),
            parent_ref=text_dict.get("parent", {}).get("$ref")
            if text_dict.get("parent")
            else None,
            children_refs=[
                child.get("$ref", "") for child in text_dict.get("children", [])
            ],
        )
        texts.append(text_element)

    # Extract picture elements
    pictures = []
    for i, pic_data in enumerate(dl_doc.pictures):
        pic_dict = (
            pic_data.model_dump() if hasattr(pic_data, "model_dump") else pic_data
        )

        prov_list = []
        if "prov" in pic_dict:
            for prov_data in pic_dict["prov"]:
                if "bbox" in prov_data:
                    bbox = BoundingBox(
                        left=prov_data["bbox"].get("l", 0),
                        top=prov_data["bbox"].get("t", 0),
                        right=prov_data["bbox"].get("r", 0),
                        bottom=prov_data["bbox"].get("b", 0),
                        coord_origin=prov_data["bbox"].get("coord_origin", "TOPLEFT"),
                    )
                    prov = Provenance(
                        page_no=prov_data.get("page_no", 1),
                        bbox=bbox,
                        charspan=prov_data.get("charspan", [0, 0]),
                    )
                    prov_list.append(prov)

        image_data = None
        if "image" in pic_dict:
            image_info = pic_dict["image"]
            image_data = ImageData(
                mimetype=image_info.get("mimetype", "image/png"),
                dpi=image_info.get("dpi", 72),
                size=image_info.get("size", {"width": 0, "height": 0}),
                uri=image_info.get("uri", ""),
            )

        picture_element = DocumentPicture(
            picture_id=f"{document_id}_picture_{i}",
            document_id=document_id,
            label=pic_dict.get("label", "picture"),
            prov=prov_list,
            image=image_data,
            captions=[cap.get("$ref", "") for cap in pic_dict.get("captions", [])],
            references=[ref.get("$ref", "") for ref in pic_dict.get("references", [])],
            footnotes=[fn.get("$ref", "") for fn in pic_dict.get("footnotes", [])],
            annotations=pic_dict.get("annotations", []),
            parent_ref=pic_dict.get("parent", {}).get("$ref")
            if pic_dict.get("parent")
            else None,
            children_refs=[
                child.get("$ref", "") for child in pic_dict.get("children", [])
            ],
        )
        pictures.append(picture_element)

    # Extract table elements
    tables = []
    for i, table_data in enumerate(dl_doc.tables):
        table_dict = (
            table_data.model_dump() if hasattr(table_data, "model_dump") else table_data
        )

        prov_list = []
        if "prov" in table_dict:
            for prov_data in table_dict["prov"]:
                if "bbox" in prov_data:
                    bbox = BoundingBox(
                        left=prov_data["bbox"].get("l", 0),
                        top=prov_data["bbox"].get("t", 0),
                        right=prov_data["bbox"].get("r", 0),
                        bottom=prov_data["bbox"].get("b", 0),
                        coord_origin=prov_data["bbox"].get("coord_origin", "TOPLEFT"),
                    )
                    prov = Provenance(
                        page_no=prov_data.get("page_no", 1),
                        bbox=bbox,
                        charspan=prov_data.get("charspan", [0, 0]),
                    )
                    prov_list.append(prov)

        table_data_obj = None
        if "data" in table_dict:
            data_info = table_dict["data"]
            table_cells = []

            for cell_data in data_info.get("table_cells", []):
                if "bbox" in cell_data:
                    cell_bbox = BoundingBox(
                        left=cell_data["bbox"].get("l", 0),
                        top=cell_data["bbox"].get("t", 0),
                        right=cell_data["bbox"].get("r", 0),
                        bottom=cell_data["bbox"].get("b", 0),
                        coord_origin=cell_data["bbox"].get("coord_origin", "TOPLEFT"),
                    )

                    table_cell = TableCell(
                        bbox=cell_bbox,
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
                    table_cells.append(table_cell)

            table_data_obj = TableData(
                table_cells=table_cells,
                num_rows=data_info.get("num_rows", 0),
                num_cols=data_info.get("num_cols", 0),
                grid=data_info.get("grid", []),
            )

        table_element = DocumentTable(
            table_id=f"{document_id}_table_{i}",
            document_id=document_id,
            label=table_dict.get("label", "table"),
            prov=prov_list,
            data=table_data_obj,
            captions=[cap.get("$ref", "") for cap in table_dict.get("captions", [])],
            references=[
                ref.get("$ref", "") for ref in table_dict.get("references", [])
            ],
            footnotes=[fn.get("$ref", "") for fn in table_dict.get("footnotes", [])],
            annotations=table_dict.get("annotations", []),
            parent_ref=table_dict.get("parent", {}).get("$ref")
            if table_dict.get("parent")
            else None,
            children_refs=[
                child.get("$ref", "") for child in table_dict.get("children", [])
            ],
        )
        tables.append(table_element)

    return document, texts, pictures, tables
