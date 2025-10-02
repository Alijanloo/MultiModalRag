from typing import List
from docling_core.types.doc.document import DoclingDocument as DLDocument
from .document import (
    DocumentOrigin,
    DoclingDocument,
    Provenance,
    DocumentText,
    ImageData,
    DocumentPicture,
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
        # pages={str(k): v.model_dump() for k, v in dl_doc.pages.items()},
    )

    texts = []
    for i, text_data in enumerate(dl_doc.texts):
        text_dict = (
            text_data.model_dump() if hasattr(text_data, "model_dump") else text_data
        )

        prov_list = []
        for prov_data in text_dict.get("prov", []):
            prov_list.append(Provenance.from_elastic_data(prov_data))

        text_id = text_dict.get("self_ref", "").split("/")[-1] if text_dict.get("self_ref") else str(i)
        text_element = DocumentText(
            text_id=f"{document_id}_text_{text_id}",
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

    pictures = []
    for i, pic_data in enumerate(dl_doc.pictures):
        pic_dict = (
            pic_data.model_dump() if hasattr(pic_data, "model_dump") else pic_data
        )

        prov_list = []
        for prov_data in pic_dict.get("prov", []):
            prov_list.append(Provenance.from_elastic_data(prov_data))

        image_data = None
        if "image" in pic_dict:
            image_data = ImageData.from_elastic_data(pic_dict["image"])

        picture_id = pic_dict.get("self_ref", "").split("/")[-1] if pic_dict.get("self_ref") else str(i)
        picture_element = DocumentPicture(
            picture_id=f"{document_id}_picture_{picture_id}",
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
        for prov_data in table_dict.get("prov", []):
            prov_list.append(Provenance.from_elastic_data(prov_data))

        table_data_obj = None
        if "data" in table_dict:
            table_data_obj = TableData.from_elastic_data(table_dict["data"])

        table_id = table_dict.get("self_ref", "").split("/")[-1] if table_dict.get("self_ref") else str(i)
        table_element = DocumentTable(
            table_id=f"{document_id}_table_{table_id}",
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
