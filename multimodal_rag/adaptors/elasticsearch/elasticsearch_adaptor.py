"""Elasticsearch adaptor for document indexing and search."""

import logging
from typing import Dict, List, Optional, Any, Tuple

from elasticsearch import AsyncElasticsearch
from elasticsearch.exceptions import NotFoundError
from elasticsearch.helpers import async_bulk

from ...usecases.interfaces.document_repository import IDocumentIndexRepository

from ...entities.document import (
    DoclingDocument,
    DocChunk,
    DocumentText,
    DocumentPicture,
    DocumentTable,
)

logger = logging.getLogger(__name__)


class ElasticsearchDocumentAdaptor(IDocumentIndexRepository):
    """Elasticsearch implementation of document indexing repository."""

    DEFAULT_INDEX = "multimodal_index"

    MAPPING = {
        "mappings": {
            "properties": {
                "document": {
                    "properties": {
                        "schema_name": {"type": "keyword"},
                        "version": {"type": "keyword"},
                        "name": {"type": "text", "analyzer": "standard"},
                        "origin": {
                            "properties": {
                                "mimetype": {"type": "keyword"},
                                "binary_hash": {"type": "long"},
                                "filename": {"type": "text", "analyzer": "standard"},
                            }
                        },
                        "furniture": {"type": "object", "enabled": False},
                        "body": {"type": "object", "enabled": False},
                        "groups": {"type": "object", "enabled": False},
                        "texts": {"type": "object", "enabled": False},
                        "pictures": {"type": "object", "enabled": False},
                        "tables": {"type": "object", "enabled": False},
                        "key_value_items": {"type": "object", "enabled": False},
                        "form_items": {"type": "object", "enabled": False},
                        "pages": {"type": "object", "enabled": False},
                    }
                },
                "chunk": {
                    "properties": {
                        "text": {
                            "type": "text",
                            "analyzer": "standard",
                            "fields": {
                                "keyword": {"type": "keyword", "ignore_above": 256}
                            },
                        },
                        "meta": {
                            "properties": {
                                "schema_name": {"type": "keyword"},
                                "version": {"type": "keyword"},
                                "headings": {"type": "text", "analyzer": "standard"},
                                "origin": {
                                    "properties": {
                                        "mimetype": {"type": "keyword"},
                                        "binary_hash": {"type": "long"},
                                        "filename": {
                                            "type": "text",
                                            "analyzer": "standard",
                                        },
                                    }
                                },
                                "doc_items": {"type": "object", "enabled": False},
                            }
                        },
                        "vector": {
                            "type": "dense_vector",
                            "dims": 1536,  # Default for OpenAI embeddings, adjust as needed
                            "index": True,
                            "similarity": "cosine",
                        },
                        "document_id": {
                            "type": "keyword"
                        },  # Reference to parent document
                    }
                },
                "text": {
                    "properties": {
                        "text_id": {"type": "keyword"},
                        "document_id": {"type": "keyword"},
                        "text": {
                            "type": "text",
                            "analyzer": "standard",
                            "fields": {
                                "keyword": {"type": "keyword", "ignore_above": 256}
                            },
                        },
                        "label": {"type": "keyword"},
                        "level": {"type": "integer"},
                        "orig": {"type": "text", "analyzer": "standard"},
                        "parent_ref": {"type": "keyword"},
                        "children_refs": {"type": "keyword"},
                        "prov": {"type": "object", "enabled": False},
                    }
                },
                "picture": {
                    "properties": {
                        "picture_id": {"type": "keyword"},
                        "document_id": {"type": "keyword"},
                        "label": {"type": "keyword"},
                        "captions": {"type": "text", "analyzer": "standard"},
                        "references": {"type": "keyword"},
                        "footnotes": {"type": "keyword"},
                        "parent_ref": {"type": "keyword"},
                        "children_refs": {"type": "keyword"},
                        "prov": {"type": "object", "enabled": False},
                        "image": {"type": "object", "enabled": False},
                        "annotations": {"type": "object", "enabled": False},
                    }
                },
                "table": {
                    "properties": {
                        "table_id": {"type": "keyword"},
                        "document_id": {"type": "keyword"},
                        "label": {"type": "keyword"},
                        "captions": {"type": "text", "analyzer": "standard"},
                        "references": {"type": "keyword"},
                        "footnotes": {"type": "keyword"},
                        "parent_ref": {"type": "keyword"},
                        "children_refs": {"type": "keyword"},
                        "prov": {"type": "object", "enabled": False},
                        "data": {"type": "object", "enabled": False},
                        "annotations": {"type": "object", "enabled": False},
                    }
                },
            }
        }
    }

    def __init__(
        self,
        elasticsearch_client: AsyncElasticsearch,
        index_name: str = DEFAULT_INDEX,
        vector_dimensions: int = 1536,
    ):
        """Initialize the Elasticsearch adaptor.

        Args:
            elasticsearch_client: AsyncElasticsearch client instance
            index_name: Name of the index for documents and chunks
            vector_dimensions: Dimension of the vector embeddings
        """
        self._es = elasticsearch_client
        self._index_name = index_name
        self._vector_dimensions = vector_dimensions

        # Update mapping with correct vector dimensions
        self.MAPPING["mappings"]["properties"]["chunk"]["properties"]["vector"][
            "dims"
        ] = vector_dimensions

    async def initialize_indices(self) -> bool:
        """Initialize the Elasticsearch index with proper mapping."""
        try:
            if not await self._es.indices.exists(index=self._index_name):
                await self._es.indices.create(index=self._index_name, body=self.MAPPING)
                logger.info(f"Created index: {self._index_name}")

            return True
        except Exception as e:
            logger.error(f"Failed to initialize index: {e}")
            return False

    async def index_document(
        self,
        document: DoclingDocument,
        document_id: str,
        index_name: Optional[str] = None,
    ) -> bool:
        """Index a single document."""
        try:
            index_name = index_name or self._index_name

            document_data = document.to_elastic_data()

            result = await self._es.index(
                index=index_name, id=document_id, document=document_data
            )

            success = result.get("result") in ["created", "updated"]
            if not success:
                logger.error(f"Failed to index document {document_id}: {result}")

            return success
        except Exception as e:
            error_msg = f"Failed to index document {document_id}: {e}"
            if hasattr(e, "meta") and hasattr(e.meta, "status"):
                if e.meta.status == 413:
                    error_msg += " - Document is too large. Consider chunking the document or increasing Elasticsearch's http.max_content_length setting."
                error_msg += f" (HTTP {e.meta.status})"
            logger.error(error_msg)
            return False

    async def index_chunk(
        self,
        chunk: DocChunk,
        chunk_id: str,
        document_id: Optional[str] = None,
        index_name: Optional[str] = None,
    ) -> bool:
        """Index a single chunk."""
        try:
            index_name = index_name or self._index_name

            chunk_data = chunk.to_elastic_data(document_id=document_id)

            result = await self._es.index(
                index=index_name, id=chunk_id, document=chunk_data
            )

            success = result.get("result") in ["created", "updated"]
            if not success:
                logger.error(f"Failed to index chunk {chunk_id}: {result}")

            return success
        except Exception as e:
            logger.error(f"Failed to index chunk {chunk_id}: {e}")
            return False

    async def index_text(
        self, text: DocumentText, index_name: Optional[str] = None
    ) -> bool:
        """Index a single text element."""
        try:
            index_name = index_name or self._index_name

            text_data = text.to_elastic_data()

            result = await self._es.index(
                index=index_name, id=text.text_id, document=text_data
            )

            success = result.get("result") in ["created", "updated"]
            if not success:
                logger.error(f"Failed to index text {text.text_id}: {result}")

            return success
        except Exception as e:
            logger.error(f"Failed to index text {text.text_id}: {e}")
            return False

    async def index_picture(
        self, picture: DocumentPicture, index_name: Optional[str] = None
    ) -> bool:
        """Index a single picture element."""
        try:
            index_name = index_name or self._index_name

            picture_data = picture.to_elastic_data()

            result = await self._es.index(
                index=index_name, id=picture.picture_id, document=picture_data
            )

            success = result.get("result") in ["created", "updated"]
            if not success:
                logger.error(f"Failed to index picture {picture.picture_id}: {result}")

            return success
        except Exception as e:
            logger.error(f"Failed to index picture {picture.picture_id}: {e}")
            return False

    async def index_table(
        self, table: DocumentTable, index_name: Optional[str] = None
    ) -> bool:
        """Index a single table element."""
        try:
            index_name = index_name or self._index_name

            table_data = table.to_elastic_data()

            result = await self._es.index(
                index=index_name, id=table.table_id, document=table_data
            )

            success = result.get("result") in ["created", "updated"]
            if not success:
                logger.error(f"Failed to index table {table.table_id}: {result}")

            return success
        except Exception as e:
            logger.error(f"Failed to index table {table.table_id}: {e}")
            return False

    async def bulk_index_chunks(
        self,
        chunks: List[DocChunk],
        document_id: str,
        index_name: Optional[str] = None,
    ) -> Tuple[int, int, List[str]]:
        """Bulk index multiple chunks."""
        index_name = index_name or self._index_name
        indexed_count = 0
        failed_count = 0
        errors = []

        actions = []
        for i, chunk in enumerate(chunks):
            try:
                chunk_id = f"{document_id}_chunk_{i}"
                chunk_data = chunk.to_elastic_data(document_id=document_id)
                
                action = {
                    "_index": index_name,
                    "_id": chunk_id,
                    "_source": chunk_data
                }
                actions.append(action)
            except Exception as e:
                failed_count += 1
                error_msg = f"Failed to prepare chunk {i}: {e}"
                logger.error(error_msg)
                errors.append(error_msg)

        if actions:
            try:
                success_count, failed_items = await async_bulk(
                    self._es, actions, stats_only=True, raise_on_error=False
                )
                indexed_count += success_count
                failed_count += len(failed_items) if failed_items else 0
                        
            except Exception as e:
                failed_count += len(actions)
                error_msg = f"Bulk indexing failed: {e}"
                logger.error(error_msg)
                errors.append(error_msg)

        return indexed_count, failed_count, errors

    async def bulk_index_texts(
        self,
        texts: List[DocumentText],
        index_name: Optional[str] = None,
    ) -> Tuple[int, int, List[str]]:
        """Bulk index multiple text elements."""
        index_name = index_name or self._index_name
        indexed_count = 0
        failed_count = 0
        errors = []

        actions = []
        for text in texts:
            try:
                text_data = text.to_elastic_data()
                
                action = {
                    "_index": index_name,
                    "_id": text.text_id,
                    "_source": text_data
                }
                actions.append(action)
            except Exception as e:
                failed_count += 1
                error_msg = f"Failed to prepare text {text.text_id}: {e}"
                logger.error(error_msg)
                errors.append(error_msg)

        if actions:
            try:
                success_count, failed_items = await async_bulk(
                    self._es, actions, stats_only=True, raise_on_error=False
                )
                indexed_count += success_count
                failed_count += len(failed_items) if failed_items else 0
                        
            except Exception as e:
                failed_count += len(actions)
                error_msg = f"Bulk indexing failed: {e}"
                logger.error(error_msg)
                errors.append(error_msg)

        return indexed_count, failed_count, errors

    async def bulk_index_pictures(
        self,
        pictures: List[DocumentPicture],
        index_name: Optional[str] = None,
    ) -> Tuple[int, int, List[str]]:
        """Bulk index multiple picture elements."""
        index_name = index_name or self._index_name
        indexed_count = 0
        failed_count = 0
        errors = []

        actions = []
        for picture in pictures:
            try:
                picture_data = picture.to_elastic_data()
                
                action = {
                    "_index": index_name,
                    "_id": picture.picture_id,
                    "_source": picture_data
                }
                actions.append(action)
            except Exception as e:
                failed_count += 1
                error_msg = f"Failed to prepare picture {picture.picture_id}: {e}"
                logger.error(error_msg)
                errors.append(error_msg)

        if actions:
            try:
                success_count, failed_items = await async_bulk(
                    self._es, actions, stats_only=True, raise_on_error=False
                )
                indexed_count += success_count
                failed_count += len(failed_items) if failed_items else 0
                        
            except Exception as e:
                failed_count += len(actions)
                error_msg = f"Bulk indexing failed: {e}"
                logger.error(error_msg)
                errors.append(error_msg)

        return indexed_count, failed_count, errors

    async def bulk_index_tables(
        self,
        tables: List[DocumentTable],
        index_name: Optional[str] = None,
    ) -> Tuple[int, int, List[str]]:
        """Bulk index multiple table elements."""
        index_name = index_name or self._index_name
        indexed_count = 0
        failed_count = 0
        errors = []

        actions = []
        for table in tables:
            try:
                table_data = table.to_elastic_data()
                
                action = {
                    "_index": index_name,
                    "_id": table.table_id,
                    "_source": table_data
                }
                actions.append(action)
            except Exception as e:
                failed_count += 1
                error_msg = f"Failed to prepare table {table.table_id}: {e}"
                logger.error(error_msg)
                errors.append(error_msg)

        if actions:
            try:
                success_count, failed_items = await async_bulk(
                    self._es, actions, stats_only=True, raise_on_error=False
                )
                indexed_count += success_count
                failed_count += len(failed_items) if failed_items else 0
                        
            except Exception as e:
                failed_count += len(actions)
                error_msg = f"Bulk indexing failed: {e}"
                logger.error(error_msg)
                errors.append(error_msg)

        return indexed_count, failed_count, errors

    async def get_document(
        self, document_id: str, index_name: Optional[str] = None
    ) -> Optional[DoclingDocument]:
        """Get a document by ID."""
        try:
            index_name = index_name or self._index_name

            result = await self._es.get(index=index_name, id=document_id)

            if result and "_source" in result and "document" in result["_source"]:
                return DoclingDocument.from_elastic_hit(result["_source"])

            return None

        except NotFoundError:
            return None
        except Exception as e:
            logger.error(f"Failed to get document {document_id}: {e}")
            return None

    async def get_picture(
        self, document_id: str, picture_id: str, index_name: Optional[str] = None
    ) -> Optional[DocumentPicture]:
        """Get a picture by document_id and picture_id."""
        try:
            index_name = index_name or self._index_name

            query = {
                "bool": {
                    "must": [
                        {"exists": {"field": "picture"}},
                        {"term": {"picture.document_id": document_id}},
                        {"term": {"picture.picture_id": picture_id}},
                    ]
                }
            }

            result = await self._es.search(index=index_name, size=1, query=query)

            if result["hits"]["hits"]:
                hit = result["hits"]["hits"][0]
                return DocumentPicture.from_elastic_hit(hit["_source"])

            return None

        except Exception as e:
            logger.error(
                f"Failed to get picture {picture_id} for document {document_id}: {e}"
            )
            return None

    async def search_chunks(
        self,
        query: Optional[str] = None,
        vector: Optional[List[float]] = None,
        filters: Optional[Dict[str, Any]] = None,
        size: int = 10,
        index_name: Optional[str] = None,
    ) -> Tuple[List[DocChunk], int]:
        """Search chunks using text or vector similarity."""
        try:
            index_name = index_name or self._index_name

            # Build query for chunks
            query_dict = self._build_chunk_search_query(query, vector, filters, size)

            search_params = {
                "index": index_name,
                "size": size,
                "query": query_dict.get("query"),
                "highlight": {
                    "fields": {
                        "chunk.text": {"fragment_size": 150, "number_of_fragments": 3}
                    }
                },
            }

            # Add vector search if provided
            if vector and "knn" in query_dict:
                search_params["knn"] = query_dict["knn"]

            result = await self._es.search(**search_params)

            chunks = []
            for hit in result["hits"]["hits"]:
                if "chunk" in hit["_source"]:
                    chunk = DocChunk.from_elastic_hit(hit["_source"])
                    chunks.append(chunk)

            return chunks, result["hits"]["total"]["value"]

        except Exception as e:
            logger.error(f"Search chunks failed: {e}")
            return [], 0

    async def search_documents(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        size: int = 10,
        index_name: Optional[str] = None,
    ) -> Tuple[List[DoclingDocument], int]:
        """Search documents using text query."""
        try:
            index_name = index_name or self._index_name

            # Build text query for documents
            query_dict = {
                "bool": {
                    "must": [
                        {"exists": {"field": "document"}},
                        {
                            "bool": {
                                "should": [
                                    {
                                        "match": {
                                            "document.name": {
                                                "query": query,
                                                "boost": 2.0,
                                            }
                                        }
                                    },
                                    {
                                        "match": {
                                            "document.origin.filename": {
                                                "query": query,
                                                "boost": 1.5,
                                            }
                                        }
                                    },
                                ]
                            }
                        },
                    ]
                }
            }

            # Add filters if provided
            if filters:
                query_dict["bool"]["filter"] = self._build_filters(filters)

            result = await self._es.search(
                index=index_name,
                size=size,
                query=query_dict,
                highlight={
                    "fields": {
                        "document.name": {
                            "fragment_size": 150,
                            "number_of_fragments": 1,
                        },
                        "document.origin.filename": {
                            "fragment_size": 150,
                            "number_of_fragments": 1,
                        },
                    }
                },
            )

            documents = []
            for hit in result["hits"]["hits"]:
                if "document" in hit["_source"]:
                    document = DoclingDocument.from_elastic_hit(hit["_source"])
                    documents.append(document)

            return documents, result["hits"]["total"]["value"]

        except Exception as e:
            logger.error(f"Search documents failed: {e}")
            return [], 0

    async def delete_document(
        self, document_id: str, index_name: Optional[str] = None
    ) -> bool:
        """Delete a document by ID."""
        try:
            index_name = index_name or self._index_name

            result = await self._es.delete(index=index_name, id=document_id)

            return result.get("result") == "deleted"

        except NotFoundError:
            return False
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            return False

    async def delete_chunk(
        self, chunk_id: str, index_name: Optional[str] = None
    ) -> bool:
        """Delete a chunk by ID."""
        try:
            index_name = index_name or self._index_name

            result = await self._es.delete(index=index_name, id=chunk_id)

            return result.get("result") == "deleted"

        except NotFoundError:
            return False
        except Exception as e:
            logger.error(f"Failed to delete chunk {chunk_id}: {e}")
            return False

    def _build_chunk_search_query(
        self,
        query: Optional[str] = None,
        vector: Optional[List[float]] = None,
        filters: Optional[Dict[str, Any]] = None,
        size: int = 10,
    ) -> Dict[str, Any]:
        """Build search query for chunks."""
        query_parts = {}

        # Text search with field existence filter for chunks
        if query:
            text_query = {
                "bool": {
                    "must": [{"exists": {"field": "chunk"}}],
                    "should": [
                        {"match": {"chunk.text": {"query": query, "boost": 1.0}}},
                        {
                            "match": {
                                "chunk.meta.headings": {
                                    "query": query,
                                    "boost": 1.5,
                                }
                            }
                        },
                    ],
                }
            }

            # Add filters if provided
            if filters:
                text_query["bool"]["filter"] = self._build_filters(filters)

            query_parts["query"] = text_query

        # Vector search
        if vector:
            knn_query = {
                "field": "chunk.vector",
                "query_vector": vector,
                "k": size,
                "num_candidates": size * 10,
                "filter": {"exists": {"field": "chunk"}},
            }

            # Add additional filters to KNN if provided
            if filters:
                additional_filters = self._build_filters(filters)
                knn_query["filter"] = {
                    "bool": {
                        "must": [{"exists": {"field": "chunk"}}, *additional_filters]
                    }
                }

            query_parts["knn"] = knn_query

        # If both text and vector, create hybrid search
        if query and vector:
            # For hybrid search, we use the KNN with a filter that includes the text match
            hybrid_filter = {
                "bool": {
                    "must": [{"exists": {"field": "chunk"}}],
                    "should": [
                        {"match": {"chunk.text": query}},
                        {"match": {"chunk.meta.headings": query}},
                    ],
                }
            }

            if filters:
                existing_filters = self._build_filters(filters)
                hybrid_filter["bool"]["must"].extend(existing_filters)

            query_parts["knn"]["filter"] = hybrid_filter
            # Remove the separate text query for hybrid
            if "query" in query_parts:
                del query_parts["query"]

        return query_parts

    def _build_filters(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build Elasticsearch filters from filter dictionary."""
        filter_list = []

        for field, value in filters.items():
            if isinstance(value, dict):
                # Handle range filters, etc.
                filter_list.append({field: value})
            elif isinstance(value, list):
                # Handle terms filter
                filter_list.append({"terms": {field: value}})
            else:
                # Handle exact match
                filter_list.append({"term": {field: value}})

        return filter_list
