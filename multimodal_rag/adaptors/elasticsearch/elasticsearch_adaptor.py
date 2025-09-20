"""Elasticsearch adaptor for document indexing and search."""

import logging
from typing import Dict, List, Optional, Any

from elasticsearch import AsyncElasticsearch
from elasticsearch.exceptions import NotFoundError

from ...usecases.interfaces.document_repository import IDocumentIndexRepository
from ...usecases.dtos import (
    IndexDocumentResponse,
    IndexChunkResponse,
    IndexTextResponse,
    IndexPictureResponse,
    IndexTableResponse,
    SearchRequest,
    SearchResponse,
    SearchHit,
    GetDocumentResponse,
)
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
    ) -> IndexDocumentResponse:
        """Index a single document."""
        try:
            index_name = index_name or self._index_name

            document_data = document.to_elastic_data()

            result = await self._es.index(
                index=index_name, id=document_id, document=document_data
            )

            success = result.get("result") in ["created", "updated"]
            message = f"Document {result.get('result', 'processed')}"

            return IndexDocumentResponse(
                document_id=document_id, success=success, message=message
            )
        except Exception as e:
            error_msg = f"Failed to index document {document_id}: {e}"
            if hasattr(e, "meta") and hasattr(e.meta, "status"):
                if e.meta.status == 413:
                    error_msg += " - Document is too large. Consider chunking the document or increasing Elasticsearch's http.max_content_length setting."
                error_msg += f" (HTTP {e.meta.status})"
            logger.error(error_msg)
            return IndexDocumentResponse(
                document_id=document_id, success=False, message=error_msg
            )

    async def index_chunk(
        self,
        chunk: DocChunk,
        chunk_id: str,
        document_id: Optional[str] = None,
        index_name: Optional[str] = None,
    ) -> IndexChunkResponse:
        """Index a single chunk."""
        try:
            index_name = index_name or self._index_name

            chunk_data = chunk.to_elastic_data(document_id=document_id)

            result = await self._es.index(
                index=index_name, id=chunk_id, document=chunk_data
            )

            success = result.get("result") in ["created", "updated"]
            message = f"Chunk {result.get('result', 'processed')}"

            return IndexChunkResponse(
                chunk_id=chunk_id, success=success, message=message
            )
        except Exception as e:
            logger.error(f"Failed to index chunk {chunk_id}: {e}")
            return IndexChunkResponse(chunk_id=chunk_id, success=False, message=str(e))

    async def index_text(
        self, text: DocumentText, index_name: Optional[str] = None
    ) -> IndexTextResponse:
        """Index a single text element."""
        try:
            index_name = index_name or self._index_name

            text_data = text.to_elastic_data()

            result = await self._es.index(
                index=index_name, id=text.text_id, document=text_data
            )

            success = result.get("result") in ["created", "updated"]
            message = f"Text {result.get('result', 'processed')}"

            return IndexTextResponse(
                text_id=text.text_id, success=success, message=message
            )
        except Exception as e:
            logger.error(f"Failed to index text {text.text_id}: {e}")
            return IndexTextResponse(
                text_id=text.text_id, success=False, message=str(e)
            )

    async def index_picture(
        self, picture: DocumentPicture, index_name: Optional[str] = None
    ) -> IndexPictureResponse:
        """Index a single picture element."""
        try:
            index_name = index_name or self._index_name

            picture_data = picture.to_elastic_data()

            result = await self._es.index(
                index=index_name, id=picture.picture_id, document=picture_data
            )

            success = result.get("result") in ["created", "updated"]
            message = f"Picture {result.get('result', 'processed')}"

            return IndexPictureResponse(
                picture_id=picture.picture_id, success=success, message=message
            )
        except Exception as e:
            logger.error(f"Failed to index picture {picture.picture_id}: {e}")
            return IndexPictureResponse(
                picture_id=picture.picture_id, success=False, message=str(e)
            )

    async def index_table(
        self, table: DocumentTable, index_name: Optional[str] = None
    ) -> IndexTableResponse:
        """Index a single table element."""
        try:
            index_name = index_name or self._index_name

            table_data = table.to_elastic_data()

            result = await self._es.index(
                index=index_name, id=table.table_id, document=table_data
            )

            success = result.get("result") in ["created", "updated"]
            message = f"Table {result.get('result', 'processed')}"

            return IndexTableResponse(
                table_id=table.table_id, success=success, message=message
            )
        except Exception as e:
            logger.error(f"Failed to index table {table.table_id}: {e}")
            return IndexTableResponse(
                table_id=table.table_id, success=False, message=str(e)
            )

    async def get_document(
        self, document_id: str, index_name: Optional[str] = None
    ) -> GetDocumentResponse:
        """Get a document by ID."""
        try:
            index_name = index_name or self._index_name

            result = await self._es.get(index=index_name, id=document_id)

            return GetDocumentResponse(
                document_id=document_id, found=True, source=result.get("_source")
            )

        except NotFoundError:
            return GetDocumentResponse(
                document_id=document_id, found=False, source=None
            )
        except Exception as e:
            logger.error(f"Failed to get document {document_id}: {e}")
            return GetDocumentResponse(
                document_id=document_id, found=False, source=None
            )

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

    async def search_chunks(self, request: SearchRequest) -> SearchResponse:
        """Search chunks using text or vector similarity."""
        try:
            index_name = request.index_name or self._index_name

            # Build query for chunks
            query = self._build_chunk_search_query(request)

            search_params = {
                "index": index_name,
                "size": request.size,
                "query": query.get("query"),
                "highlight": {
                    "fields": {
                        "chunk.text": {"fragment_size": 150, "number_of_fragments": 3}
                    }
                },
            }

            # Add vector search if provided
            if request.vector and "knn" in query:
                search_params["knn"] = query["knn"]

            result = await self._es.search(**search_params)

            hits = []
            for hit in result["hits"]["hits"]:
                search_hit = SearchHit(
                    id=hit["_id"],
                    score=hit["_score"],
                    source=hit["_source"],
                    highlight=hit.get("highlight"),
                )
                hits.append(search_hit)

            return SearchResponse(
                hits=hits,
                total=result["hits"]["total"]["value"],
                max_score=result["hits"]["max_score"],
            )

        except Exception as e:
            logger.error(f"Search chunks failed: {e}")
            return SearchResponse(hits=[], total=0, max_score=None)

    async def search_documents(self, request: SearchRequest) -> SearchResponse:
        """Search documents using text query."""
        try:
            index_name = request.index_name or self._index_name

            # Build text query for documents
            query = {
                "bool": {
                    "must": [
                        {"exists": {"field": "document"}},
                        {
                            "bool": {
                                "should": [
                                    {
                                        "match": {
                                            "document.name": {
                                                "query": request.query,
                                                "boost": 2.0,
                                            }
                                        }
                                    },
                                    {
                                        "match": {
                                            "document.origin.filename": {
                                                "query": request.query,
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
            if request.filters:
                query["bool"]["filter"] = self._build_filters(request.filters)

            result = await self._es.search(
                index=index_name,
                size=request.size,
                query=query,
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

            hits = []
            for hit in result["hits"]["hits"]:
                search_hit = SearchHit(
                    id=hit["_id"],
                    score=hit["_score"],
                    source=hit["_source"],
                    highlight=hit.get("highlight"),
                )
                hits.append(search_hit)

            return SearchResponse(
                hits=hits,
                total=result["hits"]["total"]["value"],
                max_score=result["hits"]["max_score"],
            )

        except Exception as e:
            logger.error(f"Search documents failed: {e}")
            return SearchResponse(hits=[], total=0, max_score=None)

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

    def _build_chunk_search_query(self, request: SearchRequest) -> Dict[str, Any]:
        """Build search query for chunks."""
        query_parts = {}

        # Text search with field existence filter for chunks
        if request.query:
            text_query = {
                "bool": {
                    "must": [{"exists": {"field": "chunk"}}],
                    "should": [
                        {
                            "match": {
                                "chunk.text": {"query": request.query, "boost": 1.0}
                            }
                        },
                        {
                            "match": {
                                "chunk.meta.headings": {
                                    "query": request.query,
                                    "boost": 1.5,
                                }
                            }
                        },
                    ],
                }
            }

            # Add filters if provided
            if request.filters:
                text_query["bool"]["filter"] = self._build_filters(request.filters)

            query_parts["query"] = text_query

        # Vector search
        if request.vector:
            knn_query = {
                "field": "chunk.vector",
                "query_vector": request.vector,
                "k": request.size,
                "num_candidates": request.size * 10,
                "filter": {"exists": {"field": "chunk"}},
            }

            # Add additional filters to KNN if provided
            if request.filters:
                additional_filters = self._build_filters(request.filters)
                knn_query["filter"] = {
                    "bool": {
                        "must": [{"exists": {"field": "chunk"}}, *additional_filters]
                    }
                }

            query_parts["knn"] = knn_query

        # If both text and vector, create hybrid search
        if request.query and request.vector:
            # For hybrid search, we use the KNN with a filter that includes the text match
            hybrid_filter = {
                "bool": {
                    "must": [{"exists": {"field": "chunk"}}],
                    "should": [
                        {"match": {"chunk.text": request.query}},
                        {"match": {"chunk.meta.headings": request.query}},
                    ],
                }
            }

            if request.filters:
                existing_filters = self._build_filters(request.filters)
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
