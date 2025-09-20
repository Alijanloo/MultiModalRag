"""DTOs for document indexing and retrieval operations."""

from typing import List, NamedTuple


class BulkIndexResult(NamedTuple):
    """Simple result for bulk indexing operations."""
    total_indexed: int
    total_failed: int
    errors: List[str]
