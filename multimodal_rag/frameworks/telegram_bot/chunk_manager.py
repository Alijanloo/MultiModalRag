"""Document chunk management for Telegram bot."""

from typing import Dict, List, Any, Optional
from multimodal_rag.frameworks.logging_config import get_logger

logger = get_logger(__name__)


class ChunkManager:
    """Manages document chunks for user interactions."""

    def __init__(self):
        """Initialize chunk manager."""
        self._user_chunks: Dict[str, Dict[str, Any]] = {}

    def store_chunks(
        self, user_id: str, chunk_ids: List[str], retrieved_chunks: List[Any]
    ) -> None:
        """Store chunks for a user session.

        Args:
            user_id: User identifier
            chunk_ids: List of chunk identifiers
            retrieved_chunks: List of retrieved chunk objects
        """
        if not chunk_ids or not retrieved_chunks:
            return

        if user_id not in self._user_chunks:
            self._user_chunks[user_id] = {}

        # Map chunk IDs to chunk data for this user
        for chunk_id, chunk in zip(chunk_ids, retrieved_chunks):
            self._user_chunks[user_id][chunk_id] = {
                "text": chunk.text,
                "document_id": chunk.document_id,
                "chunk_id": chunk_id,
            }

        logger.debug(
            f"Stored {len(chunk_ids)} chunks for user {user_id}: {chunk_ids}"
        )

    def get_chunk(self, user_id: str, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get chunk data for a user.

        Args:
            user_id: User identifier
            chunk_id: Chunk identifier

        Returns:
            Chunk data dictionary or None if not found
        """
        user_chunks = self._user_chunks.get(user_id, {})
        return user_chunks.get(chunk_id)

    def clear_user_chunks(self, user_id: str) -> bool:
        """Clear all chunks for a user.

        Args:
            user_id: User identifier

        Returns:
            True if chunks were found and cleared, False otherwise
        """
        if user_id in self._user_chunks:
            del self._user_chunks[user_id]
            logger.info(f"Cleared chunks for user {user_id}")
            return True
        return False

    def get_user_chunk_ids(self, user_id: str) -> List[str]:
        """Get all chunk IDs for a user.

        Args:
            user_id: User identifier

        Returns:
            List of chunk IDs
        """
        user_chunks = self._user_chunks.get(user_id, {})
        return list(user_chunks.keys())

    def format_chunk_content(
        self, chunk_data: Dict[str, Any], max_length: int = 3000
    ) -> str:
        """Format chunk content for display.

        Args:
            chunk_data: Chunk data dictionary
            max_length: Maximum length for chunk text

        Returns:
            Formatted chunk content
        """
        chunk_text = chunk_data.get("text", "No content available")
        document_id = chunk_data.get("document_id", "Unknown")
        chunk_id = chunk_data.get("chunk_id", "Unknown")

        # Truncate if too long
        if len(chunk_text) > max_length:
            chunk_text = chunk_text[:max_length] + "..."

        return (
            f"📄 **Document Chunk: {chunk_id}**\n\n"
            f"📋 **Document ID:** `{document_id}`\n\n"
            f"**Content:**\n{chunk_text}"
        )
