"""Conversation management for Telegram bot users."""

from typing import Dict, List
from multimodal_rag.usecases.langgraph_agent.dtos import ChatMessage
from multimodal_rag.frameworks.logging_config import get_logger

logger = get_logger(__name__)


class ConversationManager:
    """Manages user conversations and chat history."""

    def __init__(self, max_conversation_length: int = 10):
        """Initialize conversation manager.

        Args:
            max_conversation_length: Maximum number of messages to keep in history
        """
        self._user_conversations: Dict[str, List[ChatMessage]] = {}
        self._max_conversation_length = max_conversation_length

    def get_conversation_history(self, user_id: str) -> List[ChatMessage]:
        """Get conversation history for a user.

        Args:
            user_id: User identifier

        Returns:
            List of chat messages in conversation history
        """
        return self._user_conversations.get(user_id, [])

    def add_message(self, user_id: str, message: ChatMessage) -> None:
        """Add a message to user's conversation history.

        Args:
            user_id: User identifier
            message: Chat message to add
        """
        if user_id not in self._user_conversations:
            self._user_conversations[user_id] = []

        self._user_conversations[user_id].append(message)

        # Trim conversation to max length
        if len(self._user_conversations[user_id]) > self._max_conversation_length:
            self._user_conversations[user_id] = self._user_conversations[user_id][
                -self._max_conversation_length :
            ]

        logger.debug(f"Added message to conversation for user {user_id}")

    def add_user_message(self, user_id: str, content: str) -> None:
        """Add a user message to conversation history.

        Args:
            user_id: User identifier
            content: Message content
        """
        message = ChatMessage(role="user", content=content)
        self.add_message(user_id, message)

    def add_assistant_message(self, user_id: str, content: str) -> None:
        """Add an assistant message to conversation history.

        Args:
            user_id: User identifier
            content: Message content
        """
        message = ChatMessage(role="assistant", content=content)
        self.add_message(user_id, message)

    def clear_conversation(self, user_id: str) -> bool:
        """Clear conversation history for a user.

        Args:
            user_id: User identifier

        Returns:
            True if conversation was found and cleared, False otherwise
        """
        if user_id in self._user_conversations:
            del self._user_conversations[user_id]
            logger.info(f"Cleared conversation history for user {user_id}")
            return True
        return False

    def get_active_users(self) -> List[str]:
        """Get list of users with active conversations.

        Returns:
            List of user IDs with conversation history
        """
        return list(self._user_conversations.keys())

    def get_conversation_count(self, user_id: str) -> int:
        """Get number of messages in user's conversation.

        Args:
            user_id: User identifier

        Returns:
            Number of messages in conversation
        """
        return len(self._user_conversations.get(user_id, []))
