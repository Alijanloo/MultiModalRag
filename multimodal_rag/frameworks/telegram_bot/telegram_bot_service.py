"""Telegram Bot service for multimodal RAG integration - refactored version."""

import asyncio
from typing import Optional
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
)
from telegram import Update

from multimodal_rag.frameworks.logging_config import get_logger
from multimodal_rag.usecases.langgraph_agent.agentic_rag import AgenticRAGUseCase

from .conversation_manager import ConversationManager
from .chunk_manager import ChunkManager
from .response_formatter import ResponseFormatter
from .message_handlers import MessageHandlers

logger = get_logger(__name__)


class TelegramBotService:
    """Telegram bot service that integrates with AgenticRAG for intelligent document-based conversations."""

    def __init__(
        self,
        token: str,
        agentic_rag_use_case: AgenticRAGUseCase,
        max_message_length: int = 4096,
        max_caption_length: int = 1024,
        max_conversation_length: int = 10,
    ):
        """Initialize the Telegram bot service.

        Args:
            token: Telegram bot token
            agentic_rag_use_case: AgenticRAG use case for processing messages
            max_message_length: Maximum length for Telegram messages
            max_caption_length: Maximum length for media captions
            max_conversation_length: Maximum number of messages to keep in conversation history
        """
        self._token = token
        self._agentic_rag_use_case = agentic_rag_use_case
        self._application: Optional[Application] = None

        # Initialize component managers
        self._conversation_manager = ConversationManager(max_conversation_length)
        self._chunk_manager = ChunkManager()
        self._response_formatter = ResponseFormatter(
            max_message_length, max_caption_length, self._chunk_manager
        )
        self._message_handlers = MessageHandlers(self)

    async def initialize(self) -> None:
        """Initialize the Telegram bot application."""
        try:
            self._application = Application.builder().token(self._token).build()

            # Add handlers
            self._application.add_handler(
                CommandHandler("start", self._message_handlers.handle_start)
            )
            self._application.add_handler(
                CommandHandler("help", self._message_handlers.handle_help)
            )
            self._application.add_handler(
                CommandHandler("clear", self._message_handlers.handle_clear)
            )
            self._application.add_handler(
                MessageHandler(
                    filters.TEXT & ~filters.COMMAND,
                    self._message_handlers.handle_message,
                )
            )
            self._application.add_handler(
                CallbackQueryHandler(self._message_handlers.handle_chunk_callback)
            )

            logger.info("Telegram bot initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Telegram bot: {e}")
            raise

    async def start_polling(self) -> None:
        """Start the bot with polling mode."""
        if not self._application:
            await self.initialize()

        try:
            logger.info("Starting Telegram bot polling...")

            # Initialize the application
            await self._application.initialize()

            # Start the updater manually for better control in async context
            await self._application.updater.start_polling(
                allowed_updates=Update.ALL_TYPES,
                drop_pending_updates=True,
            )

            # Start processing updates
            await self._application.start()

            logger.info("Telegram bot is now running...")

            # Keep the bot running until stopped
            while True:
                await asyncio.sleep(1)

        except asyncio.CancelledError:
            logger.info("Bot polling cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in bot polling: {e}")
            raise

    async def shutdown(self) -> None:
        """Shutdown the bot gracefully."""
        if self._application:
            try:
                logger.info("Stopping Telegram bot...")

                await self._application.stop()
                await self._application.updater.stop()
                await self._application.shutdown()

                logger.info("Telegram bot shutdown completed")
            except Exception as e:
                logger.error(f"Error during bot shutdown: {e}")

    @property
    def conversation_manager(self) -> ConversationManager:
        """Get the conversation manager."""
        return self._conversation_manager

    @property
    def chunk_manager(self) -> ChunkManager:
        """Get the chunk manager."""
        return self._chunk_manager

    @property
    def response_formatter(self) -> ResponseFormatter:
        """Get the response formatter."""
        return self._response_formatter

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.shutdown()
