"""Telegram Bot framework service for multimodal RAG integration."""

import asyncio
import io
import base64
from typing import Dict, List, Optional, Any
from telegram import Update, ReplyKeyboardRemove
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from telegram.constants import ParseMode
from telegram.error import BadRequest, TimedOut, NetworkError

from multimodal_rag.frameworks.logging_config import get_logger
from multimodal_rag.usecases.langgraph_agent.agentic_rag import AgenticRAGUseCase
from multimodal_rag.usecases.langgraph_agent.dtos import ChatMessage

logger = get_logger(__name__)


class TelegramBotService:
    """Telegram bot service that integrates with AgenticRAG for intelligent document-based conversations."""

    def __init__(
        self,
        token: str,
        agentic_rag_use_case: AgenticRAGUseCase,
        max_message_length: int = 4096,
        max_caption_length: int = 1024,
    ):
        """Initialize the Telegram bot service.

        Args:
            token: Telegram bot token
            agentic_rag_use_case: AgenticRAG use case for processing messages
            max_message_length: Maximum length for Telegram messages
            max_caption_length: Maximum length for media captions
        """
        self._token = token
        self._agentic_rag_use_case = agentic_rag_use_case
        self._max_message_length = max_message_length
        self._max_caption_length = max_caption_length
        self._application: Optional[Application] = None
        self._user_conversations: Dict[str, List[ChatMessage]] = {}

    async def initialize(self) -> None:
        """Initialize the Telegram bot application."""
        try:
            self._application = Application.builder().token(self._token).build()

            self._application.add_handler(CommandHandler("start", self._handle_start))
            self._application.add_handler(CommandHandler("help", self._handle_help))
            self._application.add_handler(CommandHandler("clear", self._handle_clear))
            self._application.add_handler(
                MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message)
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

    async def _handle_start(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /start command."""
        user = update.message.from_user
        logger.info(f"User {user.first_name} ({user.id}) started conversation")

        welcome_message = (
            f"👋 Hello {user.first_name}!\n\n"
            "I'm your intelligent document assistant powered by multimodal RAG. "
            "I can help you find information from documents and answer your questions.\n\n"
            "📖 Simply send me a message with your question, and I'll search through "
            "the available documents to provide you with relevant answers.\n\n"
            "🖼️ If there are relevant images or diagrams in the documents, "
            "I'll include them in my response.\n\n"
            "Commands:\n"
            "• /help - Show this help message\n"
            "• /clear - Clear conversation history\n\n"
            "What would you like to know?"
        )

        await update.message.reply_text(
            welcome_message,
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=ReplyKeyboardRemove(),
        )

    async def _handle_help(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /help command."""
        help_message = (
            "🤖 *Multimodal RAG Assistant Help*\n\n"
            "*What I can do:*\n"
            "• Answer questions based on document content\n"
            "• Search through indexed documents\n"
            "• Provide relevant images and diagrams\n"
            "• Maintain conversation context\n\n"
            "*How to use:*\n"
            "1. Simply type your question\n"
            "2. I'll search relevant documents\n"
            "3. You'll get an answer with supporting images (if available)\n\n"
            "*Commands:*\n"
            "• /start - Start conversation\n"
            "• /help - Show this help\n"
            "• /clear - Clear conversation history\n\n"
            "*Tips:*\n"
            "• Be specific in your questions for better results\n"
            "• You can ask follow-up questions for clarification\n"
            "• Reference previous answers in your questions"
        )

        await update.message.reply_text(
            help_message,
            parse_mode=ParseMode.MARKDOWN,
        )

    async def _handle_clear(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /clear command to clear conversation history."""
        user_id = str(update.message.from_user.id)

        if user_id in self._user_conversations:
            del self._user_conversations[user_id]

        await update.message.reply_text(
            "🗑️ Conversation history cleared! You can start fresh.",
            parse_mode=ParseMode.MARKDOWN,
        )

    async def _handle_message(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle user messages."""
        user = update.message.from_user
        user_id = str(user.id)
        message_text = update.message.text

        logger.info(
            f"Processing message from {user.first_name} ({user_id}): {message_text}"
        )

        # Show typing action
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id, action="typing"
        )

        try:
            conversation_history = self._user_conversations.get(user_id, [])

            agent_response = await self._agentic_rag_use_case.process_message(
                message=message_text,
                chat_id=user_id,
                conversation_history=conversation_history,
            )

            conversation_history.append(ChatMessage(role="user", content=message_text))
            conversation_history.append(
                ChatMessage(role="assistant", content=agent_response.content)
            )

            self._user_conversations[user_id] = conversation_history[-10:]

            await self._send_agent_response(update, agent_response)

        except Exception as e:
            logger.error(f"Error processing message from {user_id}: {e}")
            await update.message.reply_text(
                "❌ I apologize, but I encountered an error while processing your message. "
                "Please try again or contact support if the issue persists.",
                parse_mode=ParseMode.MARKDOWN,
            )

    async def _send_agent_response(self, update: Update, agent_response) -> None:
        """Send agent response with optional images."""
        try:
            response_text = agent_response.content
            pictures = agent_response.pictures

            # If there are pictures, send them as a media group with caption
            if pictures:
                await self._send_response_with_pictures(update, response_text, pictures)
            else:
                # Send text-only response
                await self._send_text_response(update, response_text)

        except Exception as e:
            logger.error(f"Error sending agent response: {e}")
            await update.message.reply_text(
                "❌ Error sending response. Please try again.",
                parse_mode=ParseMode.MARKDOWN,
            )

    async def _send_response_with_pictures(
        self, update: Update, text: str, pictures: List[Any]
    ) -> None:
        """Send response with pictures as media group."""
        try:
            from telegram import InputMediaPhoto

            media_group = []

            # Prepare caption (truncate if necessary)
            caption = self._truncate_text(text, self._max_caption_length)

            # Process pictures and create media group
            for i, picture in enumerate(
                pictures[:10]
            ):  # Limit to 10 images per Telegram's constraint
                try:
                    if (
                        hasattr(picture, "image")
                        and hasattr(picture.image, "uri")
                        and picture.image.uri
                    ):
                        data_uri = picture.image.uri
                        if data_uri.startswith("data:image/"):
                            header, encoded = data_uri.split(",", 1)
                            image_bytes = base64.b64decode(encoded)
                            image_data = io.BytesIO(image_bytes)
                            image_data.name = f"image_{i}.jpg"

                            # Add caption only to the first image
                            if i == 0:
                                media_group.append(
                                    InputMediaPhoto(
                                        media=image_data,
                                        caption=caption,
                                        parse_mode=ParseMode.MARKDOWN,
                                    )
                                )
                            else:
                                media_group.append(InputMediaPhoto(media=image_data))
                        else:
                            logger.warning(
                                f"Invalid data URI format for picture {i}: {data_uri[:50]}..."
                            )

                except Exception as e:
                    logger.warning(f"Error processing picture {i}: {e}")
                    continue

            if media_group:
                await update.message.reply_media_group(media=media_group)

                # If text was truncated, send the remaining text
                if len(text) > self._max_caption_length:
                    remaining_text = text[self._max_caption_length :]
                    await self._send_text_response(
                        update, remaining_text, is_continuation=True
                    )
            else:
                # If no valid pictures, send text only
                await self._send_text_response(update, text)

        except (BadRequest, TimedOut, NetworkError) as e:
            logger.error(f"Telegram API error sending pictures: {e}")
            # Fallback to text-only response
            await self._send_text_response(update, text)
        except Exception as e:
            logger.error(f"Unexpected error sending pictures: {e}")
            await self._send_text_response(update, text)

    async def _send_text_response(
        self, update: Update, text: str, is_continuation: bool = False
    ) -> None:
        """Send text-only response, handling long messages."""
        try:
            if is_continuation:
                text = f"📄 *Continued:*\n\n{text}"

            # Split long messages
            if len(text) <= self._max_message_length:
                await update.message.reply_text(
                    text,
                    parse_mode=ParseMode.MARKDOWN,
                )
            else:
                # Send in chunks
                chunks = self._split_text(text, self._max_message_length)
                for i, chunk in enumerate(chunks):
                    if i > 0:
                        chunk = f"📄 *Continued ({i + 1}/{len(chunks)}):*\n\n{chunk}"

                    await update.message.reply_text(
                        chunk,
                        parse_mode=ParseMode.MARKDOWN,
                    )

                    # Small delay between messages to avoid rate limiting
                    if i < len(chunks) - 1:
                        await asyncio.sleep(0.5)

        except Exception as e:
            logger.error(f"Error sending text response: {e}")
            # Try without markdown parsing as fallback
            try:
                await update.message.reply_text(text)
            except Exception as fallback_error:
                logger.error(f"Fallback text sending also failed: {fallback_error}")

    def _truncate_text(self, text: str, max_length: int) -> str:
        """Truncate text to fit within the specified length."""
        if len(text) <= max_length:
            return text

        # Try to truncate at a sentence boundary
        truncated = text[: max_length - 3]
        last_period = truncated.rfind(".")
        last_newline = truncated.rfind("\n")

        # Use the latest sentence or paragraph break
        break_point = max(last_period, last_newline)
        if break_point > max_length * 0.8:  # Only if we don't lose too much content
            return text[: break_point + 1] + "..."
        else:
            return text[: max_length - 3] + "..."

    def _split_text(self, text: str, max_length: int) -> List[str]:
        """Split text into chunks that fit within the message length limit."""
        if len(text) <= max_length:
            return [text]

        chunks = []
        current_chunk = ""

        # Split by paragraphs first
        paragraphs = text.split("\n\n")

        for paragraph in paragraphs:
            # If adding this paragraph would exceed the limit
            if len(current_chunk) + len(paragraph) + 2 > max_length:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""

                # If the paragraph itself is too long, split it further
                if len(paragraph) > max_length:
                    # Split by sentences
                    sentences = paragraph.split(". ")
                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) + 2 > max_length:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                                current_chunk = ""

                        if current_chunk:
                            current_chunk += ". " + sentence
                        else:
                            current_chunk = sentence
                else:
                    current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.shutdown()
