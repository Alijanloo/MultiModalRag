"""Message handlers for Telegram bot commands and messages."""

import asyncio
from typing import TYPE_CHECKING
from telegram import Update, ReplyKeyboardRemove
from telegram.ext import ContextTypes
from telegram.constants import ParseMode

from multimodal_rag.frameworks.logging_config import get_logger

if TYPE_CHECKING:
    from .telegram_bot_service import TelegramBotService

logger = get_logger(__name__)


class MessageHandlers:
    """Handles different types of Telegram messages and commands."""

    def __init__(self, bot_service: "TelegramBotService"):
        """Initialize message handlers.

        Args:
            bot_service: Reference to the main bot service
        """
        self._bot_service = bot_service

    async def _show_typing_continuously(self, context: ContextTypes.DEFAULT_TYPE, chat_id: int, stop_event: asyncio.Event) -> None:
        """Show typing indicator continuously until stopped.
        
        Args:
            context: Bot context for sending chat actions
            chat_id: Chat ID to send typing action to
            stop_event: Event to signal when to stop showing typing
        """
        try:
            while not stop_event.is_set():
                await context.bot.send_chat_action(chat_id=chat_id, action="typing")
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=4.0)
                    break
                except asyncio.TimeoutError:
                    continue
        except Exception as e:
            logger.error(f"Error in continuous typing: {e}")

    async def handle_start(
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

    async def handle_help(
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

    async def handle_clear(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /clear command to clear conversation history."""
        user_id = str(update.message.from_user.id)

        self._bot_service._conversation_manager.clear_conversation(user_id)
        self._bot_service._chunk_manager.clear_user_chunks(user_id)

        await update.message.reply_text(
            "🗑️ Conversation history cleared! You can start fresh.",
            parse_mode=ParseMode.MARKDOWN,
        )

    async def handle_message(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle user messages."""
        user = update.message.from_user
        user_id = str(user.id)
        message_text = update.message.text

        logger.info(
            f"Processing message from {user.first_name} ({user_id}): {message_text}"
        )

        stop_typing_event = asyncio.Event()
        typing_task = asyncio.create_task(
            self._show_typing_continuously(
                context, update.effective_chat.id, stop_typing_event
            )
        )

        try:
            conversation_history = (
                self._bot_service._conversation_manager.get_conversation_history(
                    user_id
                )
            )

            agent_response = await self._bot_service._agentic_rag_use_case.process_message(
                message=message_text,
                chat_id=user_id,
                conversation_history=conversation_history,
            )

            self._bot_service._conversation_manager.add_user_message(
                user_id, message_text
            )
            self._bot_service._conversation_manager.add_assistant_message(
                user_id, agent_response.content
            )

            await self._bot_service._response_formatter.send_agent_response(
                update, agent_response, user_id
            )

        except Exception as e:
            logger.error(f"Error processing message from {user_id}: {e}")
            await update.message.reply_text(
                "❌ I apologize, but I encountered an error while processing your message. "
                "Please try again or contact support if the issue persists.",
                parse_mode=ParseMode.MARKDOWN,
            )
        finally:
            stop_typing_event.set()
            try:
                await typing_task
            except Exception as e:
                logger.error(f"Error stopping typing task: {e}")

    async def handle_chunk_callback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle callback queries from chunk ID buttons."""
        query = update.callback_query
        user_id = str(query.from_user.id)
        chunk_id = query.data

        logger.info(f"User {user_id} requested chunk: {chunk_id}")

        try:
            # Answer the callback query first
            await query.answer()

            # Get chunk data
            chunk_data = self._bot_service._chunk_manager.get_chunk(user_id, chunk_id)

            if not chunk_data:
                await query.edit_message_text(
                    "❌ Chunk not found or expired. Please send a new message.",
                    parse_mode=ParseMode.MARKDOWN,
                )
                return

            # Format and send chunk content
            chunk_message = self._bot_service._chunk_manager.format_chunk_content(
                chunk_data
            )

            # Send chunk content as a new message
            await context.bot.send_message(
                chat_id=query.message.chat_id,
                text=chunk_message,
                parse_mode=ParseMode.MARKDOWN,
            )

        except Exception as e:
            logger.error(f"Error handling chunk callback: {e}")
            try:
                await query.edit_message_text(
                    "❌ Error retrieving chunk content. Please try again.",
                    parse_mode=ParseMode.MARKDOWN,
                )
            except Exception:
                # If edit fails, send a new message
                await context.bot.send_message(
                    chat_id=query.message.chat_id,
                    text="❌ Error retrieving chunk content. Please try again.",
                    parse_mode=ParseMode.MARKDOWN,
                )
