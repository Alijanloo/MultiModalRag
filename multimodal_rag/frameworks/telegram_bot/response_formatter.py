"""Response formatting and sending utilities for Telegram bot."""

import asyncio
import io
import base64
from typing import List, Any, Optional
from telegram import Update, InputMediaPhoto, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.constants import ParseMode
from telegram.error import BadRequest, TimedOut, NetworkError

from multimodal_rag.frameworks.logging_config import get_logger

logger = get_logger(__name__)


class ResponseFormatter:
    """Handles formatting and sending responses to Telegram users."""

    def __init__(
        self,
        max_message_length: int = 4096,
        max_caption_length: int = 1024,
        chunk_manager=None,
    ):
        """Initialize response formatter.

        Args:
            max_message_length: Maximum length for Telegram messages
            max_caption_length: Maximum length for media captions
            chunk_manager: Reference to chunk manager for storing chunks
        """
        self._max_message_length = max_message_length
        self._max_caption_length = max_caption_length
        self._chunk_manager = chunk_manager

    async def send_agent_response(
        self, update: Update, agent_response, user_id: str
    ) -> None:
        """Send agent response with optional images and chunk buttons."""
        try:
            response_text = agent_response.content
            pictures = agent_response.pictures
            chunk_ids_used = agent_response.chunk_ids_used
            retrieved_chunks = agent_response.retrieved_chunks

            # Store chunks for callback access
            if chunk_ids_used and retrieved_chunks and self._chunk_manager:
                self._chunk_manager.store_chunks(
                    user_id, chunk_ids_used, retrieved_chunks
                )

            # Create inline keyboard for chunk IDs if available
            reply_markup = self._create_chunk_buttons(chunk_ids_used)

            # If there are pictures, send them as a media group with caption
            if pictures:
                await self._send_response_with_pictures(
                    update, response_text, pictures, reply_markup
                )
            else:
                # Send text-only response
                await self._send_text_response(
                    update, response_text, reply_markup=reply_markup
                )

        except Exception as e:
            logger.error(f"Error sending agent response: {e}")
            await update.message.reply_text(
                "❌ Error sending response. Please try again.",
                parse_mode=ParseMode.MARKDOWN,
            )

    def _create_chunk_buttons(
        self, chunk_ids_used: Optional[List[str]]
    ) -> Optional[InlineKeyboardMarkup]:
        """Create inline keyboard buttons for chunk IDs.

        Args:
            chunk_ids_used: List of chunk IDs to create buttons for

        Returns:
            InlineKeyboardMarkup or None if no chunk IDs provided
        """
        if not chunk_ids_used:
            return None

        chunk_buttons = []
        # Create buttons in rows of 2
        for i in range(0, len(chunk_ids_used), 2):
            row = []
            for j in range(i, min(i + 2, len(chunk_ids_used))):
                chunk_id = chunk_ids_used[j]
                button_text = f"📄 {chunk_id}"
                row.append(InlineKeyboardButton(button_text, callback_data=chunk_id))
            chunk_buttons.append(row)

        return InlineKeyboardMarkup(chunk_buttons)

    async def _send_response_with_pictures(
        self, update: Update, text: str, pictures: List[Any], reply_markup=None
    ) -> None:
        """Send response with pictures as media group."""
        try:
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
                await update.message.reply_media_group(
                    media=media_group, read_timeout=60
                )

                # If text was truncated, send the remaining text
                if len(text) > self._max_caption_length:
                    remaining_text = text[self._max_caption_length :]
                    await self._send_text_response(
                        update,
                        remaining_text,
                        is_continuation=True,
                        reply_markup=reply_markup,
                    )
                elif reply_markup:
                    # Send chunk buttons separately if we have them
                    await update.message.reply_text(
                        "📚 **View Source Chunks:**",
                        reply_markup=reply_markup,
                        parse_mode=ParseMode.MARKDOWN,
                    )
            else:
                # If no valid pictures, send text only
                await self._send_text_response(update, text, reply_markup=reply_markup)

        except (BadRequest, TimedOut, NetworkError) as e:
            logger.error(f"Telegram API error sending pictures: {e}")
            # Fallback to text-only response
            await self._send_text_response(update, text, reply_markup=reply_markup)
        except Exception as e:
            logger.error(f"Unexpected error sending pictures: {e}")
            await self._send_text_response(update, text, reply_markup=reply_markup)

    async def _send_text_response(
        self,
        update: Update,
        text: str,
        is_continuation: bool = False,
        reply_markup=None,
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
                    reply_markup=reply_markup,
                )
            else:
                # Send in chunks
                chunks = self._split_text(text, self._max_message_length)
                for i, chunk in enumerate(chunks):
                    if i > 0:
                        chunk = f"📄 *Continued ({i + 1}/{len(chunks)}):*\n\n{chunk}"

                    # Only add reply_markup to the last chunk
                    chunk_markup = reply_markup if i == len(chunks) - 1 else None

                    await update.message.reply_text(
                        chunk,
                        parse_mode=ParseMode.MARKDOWN,
                        reply_markup=chunk_markup,
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
