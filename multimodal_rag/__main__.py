import asyncio
import sys

from multimodal_rag.frameworks.logging_config import get_logger
from multimodal_rag.frameworks.telegram_bot.runner import TelegramBotRunner

logger = get_logger(__name__)


async def main():
    """Main function to run the Telegram bot."""
    try:
        async with TelegramBotRunner() as bot_runner:
            await bot_runner.run()
    except Exception as e:
        logger.error(f"Critical error in main: {e}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application terminated by user")
    except Exception as e:
        logger.error(f"Application failed: {e}")
        sys.exit(1)
