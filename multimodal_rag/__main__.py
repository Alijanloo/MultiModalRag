import asyncio
import signal
import sys
from typing import Optional

from multimodal_rag.frameworks.logging_config import get_logger
from multimodal_rag.container import ApplicationContainer

logger = get_logger(__name__)


class TelegramBotRunner:
    """Runner for the Telegram bot with proper lifecycle management."""

    def __init__(self):
        self.container: Optional[ApplicationContainer] = None
        self.bot_service = None
        self._shutdown_event = asyncio.Event()

    async def initialize(self) -> None:
        """Initialize the application container and bot service."""
        try:
            self.container = ApplicationContainer()

            await self.container.init_resources()

            self.bot_service = await self.container.telegram_bot_service()

            logger.info("Telegram bot runner initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize bot runner: {e}")
            raise

    async def run(self) -> None:
        """Run the Telegram bot."""
        if not self.bot_service:
            raise RuntimeError("Bot service not initialized. Call initialize() first.")

        try:
            logger.info("Starting Telegram bot...")

            # Set up signal handlers for graceful shutdown
            def signal_handler(signum, frame):
                logger.info(f"Received signal {signum}, initiating shutdown...")
                self._shutdown_event.set()

            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)

            # Start bot polling in a task
            bot_task = asyncio.create_task(self.bot_service.start_polling())

            # Wait for shutdown signal
            shutdown_task = asyncio.create_task(self._shutdown_event.wait())

            try:
                # Wait for either bot completion or shutdown signal
                done, pending = await asyncio.wait(
                    [bot_task, shutdown_task], return_when=asyncio.FIRST_COMPLETED
                )

                # Cancel pending tasks
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

            finally:
                await self.bot_service.shutdown()

        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, shutting down...")
        except Exception as e:
            logger.error(f"Error running bot: {e}")
            raise

    async def shutdown(self) -> None:
        """Shutdown the bot and cleanup resources."""
        try:
            if self.container:
                logger.info("Shutting down container resources...")
                await self.container.shutdown_resources()

            logger.info("Shutdown completed successfully")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.shutdown()


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
