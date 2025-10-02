"""Example usage of the Agentic RAG system."""

import asyncio
from typing import List

from multimodal_rag.container import ApplicationContainer
from multimodal_rag.usecases.langgraph_agent.dtos import ChatMessage
from multimodal_rag.frameworks.logging_config import get_logger

logger = get_logger(__name__)


async def main():
    """Example usage of the agentic RAG system."""
    # Initialize the application container
    container = ApplicationContainer()
    container.config.from_yaml("config.yaml")

    # Initialize resources
    await container.init_resources()

    try:
        # Get agentic RAG use case from the container
        agentic_rag = await container.agentic_rag_use_case()

        # Example conversation
        chat_id = "example_chat_001"
        conversation_history: List[ChatMessage] = []

        # Example queries
        queries = [
            # "Hello! How can you help me?",
            "how does narrow band imaging help in distinguishing the columnar mucosa from the surrounding squamous mucosa?",
        ]

        for query in queries:
            print(f"\n🤖 User: {query}")

            # Process the message
            response = await agentic_rag.process_message(
                message=query,
                chat_id=chat_id,
                conversation_history=conversation_history,
            )

            print(f"🤖 Assistant: {response.content}")

            # Display associated pictures if any
            if response.pictures:
                print(f"📸 Associated pictures: {len(response.pictures)}")
                for i, picture in enumerate(response.pictures, 1):
                    print(f"   {i}. {picture.label} (ID: {picture.picture_id})")
                    if picture.captions:
                        print(f"      Caption: {picture.captions}")

            # Display chunks used if any
            if response.chunks_used:
                print(f"📄 Used {len(response.chunks_used)} document chunks")

            # Update conversation history
            conversation_history.extend(
                [
                    ChatMessage(role="user", content=query, chat_id=chat_id),
                    ChatMessage(
                        role="assistant", content=response.content, chat_id=chat_id
                    ),
                ]
            )

            # Keep conversation history manageable
            if len(conversation_history) > 10:
                conversation_history = conversation_history[-10:]

            print("-" * 50)

    except Exception as e:
        logger.error(f"Error in agentic RAG example: {e}")
        raise
    finally:
        # Cleanup resources
        await container.shutdown_resources()


if __name__ == "__main__":
    asyncio.run(main())
