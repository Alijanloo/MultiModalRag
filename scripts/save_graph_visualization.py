"""Example script to save the agentic RAG graph visualization."""

import asyncio
from multimodal_rag.container import ApplicationContainer
from multimodal_rag.frameworks.logging_config import get_logger

logger = get_logger(__name__)


async def main():
    try:
        container = ApplicationContainer()

        await container.init_resources()

        agentic_rag = await container.agentic_rag_use_case()

        if agentic_rag._graph is None:
            logger.error("Graph not initialized. Cannot save visualization.")
            print("Failed to save graph visualization: Graph not initialized")
            return

        png_data = agentic_rag._graph.get_graph().draw_mermaid_png()

        output_path = "data/agentic_rag_workflow.png"
        with open(output_path, "wb") as f:
            f.write(png_data)

        logger.info(f"Graph visualization saved to {output_path}")
        print(f"Graph visualization saved successfully to '{output_path}'")

    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f"Error: {e}")
    finally:
        try:
            await container.shutdown_resources()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
