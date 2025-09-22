"""Example usage of bulk indexing from data/indexing directories."""

import asyncio

from multimodal_rag.container import ApplicationContainer
from multimodal_rag.frameworks.logging_config import get_logger


async def bulk_index_documents():
    """Bulk index all documents from data/indexing directories."""
    logger = get_logger(__name__)

    # Create and configure container
    container = ApplicationContainer()

    try:
        # Initialize system
        logger.info("🚀 Initializing MultiModal RAG system...")

        # Get the document repository to initialize indices
        document_repository = await container.document_repository()
        await document_repository.initialize_indices()
        logger.info("✅ Elasticsearch index initialized")

        # Get indexing use case
        indexing_use_case = await container.document_indexing_use_case()

        # Bulk index all documents from data/indexing directories
        logger.info("🔄 Starting bulk indexing from data/indexing directories...")

        responses = await indexing_use_case.bulk_index_from_directory(
            indexing_directory="data/indexing",
            index_name="multimodal_rag",
            max_tokens=512,
            generate_embeddings=True
        )

        # Summary
        total_indexed = sum(response.total_indexed for response in responses)
        total_errors = sum(len(response.errors) for response in responses if response.errors)

        logger.info("🎉 Bulk indexing completed!")
        logger.info("📊 Summary:")
        logger.info(f"  - Processed documents: {len(responses)}")
        logger.info(f"  - Total items indexed: {total_indexed}")
        logger.info(f"  - Total errors: {total_errors}")

        if total_errors > 0:
            logger.warning("⚠️  Some errors occurred during indexing. Check logs for details.")

        # Verify indexing by searching
        logger.info("🔍 Verifying indexing with a test search...")
        search_use_case = await container.document_search_use_case()

        search_results = await search_use_case.search_chunks_by_text(
            query="disease", size=5, index_name="multimodal_rag"
        )

        logger.info(
            f"📝 Search verification: Found {len(search_results)} chunks matching 'disease'"
        )

        # Show sample results
        for i, chunk in enumerate(search_results[:3], 1):
            logger.info(f"  Sample {i}: {chunk.text[:100]}...")

    except Exception as e:
        logger.error(f"❌ Error during bulk indexing: {e}")
        import traceback

        traceback.print_exc()


async def main():
    """Main function to run bulk indexing example."""
    await bulk_index_documents()


if __name__ == "__main__":
    asyncio.run(main())
