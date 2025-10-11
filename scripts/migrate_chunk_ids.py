"""Migration script to add chunk_id field to all chunk documents using their Elasticsearch _id."""

import asyncio
import time
from typing import List, Dict, Any

from multimodal_rag.container import ApplicationContainer
from multimodal_rag.frameworks.logging_config import get_logger


class ChunkIdMigration:
    """Migration class to add chunk_id field to existing chunks."""

    def __init__(self):
        """Initialize the migration with application container."""
        self.container = ApplicationContainer()
        self.logger = get_logger(__name__)
        self.es_client = None
        self.index_name = None

    async def setup(self):
        """Setup the migration by initializing dependencies."""
        await self.container.init_resources()
        self.es_client = await self.container.elasticsearch_client()
        es_config = self.container.elasticsearch_config()
        self.index_name = es_config.index_name
        self.logger.info(f"Migration setup completed for index: {self.index_name}")

    async def cleanup(self):
        """Cleanup resources after migration."""
        await self.container.shutdown_resources()
        self.logger.info("Migration cleanup completed")

    async def get_all_chunks_without_chunk_id(
        self, size: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Get all chunk documents that either don't have chunk_id or have null/empty chunk_id.

        Args:
            size: Batch size for processing

        Returns:
            List of chunk documents with their metadata
        """
        chunks = []

        query = {
            "bool": {
                "must": [{"exists": {"field": "chunk"}}],
                "must_not": [{"exists": {"field": "chunk.chunk_id"}}],
            }
        }

        try:
            response = await self.es_client.search(
                index=self.index_name, query=query, size=size, scroll="2m", _source=True
            )

            scroll_id = response.get("_scroll_id")
            hits = response["hits"]["hits"]

            while hits:
                for hit in hits:
                    chunks.append(
                        {
                            "id": hit["_id"],
                            "source": hit["_source"],
                            "score": hit.get("_score"),
                        }
                    )

                if not scroll_id:
                    break

                response = await self.es_client.scroll(scroll_id=scroll_id, scroll="2m")
                hits = response["hits"]["hits"]
                scroll_id = response.get("_scroll_id")

            if scroll_id:
                await self.es_client.clear_scroll(scroll_id=scroll_id)

        except Exception as e:
            self.logger.error(f"Error retrieving chunks: {e}")
            raise

        return chunks

    async def update_chunk_with_id(self, chunk_doc: Dict[str, Any]) -> bool:
        """
        Update a single chunk document with chunk_id field.

        Args:
            chunk_doc: Chunk document with id and source

        Returns:
            True if successful, False otherwise
        """
        try:
            doc_id = chunk_doc["id"]
            source = chunk_doc["source"]

            if "chunk" not in source:
                self.logger.warning(f"Document {doc_id} does not contain chunk field")
                return False

            source["chunk"]["chunk_id"] = doc_id

            result = await self.es_client.index(
                index=self.index_name, id=doc_id, document=source
            )

            if result.get("result") in ["updated", "created"]:
                self.logger.debug(f"Successfully updated chunk {doc_id}")
                return True
            else:
                self.logger.warning(f"Unexpected result for chunk {doc_id}: {result}")
                return False

        except Exception as e:
            self.logger.error(
                f"Error updating chunk {chunk_doc.get('id', 'unknown')}: {e}"
            )
            return False

    async def bulk_update_chunks(
        self, chunks: List[Dict[str, Any]], batch_size: int = 100
    ) -> Dict[str, int]:
        """
        Update chunks in bulk batches.

        Args:
            chunks: List of chunk documents to update
            batch_size: Size of each bulk update batch

        Returns:
            Dictionary with success and failure counts
        """
        results = {"success": 0, "failed": 0, "total": len(chunks)}

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            batch_results = await self._process_batch(batch)
            results["success"] += batch_results["success"]
            results["failed"] += batch_results["failed"]

            self.logger.info(
                f"Processed batch {i // batch_size + 1}: "
                f"{batch_results['success']} successful, {batch_results['failed']} failed"
            )

            await asyncio.sleep(0.1)

        return results

    async def _process_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, int]:
        """Process a single batch of chunks."""
        batch_results = {"success": 0, "failed": 0}

        actions = []
        for chunk_doc in batch:
            try:
                doc_id = chunk_doc["id"]
                source = chunk_doc["source"]

                if "chunk" not in source:
                    batch_results["failed"] += 1
                    continue

                source["chunk"]["chunk_id"] = doc_id

                action = {"_index": self.index_name, "_id": doc_id, "_source": source}
                actions.append(action)

            except Exception as e:
                self.logger.error(
                    f"Error preparing action for chunk {chunk_doc.get('id', 'unknown')}: {e}"
                )
                batch_results["failed"] += 1

        if actions:
            try:
                from elasticsearch.helpers import async_bulk

                success_count, failed_items = await async_bulk(
                    self.es_client,
                    actions,
                    index=self.index_name,
                    stats_only=True,
                    raise_on_error=False,
                )

                batch_results["success"] += success_count
                batch_results["failed"] += len(failed_items) if failed_items else 0

            except Exception as e:
                self.logger.error(f"Bulk update failed: {e}")
                batch_results["failed"] += len(actions)

        return batch_results

    async def verify_migration(self) -> Dict[str, int]:
        """
        Verify the migration by checking chunks with and without chunk_id.

        Returns:
            Dictionary with counts of chunks with and without chunk_id
        """
        try:
            with_chunk_id_query = {
                "bool": {
                    "must": [
                        {"exists": {"field": "chunk"}},
                        {"exists": {"field": "chunk.chunk_id"}},
                    ],
                    "must_not": [{"term": {"chunk.chunk_id": ""}}],
                }
            }

            with_chunk_id_response = await self.es_client.count(
                index=self.index_name, query=with_chunk_id_query
            )

            without_chunk_id_query = {
                "bool": {
                    "must": [{"exists": {"field": "chunk"}}],
                    "must_not": [{"exists": {"field": "chunk.chunk_id"}}],
                }
            }

            without_chunk_id_response = await self.es_client.count(
                index=self.index_name, query=without_chunk_id_query
            )

            total_chunks_query = {"exists": {"field": "chunk"}}
            total_chunks_response = await self.es_client.count(
                index=self.index_name, query=total_chunks_query
            )

            return {
                "total_chunks": total_chunks_response["count"],
                "with_chunk_id": with_chunk_id_response["count"],
                "without_chunk_id": without_chunk_id_response["count"],
            }

        except Exception as e:
            self.logger.error(f"Error verifying migration: {e}")
            raise

    async def run_migration(
        self, batch_size: int = 100, verify_before: bool = True
    ) -> Dict[str, Any]:
        """
        Run the complete migration process.

        Args:
            batch_size: Size of batches for bulk operations
            verify_before: Whether to verify counts before migration

        Returns:
            Migration results summary
        """
        start_time = time.time()

        try:
            await self.setup()

            if verify_before:
                pre_migration_counts = await self.verify_migration()
                self.logger.info(f"Pre-migration verification: {pre_migration_counts}")

                if pre_migration_counts["without_chunk_id"] == 0:
                    self.logger.info(
                        "No chunks need migration. All chunks already have chunk_id."
                    )
                    return {
                        "status": "completed",
                        "reason": "no_chunks_to_migrate",
                        "pre_migration_counts": pre_migration_counts,
                        "execution_time": time.time() - start_time,
                    }

            self.logger.info("Retrieving chunks that need chunk_id migration...")
            chunks_to_update = await self.get_all_chunks_without_chunk_id()

            if not chunks_to_update:
                self.logger.info("No chunks found that need migration.")
                return {
                    "status": "completed",
                    "reason": "no_chunks_found",
                    "execution_time": time.time() - start_time,
                }

            self.logger.info(f"Found {len(chunks_to_update)} chunks to update")

            self.logger.info("Starting bulk update of chunks...")
            update_results = await self.bulk_update_chunks(chunks_to_update, batch_size)

            post_migration_counts = await self.verify_migration()
            self.logger.info(f"Post-migration verification: {post_migration_counts}")

            execution_time = time.time() - start_time

            results = {
                "status": "completed",
                "execution_time": execution_time,
                "chunks_processed": len(chunks_to_update),
                "update_results": update_results,
                "post_migration_counts": post_migration_counts,
            }

            if verify_before:
                results["pre_migration_counts"] = pre_migration_counts

            self.logger.info(f"Migration completed in {execution_time:.2f} seconds")
            self.logger.info(f"Results: {results}")

            return results

        except Exception as e:
            self.logger.error(f"Migration failed: {e}")
            raise
        finally:
            await self.cleanup()


async def main():
    """Main function to run the chunk ID migration."""
    migration = ChunkIdMigration()

    try:
        results = await migration.run_migration(batch_size=100, verify_before=True)

        print("\n" + "=" * 60)
        print("CHUNK ID MIGRATION COMPLETED")
        print("=" * 60)
        print(f"Status: {results['status']}")
        print(f"Execution time: {results['execution_time']:.2f} seconds")

        if "chunks_processed" in results:
            print(f"Chunks processed: {results['chunks_processed']}")
            print(f"Successful updates: {results['update_results']['success']}")
            print(f"Failed updates: {results['update_results']['failed']}")

        if "post_migration_counts" in results:
            counts = results["post_migration_counts"]
            print("\nFinal state:")
            print(f"  Total chunks: {counts['total_chunks']}")
            print(f"  With chunk_id: {counts['with_chunk_id']}")
            print(f"  Without chunk_id: {counts['without_chunk_id']}")

        print("=" * 60)

    except Exception as e:
        print(f"Migration failed with error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
