"""Command-line interface for running the YouTube RAG pipeline."""

import argparse
import asyncio

from src.utils.logging import get_logger

from .config import get_config
from .pipeline import YouTubeRAGPipeline

logger = get_logger(__name__)


async def main() -> None:
    """CLI entry point for YouTube RAG pipeline.

    This function parses command-line arguments, initializes the pipeline,
    runs it, and displays results to the user.
    """
    parser = argparse.ArgumentParser(
        description="YouTube RAG Pipeline - Extract and index video transcripts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process videos from last 7 days (using .env config)
  python -m src.rag_pipeline.cli

  # Override channel ID
  python -m src.rag_pipeline.cli --channel-id UCxxxxxxxxxxxxx

  # Process videos from last 14 days
  python -m src.rag_pipeline.cli --days-back 14

  # Dry run mode (no database writes)
  python -m src.rag_pipeline.cli --dry-run
        """,
    )

    parser.add_argument(
        "--channel-id",
        type=str,
        help="Override YouTube channel ID from environment",
    )
    parser.add_argument(
        "--days-back",
        type=int,
        help="Number of days to look back for videos",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode - fetch and process but don't write to database",
    )

    args = parser.parse_args()

    # Load configuration
    config = get_config()

    # Override config with CLI arguments
    if args.channel_id:
        config.youtube_channel_id = args.channel_id
    if args.days_back:
        config.days_back = args.days_back

    logger.info(
        "cli_started",
        channel_id=config.youtube_channel_id,
        days_back=config.days_back,
        dry_run=args.dry_run,
    )

    # Display configuration
    print("\n" + "=" * 60)
    print("YouTube RAG Pipeline")
    print("=" * 60)
    print(f"Channel ID: {config.youtube_channel_id}")
    print(f"Days back: {config.days_back}")
    print(f"Embedding provider: {config.embedding_provider}")
    print(f"Embedding model: {config.embedding_model}")
    print(f"Chunk size: {config.min_tokens}-{config.max_tokens} tokens")
    if args.dry_run:
        print("\n⚠️  DRY RUN MODE - No database writes will occur")
    print("=" * 60 + "\n")

    # Run pipeline
    pipeline = YouTubeRAGPipeline(config)

    try:
        result = await pipeline.process_channel()
    except Exception as e:
        logger.exception("pipeline_execution_failed", error_type=type(e).__name__)
        print(f"\n❌ Pipeline failed: {str(e)}")
        return

    # Display results
    print("\n" + "=" * 60)
    print("Pipeline Results")
    print("=" * 60)
    print(f"Total videos found: {result.total_videos}")
    print(f"Successfully processed: {result.processed}")
    print(f"Failed: {result.failed}")
    print(f"Skipped (already processed): {result.skipped}")
    print(f"Total chunks created: {result.chunks_created}")

    if result.errors:
        print("\nErrors encountered:")
        for error in result.errors:
            print(f"  ❌ {error}")
    else:
        print("\n✅ No errors encountered")

    print("=" * 60 + "\n")

    logger.info(
        "cli_completed",
        total_videos=result.total_videos,
        processed=result.processed,
        failed=result.failed,
        skipped=result.skipped,
        chunks_created=result.chunks_created,
    )


if __name__ == "__main__":
    asyncio.run(main())
