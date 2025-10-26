"""Main pipeline orchestrator for YouTube RAG processing."""

from typing import Any

from src.utils.logging import get_logger

from .chunking_service import ChunkingService
from .config import YouTubeRAGConfig, get_config
from .embedding_service import EmbeddingService
from .schemas import ChunkWithEmbedding, PipelineResult, VideoMetadata
from .storage_service import StorageService
from .youtube_service import YouTubeService

logger = get_logger(__name__)


class YouTubeRAGPipeline:
    """Orchestrates the YouTube RAG pipeline execution.

    This class coordinates all services to fetch YouTube transcripts, chunk them,
    generate embeddings, and store in the vector database. It handles deduplication,
    retry logic, and comprehensive result tracking.
    """

    def __init__(self, config: YouTubeRAGConfig | None = None):
        """Initialize pipeline with all required services.

        Args:
            config: Configuration object. If None, loads from environment.
        """
        self.config = config or get_config()
        self.youtube_service = YouTubeService(self.config)
        self.chunking_service = ChunkingService(self.config)
        self.embedding_service = EmbeddingService(self.config)
        self.storage_service = StorageService(self.config)

        logger.info(
            "pipeline_initialized",
            channel_id=self.config.youtube_channel_id,
            days_back=self.config.days_back,
        )

    async def process_channel(self) -> PipelineResult:
        """Process all recent videos from the configured channel.

        This is the main entry point for the pipeline. It fetches recent videos
        from the channel and processes each one through the complete pipeline.

        Returns:
            PipelineResult with statistics and any errors encountered.

        Raises:
            Exception: If critical pipeline failure occurs.
        """
        logger.info(
            "pipeline_started",
            channel_id=self.config.youtube_channel_id,
            days_back=self.config.days_back,
        )

        result = PipelineResult(
            total_videos=0,
            processed=0,
            failed=0,
            skipped=0,
            chunks_created=0,
        )

        try:
            # 1. Fetch recent videos from channel
            videos = await self.youtube_service.get_recent_videos(
                channel_id=self.config.youtube_channel_id,
                days_back=self.config.days_back,
            )
            result.total_videos = len(videos)

            logger.info("videos_fetched", count=len(videos))

            # 2. Process each video through the pipeline
            for video in videos:
                video_result = await self._process_video(video)

                if video_result["status"] == "completed":
                    result.processed += 1
                    result.chunks_created += video_result["chunks"]
                elif video_result["status"] == "skipped":
                    result.skipped += 1
                else:
                    result.failed += 1
                    error_msg = f"{video.id}: {video_result.get('error', 'Unknown error')}"
                    result.errors.append(error_msg)

            logger.info(
                "pipeline_completed",
                processed=result.processed,
                failed=result.failed,
                skipped=result.skipped,
                chunks_created=result.chunks_created,
            )

            return result

        except Exception as e:
            logger.exception("pipeline_failed", error_type=type(e).__name__)
            raise

    async def _process_video(self, video: VideoMetadata) -> dict[str, Any]:
        """Process a single video through the complete pipeline.

        This method:
        1. Checks if video was already processed (deduplication)
        2. Fetches transcript with retry logic
        3. Chunks transcript with token awareness
        4. Generates embeddings for chunks
        5. Stores chunks in vector database
        6. Updates video status

        Args:
            video: Video metadata to process.

        Returns:
            Dictionary with processing result: {"status": str, "chunks": int, "error": str}.
        """
        logger.info("processing_video", video_id=video.id)

        # Check if already processed (deduplication)
        if await self.storage_service.is_video_processed(video.id):
            logger.info("video_already_processed", video_id=video.id)
            return {"status": "skipped", "chunks": 0}

        # Update status to processing
        await self.storage_service.save_video(video, status="processing")

        # Simple retry: try once, if fails try one more time, then give up
        for attempt in range(2):  # 0 and 1
            try:
                # 1. Fetch transcript
                transcript = await self.youtube_service.get_transcript(
                    video_id=video.id,
                    retry=(attempt > 0),
                )

                if transcript is None:
                    # Transcript unavailable - don't retry, just fail
                    await self.storage_service.update_video_status(
                        video.id,
                        status="failed",
                        error_message="Transcript unavailable",
                        retry_count=0,
                    )
                    return {
                        "status": "failed",
                        "error": "Transcript unavailable",
                        "chunks": 0,
                    }

                # 2. Chunk transcript
                chunks = self.chunking_service.chunk_transcript(transcript, video)

                # 3. Generate embeddings
                texts = [chunk.text_content for chunk in chunks]
                embeddings = await self.embedding_service.embed_batch(
                    texts, batch_size=self.config.batch_size
                )

                # 4. Combine chunks with embeddings
                chunks_with_embeddings = [
                    ChunkWithEmbedding(**chunk.model_dump(), embedding=embedding)
                    for chunk, embedding in zip(chunks, embeddings, strict=False)
                ]

                # 5. Store in database
                await self.storage_service.save_chunks(chunks_with_embeddings)

                # 6. Update status to completed
                await self.storage_service.update_video_status(
                    video.id,
                    status="completed",
                    retry_count=attempt,
                )

                logger.info("video_processed", video_id=video.id, chunks=len(chunks))
                return {"status": "completed", "chunks": len(chunks)}

            except Exception as e:
                logger.warning(
                    "video_processing_attempt_failed",
                    video_id=video.id,
                    attempt=attempt + 1,
                    error_type=type(e).__name__,
                    error=str(e),
                )

                # If this was the last attempt, fail permanently
                if attempt >= 1:
                    await self.storage_service.update_video_status(
                        video.id,
                        status="failed",
                        error_message=str(e),
                        retry_count=attempt + 1,
                    )
                    logger.error("video_processing_failed", video_id=video.id, error=str(e))
                    return {"status": "failed", "error": str(e), "chunks": 0}

                # Otherwise, log and retry
                logger.info("retrying_video", video_id=video.id)

        # Should never reach here, but just in case
        return {"status": "failed", "error": "Unknown error", "chunks": 0}
