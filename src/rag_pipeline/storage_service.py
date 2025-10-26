"""Storage service for managing videos and chunks in Supabase vector database."""

from datetime import datetime
from typing import Any

from supabase import Client, create_client

from src.utils.logging import get_logger

from .config import YouTubeRAGConfig
from .schemas import ChunkWithEmbedding, VideoMetadata

logger = get_logger(__name__)


class StorageService:
    """Service for storing videos and transcript chunks in Supabase.

    This service handles all database operations including video metadata storage,
    chunk storage with embeddings, deduplication checks, and vector similarity search.
    """

    def __init__(self, config: YouTubeRAGConfig):
        """Initialize storage service with configuration.

        Args:
            config: Configuration object with Supabase credentials.
        """
        self.config = config
        self.client: Client = create_client(
            config.supabase_url,
            config.supabase_key,
        )
        logger.info(
            "storage_service_initialized",
            supabase_url=config.supabase_url,
        )

    async def is_video_processed(self, video_id: str) -> bool:
        """Check if a video has already been processed.

        This method queries the database to see if the video exists and has
        completed processing. Used for deduplication.

        Args:
            video_id: YouTube video ID to check.

        Returns:
            True if video exists with 'completed' status, False otherwise.
        """
        try:
            response = (
                self.client.table("videos")
                .select("id, status")
                .eq("id", video_id)
                .execute()
            )

            if response.data:
                status = response.data[0].get("status")
                is_completed = bool(status == "completed")
                logger.debug(
                    "video_check_completed",
                    video_id=video_id,
                    status=status,
                    is_processed=is_completed,
                )
                return is_completed

            logger.debug("video_not_found", video_id=video_id)
            return False

        except Exception as e:
            logger.exception(
                "video_check_failed",
                video_id=video_id,
                error_type=type(e).__name__,
            )
            # Assume not processed on error to be safe
            return False

    async def save_video(self, video: VideoMetadata, status: str = "processing") -> None:
        """Save or update video metadata in the database.

        Args:
            video: Video metadata to save.
            status: Processing status (default: "processing").

        Raises:
            Exception: If database operation fails.
        """
        try:
            data = {
                "id": video.id,
                "channel_id": video.channel_id,
                "title": video.title,
                "url": video.url,
                "published_at": (
                    video.published_at.isoformat() if video.published_at else None
                ),
                "duration_seconds": video.duration_seconds,
                "status": status,
                "updated_at": datetime.now().isoformat(),
            }

            self.client.table("videos").upsert(data).execute()
            logger.info("video_saved", video_id=video.id, status=status)

        except Exception as e:
            logger.exception(
                "video_save_failed",
                video_id=video.id,
                error_type=type(e).__name__,
            )
            raise

    async def save_chunks(self, chunks: list[ChunkWithEmbedding]) -> None:
        """Save transcript chunks with embeddings to the database.

        Args:
            chunks: List of chunks with embeddings to save.

        Raises:
            Exception: If database operation fails.
        """
        try:
            data = [
                {
                    "video_id": chunk.video_id,
                    "chunk_index": chunk.chunk_index,
                    "text_content": chunk.text_content,
                    "start_offset_ms": chunk.start_offset_ms,
                    "end_offset_ms": chunk.end_offset_ms,
                    "duration_ms": chunk.duration_ms,
                    "token_count": chunk.token_count,
                    "embedding": chunk.embedding,
                    "metadata": chunk.metadata,
                }
                for chunk in chunks
            ]

            self.client.table("transcript_chunks").insert(data).execute()
            logger.info(
                "chunks_saved",
                count=len(chunks),
                video_id=chunks[0].video_id if chunks else None,
            )

        except Exception as e:
            logger.exception(
                "chunks_save_failed",
                count=len(chunks),
                error_type=type(e).__name__,
            )
            raise

    async def update_video_status(
        self,
        video_id: str,
        status: str,
        error_message: str | None = None,
        retry_count: int = 0,
    ) -> None:
        """Update video processing status in the database.

        Args:
            video_id: YouTube video ID.
            status: New status (completed, failed, etc.).
            error_message: Error message if failed (optional).
            retry_count: Number of retries attempted.

        Raises:
            Exception: If database operation fails.
        """
        try:
            data = {
                "status": status,
                "error_message": error_message,
                "retry_count": retry_count,
                "updated_at": datetime.now().isoformat(),
            }

            if status == "completed":
                data["processed_at"] = datetime.now().isoformat()
                data["transcript_available"] = True

            self.client.table("videos").update(data).eq("id", video_id).execute()
            logger.info(
                "video_status_updated",
                video_id=video_id,
                status=status,
                retry_count=retry_count,
            )

        except Exception as e:
            logger.exception(
                "status_update_failed",
                video_id=video_id,
                error_type=type(e).__name__,
            )
            raise

    async def search_chunks(
        self,
        query_embedding: list[float],
        match_count: int = 5,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar chunks using vector similarity.

        This method uses the Supabase RPC function to perform vector similarity
        search with optional metadata filtering.

        Args:
            query_embedding: Query embedding vector.
            match_count: Number of results to return (default: 5).
            filter_metadata: Optional JSONB filter for metadata (default: None).

        Returns:
            List of matching chunks with similarity scores.

        Raises:
            Exception: If search operation fails.
        """
        try:
            filter_json = filter_metadata or {}

            response = self.client.rpc(
                "match_transcript_chunks",
                {
                    "query_embedding": query_embedding,
                    "match_count": match_count,
                    "filter": filter_json,
                },
            ).execute()

            results: list[dict[str, Any]] = response.data
            logger.info(
                "vector_search_completed",
                results=len(results),
                match_count=match_count,
            )
            return results

        except Exception as e:
            logger.exception(
                "vector_search_failed",
                error_type=type(e).__name__,
            )
            raise
