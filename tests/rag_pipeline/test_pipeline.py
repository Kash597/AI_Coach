"""Unit tests for YouTube RAG pipeline orchestrator."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.rag_pipeline.config import YouTubeRAGConfig
from src.rag_pipeline.pipeline import YouTubeRAGPipeline
from src.rag_pipeline.schemas import (
    Chunk,
    Transcript,
    TranscriptSegment,
    VideoMetadata,
)


@pytest.mark.unit
class TestYouTubeRAGPipeline:
    """Test suite for YouTubeRAGPipeline class."""

    @pytest.fixture
    def config(self) -> YouTubeRAGConfig:
        """Create test configuration."""
        return YouTubeRAGConfig(
            youtube_channel_id="UCtest123",
            days_back=7,
            max_retries=1,
            batch_size=5,
            min_tokens=50,
            max_tokens=150,
            supabase_url="https://test.supabase.co",
            supabase_key="test_key",
        )

    @pytest.fixture
    def sample_videos(self) -> list[VideoMetadata]:
        """Create sample video metadata list."""
        return [
            VideoMetadata(
                id="video_1",
                channel_id="UCtest123",
                title="First Video",
                url="https://youtube.com/watch?v=video_1",
                published_at=datetime(2024, 1, 1),
                duration_seconds=300,
            ),
            VideoMetadata(
                id="video_2",
                channel_id="UCtest123",
                title="Second Video",
                url="https://youtube.com/watch?v=video_2",
                published_at=datetime(2024, 1, 2),
                duration_seconds=250,
            ),
        ]

    @pytest.fixture
    def sample_transcript(self) -> Transcript:
        """Create sample transcript."""
        return Transcript(
            video_id="video_1",
            segments=[
                TranscriptSegment(
                    text="Sample transcript segment.",
                    offset_ms=0,
                    duration_ms=2000,
                    lang="en",
                )
            ],
            lang="en",
            available_langs=["en"],
        )

    @pytest.fixture
    def sample_chunks(self) -> list[Chunk]:
        """Create sample chunks."""
        return [
            Chunk(
                video_id="video_1",
                chunk_index=0,
                text_content="Sample transcript segment.",
                start_offset_ms=0,
                end_offset_ms=2000,
                duration_ms=2000,
                token_count=50,
                metadata={"video_title": "First Video"},
            )
        ]

    def test_pipeline_initialization(self, config: YouTubeRAGConfig) -> None:
        """Test pipeline initialization with all services."""
        with (
            patch("src.rag_pipeline.pipeline.YouTubeService"),
            patch("src.rag_pipeline.pipeline.ChunkingService"),
            patch("src.rag_pipeline.pipeline.EmbeddingService"),
            patch("src.rag_pipeline.pipeline.StorageService"),
        ):
            pipeline = YouTubeRAGPipeline(config)

            assert pipeline.config == config
            assert pipeline.youtube_service is not None
            assert pipeline.chunking_service is not None
            assert pipeline.embedding_service is not None
            assert pipeline.storage_service is not None

    def test_pipeline_initialization_without_config(self) -> None:
        """Test pipeline can initialize without explicit config."""
        with (
            patch("src.rag_pipeline.pipeline.get_config") as mock_get_config,
            patch("src.rag_pipeline.pipeline.YouTubeService"),
            patch("src.rag_pipeline.pipeline.ChunkingService"),
            patch("src.rag_pipeline.pipeline.EmbeddingService"),
            patch("src.rag_pipeline.pipeline.StorageService"),
        ):
            mock_config = MagicMock()
            mock_get_config.return_value = mock_config

            pipeline = YouTubeRAGPipeline()

            assert pipeline.config == mock_config
            mock_get_config.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_channel_success(
        self,
        config: YouTubeRAGConfig,
        sample_videos: list[VideoMetadata],
        sample_transcript: Transcript,
        sample_chunks: list[Chunk],
    ) -> None:
        """Test successful channel processing."""
        with (
            patch("src.rag_pipeline.pipeline.YouTubeService") as mock_youtube,
            patch("src.rag_pipeline.pipeline.ChunkingService") as mock_chunking,
            patch("src.rag_pipeline.pipeline.EmbeddingService") as mock_embedding,
            patch("src.rag_pipeline.pipeline.StorageService") as mock_storage,
        ):
            # Setup mocks
            mock_youtube_instance = MagicMock()
            mock_youtube_instance.get_recent_videos = AsyncMock(return_value=sample_videos)
            mock_youtube_instance.get_transcript = AsyncMock(return_value=sample_transcript)
            mock_youtube.return_value = mock_youtube_instance

            mock_chunking_instance = MagicMock()
            mock_chunking_instance.chunk_transcript.return_value = sample_chunks
            mock_chunking.return_value = mock_chunking_instance

            mock_embedding_instance = MagicMock()
            mock_embedding_instance.embed_batch = AsyncMock(
                return_value=[[0.1, 0.2, 0.3]]
            )
            mock_embedding.return_value = mock_embedding_instance

            mock_storage_instance = MagicMock()
            mock_storage_instance.is_video_processed = AsyncMock(return_value=False)
            mock_storage_instance.save_video = AsyncMock()
            mock_storage_instance.save_chunks = AsyncMock()
            mock_storage_instance.update_video_status = AsyncMock()
            mock_storage.return_value = mock_storage_instance

            # Run pipeline
            pipeline = YouTubeRAGPipeline(config)
            result = await pipeline.process_channel()

            # Verify results
            assert result.total_videos == 2
            assert result.processed == 2
            assert result.failed == 0
            assert result.skipped == 0
            assert result.chunks_created == 2  # 1 chunk per video

    @pytest.mark.asyncio
    async def test_process_channel_with_skipped_video(
        self, config: YouTubeRAGConfig, sample_videos: list[VideoMetadata]
    ) -> None:
        """Test channel processing with already processed video."""
        with (
            patch("src.rag_pipeline.pipeline.YouTubeService") as mock_youtube,
            patch("src.rag_pipeline.pipeline.ChunkingService"),
            patch("src.rag_pipeline.pipeline.EmbeddingService"),
            patch("src.rag_pipeline.pipeline.StorageService") as mock_storage,
        ):
            mock_youtube_instance = MagicMock()
            mock_youtube_instance.get_recent_videos = AsyncMock(return_value=sample_videos)
            mock_youtube.return_value = mock_youtube_instance

            mock_storage_instance = MagicMock()
            # First video already processed, second video not processed
            mock_storage_instance.is_video_processed = AsyncMock(
                side_effect=[True, False, False]
            )
            mock_storage_instance.save_video = AsyncMock()
            mock_storage_instance.save_chunks = AsyncMock()
            mock_storage_instance.update_video_status = AsyncMock()
            mock_storage.return_value = mock_storage_instance

            pipeline = YouTubeRAGPipeline(config)

            # Mock other methods to prevent actual processing
            pipeline.youtube_service.get_transcript = AsyncMock(return_value=None)

            result = await pipeline.process_channel()

            # First video should be skipped
            assert result.skipped >= 1

    @pytest.mark.asyncio
    async def test_process_channel_empty_results(self, config: YouTubeRAGConfig) -> None:
        """Test channel processing with no videos."""
        with (
            patch("src.rag_pipeline.pipeline.YouTubeService") as mock_youtube,
            patch("src.rag_pipeline.pipeline.ChunkingService"),
            patch("src.rag_pipeline.pipeline.EmbeddingService"),
            patch("src.rag_pipeline.pipeline.StorageService"),
        ):
            mock_youtube_instance = MagicMock()
            mock_youtube_instance.get_recent_videos = AsyncMock(return_value=[])
            mock_youtube.return_value = mock_youtube_instance

            pipeline = YouTubeRAGPipeline(config)
            result = await pipeline.process_channel()

            assert result.total_videos == 0
            assert result.processed == 0
            assert result.failed == 0
            assert result.skipped == 0

    @pytest.mark.asyncio
    async def test_process_video_transcript_unavailable(
        self, config: YouTubeRAGConfig, sample_videos: list[VideoMetadata]
    ) -> None:
        """Test video processing when transcript is unavailable."""
        with (
            patch("src.rag_pipeline.pipeline.YouTubeService") as mock_youtube,
            patch("src.rag_pipeline.pipeline.ChunkingService"),
            patch("src.rag_pipeline.pipeline.EmbeddingService"),
            patch("src.rag_pipeline.pipeline.StorageService") as mock_storage,
        ):
            mock_youtube_instance = MagicMock()
            mock_youtube_instance.get_transcript = AsyncMock(return_value=None)
            mock_youtube.return_value = mock_youtube_instance

            mock_storage_instance = MagicMock()
            mock_storage_instance.is_video_processed = AsyncMock(return_value=False)
            mock_storage_instance.save_video = AsyncMock()
            mock_storage_instance.update_video_status = AsyncMock()
            mock_storage.return_value = mock_storage_instance

            pipeline = YouTubeRAGPipeline(config)
            result = await pipeline._process_video(sample_videos[0])

            assert result["status"] == "failed"
            assert "Transcript unavailable" in result["error"]
            assert result["chunks"] == 0

            # Verify video status updated to failed
            mock_storage_instance.update_video_status.assert_called_once()
            # Check using kwargs instead of positional args
            call_kwargs = mock_storage_instance.update_video_status.call_args.kwargs
            assert call_kwargs["status"] == "failed"

    @pytest.mark.asyncio
    async def test_process_video_with_retry(
        self,
        config: YouTubeRAGConfig,
        sample_videos: list[VideoMetadata],
        sample_transcript: Transcript,
        sample_chunks: list[Chunk],
    ) -> None:
        """Test video processing with retry logic."""
        with (
            patch("src.rag_pipeline.pipeline.YouTubeService") as mock_youtube,
            patch("src.rag_pipeline.pipeline.ChunkingService") as mock_chunking,
            patch("src.rag_pipeline.pipeline.EmbeddingService") as mock_embedding,
            patch("src.rag_pipeline.pipeline.StorageService") as mock_storage,
        ):
            # First attempt fails, second succeeds
            mock_youtube_instance = MagicMock()
            mock_youtube_instance.get_transcript = AsyncMock(
                side_effect=[Exception("Network error"), sample_transcript]
            )
            mock_youtube.return_value = mock_youtube_instance

            mock_chunking_instance = MagicMock()
            mock_chunking_instance.chunk_transcript.return_value = sample_chunks
            mock_chunking.return_value = mock_chunking_instance

            mock_embedding_instance = MagicMock()
            mock_embedding_instance.embed_batch = AsyncMock(
                return_value=[[0.1, 0.2, 0.3]]
            )
            mock_embedding.return_value = mock_embedding_instance

            mock_storage_instance = MagicMock()
            mock_storage_instance.is_video_processed = AsyncMock(return_value=False)
            mock_storage_instance.save_video = AsyncMock()
            mock_storage_instance.save_chunks = AsyncMock()
            mock_storage_instance.update_video_status = AsyncMock()
            mock_storage.return_value = mock_storage_instance

            pipeline = YouTubeRAGPipeline(config)
            result = await pipeline._process_video(sample_videos[0])

            # Should succeed after retry
            assert result["status"] == "completed"
            assert result["chunks"] == 1

            # Should have called get_transcript twice
            assert mock_youtube_instance.get_transcript.call_count == 2

    @pytest.mark.asyncio
    async def test_process_video_max_retries_exceeded(
        self, config: YouTubeRAGConfig, sample_videos: list[VideoMetadata]
    ) -> None:
        """Test video processing when max retries exceeded."""
        with (
            patch("src.rag_pipeline.pipeline.YouTubeService") as mock_youtube,
            patch("src.rag_pipeline.pipeline.ChunkingService"),
            patch("src.rag_pipeline.pipeline.EmbeddingService"),
            patch("src.rag_pipeline.pipeline.StorageService") as mock_storage,
        ):
            # Always fail
            mock_youtube_instance = MagicMock()
            mock_youtube_instance.get_transcript = AsyncMock(
                side_effect=Exception("Persistent error")
            )
            mock_youtube.return_value = mock_youtube_instance

            mock_storage_instance = MagicMock()
            mock_storage_instance.is_video_processed = AsyncMock(return_value=False)
            mock_storage_instance.save_video = AsyncMock()
            mock_storage_instance.update_video_status = AsyncMock()
            mock_storage.return_value = mock_storage_instance

            pipeline = YouTubeRAGPipeline(config)
            result = await pipeline._process_video(sample_videos[0])

            # Should fail after max retries
            assert result["status"] == "failed"
            assert "error" in result

            # Should have attempted max_retries + 1 times (1 original + 1 retry)
            assert mock_youtube_instance.get_transcript.call_count == config.max_retries + 1

    @pytest.mark.asyncio
    async def test_process_video_embedding_failure(
        self,
        config: YouTubeRAGConfig,
        sample_videos: list[VideoMetadata],
        sample_transcript: Transcript,
        sample_chunks: list[Chunk],
    ) -> None:
        """Test video processing when embedding generation fails."""
        with (
            patch("src.rag_pipeline.pipeline.YouTubeService") as mock_youtube,
            patch("src.rag_pipeline.pipeline.ChunkingService") as mock_chunking,
            patch("src.rag_pipeline.pipeline.EmbeddingService") as mock_embedding,
            patch("src.rag_pipeline.pipeline.StorageService") as mock_storage,
        ):
            mock_youtube_instance = MagicMock()
            mock_youtube_instance.get_transcript = AsyncMock(return_value=sample_transcript)
            mock_youtube.return_value = mock_youtube_instance

            mock_chunking_instance = MagicMock()
            mock_chunking_instance.chunk_transcript.return_value = sample_chunks
            mock_chunking.return_value = mock_chunking_instance

            # Embedding fails
            mock_embedding_instance = MagicMock()
            mock_embedding_instance.embed_batch = AsyncMock(
                side_effect=Exception("Embedding API error")
            )
            mock_embedding.return_value = mock_embedding_instance

            mock_storage_instance = MagicMock()
            mock_storage_instance.is_video_processed = AsyncMock(return_value=False)
            mock_storage_instance.save_video = AsyncMock()
            mock_storage_instance.update_video_status = AsyncMock()
            mock_storage.return_value = mock_storage_instance

            pipeline = YouTubeRAGPipeline(config)
            result = await pipeline._process_video(sample_videos[0])

            # Should fail due to embedding error
            assert result["status"] == "failed"
            assert "error" in result

    @pytest.mark.asyncio
    async def test_process_video_storage_failure(
        self,
        config: YouTubeRAGConfig,
        sample_videos: list[VideoMetadata],
        sample_transcript: Transcript,
        sample_chunks: list[Chunk],
    ) -> None:
        """Test video processing when storage fails."""
        with (
            patch("src.rag_pipeline.pipeline.YouTubeService") as mock_youtube,
            patch("src.rag_pipeline.pipeline.ChunkingService") as mock_chunking,
            patch("src.rag_pipeline.pipeline.EmbeddingService") as mock_embedding,
            patch("src.rag_pipeline.pipeline.StorageService") as mock_storage,
        ):
            mock_youtube_instance = MagicMock()
            mock_youtube_instance.get_transcript = AsyncMock(return_value=sample_transcript)
            mock_youtube.return_value = mock_youtube_instance

            mock_chunking_instance = MagicMock()
            mock_chunking_instance.chunk_transcript.return_value = sample_chunks
            mock_chunking.return_value = mock_chunking_instance

            mock_embedding_instance = MagicMock()
            mock_embedding_instance.embed_batch = AsyncMock(
                return_value=[[0.1, 0.2, 0.3]]
            )
            mock_embedding.return_value = mock_embedding_instance

            # Storage fails
            mock_storage_instance = MagicMock()
            mock_storage_instance.is_video_processed = AsyncMock(return_value=False)
            mock_storage_instance.save_video = AsyncMock()
            mock_storage_instance.save_chunks = AsyncMock(
                side_effect=Exception("Database error")
            )
            mock_storage_instance.update_video_status = AsyncMock()
            mock_storage.return_value = mock_storage_instance

            pipeline = YouTubeRAGPipeline(config)
            result = await pipeline._process_video(sample_videos[0])

            # Should fail due to storage error
            assert result["status"] == "failed"
            assert "error" in result

    @pytest.mark.asyncio
    async def test_process_channel_handles_pipeline_error(
        self, config: YouTubeRAGConfig
    ) -> None:
        """Test that process_channel handles critical pipeline errors."""
        with (
            patch("src.rag_pipeline.pipeline.YouTubeService") as mock_youtube,
            patch("src.rag_pipeline.pipeline.ChunkingService"),
            patch("src.rag_pipeline.pipeline.EmbeddingService"),
            patch("src.rag_pipeline.pipeline.StorageService"),
        ):
            # Simulate critical error fetching videos
            mock_youtube_instance = MagicMock()
            mock_youtube_instance.get_recent_videos = AsyncMock(
                side_effect=Exception("Critical API error")
            )
            mock_youtube.return_value = mock_youtube_instance

            pipeline = YouTubeRAGPipeline(config)

            # Should raise exception for critical errors
            with pytest.raises(Exception, match="Critical API error"):
                await pipeline.process_channel()
