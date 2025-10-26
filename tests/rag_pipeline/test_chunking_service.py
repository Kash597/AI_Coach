"""Unit tests for chunking service."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.rag_pipeline.chunking_service import ChunkingService
from src.rag_pipeline.config import YouTubeRAGConfig
from src.rag_pipeline.schemas import Transcript, TranscriptSegment, VideoMetadata


@pytest.mark.unit
class TestChunkingService:
    """Test suite for ChunkingService class."""

    @pytest.fixture
    def config(self) -> YouTubeRAGConfig:
        """Create test configuration."""
        return YouTubeRAGConfig(
            min_tokens=50,
            max_tokens=150,
            embedding_model="text-embedding-3-small",
        )

    @pytest.fixture
    def mock_tokenizer(self) -> MagicMock:
        """Create a mock tokenizer."""
        tokenizer = MagicMock()
        # Mock encode to return token count based on text length (simplified)
        tokenizer.encode = lambda text: ["token"] * (len(text.split()) * 2)
        return tokenizer

    @pytest.fixture
    def chunking_service(self, config: YouTubeRAGConfig, mock_tokenizer: MagicMock) -> ChunkingService:
        """Create chunking service with mocked tokenizer."""
        with patch("src.rag_pipeline.chunking_service.AutoTokenizer") as mock_auto:
            mock_auto.from_pretrained.return_value = mock_tokenizer
            service = ChunkingService(config)
            service.tokenizer = mock_tokenizer
            return service

    @pytest.fixture
    def sample_video(self) -> VideoMetadata:
        """Create sample video metadata."""
        return VideoMetadata(
            id="test_video_123",
            channel_id="UCtest",
            title="Test Video Title",
            url="https://youtube.com/watch?v=test_video_123",
            published_at=datetime(2024, 1, 1),
            duration_seconds=300,
        )

    @pytest.fixture
    def sample_transcript(self) -> Transcript:
        """Create sample transcript with multiple segments."""
        segments = [
            TranscriptSegment(
                text="This is the first segment of the transcript.",
                offset_ms=0,
                duration_ms=3000,
                lang="en",
            ),
            TranscriptSegment(
                text="This is the second segment with more content.",
                offset_ms=3000,
                duration_ms=3000,
                lang="en",
            ),
            TranscriptSegment(
                text="And this is the third segment continuing the discussion.",
                offset_ms=6000,
                duration_ms=4000,
                lang="en",
            ),
            TranscriptSegment(
                text="Finally, the fourth segment concludes the transcript.",
                offset_ms=10000,
                duration_ms=3000,
                lang="en",
            ),
        ]

        return Transcript(
            video_id="test_video_123",
            segments=segments,
            lang="en",
            available_langs=["en"],
        )

    def test_service_initialization(self, config: YouTubeRAGConfig) -> None:
        """Test that service initializes correctly."""
        with patch("src.rag_pipeline.chunking_service.AutoTokenizer") as mock_auto:
            mock_auto.from_pretrained.return_value = MagicMock()
            service = ChunkingService(config)

            assert service.config == config
            assert service.tokenizer is not None
            mock_auto.from_pretrained.assert_called_once()

    def test_chunk_transcript_basic(
        self,
        chunking_service: ChunkingService,
        sample_transcript: Transcript,
        sample_video: VideoMetadata,
    ) -> None:
        """Test basic transcript chunking functionality."""
        chunks = chunking_service.chunk_transcript(sample_transcript, sample_video)

        # Should create at least one chunk
        assert len(chunks) > 0

        # Verify chunk structure
        for chunk in chunks:
            assert chunk.video_id == sample_video.id
            assert len(chunk.text_content) > 0
            assert chunk.token_count > 0
            assert chunk.start_offset_ms >= 0
            assert chunk.end_offset_ms > chunk.start_offset_ms
            assert chunk.duration_ms > 0
            assert "video_title" in chunk.metadata
            assert "video_url" in chunk.metadata
            assert "timestamp_url" in chunk.metadata

    def test_chunk_preserves_timestamps(
        self,
        chunking_service: ChunkingService,
        sample_transcript: Transcript,
        sample_video: VideoMetadata,
    ) -> None:
        """Test that chunks preserve timestamp information."""
        chunks = chunking_service.chunk_transcript(sample_transcript, sample_video)

        # First chunk should start at beginning
        assert chunks[0].start_offset_ms == 0

        # Chunks should be sequential (no gaps or overlaps)
        for i in range(len(chunks) - 1):
            current_chunk = chunks[i]
            next_chunk = chunks[i + 1]
            # Next chunk should start where current ends or shortly after
            assert next_chunk.start_offset_ms >= current_chunk.start_offset_ms

    def test_chunk_metadata_enrichment(
        self,
        chunking_service: ChunkingService,
        sample_transcript: Transcript,
        sample_video: VideoMetadata,
    ) -> None:
        """Test that chunks include video metadata."""
        chunks = chunking_service.chunk_transcript(sample_transcript, sample_video)

        for chunk in chunks:
            assert chunk.metadata["video_title"] == sample_video.title
            assert chunk.metadata["video_url"] == sample_video.url
            assert chunk.metadata["channel_id"] == sample_video.channel_id
            # Timestamp URL should include video URL and timestamp parameter
            assert "youtube.com" in chunk.metadata["timestamp_url"]
            assert "&t=" in chunk.metadata["timestamp_url"]

    def test_chunk_empty_transcript(
        self,
        chunking_service: ChunkingService,
        sample_video: VideoMetadata,
    ) -> None:
        """Test handling of empty transcript."""
        empty_transcript = Transcript(
            video_id="test_video_123",
            segments=[],
            lang="en",
            available_langs=["en"],
        )

        chunks = chunking_service.chunk_transcript(empty_transcript, sample_video)

        # Empty transcript should produce no chunks
        assert len(chunks) == 0

    def test_chunk_single_segment(
        self,
        chunking_service: ChunkingService,
        sample_video: VideoMetadata,
    ) -> None:
        """Test chunking with single segment."""
        single_segment_transcript = Transcript(
            video_id="test_video_123",
            segments=[
                TranscriptSegment(
                    text="This is a single short segment.",
                    offset_ms=0,
                    duration_ms=2000,
                    lang="en",
                )
            ],
            lang="en",
            available_langs=["en"],
        )

        chunks = chunking_service.chunk_transcript(single_segment_transcript, sample_video)

        # Should create exactly one chunk
        assert len(chunks) == 1
        assert chunks[0].text_content == "This is a single short segment."
        assert chunks[0].chunk_index == 0

    def test_chunk_respects_max_tokens(
        self,
        config: YouTubeRAGConfig,
        mock_tokenizer: MagicMock,
        sample_video: VideoMetadata,
    ) -> None:
        """Test that chunks respect max token limit."""
        # Create service with small max_tokens
        config.max_tokens = 20
        config.min_tokens = 5

        with patch("src.rag_pipeline.chunking_service.AutoTokenizer") as mock_auto:
            mock_auto.from_pretrained.return_value = mock_tokenizer
            service = ChunkingService(config)
            service.tokenizer = mock_tokenizer

            # Create transcript with segments that would exceed max_tokens
            long_transcript = Transcript(
                video_id="test_video_123",
                segments=[
                    TranscriptSegment(
                        text="Word " * 50,  # Long segment
                        offset_ms=i * 1000,
                        duration_ms=1000,
                        lang="en",
                    )
                    for i in range(5)
                ],
                lang="en",
                available_langs=["en"],
            )

            chunks = service.chunk_transcript(long_transcript, sample_video)

            # Should create multiple chunks due to token limit
            assert len(chunks) > 1

    def test_chunk_indices_sequential(
        self,
        chunking_service: ChunkingService,
        sample_transcript: Transcript,
        sample_video: VideoMetadata,
    ) -> None:
        """Test that chunk indices are sequential."""
        chunks = chunking_service.chunk_transcript(sample_transcript, sample_video)

        # After merging, indices should be sequential starting from 0
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_merge_small_chunks(self, chunking_service: ChunkingService) -> None:
        """Test that small chunks are merged properly."""
        from src.rag_pipeline.schemas import Chunk

        # Create small chunks that should be merged
        small_chunks = [
            Chunk(
                video_id="test",
                chunk_index=0,
                text_content="Small chunk one.",
                start_offset_ms=0,
                end_offset_ms=1000,
                duration_ms=1000,
                token_count=10,  # Below min_tokens (50)
                metadata={"video_title": "Test"},
            ),
            Chunk(
                video_id="test",
                chunk_index=1,
                text_content="Small chunk two.",
                start_offset_ms=1000,
                end_offset_ms=2000,
                duration_ms=1000,
                token_count=10,  # Below min_tokens (50)
                metadata={"video_title": "Test"},
            ),
        ]

        merged = chunking_service._merge_small_chunks(small_chunks)

        # Should merge into one chunk
        assert len(merged) == 1
        assert "Small chunk one." in merged[0].text_content
        assert "Small chunk two." in merged[0].text_content
        assert merged[0].token_count == 20

    def test_no_merge_when_exceeds_max(self, chunking_service: ChunkingService) -> None:
        """Test that chunks are not merged if result would exceed max_tokens."""
        from src.rag_pipeline.schemas import Chunk

        chunks = [
            Chunk(
                video_id="test",
                chunk_index=0,
                text_content="Chunk one.",
                start_offset_ms=0,
                end_offset_ms=1000,
                duration_ms=1000,
                token_count=40,  # Below min but merging would exceed max (150)
                metadata={"video_title": "Test"},
            ),
            Chunk(
                video_id="test",
                chunk_index=1,
                text_content="Chunk two.",
                start_offset_ms=1000,
                end_offset_ms=2000,
                duration_ms=1000,
                token_count=120,  # Would exceed max_tokens if merged
                metadata={"video_title": "Test"},
            ),
        ]

        merged = chunking_service._merge_small_chunks(chunks)

        # Should not merge - would exceed max_tokens
        assert len(merged) == 2
