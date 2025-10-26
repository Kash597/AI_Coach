"""Unit tests for storage service."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.rag_pipeline.config import YouTubeRAGConfig
from src.rag_pipeline.schemas import ChunkWithEmbedding, VideoMetadata
from src.rag_pipeline.storage_service import StorageService


@pytest.mark.unit
class TestStorageService:
    """Test suite for StorageService class."""

    @pytest.fixture
    def config(self) -> YouTubeRAGConfig:
        """Create test configuration."""
        return YouTubeRAGConfig(
            supabase_url="https://test.supabase.co",
            supabase_key="test_key",
        )

    @pytest.fixture
    def mock_supabase_client(self) -> MagicMock:
        """Create mock Supabase client."""
        return MagicMock()

    @pytest.fixture
    def storage_service(
        self, config: YouTubeRAGConfig, mock_supabase_client: MagicMock
    ) -> StorageService:
        """Create storage service with mocked Supabase client."""
        with patch("src.rag_pipeline.storage_service.create_client") as mock_create:
            mock_create.return_value = mock_supabase_client
            service = StorageService(config)
            service.client = mock_supabase_client
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
    def sample_chunks(self) -> list[ChunkWithEmbedding]:
        """Create sample chunks with embeddings."""
        return [
            ChunkWithEmbedding(
                video_id="test_video_123",
                chunk_index=0,
                text_content="First chunk content",
                start_offset_ms=0,
                end_offset_ms=3000,
                duration_ms=3000,
                token_count=50,
                embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
                metadata={"video_title": "Test Video"},
            ),
            ChunkWithEmbedding(
                video_id="test_video_123",
                chunk_index=1,
                text_content="Second chunk content",
                start_offset_ms=3000,
                end_offset_ms=6000,
                duration_ms=3000,
                token_count=45,
                embedding=[0.2, 0.3, 0.4, 0.5, 0.6],
                metadata={"video_title": "Test Video"},
            ),
        ]

    def test_service_initialization(self, config: YouTubeRAGConfig) -> None:
        """Test service initialization."""
        with patch("src.rag_pipeline.storage_service.create_client") as mock_create:
            mock_client = MagicMock()
            mock_create.return_value = mock_client

            service = StorageService(config)

            assert service.config == config
            assert service.client == mock_client
            mock_create.assert_called_once_with(
                config.supabase_url,
                config.supabase_key,
            )

    @pytest.mark.asyncio
    async def test_is_video_processed_true(
        self, storage_service: StorageService, mock_supabase_client: MagicMock
    ) -> None:
        """Test checking if video is processed returns True."""
        # Mock response for completed video
        mock_response = MagicMock()
        mock_response.data = [{"id": "test_video_123", "status": "completed"}]

        mock_query = MagicMock()
        mock_query.select.return_value.eq.return_value.execute.return_value = mock_response
        mock_supabase_client.table.return_value = mock_query

        result = await storage_service.is_video_processed("test_video_123")

        assert result is True
        mock_supabase_client.table.assert_called_once_with("videos")

    @pytest.mark.asyncio
    async def test_is_video_processed_false(
        self, storage_service: StorageService, mock_supabase_client: MagicMock
    ) -> None:
        """Test checking if video is processed returns False."""
        # Mock response for processing video
        mock_response = MagicMock()
        mock_response.data = [{"id": "test_video_123", "status": "processing"}]

        mock_query = MagicMock()
        mock_query.select.return_value.eq.return_value.execute.return_value = mock_response
        mock_supabase_client.table.return_value = mock_query

        result = await storage_service.is_video_processed("test_video_123")

        assert result is False

    @pytest.mark.asyncio
    async def test_is_video_processed_not_found(
        self, storage_service: StorageService, mock_supabase_client: MagicMock
    ) -> None:
        """Test checking if video not found in database."""
        # Mock empty response
        mock_response = MagicMock()
        mock_response.data = []

        mock_query = MagicMock()
        mock_query.select.return_value.eq.return_value.execute.return_value = mock_response
        mock_supabase_client.table.return_value = mock_query

        result = await storage_service.is_video_processed("nonexistent_video")

        assert result is False

    @pytest.mark.asyncio
    async def test_is_video_processed_error(
        self, storage_service: StorageService, mock_supabase_client: MagicMock
    ) -> None:
        """Test checking video handles database errors."""
        # Mock exception
        mock_query = MagicMock()
        mock_query.select.return_value.eq.return_value.execute.side_effect = Exception(
            "Database error"
        )
        mock_supabase_client.table.return_value = mock_query

        # Should return False on error (safe default)
        result = await storage_service.is_video_processed("test_video_123")
        assert result is False

    @pytest.mark.asyncio
    async def test_save_video_success(
        self,
        storage_service: StorageService,
        mock_supabase_client: MagicMock,
        sample_video: VideoMetadata,
    ) -> None:
        """Test saving video metadata."""
        mock_query = MagicMock()
        mock_query.upsert.return_value.execute.return_value = None
        mock_supabase_client.table.return_value = mock_query

        await storage_service.save_video(sample_video, status="processing")

        # Verify table and upsert called
        mock_supabase_client.table.assert_called_once_with("videos")
        mock_query.upsert.assert_called_once()

        # Check data structure
        call_args = mock_query.upsert.call_args[0][0]
        assert call_args["id"] == "test_video_123"
        assert call_args["channel_id"] == "UCtest"
        assert call_args["title"] == "Test Video Title"
        assert call_args["status"] == "processing"

    @pytest.mark.asyncio
    async def test_save_video_with_completed_status(
        self,
        storage_service: StorageService,
        mock_supabase_client: MagicMock,
        sample_video: VideoMetadata,
    ) -> None:
        """Test saving video with completed status."""
        mock_query = MagicMock()
        mock_query.upsert.return_value.execute.return_value = None
        mock_supabase_client.table.return_value = mock_query

        await storage_service.save_video(sample_video, status="completed")

        call_args = mock_query.upsert.call_args[0][0]
        assert call_args["status"] == "completed"

    @pytest.mark.asyncio
    async def test_save_video_error(
        self,
        storage_service: StorageService,
        mock_supabase_client: MagicMock,
        sample_video: VideoMetadata,
    ) -> None:
        """Test save video handles database errors."""
        mock_query = MagicMock()
        mock_query.upsert.return_value.execute.side_effect = Exception("Database error")
        mock_supabase_client.table.return_value = mock_query

        # Should raise exception
        with pytest.raises(Exception, match="Database error"):
            await storage_service.save_video(sample_video)

    @pytest.mark.asyncio
    async def test_save_chunks_success(
        self,
        storage_service: StorageService,
        mock_supabase_client: MagicMock,
        sample_chunks: list[ChunkWithEmbedding],
    ) -> None:
        """Test saving chunks with embeddings."""
        mock_query = MagicMock()
        mock_query.insert.return_value.execute.return_value = None
        mock_supabase_client.table.return_value = mock_query

        await storage_service.save_chunks(sample_chunks)

        # Verify table and insert called
        mock_supabase_client.table.assert_called_once_with("transcript_chunks")
        mock_query.insert.assert_called_once()

        # Check data structure
        call_args = mock_query.insert.call_args[0][0]
        assert len(call_args) == 2
        assert call_args[0]["video_id"] == "test_video_123"
        assert call_args[0]["chunk_index"] == 0
        assert call_args[0]["text_content"] == "First chunk content"
        assert call_args[0]["embedding"] == [0.1, 0.2, 0.3, 0.4, 0.5]

    @pytest.mark.asyncio
    async def test_save_chunks_empty_list(
        self,
        storage_service: StorageService,
        mock_supabase_client: MagicMock,
    ) -> None:
        """Test saving empty chunks list."""
        mock_query = MagicMock()
        mock_query.insert.return_value.execute.return_value = None
        mock_supabase_client.table.return_value = mock_query

        await storage_service.save_chunks([])

        # Should still call insert with empty list
        mock_query.insert.assert_called_once_with([])

    @pytest.mark.asyncio
    async def test_save_chunks_error(
        self,
        storage_service: StorageService,
        mock_supabase_client: MagicMock,
        sample_chunks: list[ChunkWithEmbedding],
    ) -> None:
        """Test save chunks handles database errors."""
        mock_query = MagicMock()
        mock_query.insert.return_value.execute.side_effect = Exception("Database error")
        mock_supabase_client.table.return_value = mock_query

        # Should raise exception
        with pytest.raises(Exception, match="Database error"):
            await storage_service.save_chunks(sample_chunks)

    @pytest.mark.asyncio
    async def test_update_video_status_completed(
        self, storage_service: StorageService, mock_supabase_client: MagicMock
    ) -> None:
        """Test updating video status to completed."""
        mock_query = MagicMock()
        mock_query.update.return_value.eq.return_value.execute.return_value = None
        mock_supabase_client.table.return_value = mock_query

        await storage_service.update_video_status(
            "test_video_123", status="completed", retry_count=0
        )

        # Verify update called
        mock_supabase_client.table.assert_called_once_with("videos")
        mock_query.update.assert_called_once()

        # Check data includes completed fields
        call_args = mock_query.update.call_args[0][0]
        assert call_args["status"] == "completed"
        assert "processed_at" in call_args
        assert call_args["transcript_available"] is True

    @pytest.mark.asyncio
    async def test_update_video_status_failed(
        self, storage_service: StorageService, mock_supabase_client: MagicMock
    ) -> None:
        """Test updating video status to failed with error message."""
        mock_query = MagicMock()
        mock_query.update.return_value.eq.return_value.execute.return_value = None
        mock_supabase_client.table.return_value = mock_query

        await storage_service.update_video_status(
            "test_video_123",
            status="failed",
            error_message="Transcript unavailable",
            retry_count=2,
        )

        call_args = mock_query.update.call_args[0][0]
        assert call_args["status"] == "failed"
        assert call_args["error_message"] == "Transcript unavailable"
        assert call_args["retry_count"] == 2
        assert "processed_at" not in call_args

    @pytest.mark.asyncio
    async def test_update_video_status_error(
        self, storage_service: StorageService, mock_supabase_client: MagicMock
    ) -> None:
        """Test update video status handles database errors."""
        mock_query = MagicMock()
        mock_query.update.return_value.eq.return_value.execute.side_effect = Exception(
            "Database error"
        )
        mock_supabase_client.table.return_value = mock_query

        # Should raise exception
        with pytest.raises(Exception, match="Database error"):
            await storage_service.update_video_status("test_video_123", status="completed")

    @pytest.mark.asyncio
    async def test_search_chunks_success(
        self, storage_service: StorageService, mock_supabase_client: MagicMock
    ) -> None:
        """Test vector similarity search."""
        # Mock search results
        mock_response = MagicMock()
        mock_response.data = [
            {
                "video_id": "test_video_123",
                "text_content": "Matching chunk",
                "similarity": 0.95,
            },
            {
                "video_id": "test_video_456",
                "text_content": "Another match",
                "similarity": 0.88,
            },
        ]

        mock_supabase_client.rpc.return_value.execute.return_value = mock_response

        query_embedding = [0.1, 0.2, 0.3]
        results = await storage_service.search_chunks(
            query_embedding=query_embedding,
            match_count=5,
        )

        # Verify results
        assert len(results) == 2
        assert results[0]["similarity"] == 0.95
        assert results[1]["text_content"] == "Another match"

        # Verify RPC called correctly
        mock_supabase_client.rpc.assert_called_once_with(
            "match_transcript_chunks",
            {
                "query_embedding": query_embedding,
                "match_count": 5,
                "filter": {},
            },
        )

    @pytest.mark.asyncio
    async def test_search_chunks_with_filter(
        self, storage_service: StorageService, mock_supabase_client: MagicMock
    ) -> None:
        """Test vector search with metadata filter."""
        mock_response = MagicMock()
        mock_response.data = []
        mock_supabase_client.rpc.return_value.execute.return_value = mock_response

        filter_metadata = {"channel_id": "UCtest"}
        await storage_service.search_chunks(
            query_embedding=[0.1, 0.2, 0.3],
            match_count=3,
            filter_metadata=filter_metadata,
        )

        # Verify filter passed to RPC
        call_args = mock_supabase_client.rpc.call_args[0][1]
        assert call_args["filter"] == {"channel_id": "UCtest"}

    @pytest.mark.asyncio
    async def test_search_chunks_empty_results(
        self, storage_service: StorageService, mock_supabase_client: MagicMock
    ) -> None:
        """Test search with no results."""
        mock_response = MagicMock()
        mock_response.data = []
        mock_supabase_client.rpc.return_value.execute.return_value = mock_response

        results = await storage_service.search_chunks([0.1, 0.2, 0.3])

        assert results == []

    @pytest.mark.asyncio
    async def test_search_chunks_error(
        self, storage_service: StorageService, mock_supabase_client: MagicMock
    ) -> None:
        """Test search handles database errors."""
        mock_supabase_client.rpc.return_value.execute.side_effect = Exception(
            "Search error"
        )

        # Should raise exception
        with pytest.raises(Exception, match="Search error"):
            await storage_service.search_chunks([0.1, 0.2, 0.3])
