"""Unit tests for RAG tools service layer.

Tests helper functions and tool implementation functions for searching
YouTube transcript chunks and retrieving full video transcripts.
"""

from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from src.tools.rag_tools.service import (
    format_duration,
    format_timestamp_display,
    format_video_url,
    get_embedding,
    get_full_video_transcript,
    search_transcript_chunks,
)

# ==============================================================================
# Helper Function Tests
# ==============================================================================


@pytest.mark.unit
class TestFormatVideoUrl:
    """Test format_video_url helper function."""

    def test_format_video_url_without_timestamp(self) -> None:
        """Test formatting video URL without timestamp."""
        result = format_video_url("dQw4w9WgXcQ")
        assert result == "https://youtube.com/watch?v=dQw4w9WgXcQ"

    def test_format_video_url_with_timestamp(self) -> None:
        """Test formatting video URL with timestamp in milliseconds."""
        result = format_video_url("dQw4w9WgXcQ", 120000)
        assert result == "https://youtube.com/watch?v=dQw4w9WgXcQ&t=120s"

    def test_format_video_url_with_zero_timestamp(self) -> None:
        """Test that zero timestamp is ignored (treated as no timestamp)."""
        result = format_video_url("dQw4w9WgXcQ", 0)
        assert result == "https://youtube.com/watch?v=dQw4w9WgXcQ"

    def test_format_video_url_with_fractional_seconds(self) -> None:
        """Test that milliseconds are properly converted to integer seconds."""
        result = format_video_url("dQw4w9WgXcQ", 125500)
        assert result == "https://youtube.com/watch?v=dQw4w9WgXcQ&t=125s"


@pytest.mark.unit
class TestFormatTimestampDisplay:
    """Test format_timestamp_display helper function."""

    def test_format_timestamp_display_seconds_only(self) -> None:
        """Test formatting timestamp less than 1 minute."""
        result = format_timestamp_display(45000)  # 45 seconds
        assert result == "[00:45]"

    def test_format_timestamp_display_minutes_and_seconds(self) -> None:
        """Test formatting timestamp in MM:SS format."""
        result = format_timestamp_display(125000)  # 2:05
        assert result == "[02:05]"

    def test_format_timestamp_display_hours_minutes_seconds(self) -> None:
        """Test formatting timestamp in HH:MM:SS format."""
        result = format_timestamp_display(3725000)  # 1:02:05
        assert result == "[01:02:05]"

    def test_format_timestamp_display_zero(self) -> None:
        """Test formatting zero timestamp."""
        result = format_timestamp_display(0)
        assert result == "[00:00]"

    def test_format_timestamp_display_exactly_one_hour(self) -> None:
        """Test formatting exactly one hour."""
        result = format_timestamp_display(3600000)
        assert result == "[01:00:00]"


@pytest.mark.unit
class TestFormatDuration:
    """Test format_duration helper function."""

    def test_format_duration_seconds_only(self) -> None:
        """Test formatting duration less than 1 minute."""
        result = format_duration(45)
        assert result == "00:45"

    def test_format_duration_minutes_and_seconds(self) -> None:
        """Test formatting duration in MM:SS format."""
        result = format_duration(125)  # 2:05
        assert result == "02:05"

    def test_format_duration_hours_minutes_seconds(self) -> None:
        """Test formatting duration in HH:MM:SS format."""
        result = format_duration(3725)  # 1:02:05
        assert result == "01:02:05"

    def test_format_duration_zero(self) -> None:
        """Test formatting zero duration."""
        result = format_duration(0)
        assert result == "00:00"

    def test_format_duration_exactly_one_hour(self) -> None:
        """Test formatting exactly one hour."""
        result = format_duration(3600)
        assert result == "01:00:00"


@pytest.mark.unit
class TestGetEmbedding:
    """Test get_embedding helper function."""

    @pytest.mark.asyncio
    async def test_get_embedding_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test successful embedding generation."""
        # Mock environment variable
        monkeypatch.setenv("EMBEDDING_MODEL_CHOICE", "text-embedding-3-small")

        # Create mock client
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)

        # Test
        result = await get_embedding("test text", mock_client)

        # Assertions
        assert result == [0.1, 0.2, 0.3]
        mock_client.embeddings.create.assert_called_once_with(
            input=["test text"],
            model="text-embedding-3-small",
        )

    @pytest.mark.asyncio
    async def test_get_embedding_uses_default_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that default model is used when env var not set."""
        # Ensure env var is not set
        monkeypatch.delenv("EMBEDDING_MODEL_CHOICE", raising=False)

        # Create mock client
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2])]
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)

        # Test
        result = await get_embedding("test", mock_client)

        # Assertions
        assert result == [0.1, 0.2]
        mock_client.embeddings.create.assert_called_once_with(
            input=["test"],
            model="text-embedding-3-small",
        )

    @pytest.mark.asyncio
    async def test_get_embedding_raises_on_error(self) -> None:
        """Test that exceptions are properly raised."""
        # Create mock client that raises an exception
        mock_client = AsyncMock()
        mock_client.embeddings.create = AsyncMock(
            side_effect=Exception("API Error")
        )

        # Test and expect exception
        with pytest.raises(Exception, match="API Error"):
            await get_embedding("test", mock_client)


# ==============================================================================
# Tool Implementation Function Tests
# ==============================================================================


@pytest.mark.unit
class TestSearchTranscriptChunks:
    """Test search_transcript_chunks tool implementation."""

    @pytest.mark.asyncio
    async def test_search_transcript_chunks_success(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test successful transcript search with results."""
        # Mock embedding client
        mock_embedding_client = AsyncMock()
        mock_embedding_client.embeddings.create = AsyncMock(
            return_value=Mock(data=[Mock(embedding=[0.1, 0.2, 0.3])])
        )

        # Mock Supabase client
        mock_supabase = MagicMock()

        # Mock RPC response for transcript chunks
        mock_rpc_response = Mock()
        mock_rpc_response.data = [
            {
                "video_id": "video123",
                "text_content": "This is great coaching advice about goals.",
                "start_offset_ms": 60000,
                "similarity": 0.95,
            }
        ]
        mock_rpc_response.execute = Mock(return_value=mock_rpc_response)
        mock_supabase.rpc = Mock(return_value=mock_rpc_response)

        # Mock video metadata response
        mock_video_response = Mock()
        mock_video_response.data = [
            {
                "id": "video123",
                "title": "Goal Setting Masterclass",
                "url": "https://youtube.com/watch?v=video123",
                "duration_seconds": 600,
            }
        ]
        mock_video_select = Mock()
        mock_video_select.in_ = Mock(return_value=Mock(execute=Mock(return_value=mock_video_response)))
        mock_video_select.select = Mock(return_value=mock_video_select)
        mock_supabase.table = Mock(return_value=mock_video_select)

        # Test
        result = await search_transcript_chunks(
            mock_supabase, mock_embedding_client, "how to set goals", match_count=5
        )

        # Assertions
        assert "Result 1" in result
        assert "Goal Setting Masterclass" in result
        assert "95.00%" in result  # Similarity percentage
        assert "[01:00]" in result  # Timestamp display
        assert "This is great coaching advice about goals." in result

    @pytest.mark.asyncio
    async def test_search_transcript_chunks_no_results(self) -> None:
        """Test search with no matching results."""
        # Mock embedding client
        mock_embedding_client = AsyncMock()
        mock_embedding_client.embeddings.create = AsyncMock(
            return_value=Mock(data=[Mock(embedding=[0.1, 0.2])])
        )

        # Mock Supabase client with no results
        mock_supabase = MagicMock()
        mock_rpc_response = Mock()
        mock_rpc_response.data = []
        mock_rpc_response.execute = Mock(return_value=mock_rpc_response)
        mock_supabase.rpc = Mock(return_value=mock_rpc_response)

        # Test
        result = await search_transcript_chunks(
            mock_supabase, mock_embedding_client, "nonexistent query"
        )

        # Assertions
        assert "No relevant coaching content found" in result

    @pytest.mark.asyncio
    async def test_search_transcript_chunks_multiple_results(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test search with multiple results from different videos."""
        # Mock embedding client
        mock_embedding_client = AsyncMock()
        mock_embedding_client.embeddings.create = AsyncMock(
            return_value=Mock(data=[Mock(embedding=[0.1, 0.2])])
        )

        # Mock Supabase client
        mock_supabase = MagicMock()

        # Mock RPC response with multiple chunks
        mock_rpc_response = Mock()
        mock_rpc_response.data = [
            {
                "video_id": "video1",
                "text_content": "First result",
                "start_offset_ms": 30000,
                "similarity": 0.95,
            },
            {
                "video_id": "video2",
                "text_content": "Second result",
                "start_offset_ms": 45000,
                "similarity": 0.90,
            },
        ]
        mock_rpc_response.execute = Mock(return_value=mock_rpc_response)
        mock_supabase.rpc = Mock(return_value=mock_rpc_response)

        # Mock video metadata response
        mock_video_response = Mock()
        mock_video_response.data = [
            {
                "id": "video1",
                "title": "Video One",
                "url": "https://youtube.com/watch?v=video1",
                "duration_seconds": 300,
            },
            {
                "id": "video2",
                "title": "Video Two",
                "url": "https://youtube.com/watch?v=video2",
                "duration_seconds": 400,
            },
        ]
        mock_video_select = Mock()
        mock_video_select.in_ = Mock(return_value=Mock(execute=Mock(return_value=mock_video_response)))
        mock_video_select.select = Mock(return_value=mock_video_select)
        mock_supabase.table = Mock(return_value=mock_video_select)

        # Test
        result = await search_transcript_chunks(
            mock_supabase, mock_embedding_client, "test query", match_count=5
        )

        # Assertions
        assert "Result 1" in result
        assert "Result 2" in result
        assert "Video One" in result
        assert "Video Two" in result
        assert "First result" in result
        assert "Second result" in result

    @pytest.mark.asyncio
    async def test_search_transcript_chunks_error_handling(self) -> None:
        """Test error handling when embedding generation fails."""
        # Mock embedding client that raises an exception
        mock_embedding_client = AsyncMock()
        mock_embedding_client.embeddings.create = AsyncMock(
            side_effect=Exception("Embedding API Error")
        )

        mock_supabase = MagicMock()

        # Test and expect exception
        with pytest.raises(Exception, match="Embedding API Error"):
            await search_transcript_chunks(
                mock_supabase, mock_embedding_client, "test query"
            )


@pytest.mark.unit
class TestGetFullVideoTranscript:
    """Test get_full_video_transcript tool implementation."""

    @pytest.mark.asyncio
    async def test_get_full_video_transcript_success(self) -> None:
        """Test successful full transcript retrieval."""
        # Mock Supabase client
        mock_supabase = MagicMock()

        # Mock video metadata response
        mock_video_response = Mock()
        mock_video_response.data = {
            "id": "video123",
            "title": "Test Video",
            "url": "https://youtube.com/watch?v=video123",
            "duration_seconds": 300,
            "transcript_available": True,
        }
        mock_video_eq = Mock()
        mock_video_eq.single = Mock(return_value=Mock(execute=Mock(return_value=mock_video_response)))
        mock_video_eq.eq = Mock(return_value=mock_video_eq)
        mock_video_select = Mock()
        mock_video_select.select = Mock(return_value=mock_video_eq)

        # Mock transcript chunks response
        mock_chunks_response = Mock()
        mock_chunks_response.data = [
            {
                "chunk_index": 0,
                "text_content": "First chunk of transcript.",
                "start_offset_ms": 0,
                "end_offset_ms": 30000,
            },
            {
                "chunk_index": 1,
                "text_content": "Second chunk of transcript.",
                "start_offset_ms": 30000,
                "end_offset_ms": 60000,
            },
        ]
        mock_chunks_order = Mock()
        mock_chunks_order.execute = Mock(return_value=mock_chunks_response)
        mock_chunks_order.order = Mock(return_value=mock_chunks_order)
        mock_chunks_eq = Mock()
        mock_chunks_eq.eq = Mock(return_value=mock_chunks_order)
        mock_chunks_select = Mock()
        mock_chunks_select.select = Mock(return_value=mock_chunks_eq)

        # Configure mock to return different responses based on which table is called
        def table_side_effect(table_name: str) -> Mock:
            if table_name == "videos":
                return mock_video_select
            elif table_name == "transcript_chunks":
                return mock_chunks_select
            return Mock()

        mock_supabase.table = Mock(side_effect=table_side_effect)

        # Test
        result = await get_full_video_transcript(
            mock_supabase, "video123", max_chars=20000
        )

        # Assertions
        assert "Test Video" in result
        assert "https://youtube.com/watch?v=video123" in result
        assert "05:00" in result  # Duration
        assert "First chunk of transcript." in result
        assert "Second chunk of transcript." in result
        assert "Transcript Chunks:** 2" in result

    @pytest.mark.asyncio
    async def test_get_full_video_transcript_not_found(self) -> None:
        """Test behavior when video is not found."""
        # Mock Supabase client
        mock_supabase = MagicMock()

        # Mock video metadata response with no data
        mock_video_response = Mock()
        mock_video_response.data = None
        mock_video_eq = Mock()
        mock_video_eq.single = Mock(return_value=Mock(execute=Mock(return_value=mock_video_response)))
        mock_video_eq.eq = Mock(return_value=mock_video_eq)
        mock_video_select = Mock()
        mock_video_select.select = Mock(return_value=mock_video_eq)
        mock_supabase.table = Mock(return_value=mock_video_select)

        # Test
        result = await get_full_video_transcript(mock_supabase, "nonexistent123")

        # Assertions
        assert "Video with ID 'nonexistent123' not found" in result

    @pytest.mark.asyncio
    async def test_get_full_video_transcript_not_available(self) -> None:
        """Test behavior when transcript is not available."""
        # Mock Supabase client
        mock_supabase = MagicMock()

        # Mock video metadata response with transcript_available=False
        mock_video_response = Mock()
        mock_video_response.data = {
            "id": "video123",
            "title": "Test Video",
            "transcript_available": False,
        }
        mock_video_eq = Mock()
        mock_video_eq.single = Mock(return_value=Mock(execute=Mock(return_value=mock_video_response)))
        mock_video_eq.eq = Mock(return_value=mock_video_eq)
        mock_video_select = Mock()
        mock_video_select.select = Mock(return_value=mock_video_eq)
        mock_supabase.table = Mock(return_value=mock_video_select)

        # Test
        result = await get_full_video_transcript(mock_supabase, "video123")

        # Assertions
        assert "Transcript is not available" in result
        assert "Test Video" in result

    @pytest.mark.asyncio
    async def test_get_full_video_transcript_no_chunks(self) -> None:
        """Test behavior when no transcript chunks found."""
        # Mock Supabase client
        mock_supabase = MagicMock()

        # Mock video metadata response
        mock_video_response = Mock()
        mock_video_response.data = {
            "id": "video123",
            "title": "Test Video",
            "transcript_available": True,
        }
        mock_video_eq = Mock()
        mock_video_eq.single = Mock(return_value=Mock(execute=Mock(return_value=mock_video_response)))
        mock_video_eq.eq = Mock(return_value=mock_video_eq)
        mock_video_select = Mock()
        mock_video_select.select = Mock(return_value=mock_video_eq)

        # Mock transcript chunks response with no data
        mock_chunks_response = Mock()
        mock_chunks_response.data = []
        mock_chunks_order = Mock()
        mock_chunks_order.execute = Mock(return_value=mock_chunks_response)
        mock_chunks_order.order = Mock(return_value=mock_chunks_order)
        mock_chunks_eq = Mock()
        mock_chunks_eq.eq = Mock(return_value=mock_chunks_order)
        mock_chunks_select = Mock()
        mock_chunks_select.select = Mock(return_value=mock_chunks_eq)

        # Configure mock to return different responses
        def table_side_effect(table_name: str) -> Mock:
            if table_name == "videos":
                return mock_video_select
            elif table_name == "transcript_chunks":
                return mock_chunks_select
            return Mock()

        mock_supabase.table = Mock(side_effect=table_side_effect)

        # Test
        result = await get_full_video_transcript(mock_supabase, "video123")

        # Assertions
        assert "No transcript chunks found" in result

    @pytest.mark.asyncio
    async def test_get_full_video_transcript_truncation(self) -> None:
        """Test transcript truncation when exceeding max_chars."""
        # Mock Supabase client
        mock_supabase = MagicMock()

        # Mock video metadata response
        mock_video_response = Mock()
        mock_video_response.data = {
            "id": "video123",
            "title": "Long Video",
            "url": "https://youtube.com/watch?v=video123",
            "duration_seconds": 1200,
            "transcript_available": True,
        }
        mock_video_eq = Mock()
        mock_video_eq.single = Mock(return_value=Mock(execute=Mock(return_value=mock_video_response)))
        mock_video_eq.eq = Mock(return_value=mock_video_eq)
        mock_video_select = Mock()
        mock_video_select.select = Mock(return_value=mock_video_eq)

        # Mock transcript chunks response with long text
        long_text = "A" * 5000
        mock_chunks_response = Mock()
        mock_chunks_response.data = [
            {
                "chunk_index": 0,
                "text_content": long_text,
                "start_offset_ms": 0,
                "end_offset_ms": 30000,
            },
        ]
        mock_chunks_order = Mock()
        mock_chunks_order.execute = Mock(return_value=mock_chunks_response)
        mock_chunks_order.order = Mock(return_value=mock_chunks_order)
        mock_chunks_eq = Mock()
        mock_chunks_eq.eq = Mock(return_value=mock_chunks_order)
        mock_chunks_select = Mock()
        mock_chunks_select.select = Mock(return_value=mock_chunks_eq)

        # Configure mock
        def table_side_effect(table_name: str) -> Mock:
            if table_name == "videos":
                return mock_video_select
            elif table_name == "transcript_chunks":
                return mock_chunks_select
            return Mock()

        mock_supabase.table = Mock(side_effect=table_side_effect)

        # Test with small max_chars
        result = await get_full_video_transcript(
            mock_supabase, "video123", max_chars=100
        )

        # Assertions
        assert "[TRUNCATED: Transcript exceeded 100 characters limit]" in result

    @pytest.mark.asyncio
    async def test_get_full_video_transcript_error_handling(self) -> None:
        """Test error handling when database query fails."""
        # Mock Supabase client that raises an exception
        mock_supabase = MagicMock()
        mock_supabase.table = Mock(side_effect=Exception("Database Error"))

        # Test and expect exception
        with pytest.raises(Exception, match="Database Error"):
            await get_full_video_transcript(mock_supabase, "video123")
