"""Unit tests for embedding service."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.rag_pipeline.config import YouTubeRAGConfig
from src.rag_pipeline.embedding_service import EmbeddingService


@pytest.mark.unit
class TestEmbeddingService:
    """Test suite for EmbeddingService class."""

    @pytest.fixture
    def config_openai(self) -> YouTubeRAGConfig:
        """Create test configuration for OpenAI provider."""
        return YouTubeRAGConfig(
            embedding_provider="openai",
            embedding_base_url="https://api.openai.com/v1",
            embedding_api_key="test_api_key",
            embedding_model="text-embedding-3-small",
        )

    @pytest.fixture
    def config_ollama(self) -> YouTubeRAGConfig:
        """Create test configuration for Ollama provider."""
        return YouTubeRAGConfig(
            embedding_provider="ollama",
            embedding_base_url="http://localhost:11434/v1",
            embedding_model="nomic-embed-text",
        )

    @pytest.fixture
    def mock_openai_client(self) -> MagicMock:
        """Create mock OpenAI client."""
        mock_client = MagicMock()
        mock_embeddings = MagicMock()
        mock_client.embeddings = mock_embeddings
        return mock_client

    def test_service_initialization_openai(self, config_openai: YouTubeRAGConfig) -> None:
        """Test service initialization with OpenAI provider."""
        with patch("src.rag_pipeline.embedding_service.AsyncOpenAI") as mock_openai:
            service = EmbeddingService(config_openai)

            assert service.config == config_openai
            mock_openai.assert_called_once_with(
                base_url="https://api.openai.com/v1",
                api_key="test_api_key",
            )

    def test_service_initialization_ollama(self, config_ollama: YouTubeRAGConfig) -> None:
        """Test service initialization with Ollama provider."""
        with patch("src.rag_pipeline.embedding_service.AsyncOpenAI") as mock_openai:
            service = EmbeddingService(config_ollama)

            assert service.config == config_ollama
            # Ollama should use "ollama" as API key
            mock_openai.assert_called_once_with(
                base_url="http://localhost:11434/v1",
                api_key="ollama",
            )

    @pytest.mark.asyncio
    async def test_embed_text_success(self, config_openai: YouTubeRAGConfig) -> None:
        """Test successful text embedding generation."""
        with patch("src.rag_pipeline.embedding_service.AsyncOpenAI") as mock_openai:
            # Create mock response
            mock_response = MagicMock()
            mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3, 0.4, 0.5])]

            # Setup mock client
            mock_client = MagicMock()
            mock_client.embeddings.create = AsyncMock(return_value=mock_response)
            mock_openai.return_value = mock_client

            service = EmbeddingService(config_openai)
            embedding = await service.embed_text("Test text to embed")

            # Verify result
            assert embedding == [0.1, 0.2, 0.3, 0.4, 0.5]
            assert len(embedding) == 5

            # Verify API call
            mock_client.embeddings.create.assert_called_once_with(
                input="Test text to embed",
                model="text-embedding-3-small",
            )

    @pytest.mark.asyncio
    async def test_embed_text_failure(self, config_openai: YouTubeRAGConfig) -> None:
        """Test embedding generation handles errors properly."""
        with patch("src.rag_pipeline.embedding_service.AsyncOpenAI") as mock_openai:
            # Setup mock to raise exception
            mock_client = MagicMock()
            mock_client.embeddings.create = AsyncMock(
                side_effect=Exception("API Error")
            )
            mock_openai.return_value = mock_client

            service = EmbeddingService(config_openai)

            # Should raise exception
            with pytest.raises(Exception, match="API Error"):
                await service.embed_text("Test text")

    @pytest.mark.asyncio
    async def test_embed_text_empty_string(self, config_openai: YouTubeRAGConfig) -> None:
        """Test embedding generation with empty string."""
        with patch("src.rag_pipeline.embedding_service.AsyncOpenAI") as mock_openai:
            # Create mock response
            mock_response = MagicMock()
            mock_response.data = [MagicMock(embedding=[0.0] * 1536)]

            mock_client = MagicMock()
            mock_client.embeddings.create = AsyncMock(return_value=mock_response)
            mock_openai.return_value = mock_client

            service = EmbeddingService(config_openai)
            embedding = await service.embed_text("")

            # Should still return embedding (API handles empty strings)
            assert isinstance(embedding, list)
            assert len(embedding) == 1536

    @pytest.mark.asyncio
    async def test_embed_batch_success(self, config_openai: YouTubeRAGConfig) -> None:
        """Test successful batch embedding generation."""
        with patch("src.rag_pipeline.embedding_service.AsyncOpenAI") as mock_openai:
            # Create mock responses for each text
            def create_mock_response(text: str) -> MagicMock:
                mock_response = MagicMock()
                # Return different embeddings for different texts
                mock_response.data = [
                    MagicMock(embedding=[float(ord(text[0])), 0.2, 0.3])
                ]
                return mock_response

            mock_client = MagicMock()
            mock_client.embeddings.create = AsyncMock(side_effect=lambda input, model: create_mock_response(input))
            mock_openai.return_value = mock_client

            service = EmbeddingService(config_openai)
            texts = ["First text", "Second text", "Third text"]
            embeddings = await service.embed_batch(texts, batch_size=2)

            # Should return same number of embeddings as texts
            assert len(embeddings) == 3

            # Each embedding should be a list of floats
            for embedding in embeddings:
                assert isinstance(embedding, list)
                assert len(embedding) == 3
                assert all(isinstance(x, float) for x in embedding)

            # Should have called embed for each text
            assert mock_client.embeddings.create.call_count == 3

    @pytest.mark.asyncio
    async def test_embed_batch_empty_list(self, config_openai: YouTubeRAGConfig) -> None:
        """Test batch embedding with empty list."""
        with patch("src.rag_pipeline.embedding_service.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            service = EmbeddingService(config_openai)
            embeddings = await service.embed_batch([])

            # Should return empty list
            assert embeddings == []

            # Should not call API
            mock_client.embeddings.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_embed_batch_single_item(self, config_openai: YouTubeRAGConfig) -> None:
        """Test batch embedding with single item."""
        with patch("src.rag_pipeline.embedding_service.AsyncOpenAI") as mock_openai:
            mock_response = MagicMock()
            mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]

            mock_client = MagicMock()
            mock_client.embeddings.create = AsyncMock(return_value=mock_response)
            mock_openai.return_value = mock_client

            service = EmbeddingService(config_openai)
            embeddings = await service.embed_batch(["Single text"], batch_size=10)

            # Should return one embedding
            assert len(embeddings) == 1
            assert embeddings[0] == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_embed_batch_with_batch_size(self, config_openai: YouTubeRAGConfig) -> None:
        """Test batch embedding respects batch_size parameter."""
        with patch("src.rag_pipeline.embedding_service.AsyncOpenAI") as mock_openai:
            call_count = 0

            async def mock_embed(*args: object, **kwargs: object) -> MagicMock:
                nonlocal call_count
                call_count += 1
                mock_response = MagicMock()
                mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
                return mock_response

            mock_client = MagicMock()
            mock_client.embeddings.create = mock_embed
            mock_openai.return_value = mock_client

            service = EmbeddingService(config_openai)

            # Process 7 texts with batch_size=3 (should create 3 batches)
            texts = [f"Text {i}" for i in range(7)]
            embeddings = await service.embed_batch(texts, batch_size=3)

            # Should return all embeddings
            assert len(embeddings) == 7

            # Should have called embed for each text (7 times)
            assert call_count == 7

    @pytest.mark.asyncio
    async def test_embed_batch_handles_error(self, config_openai: YouTubeRAGConfig) -> None:
        """Test batch embedding handles errors in batch processing."""
        with patch("src.rag_pipeline.embedding_service.AsyncOpenAI") as mock_openai:
            # First call succeeds, second fails
            call_count = 0

            async def mock_embed(*args: object, **kwargs: object) -> MagicMock:
                nonlocal call_count
                call_count += 1
                if call_count > 2:
                    raise Exception("Batch processing error")
                mock_response = MagicMock()
                mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
                return mock_response

            mock_client = MagicMock()
            mock_client.embeddings.create = mock_embed
            mock_openai.return_value = mock_client

            service = EmbeddingService(config_openai)
            texts = [f"Text {i}" for i in range(5)]

            # Should raise exception when batch fails
            with pytest.raises(Exception, match="Batch processing error"):
                await service.embed_batch(texts, batch_size=2)

    @pytest.mark.asyncio
    async def test_embed_batch_large_batch(self, config_openai: YouTubeRAGConfig) -> None:
        """Test batch embedding with large number of texts."""
        with patch("src.rag_pipeline.embedding_service.AsyncOpenAI") as mock_openai:
            mock_response = MagicMock()
            mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]

            mock_client = MagicMock()
            mock_client.embeddings.create = AsyncMock(return_value=mock_response)
            mock_openai.return_value = mock_client

            service = EmbeddingService(config_openai)

            # Test with 100 texts
            texts = [f"Text {i}" for i in range(100)]
            embeddings = await service.embed_batch(texts, batch_size=10)

            # Should return all embeddings
            assert len(embeddings) == 100

            # Should have been called 100 times
            assert mock_client.embeddings.create.call_count == 100
