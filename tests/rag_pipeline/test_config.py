"""Unit tests for YouTube RAG pipeline configuration."""


import pytest

from src.rag_pipeline.config import YouTubeRAGConfig, get_config


@pytest.mark.unit
class TestYouTubeRAGConfig:
    """Test suite for YouTubeRAGConfig class."""

    def test_config_with_defaults(self) -> None:
        """Test config creation with default values."""
        config = YouTubeRAGConfig()

        # Should create config even with empty/missing env vars
        assert isinstance(config, YouTubeRAGConfig)
        assert config.days_back == 7  # Default from env or fallback
        assert config.max_retries == 1
        assert config.batch_size == 5
        assert config.min_tokens == 400
        assert config.max_tokens == 1000
        assert config.embedding_provider == "openai"
        assert config.embedding_model == "text-embedding-3-small"

    def test_config_with_explicit_values(self) -> None:
        """Test config creation with explicit parameter values."""
        config = YouTubeRAGConfig(
            supadata_api_key="test_api_key",
            youtube_channel_id="UCtest123",
            days_back=14,
            max_retries=3,
            batch_size=10,
            min_tokens=200,
            max_tokens=800,
            embedding_provider="ollama",
            embedding_model="nomic-embed-text",
            supabase_url="https://test.supabase.co",
            supabase_key="test_key",
        )

        assert config.supadata_api_key == "test_api_key"
        assert config.youtube_channel_id == "UCtest123"
        assert config.days_back == 14
        assert config.max_retries == 3
        assert config.batch_size == 10
        assert config.min_tokens == 200
        assert config.max_tokens == 800
        assert config.embedding_provider == "ollama"
        assert config.embedding_model == "nomic-embed-text"
        assert config.supabase_url == "https://test.supabase.co"
        assert config.supabase_key == "test_key"

    def test_config_from_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test config loads from environment variables."""
        # Set environment variables
        monkeypatch.setenv("SUPADATA_API_KEY", "env_api_key")
        monkeypatch.setenv("YOUTUBE_CHANNEL_ID", "UCenv123")
        monkeypatch.setenv("YOUTUBE_DAYS_BACK", "30")
        monkeypatch.setenv("YOUTUBE_MAX_RETRIES", "5")
        monkeypatch.setenv("MIN_CHUNK_TOKENS", "300")
        monkeypatch.setenv("MAX_CHUNK_TOKENS", "1200")
        monkeypatch.setenv("EMBEDDING_PROVIDER", "openrouter")
        monkeypatch.setenv("EMBEDDING_MODEL_CHOICE", "custom-model")

        config = YouTubeRAGConfig()

        assert config.supadata_api_key == "env_api_key"
        assert config.youtube_channel_id == "UCenv123"
        assert config.days_back == 30
        assert config.max_retries == 5
        assert config.min_tokens == 300
        assert config.max_tokens == 1200
        assert config.embedding_provider == "openrouter"
        assert config.embedding_model == "custom-model"

    def test_get_config_function(self) -> None:
        """Test get_config helper function returns valid config."""
        config = get_config()

        assert isinstance(config, YouTubeRAGConfig)
        assert config.days_back > 0
        assert config.max_retries >= 0
        assert config.min_tokens > 0
        assert config.max_tokens > config.min_tokens

    def test_config_token_limits_validation(self) -> None:
        """Test that token limits are positive integers."""
        config = YouTubeRAGConfig(
            min_tokens=100,
            max_tokens=500,
        )

        assert config.min_tokens == 100
        assert config.max_tokens == 500
        assert config.max_tokens > config.min_tokens

    def test_config_empty_strings_allowed(self) -> None:
        """Test that empty strings are allowed for optional API keys."""
        # This should not raise an error - empty strings are valid defaults
        config = YouTubeRAGConfig(
            supadata_api_key="",
            embedding_api_key="",
        )

        assert config.supadata_api_key == ""
        assert config.embedding_api_key == ""
