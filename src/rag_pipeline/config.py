"""Configuration module for YouTube RAG pipeline."""

import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables from .env file
load_dotenv()


class YouTubeRAGConfig(BaseModel):
    """Configuration for YouTube RAG pipeline.

    This configuration class manages all settings for the YouTube transcript
    fetching, chunking, embedding, and storage pipeline. All settings can be
    overridden via environment variables.
    """

    # Supadata API settings
    supadata_api_key: str = Field(
        default_factory=lambda: os.getenv("SUPADATA_API_KEY", "")
    )
    youtube_channel_id: str = Field(
        default_factory=lambda: os.getenv("YOUTUBE_CHANNEL_ID", "")
    )

    # Processing settings
    days_back: int = Field(
        default_factory=lambda: int(os.getenv("YOUTUBE_DAYS_BACK", "7"))
    )
    max_retries: int = Field(
        default_factory=lambda: int(os.getenv("YOUTUBE_MAX_RETRIES", "1"))
    )
    batch_size: int = Field(
        default_factory=lambda: int(os.getenv("YOUTUBE_BATCH_SIZE", "5"))
    )

    # Chunking settings (token-based)
    min_tokens: int = Field(
        default_factory=lambda: int(os.getenv("MIN_CHUNK_TOKENS", "400"))
    )
    max_tokens: int = Field(
        default_factory=lambda: int(os.getenv("MAX_CHUNK_TOKENS", "1000"))
    )

    # Embedding settings (reuse existing env vars)
    embedding_provider: str = Field(
        default_factory=lambda: os.getenv("EMBEDDING_PROVIDER", "openai")
    )
    embedding_base_url: str = Field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_BASE_URL", "https://api.openai.com/v1"
        )
    )
    embedding_api_key: str = Field(
        default_factory=lambda: os.getenv("EMBEDDING_API_KEY", "")
    )
    embedding_model: str = Field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL_CHOICE", "text-embedding-3-small"
        )
    )

    # Database settings (reuse existing Supabase config)
    supabase_url: str = Field(default_factory=lambda: os.getenv("SUPABASE_URL", ""))
    supabase_key: str = Field(
        default_factory=lambda: os.getenv("SUPABASE_SERVICE_KEY", "")
    )


def get_config() -> YouTubeRAGConfig:
    """Get validated configuration instance.

    Returns:
        YouTubeRAGConfig: Validated configuration object with all settings.

    Raises:
        ValidationError: If required environment variables are missing or invalid.
    """
    return YouTubeRAGConfig()
