"""Embedding service for generating text embeddings via OpenAI-compatible APIs."""

import asyncio

from openai import AsyncOpenAI

from src.utils.logging import get_logger

from .config import YouTubeRAGConfig

logger = get_logger(__name__)


class EmbeddingService:
    """Service for generating text embeddings.

    This service supports multiple embedding providers (OpenAI, Ollama, OpenRouter)
    through OpenAI-compatible APIs. It provides both single and batch embedding
    generation with configurable parallelism.
    """

    def __init__(self, config: YouTubeRAGConfig):
        """Initialize embedding service with configuration.

        Args:
            config: Configuration object with embedding provider settings.
        """
        self.config = config
        self.client = self._get_client()
        logger.info(
            "embedding_service_initialized",
            provider=config.embedding_provider,
            model=config.embedding_model,
            base_url=config.embedding_base_url,
        )

    def _get_client(self) -> AsyncOpenAI:
        """Initialize OpenAI-compatible client based on provider.

        Returns:
            Configured AsyncOpenAI client instance.
        """
        if self.config.embedding_provider == "ollama":
            # Ollama doesn't require a real API key
            return AsyncOpenAI(
                base_url=self.config.embedding_base_url,
                api_key="ollama",
            )
        else:
            # OpenAI, OpenRouter, or other compatible providers
            return AsyncOpenAI(
                base_url=self.config.embedding_base_url,
                api_key=self.config.embedding_api_key,
            )

    async def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Text content to embed.

        Returns:
            Embedding vector as a list of floats.

        Raises:
            Exception: If embedding generation fails.
        """
        try:
            response = await self.client.embeddings.create(
                input=text,
                model=self.config.embedding_model,
            )
            embedding = response.data[0].embedding
            logger.debug(
                "embedding_generated",
                text_length=len(text),
                embedding_dim=len(embedding),
            )
            return embedding

        except Exception as e:
            logger.exception(
                "embedding_failed",
                text_length=len(text),
                error_type=type(e).__name__,
            )
            raise

    async def embed_batch(
        self, texts: list[str], batch_size: int = 10
    ) -> list[list[float]]:
        """Generate embeddings for multiple texts with batching.

        This method processes texts in batches to optimize throughput while
        respecting API rate limits. Batches are processed in parallel.

        Args:
            texts: List of text strings to embed.
            batch_size: Number of texts to embed in parallel (default: 10).

        Returns:
            List of embedding vectors in the same order as input texts.

        Raises:
            Exception: If batch embedding fails.
        """
        logger.info(
            "batch_embedding_started",
            count=len(texts),
            batch_size=batch_size,
        )

        embeddings: list[list[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            try:
                # Embed batch in parallel
                batch_embeddings = await asyncio.gather(
                    *[self.embed_text(text) for text in batch]
                )
                embeddings.extend(batch_embeddings)

                logger.debug(
                    "batch_completed",
                    batch_num=i // batch_size + 1,
                    count=len(batch),
                )

            except Exception as e:
                logger.exception(
                    "batch_embedding_failed",
                    batch_num=i // batch_size + 1,
                    error_type=type(e).__name__,
                )
                raise

        logger.info(
            "batch_embedding_completed",
            total_embeddings=len(embeddings),
        )
        return embeddings
