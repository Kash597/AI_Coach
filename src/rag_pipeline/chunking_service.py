"""Chunking service for token-aware transcript segmentation."""

from typing import Any

from transformers import AutoTokenizer

from src.utils.logging import get_logger

from .config import YouTubeRAGConfig
from .schemas import Chunk, Transcript, TranscriptSegment, VideoMetadata

logger = get_logger(__name__)


class ChunkingService:
    """Service for chunking transcripts with token awareness.

    This service splits YouTube transcripts into chunks that respect token
    limits for embedding models while preserving timestamp information and
    semantic boundaries.
    """

    def __init__(self, config: YouTubeRAGConfig):
        """Initialize chunking service with configuration.

        Args:
            config: Configuration object with token limits and embedding model.
        """
        self.config = config
        self.tokenizer = self._get_tokenizer(config.embedding_model)
        logger.info(
            "chunking_service_initialized",
            min_tokens=config.min_tokens,
            max_tokens=config.max_tokens,
            model=config.embedding_model,
        )

    def _get_tokenizer(self, embedding_model: str) -> Any:
        """Get appropriate tokenizer for the embedding model.

        Maps embedding model names to compatible HuggingFace tokenizers.

        Args:
            embedding_model: Name of the embedding model.

        Returns:
            Configured AutoTokenizer instance (untyped due to transformers library).
        """
        # Map embedding models to tokenizers
        tokenizer_map = {
            "text-embedding-3-small": "sentence-transformers/all-MiniLM-L6-v2",
            "text-embedding-3-large": "sentence-transformers/all-MiniLM-L6-v2",
            "nomic-embed-text": "bert-base-uncased",
            "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
        }

        tokenizer_name = tokenizer_map.get(
            embedding_model, "sentence-transformers/all-MiniLM-L6-v2"
        )
        logger.info("loading_tokenizer", tokenizer=tokenizer_name)
        return AutoTokenizer.from_pretrained(tokenizer_name)  # type: ignore

    def chunk_transcript(
        self, transcript: Transcript, video_metadata: VideoMetadata
    ) -> list[Chunk]:
        """Chunk transcript with token awareness and timestamp preservation.

        This method splits a transcript into chunks that:
        - Respect min/max token limits
        - Preserve timestamp information for navigation
        - Merge small chunks to avoid fragmentation
        - Include video metadata for context

        Args:
            transcript: Full transcript with timed segments.
            video_metadata: Video metadata for context enrichment.

        Returns:
            List of Chunk objects ready for embedding generation.
        """
        logger.info(
            "chunking_started",
            video_id=transcript.video_id,
            segments=len(transcript.segments),
        )

        chunks: list[Chunk] = []
        current_chunk_segments: list[TranscriptSegment] = []
        current_tokens = 0
        chunk_index = 0

        for segment in transcript.segments:
            segment_text = segment.text
            segment_tokens = len(self.tokenizer.encode(segment_text))

            # Check if adding this segment exceeds max_tokens
            if (
                current_tokens + segment_tokens > self.config.max_tokens
                and current_chunk_segments
            ):
                # Save current chunk
                chunk = self._create_chunk(
                    video_metadata=video_metadata,
                    segments=current_chunk_segments,
                    chunk_index=chunk_index,
                )
                chunks.append(chunk)

                # Reset for next chunk
                current_chunk_segments = []
                current_tokens = 0
                chunk_index += 1

            # Add segment to current chunk
            current_chunk_segments.append(segment)
            current_tokens += segment_tokens

            # If single segment exceeds max_tokens, split it anyway
            if current_tokens > self.config.max_tokens:
                chunk = self._create_chunk(
                    video_metadata=video_metadata,
                    segments=current_chunk_segments,
                    chunk_index=chunk_index,
                )
                chunks.append(chunk)
                current_chunk_segments = []
                current_tokens = 0
                chunk_index += 1

        # Add final chunk if any segments remain
        if current_chunk_segments:
            chunk = self._create_chunk(
                video_metadata=video_metadata,
                segments=current_chunk_segments,
                chunk_index=chunk_index,
            )
            chunks.append(chunk)

        # Merge small chunks to avoid fragmentation
        chunks = self._merge_small_chunks(chunks)

        logger.info(
            "chunking_completed",
            video_id=transcript.video_id,
            chunks_created=len(chunks),
        )
        return chunks

    def _create_chunk(
        self,
        video_metadata: VideoMetadata,
        segments: list[TranscriptSegment],
        chunk_index: int,
    ) -> Chunk:
        """Create a chunk from transcript segments.

        Args:
            video_metadata: Video metadata for context.
            segments: List of transcript segments to combine.
            chunk_index: Index of this chunk in the sequence.

        Returns:
            Chunk object with combined text and metadata.
        """
        text_content = " ".join([s.text for s in segments])
        token_count = len(self.tokenizer.encode(text_content))

        # Calculate timestamp range
        start_offset_ms = segments[0].offset_ms
        end_offset_ms = segments[-1].offset_ms + segments[-1].duration_ms
        duration_ms = sum(s.duration_ms for s in segments)

        # Create timestamp URL for easy navigation
        timestamp_url = f"{video_metadata.url}&t={start_offset_ms // 1000}s"

        return Chunk(
            video_id=video_metadata.id,
            chunk_index=chunk_index,
            text_content=text_content,
            start_offset_ms=start_offset_ms,
            end_offset_ms=end_offset_ms,
            duration_ms=duration_ms,
            token_count=token_count,
            metadata={
                "video_title": video_metadata.title,
                "video_url": video_metadata.url,
                "channel_id": video_metadata.channel_id,
                "timestamp_url": timestamp_url,
            },
        )

    def _merge_small_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Merge chunks that are below min_tokens threshold.

        This method iterates through chunks and merges adjacent small chunks
        to avoid having many tiny fragments that provide limited context.

        Args:
            chunks: List of chunks to potentially merge.

        Returns:
            List of chunks with small ones merged where possible.
        """
        if not chunks:
            return chunks

        merged: list[Chunk] = []
        i = 0

        while i < len(chunks):
            current = chunks[i]

            # If chunk is too small and not the last one, try to merge with next
            if current.token_count < self.config.min_tokens and i < len(chunks) - 1:
                next_chunk = chunks[i + 1]
                combined_tokens = current.token_count + next_chunk.token_count

                # Merge if combined doesn't exceed max
                if combined_tokens <= self.config.max_tokens:
                    merged_chunk = Chunk(
                        video_id=current.video_id,
                        chunk_index=len(merged),
                        text_content=f"{current.text_content} {next_chunk.text_content}",
                        start_offset_ms=current.start_offset_ms,
                        end_offset_ms=next_chunk.end_offset_ms,
                        duration_ms=current.duration_ms + next_chunk.duration_ms,
                        token_count=combined_tokens,
                        metadata=current.metadata,  # Use first chunk's metadata
                    )
                    merged.append(merged_chunk)
                    i += 2  # Skip both chunks
                    continue

            merged.append(current)
            i += 1

        logger.info(
            "chunks_merged",
            original_count=len(chunks),
            merged_count=len(merged),
        )
        return merged
