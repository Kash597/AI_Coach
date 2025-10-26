"""Pydantic schemas for YouTube RAG pipeline."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class VideoMetadata(BaseModel):
    """YouTube video metadata.

    This schema represents basic information about a YouTube video that will
    be processed by the RAG pipeline.
    """

    id: str
    channel_id: str
    title: str
    url: str
    published_at: datetime | None = None
    duration_seconds: int | None = None


class TranscriptSegment(BaseModel):
    """Single transcript segment with timestamp.

    Each segment represents a portion of the video transcript with timing
    information for precise navigation and timestamp preservation during chunking.
    """

    text: str
    offset_ms: int  # Start time in milliseconds
    duration_ms: int  # Duration in milliseconds
    lang: str = "en"


class Transcript(BaseModel):
    """Full video transcript with all segments.

    Contains the complete transcript as a list of timed segments, plus language
    metadata for internationalization support.
    """

    video_id: str
    segments: list[TranscriptSegment]
    lang: str
    available_langs: list[str]


class Chunk(BaseModel):
    """Chunked transcript segment.

    Represents a token-aware chunk of transcript text with preserved timestamps
    and enriched metadata for vector search and retrieval.
    """

    video_id: str
    chunk_index: int
    text_content: str
    start_offset_ms: int
    end_offset_ms: int
    duration_ms: int
    token_count: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChunkWithEmbedding(Chunk):
    """Chunk with embedding vector.

    Extends Chunk with the embedding vector generated for semantic search.
    This is what gets stored in the vector database.
    """

    embedding: list[float]


class VideoProcessingStatus(BaseModel):
    """Status of video processing.

    Tracks the processing state of a video through the pipeline, including
    error information and metrics.
    """

    video_id: str
    status: str  # pending, processing, completed, failed
    chunks_created: int = 0
    error_message: str | None = None
    processed_at: datetime | None = None


class PipelineResult(BaseModel):
    """Result of pipeline execution.

    Summary statistics and error information for a complete pipeline run.
    Used for reporting and monitoring.
    """

    total_videos: int
    processed: int
    failed: int
    skipped: int
    chunks_created: int
    errors: list[str] = Field(default_factory=list)
