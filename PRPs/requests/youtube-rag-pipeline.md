# Implementation Plan: YouTube RAG Pipeline

## Overview

Build a production-ready RAG (Retrieval-Augmented Generation) pipeline that extracts YouTube video transcripts with timestamps, chunks them intelligently using Dockling's hybrid chunking strategy, generates embeddings, and stores them in a Supabase vector database for semantic search by the AI coach agent.

**IMPORTANT:** This implementation creates NEW code in `src/` directory only. Do NOT modify or create any files in `PRPs/examples/` - that folder is for reference only.

**Tech Stack:**
- **Supadata**: YouTube transcript fetching with timestamps
- **Dockling**: Hybrid chunking (token-aware, semantic boundaries)
- **Supabase + pgvector**: Vector database for embeddings
- **OpenAI-compatible API**: Configurable embedding generation
- **Python + FastAPI**: Backend infrastructure

**Scope:** This plan focuses ONLY on the RAG pipeline (ingestion). Agent tools for querying the knowledge base will be implemented separately in a future phase.

---

## Requirements Summary

### Core Requirements
- Fetch YouTube transcripts from a specific channel with timestamps
- Only process videos from the last 7 days (configurable)
- Check database to avoid reprocessing videos
- Chunk transcripts using hybrid chunking (400-1000 tokens)
- Generate embeddings and store in Supabase vector database
- Retry failed transcripts once before marking as error
- Metadata: video URL, timestamp for chunks, channel ID
- CLI-based triggering for MVP
- Configurable embedding provider (OpenAI, Ollama, OpenRouter)

### Architecture Requirements (from CLAUDE.md)
- Follow vertical slice architecture (`src/tools/`, `src/rag_pipeline/`)
- Type safety: 100% type annotations, strict mypy
- Google-style docstrings for all functions
- Structured logging with context
- Unit and integration tests mirroring source structure
- KISS principle: prefer simple, readable solutions

### Future Extensibility
- Easy to add other data sources (podcasts, Twitter, etc.)
- Support for multiple channels
- Scheduled automation (cron/FastAPI background tasks)

---

## Research Findings

### Supadata API Capabilities

**Channel Videos Endpoint:**
- `GET /v1/youtube/channel/videos` - Fetch video IDs from channel
- Parameters: `id` (channel URL/ID/handle), `limit` (max: 5000), `type` (all/video/short/live)
- Returns: Lists of video IDs categorized by type
- Pricing: 1 credit per request

**Transcript Endpoint:**
- `GET /v1/youtube/transcript` - Fetch transcript with timestamps
- Parameters: `url` or `videoId`, `lang`, `text` (boolean), `chunkSize`
- When `text=false`: Returns array of `{text, offset, duration, lang}` objects
- Offset/duration in milliseconds
- Pricing: 1 credit per transcript

**Rate Limiting:**
- Plan-based rate limits (specific limits not documented publicly)
- Error code `limit-exceeded` when exceeded
- Recommendation: Implement rate limiting and batch processing
- Support for Auto Recharge

**Best Practices:**
- Use batch endpoints when available
- Implement exponential backoff for rate limit errors
- Track API credit usage
- Handle 206 status (transcript unavailable)

### Dockling Hybrid Chunking

**From docling_hybrid_chunking.py example:**
```python
from docling.chunking import HybridChunker
from transformers import AutoTokenizer

chunker = HybridChunker(
    tokenizer=AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2"),
    max_tokens=512,
    merge_peers=True  # Merge small adjacent chunks
)
```

**Key Features:**
- Token-aware splitting (respects max_tokens limit)
- Semantic boundary preservation (doesn't split mid-sentence)
- Hierarchical structure awareness
- `contextualize()` method preserves headings and context
- Works with any HuggingFace tokenizer

**Adaptation for YouTube Transcripts:**
- Convert transcript segments to Dockling document format
- Preserve timestamp information in metadata
- Use `contextualize()` to include video title and context
- Set `max_tokens` based on embedding model (400-1000 tokens)

### Supabase + pgvector Setup

**Extension Setup:**
1. Enable pgvector extension in Supabase dashboard
2. Create tables with `vector(dimension)` columns
3. Create IVFFlat or HNSW indexes for similarity search
4. Use RPC functions for vector similarity queries

**Python Client Options:**
- **supabase-py**: Official Supabase client
- **vecs**: Supabase's Python vector client (higher-level abstraction)
- **Direct SQL**: Using psycopg2 for complex queries

**Best Practices:**
- Use `ivfflat` index with appropriate `lists` parameter (sqrt(rows) is common)
- Store embeddings as `vector(384)` for MiniLM, `vector(1536)` for OpenAI
- Include metadata as JSONB for flexible filtering
- Implement Row Level Security (RLS) policies
- Use connection pooling for performance

### Embedding Model Considerations

**Options from .env.example:**
- OpenAI: `text-embedding-3-small` (1536 dimensions)
- Ollama: `nomic-embed-text` (768 dimensions)
- Sentence Transformers: `all-MiniLM-L6-v2` (384 dimensions)

**Token Limits:**
- MiniLM-L6-v2: 512 tokens max
- OpenAI text-embedding-3-small: 8191 tokens max
- nomic-embed-text: 2048 tokens max

**Recommendation:**
- Start with configurable provider via env vars (matching existing pattern)
- Default to `all-MiniLM-L6-v2` for local/free option
- Support OpenAI for production with better quality
- Store dimension in database schema based on EMBEDDING_MODEL_CHOICE

---

## Codebase Integration Points

### Existing Patterns to Follow

**Configuration Management** (from `PRPs/examples/backend_agent_api/clients.py`):
```python
def get_agent_clients():
    embedding_provider = os.getenv('EMBEDDING_PROVIDER', 'openai')
    base_url = os.getenv('EMBEDDING_BASE_URL', 'https://api.openai.com/v1')
    api_key = os.getenv('EMBEDDING_API_KEY')

    if embedding_provider == 'ollama':
        # Special handling for Ollama
        embedding_client = AsyncOpenAI(base_url=base_url, api_key="ollama")
    else:
        embedding_client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    return embedding_client
```

**Database Operations** (from `db_utils.py`):
```python
async def fetch_conversation_history(supabase: Client, session_id: str, limit: int = 10):
    try:
        response = supabase.table("messages") \
            .select("*") \
            .eq("session_id", session_id) \
            .order("created_at", desc=True) \
            .limit(limit) \
            .execute()
        return response.data[::-1]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch: {str(e)}")
```

**Logging** (from `PRPs/ai_docs/logging_guide.md`):
```python
from src.utils.logging import get_logger
logger = get_logger(__name__)

logger.info("video_processing_started", video_id="abc123", title="Test Video")
logger.exception("transcript_fetch_failed", video_id=video_id, error_type=type(e).__name__)
```

**Tool Structure** (from `tools.py` + `agent.py`):
- Implementation in `tools.py`
- Decorator in `agent.py`
- Comprehensive docstrings with "Use this when" / "Do NOT use"
- Performance notes and token estimates

### Files to Create

**IMPORTANT:** Create files ONLY in `src/` directory. Do NOT modify or create anything in `PRPs/examples/`.

```
src/
├── rag_pipeline/
│   ├── __init__.py
│   ├── config.py            # Environment configuration
│   ├── schemas.py           # Data models (Video, Chunk, etc.)
│   ├── youtube_service.py   # Supadata API client
│   ├── chunking_service.py  # Dockling hybrid chunking
│   ├── embedding_service.py # Embedding generation
│   ├── storage_service.py   # Supabase vector operations
│   ├── pipeline.py          # Main orchestration logic
│   └── cli.py              # CLI interface
└── utils/
    ├── logging.py           # Shared logging utilities (if doesn't exist)
    └── config.py            # Global config loader (if doesn't exist)

tests/
└── rag_pipeline/
    ├── test_youtube_service.py
    ├── test_chunking_service.py
    ├── test_embedding_service.py
    ├── test_storage_service.py
    ├── test_pipeline.py
    └── test_cli.py
```

**Note:** Agent RAG tools (`src/tools/rag_tools/`) will be created in a future phase.

### Database Schema (Supabase Initial Setup)

**Execute this SQL in Supabase Dashboard > SQL Editor:**

```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Channels table
CREATE TABLE IF NOT EXISTS channels (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    url TEXT,
    last_processed_at TIMESTAMPTZ,
    active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Videos table
CREATE TABLE IF NOT EXISTS videos (
    id TEXT PRIMARY KEY,
    channel_id TEXT REFERENCES channels(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    url TEXT NOT NULL,
    published_at TIMESTAMPTZ,
    duration_seconds INTEGER,
    status TEXT DEFAULT 'pending', -- pending, processing, completed, failed
    transcript_available BOOLEAN DEFAULT FALSE,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    processed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Transcript chunks with embeddings
CREATE TABLE IF NOT EXISTS transcript_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id TEXT REFERENCES videos(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    text_content TEXT NOT NULL,
    start_offset_ms INTEGER,
    end_offset_ms INTEGER,
    duration_ms INTEGER,
    token_count INTEGER,
    embedding vector(384), -- Adjust dimension based on embedding model
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(video_id, chunk_index)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_videos_channel_id ON videos(channel_id);
CREATE INDEX IF NOT EXISTS idx_videos_status ON videos(status);
CREATE INDEX IF NOT EXISTS idx_videos_published_at ON videos(published_at DESC);
CREATE INDEX IF NOT EXISTS idx_chunks_video_id ON transcript_chunks(video_id);

-- Vector similarity index (IVFFlat)
CREATE INDEX IF NOT EXISTS idx_chunks_embedding
ON transcript_chunks USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- RPC function for vector search
CREATE OR REPLACE FUNCTION match_transcript_chunks(
    query_embedding vector(384),
    match_count INT DEFAULT 5,
    filter JSONB DEFAULT '{}'::jsonb
)
RETURNS TABLE (
    id UUID,
    video_id TEXT,
    text_content TEXT,
    start_offset_ms INTEGER,
    similarity FLOAT,
    metadata JSONB
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        tc.id,
        tc.video_id,
        tc.text_content,
        tc.start_offset_ms,
        1 - (tc.embedding <=> query_embedding) AS similarity,
        tc.metadata
    FROM transcript_chunks tc
    WHERE
        CASE
            WHEN filter != '{}'::jsonb THEN tc.metadata @> filter
            ELSE TRUE
        END
    ORDER BY tc.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- Row Level Security (optional but recommended)
ALTER TABLE channels ENABLE ROW LEVEL SECURITY;
ALTER TABLE videos ENABLE ROW LEVEL SECURITY;
ALTER TABLE transcript_chunks ENABLE ROW LEVEL SECURITY;

-- Example RLS policy (adjust based on your auth strategy)
CREATE POLICY "Public read access" ON transcript_chunks
    FOR SELECT USING (true);
```

---

## Implementation Tasks

### Phase 1: Foundation & Configuration (4-6 hours)

#### Task 1.1: Environment Configuration
**Description:** Set up environment variables and configuration module.

**Files to create:**
- `src/rag_pipeline/config.py`
- `.env.example` (update with YouTube RAG variables)

**Implementation:**
```python
# src/rag_pipeline/config.py
import os
from typing import Optional
from pydantic import BaseModel, Field

class YouTubeRAGConfig(BaseModel):
    """Configuration for YouTube RAG pipeline."""

    # Supadata
    supadata_api_key: str = Field(default_factory=lambda: os.getenv('SUPADATA_API_KEY', ''))
    youtube_channel_id: str = Field(default_factory=lambda: os.getenv('YOUTUBE_CHANNEL_ID', ''))

    # Processing
    days_back: int = Field(default_factory=lambda: int(os.getenv('YOUTUBE_DAYS_BACK', '7')))
    max_retries: int = Field(default_factory=lambda: int(os.getenv('YOUTUBE_MAX_RETRIES', '1')))
    batch_size: int = Field(default_factory=lambda: int(os.getenv('YOUTUBE_BATCH_SIZE', '5')))

    # Chunking
    min_tokens: int = Field(default_factory=lambda: int(os.getenv('MIN_CHUNK_TOKENS', '400')))
    max_tokens: int = Field(default_factory=lambda: int(os.getenv('MAX_CHUNK_TOKENS', '1000')))

    # Embeddings (reuse existing env vars)
    embedding_provider: str = Field(default_factory=lambda: os.getenv('EMBEDDING_PROVIDER', 'openai'))
    embedding_base_url: str = Field(default_factory=lambda: os.getenv('EMBEDDING_BASE_URL', 'https://api.openai.com/v1'))
    embedding_api_key: str = Field(default_factory=lambda: os.getenv('EMBEDDING_API_KEY', ''))
    embedding_model: str = Field(default_factory=lambda: os.getenv('EMBEDDING_MODEL_CHOICE', 'text-embedding-3-small'))

    # Database (reuse existing)
    supabase_url: str = Field(default_factory=lambda: os.getenv('SUPABASE_URL', ''))
    supabase_key: str = Field(default_factory=lambda: os.getenv('SUPABASE_SERVICE_KEY', ''))

def get_config() -> YouTubeRAGConfig:
    """Get validated configuration."""
    return YouTubeRAGConfig()
```

**Add to .env.example:**
```bash
# YouTube RAG Pipeline
SUPADATA_API_KEY=your_supadata_key_here
YOUTUBE_CHANNEL_ID=UCxxxxxxxxxxxxx
YOUTUBE_DAYS_BACK=7
YOUTUBE_MAX_RETRIES=1
YOUTUBE_BATCH_SIZE=5

# Chunking (token-based)
MIN_CHUNK_TOKENS=400
MAX_CHUNK_TOKENS=1000
```

**Dependencies:** None
**Estimated effort:** 30 minutes

---

#### Task 1.2: Pydantic Schemas
**Description:** Define type-safe data models for videos, transcripts, and chunks.

**Files to create:**
- `src/rag_pipeline/schemas.py`

**Implementation:**
```python
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

class VideoMetadata(BaseModel):
    """YouTube video metadata."""
    id: str
    channel_id: str
    title: str
    url: str
    published_at: Optional[datetime] = None
    duration_seconds: Optional[int] = None

class TranscriptSegment(BaseModel):
    """Single transcript segment with timestamp."""
    text: str
    offset_ms: int  # Start time in milliseconds
    duration_ms: int  # Duration in milliseconds
    lang: str = "en"

class Transcript(BaseModel):
    """Full video transcript."""
    video_id: str
    segments: List[TranscriptSegment]
    lang: str
    available_langs: List[str]

class Chunk(BaseModel):
    """Chunked transcript segment."""
    video_id: str
    chunk_index: int
    text_content: str
    start_offset_ms: int
    end_offset_ms: int
    duration_ms: int
    token_count: int
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ChunkWithEmbedding(Chunk):
    """Chunk with embedding vector."""
    embedding: List[float]

class VideoProcessingStatus(BaseModel):
    """Status of video processing."""
    video_id: str
    status: str  # pending, processing, completed, failed
    chunks_created: int = 0
    error_message: Optional[str] = None
    processed_at: Optional[datetime] = None

class PipelineResult(BaseModel):
    """Result of pipeline execution."""
    total_videos: int
    processed: int
    failed: int
    skipped: int
    chunks_created: int
    errors: List[str] = Field(default_factory=list)
```

**Dependencies:** Task 1.1
**Estimated effort:** 45 minutes

---

#### Task 1.3: Database Setup
**Description:** Set up Supabase tables and vector search function for fresh database.

**Steps:**
1. Open Supabase Dashboard > SQL Editor
2. Copy the SQL from "Database Schema (Supabase Initial Setup)" section above
3. Execute the SQL to create:
   - `channels` table
   - `videos` table
   - `transcript_chunks` table with vector embeddings
   - Indexes for performance
   - `match_transcript_chunks()` RPC function for vector search
   - Row Level Security policies

**Verification:**
```sql
-- Verify tables created
SELECT table_name FROM information_schema.tables
WHERE table_schema = 'public'
AND table_name IN ('channels', 'videos', 'transcript_chunks');

-- Verify vector extension enabled
SELECT * FROM pg_extension WHERE extname = 'vector';
```

**Dependencies:** Task 1.1
**Estimated effort:** 30 minutes

---

#### Task 1.4: Logging Setup
**Description:** Create shared logging utilities following existing patterns.

**Files to create/modify:**
- `src/utils/logging.py` (if doesn't exist)

**Implementation:**
```python
# src/utils/logging.py
import logging
import sys
from typing import Any
import structlog

def get_logger(name: str) -> structlog.BoundLogger:
    """Get structured logger instance.

    Args:
        name: Logger name (typically __name__).

    Returns:
        Configured structlog logger instance.
    """
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    return structlog.get_logger(name)
```

**Dependencies:** None
**Estimated effort:** 30 minutes

---

### Phase 2: Core Services (8-12 hours)

#### Task 2.1: YouTube Service (Supadata Client)
**Description:** Implement Supadata API client for fetching channel videos and transcripts.

**Files to create:**
- `src/rag_pipeline/youtube_service.py`

**Implementation highlights:**
```python
from supadata import Supadata
from typing import List, Optional
from .schemas import VideoMetadata, Transcript, TranscriptSegment
from .config import YouTubeRAGConfig
from src.utils.logging import get_logger
from datetime import datetime, timedelta

logger = get_logger(__name__)

class YouTubeService:
    """Service for fetching YouTube data via Supadata API."""

    def __init__(self, config: YouTubeRAGConfig):
        self.config = config
        self.client = Supadata(api_key=config.supadata_api_key)

    async def get_recent_videos(
        self,
        channel_id: str,
        days_back: int = 7
    ) -> List[VideoMetadata]:
        """Fetch recent videos from channel.

        Args:
            channel_id: YouTube channel ID.
            days_back: Number of days to look back.

        Returns:
            List of video metadata objects.

        Raises:
            SupadataError: If API request fails.
        """
        logger.info("fetching_channel_videos", channel_id=channel_id, days_back=days_back)

        try:
            # Fetch video IDs
            response = self.client.youtube.channel.videos(
                id=channel_id,
                type="video",
                limit=100  # Adjust based on channel upload frequency
            )

            video_ids = response.video_ids
            logger.info("videos_fetched", count=len(video_ids))

            # Filter by date (requires fetching metadata)
            cutoff_date = datetime.now() - timedelta(days=days_back)
            recent_videos = []

            for video_id in video_ids:
                # Note: Supadata doesn't provide publish date in bulk
                # May need to use YouTube Data API or check after transcript fetch
                video = VideoMetadata(
                    id=video_id,
                    channel_id=channel_id,
                    title="",  # Will be enriched from transcript metadata
                    url=f"https://youtube.com/watch?v={video_id}"
                )
                recent_videos.append(video)

            return recent_videos

        except Exception as e:
            logger.exception("channel_fetch_failed", channel_id=channel_id, error=str(e))
            raise

    async def get_transcript(
        self,
        video_id: str,
        retry: bool = False
    ) -> Optional[Transcript]:
        """Fetch transcript for video with timestamps.

        Args:
            video_id: YouTube video ID.
            retry: Whether this is a retry attempt.

        Returns:
            Transcript object or None if unavailable.

        Raises:
            SupadataError: If API request fails (non-transcript errors).
        """
        logger.info("fetching_transcript", video_id=video_id, retry=retry)

        try:
            response = self.client.youtube.transcript(
                video_id=video_id,
                text=False  # Get segments with timestamps
            )

            segments = [
                TranscriptSegment(
                    text=seg["text"],
                    offset_ms=seg["offset"],
                    duration_ms=seg["duration"],
                    lang=seg.get("lang", "en")
                )
                for seg in response.content
            ]

            transcript = Transcript(
                video_id=video_id,
                segments=segments,
                lang=response.lang,
                available_langs=response.available_langs
            )

            logger.info("transcript_fetched", video_id=video_id, segments=len(segments))
            return transcript

        except Exception as e:
            # Check if transcript unavailable (206 status)
            if "transcript-unavailable" in str(e).lower():
                logger.warning("transcript_unavailable", video_id=video_id)
                return None

            logger.exception("transcript_fetch_error", video_id=video_id, error=str(e))
            raise
```

**Key features:**
- Fetch channel videos using Supadata
- Get transcripts with timestamp segments
- Handle transcript unavailable errors gracefully
- Retry logic support
- Structured logging

**Dependencies:** Task 1.1, 1.2, 1.4
**Estimated effort:** 2-3 hours

---

#### Task 2.2: Chunking Service
**Description:** Implement token-aware hybrid chunking for transcripts.

**Files to create:**
- `src/rag_pipeline/chunking_service.py`

**Implementation highlights:**
```python
from transformers import AutoTokenizer
from typing import List
from .schemas import Transcript, Chunk, VideoMetadata
from .config import YouTubeRAGConfig
from src.utils.logging import get_logger

logger = get_logger(__name__)

class ChunkingService:
    """Service for chunking transcripts with token awareness."""

    def __init__(self, config: YouTubeRAGConfig):
        self.config = config
        # Load tokenizer based on embedding model
        self.tokenizer = self._get_tokenizer(config.embedding_model)
        logger.info("chunking_service_initialized",
                   min_tokens=config.min_tokens,
                   max_tokens=config.max_tokens)

    def _get_tokenizer(self, embedding_model: str):
        """Get appropriate tokenizer for embedding model."""
        # Map embedding models to tokenizers
        tokenizer_map = {
            "text-embedding-3-small": "sentence-transformers/all-MiniLM-L6-v2",
            "text-embedding-3-large": "sentence-transformers/all-MiniLM-L6-v2",
            "nomic-embed-text": "bert-base-uncased",  # Adjust as needed
            "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2"
        }

        tokenizer_name = tokenizer_map.get(embedding_model, "sentence-transformers/all-MiniLM-L6-v2")
        return AutoTokenizer.from_pretrained(tokenizer_name)

    def chunk_transcript(
        self,
        transcript: Transcript,
        video_metadata: VideoMetadata
    ) -> List[Chunk]:
        """Chunk transcript with token awareness and timestamp preservation.

        Args:
            transcript: Full transcript with segments.
            video_metadata: Video metadata for context.

        Returns:
            List of chunks with embeddings metadata.
        """
        logger.info("chunking_started", video_id=transcript.video_id, segments=len(transcript.segments))

        chunks = []
        current_chunk_segments = []
        current_tokens = 0
        chunk_index = 0

        for segment in transcript.segments:
            segment_text = segment.text
            segment_tokens = len(self.tokenizer.encode(segment_text))

            # Check if adding this segment exceeds max_tokens
            if current_tokens + segment_tokens > self.config.max_tokens and current_chunk_segments:
                # Save current chunk
                chunk = self._create_chunk(
                    video_metadata=video_metadata,
                    segments=current_chunk_segments,
                    chunk_index=chunk_index
                )
                chunks.append(chunk)

                # Reset for next chunk
                current_chunk_segments = []
                current_tokens = 0
                chunk_index += 1

            # Add segment to current chunk
            current_chunk_segments.append(segment)
            current_tokens += segment_tokens

            # If single segment exceeds max_tokens, split it
            if current_tokens > self.config.max_tokens:
                chunk = self._create_chunk(
                    video_metadata=video_metadata,
                    segments=current_chunk_segments,
                    chunk_index=chunk_index
                )
                chunks.append(chunk)
                current_chunk_segments = []
                current_tokens = 0
                chunk_index += 1

        # Add final chunk
        if current_chunk_segments:
            chunk = self._create_chunk(
                video_metadata=video_metadata,
                segments=current_chunk_segments,
                chunk_index=chunk_index
            )
            chunks.append(chunk)

        # Merge small chunks (under min_tokens)
        chunks = self._merge_small_chunks(chunks)

        logger.info("chunking_completed", video_id=transcript.video_id, chunks_created=len(chunks))
        return chunks

    def _create_chunk(self, video_metadata: VideoMetadata, segments: List, chunk_index: int) -> Chunk:
        """Create chunk from segments."""
        text_content = " ".join([s.text for s in segments])
        token_count = len(self.tokenizer.encode(text_content))

        return Chunk(
            video_id=video_metadata.id,
            chunk_index=chunk_index,
            text_content=text_content,
            start_offset_ms=segments[0].offset_ms,
            end_offset_ms=segments[-1].offset_ms + segments[-1].duration_ms,
            duration_ms=sum(s.duration_ms for s in segments),
            token_count=token_count,
            metadata={
                "video_title": video_metadata.title,
                "video_url": video_metadata.url,
                "channel_id": video_metadata.channel_id,
                "timestamp_url": f"{video_metadata.url}&t={segments[0].offset_ms // 1000}s"
            }
        )

    def _merge_small_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Merge chunks that are below min_tokens threshold."""
        if not chunks:
            return chunks

        merged = []
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
                        metadata=current.metadata
                    )
                    merged.append(merged_chunk)
                    i += 2  # Skip both chunks
                    continue

            merged.append(current)
            i += 1

        return merged
```

**Key features:**
- Token-aware chunking respecting min/max limits
- Timestamp preservation across chunks
- Metadata enrichment (video title, URL, timestamp link)
- Merge small chunks for efficiency
- Configurable tokenizer based on embedding model

**Dependencies:** Task 1.1, 1.2, 1.4
**Estimated effort:** 3-4 hours

---

#### Task 2.3: Embedding Service
**Description:** Generate embeddings using configurable provider (OpenAI/Ollama/local).

**Files to create:**
- `src/rag_pipeline/embedding_service.py`

**Implementation highlights:**
```python
from openai import AsyncOpenAI
from typing import List
from .config import YouTubeRAGConfig
from src.utils.logging import get_logger
import asyncio

logger = get_logger(__name__)

class EmbeddingService:
    """Service for generating text embeddings."""

    def __init__(self, config: YouTubeRAGConfig):
        self.config = config
        self.client = self._get_client()
        logger.info("embedding_service_initialized",
                   provider=config.embedding_provider,
                   model=config.embedding_model)

    def _get_client(self) -> AsyncOpenAI:
        """Initialize OpenAI-compatible client based on provider."""
        if self.config.embedding_provider == 'ollama':
            return AsyncOpenAI(
                base_url=self.config.embedding_base_url,
                api_key="ollama"  # Ollama doesn't require real API key
            )
        else:
            return AsyncOpenAI(
                base_url=self.config.embedding_base_url,
                api_key=self.config.embedding_api_key
            )

    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for single text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector as list of floats.

        Raises:
            OpenAIError: If embedding generation fails.
        """
        try:
            response = await self.client.embeddings.create(
                input=text,
                model=self.config.embedding_model
            )
            return response.data[0].embedding
        except Exception as e:
            logger.exception("embedding_failed", text_length=len(text), error=str(e))
            raise

    async def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 10
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts with batching.

        Args:
            texts: List of texts to embed.
            batch_size: Number of texts to embed in parallel.

        Returns:
            List of embedding vectors.
        """
        logger.info("batch_embedding_started", count=len(texts), batch_size=batch_size)

        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            try:
                # Embed batch in parallel
                batch_embeddings = await asyncio.gather(*[
                    self.embed_text(text) for text in batch
                ])
                embeddings.extend(batch_embeddings)

                logger.debug("batch_completed", batch_num=i // batch_size + 1, count=len(batch))

            except Exception as e:
                logger.exception("batch_embedding_failed", batch_num=i // batch_size + 1, error=str(e))
                raise

        logger.info("batch_embedding_completed", total_embeddings=len(embeddings))
        return embeddings
```

**Key features:**
- Support for OpenAI, Ollama, and OpenRouter
- Batch embedding with configurable parallelism
- Error handling and retry logic
- Performance logging

**Dependencies:** Task 1.1, 1.4
**Estimated effort:** 2 hours

---

#### Task 2.4: Storage Service
**Description:** Store chunks with embeddings in Supabase vector database.

**Files to create:**
- `src/rag_pipeline/storage_service.py`

**Implementation highlights:**
```python
from supabase import Client, create_client
from typing import List, Optional
from datetime import datetime
from .schemas import VideoMetadata, Chunk, ChunkWithEmbedding, VideoProcessingStatus
from .config import YouTubeRAGConfig
from src.utils.logging import get_logger

logger = get_logger(__name__)

class StorageService:
    """Service for storing videos and chunks in Supabase."""

    def __init__(self, config: YouTubeRAGConfig):
        self.config = config
        self.client = create_client(config.supabase_url, config.supabase_key)
        logger.info("storage_service_initialized", supabase_url=config.supabase_url)

    async def is_video_processed(self, video_id: str) -> bool:
        """Check if video has already been processed.

        Args:
            video_id: YouTube video ID.

        Returns:
            True if video exists in database with 'completed' status.
        """
        try:
            response = self.client.table("videos") \
                .select("id, status") \
                .eq("id", video_id) \
                .execute()

            if response.data:
                status = response.data[0].get("status")
                return status == "completed"

            return False

        except Exception as e:
            logger.exception("video_check_failed", video_id=video_id, error=str(e))
            return False  # Assume not processed on error

    async def save_video(
        self,
        video: VideoMetadata,
        status: str = "processing"
    ) -> None:
        """Save or update video metadata.

        Args:
            video: Video metadata.
            status: Processing status.
        """
        try:
            data = {
                "id": video.id,
                "channel_id": video.channel_id,
                "title": video.title,
                "url": video.url,
                "published_at": video.published_at.isoformat() if video.published_at else None,
                "duration_seconds": video.duration_seconds,
                "status": status,
                "updated_at": datetime.now().isoformat()
            }

            self.client.table("videos").upsert(data).execute()
            logger.info("video_saved", video_id=video.id, status=status)

        except Exception as e:
            logger.exception("video_save_failed", video_id=video.id, error=str(e))
            raise

    async def save_chunks(
        self,
        chunks: List[ChunkWithEmbedding]
    ) -> None:
        """Save transcript chunks with embeddings.

        Args:
            chunks: List of chunks with embeddings.
        """
        try:
            data = [
                {
                    "video_id": chunk.video_id,
                    "chunk_index": chunk.chunk_index,
                    "text_content": chunk.text_content,
                    "start_offset_ms": chunk.start_offset_ms,
                    "end_offset_ms": chunk.end_offset_ms,
                    "duration_ms": chunk.duration_ms,
                    "token_count": chunk.token_count,
                    "embedding": chunk.embedding,
                    "metadata": chunk.metadata
                }
                for chunk in chunks
            ]

            self.client.table("transcript_chunks").insert(data).execute()
            logger.info("chunks_saved", count=len(chunks), video_id=chunks[0].video_id if chunks else None)

        except Exception as e:
            logger.exception("chunks_save_failed", count=len(chunks), error=str(e))
            raise

    async def update_video_status(
        self,
        video_id: str,
        status: str,
        error_message: Optional[str] = None,
        retry_count: int = 0
    ) -> None:
        """Update video processing status.

        Args:
            video_id: YouTube video ID.
            status: New status (completed, failed, etc.).
            error_message: Error message if failed.
            retry_count: Number of retries attempted.
        """
        try:
            data = {
                "status": status,
                "error_message": error_message,
                "retry_count": retry_count,
                "updated_at": datetime.now().isoformat()
            }

            if status == "completed":
                data["processed_at"] = datetime.now().isoformat()
                data["transcript_available"] = True

            self.client.table("videos").update(data).eq("id", video_id).execute()
            logger.info("video_status_updated", video_id=video_id, status=status)

        except Exception as e:
            logger.exception("status_update_failed", video_id=video_id, error=str(e))
            raise

    async def search_chunks(
        self,
        query_embedding: List[float],
        match_count: int = 5,
        filter_metadata: Optional[dict] = None
    ) -> List[dict]:
        """Search for similar chunks using vector similarity.

        Args:
            query_embedding: Query embedding vector.
            match_count: Number of results to return.
            filter_metadata: Optional JSONB filter.

        Returns:
            List of matching chunks with similarity scores.
        """
        try:
            filter_json = filter_metadata or {}

            response = self.client.rpc(
                "match_transcript_chunks",
                {
                    "query_embedding": query_embedding,
                    "match_count": match_count,
                    "filter": filter_json
                }
            ).execute()

            logger.info("vector_search_completed", results=len(response.data), match_count=match_count)
            return response.data

        except Exception as e:
            logger.exception("vector_search_failed", error=str(e))
            raise
```

**Key features:**
- Deduplication check for processed videos
- Video and chunk storage
- Status tracking with error messages
- Vector similarity search
- Comprehensive error handling

**Dependencies:** Task 1.1, 1.2, 1.3, 1.4
**Estimated effort:** 2-3 hours

---

### Phase 3: Pipeline Orchestration (4-6 hours)

#### Task 3.1: Pipeline Orchestrator
**Description:** Main pipeline logic coordinating all services.

**Files to create:**
- `src/rag_pipeline/pipeline.py`

**Implementation highlights:**
```python
from typing import List
from .config import YouTubeRAGConfig, get_config
from .schemas import VideoMetadata, PipelineResult, ChunkWithEmbedding
from .youtube_service import YouTubeService
from .chunking_service import ChunkingService
from .embedding_service import EmbeddingService
from .storage_service import StorageService
from src.utils.logging import get_logger

logger = get_logger(__name__)

class YouTubeRAGPipeline:
    """Orchestrates YouTube RAG pipeline execution."""

    def __init__(self, config: Optional[YouTubeRAGConfig] = None):
        self.config = config or get_config()
        self.youtube_service = YouTubeService(self.config)
        self.chunking_service = ChunkingService(self.config)
        self.embedding_service = EmbeddingService(self.config)
        self.storage_service = StorageService(self.config)

        logger.info("pipeline_initialized", channel_id=self.config.youtube_channel_id)

    async def process_channel(self) -> PipelineResult:
        """Process all recent videos from configured channel.

        Returns:
            Pipeline execution results.
        """
        logger.info("pipeline_started", channel_id=self.config.youtube_channel_id, days_back=self.config.days_back)

        result = PipelineResult(
            total_videos=0,
            processed=0,
            failed=0,
            skipped=0,
            chunks_created=0
        )

        try:
            # 1. Fetch recent videos
            videos = await self.youtube_service.get_recent_videos(
                channel_id=self.config.youtube_channel_id,
                days_back=self.config.days_back
            )
            result.total_videos = len(videos)

            logger.info("videos_fetched", count=len(videos))

            # 2. Process each video
            for video in videos:
                video_result = await self._process_video(video)

                if video_result["status"] == "completed":
                    result.processed += 1
                    result.chunks_created += video_result["chunks"]
                elif video_result["status"] == "skipped":
                    result.skipped += 1
                else:
                    result.failed += 1
                    result.errors.append(f"{video.id}: {video_result.get('error', 'Unknown error')}")

            logger.info("pipeline_completed",
                       processed=result.processed,
                       failed=result.failed,
                       skipped=result.skipped,
                       chunks_created=result.chunks_created)

            return result

        except Exception as e:
            logger.exception("pipeline_failed", error=str(e))
            raise

    async def _process_video(self, video: VideoMetadata) -> dict:
        """Process a single video.

        Args:
            video: Video metadata.

        Returns:
            Processing result dict.
        """
        logger.info("processing_video", video_id=video.id)

        # Check if already processed
        if await self.storage_service.is_video_processed(video.id):
            logger.info("video_already_processed", video_id=video.id)
            return {"status": "skipped", "chunks": 0}

        # Update status to processing
        await self.storage_service.save_video(video, status="processing")

        retry_count = 0
        last_error = None

        for attempt in range(self.config.max_retries + 1):
            try:
                # 1. Fetch transcript
                transcript = await self.youtube_service.get_transcript(
                    video_id=video.id,
                    retry=(attempt > 0)
                )

                if transcript is None:
                    # Transcript unavailable
                    await self.storage_service.update_video_status(
                        video.id,
                        status="failed",
                        error_message="Transcript unavailable",
                        retry_count=retry_count
                    )
                    return {"status": "failed", "error": "Transcript unavailable", "chunks": 0}

                # Enrich video metadata from transcript
                # (title might be available from Supadata response metadata)

                # 2. Chunk transcript
                chunks = self.chunking_service.chunk_transcript(transcript, video)

                # 3. Generate embeddings
                texts = [chunk.text_content for chunk in chunks]
                embeddings = await self.embedding_service.embed_batch(texts)

                # 4. Combine chunks with embeddings
                chunks_with_embeddings = [
                    ChunkWithEmbedding(**chunk.dict(), embedding=embedding)
                    for chunk, embedding in zip(chunks, embeddings)
                ]

                # 5. Store in database
                await self.storage_service.save_chunks(chunks_with_embeddings)

                # 6. Update status to completed
                await self.storage_service.update_video_status(
                    video.id,
                    status="completed",
                    retry_count=retry_count
                )

                logger.info("video_processed", video_id=video.id, chunks=len(chunks))
                return {"status": "completed", "chunks": len(chunks)}

            except Exception as e:
                retry_count += 1
                last_error = str(e)
                logger.warning("video_processing_attempt_failed",
                             video_id=video.id,
                             attempt=attempt + 1,
                             error=str(e))

                if attempt < self.config.max_retries:
                    logger.info("retrying_video", video_id=video.id, attempt=attempt + 2)
                    continue
                else:
                    # Max retries exceeded
                    await self.storage_service.update_video_status(
                        video.id,
                        status="failed",
                        error_message=last_error,
                        retry_count=retry_count
                    )
                    logger.error("video_processing_failed", video_id=video.id, error=last_error)
                    return {"status": "failed", "error": last_error, "chunks": 0}

        return {"status": "failed", "error": "Unknown error", "chunks": 0}
```

**Key features:**
- Orchestrates all services
- Deduplication and status tracking
- Retry logic for failed transcripts
- Comprehensive result tracking
- Error handling and logging

**Dependencies:** Tasks 2.1, 2.2, 2.3, 2.4
**Estimated effort:** 3-4 hours

---

#### Task 3.2: CLI Interface
**Description:** Command-line interface for running the pipeline.

**Files to create:**
- `src/rag_pipeline/cli.py`

**Implementation:**
```python
import asyncio
import argparse
from .pipeline import YouTubeRAGPipeline
from .config import get_config
from src.utils.logging import get_logger

logger = get_logger(__name__)

async def main():
    """CLI entry point for YouTube RAG pipeline."""
    parser = argparse.ArgumentParser(description="YouTube RAG Pipeline")
    parser.add_argument("--channel-id", type=str, help="Override YouTube channel ID")
    parser.add_argument("--days-back", type=int, help="Number of days to look back")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode (no database writes)")

    args = parser.parse_args()

    # Load config
    config = get_config()

    # Override config with CLI args
    if args.channel_id:
        config.youtube_channel_id = args.channel_id
    if args.days_back:
        config.days_back = args.days_back

    logger.info("cli_started", channel_id=config.youtube_channel_id, days_back=config.days_back)

    # Run pipeline
    pipeline = YouTubeRAGPipeline(config)
    result = await pipeline.process_channel()

    # Print results
    print("\n" + "=" * 60)
    print("YouTube RAG Pipeline Results")
    print("=" * 60)
    print(f"Total videos: {result.total_videos}")
    print(f"Processed: {result.processed}")
    print(f"Failed: {result.failed}")
    print(f"Skipped: {result.skipped}")
    print(f"Chunks created: {result.chunks_created}")

    if result.errors:
        print("\nErrors:")
        for error in result.errors:
            print(f"  - {error}")

    print("=" * 60)
    logger.info("cli_completed", result=result.dict())

if __name__ == "__main__":
    asyncio.run(main())
```

**Usage:**
```bash
# Use env vars
uv run python -m src.rag_pipeline.cli

# Override channel
uv run python -m src.rag_pipeline.cli --channel-id UCxxxxx

# Custom days back
uv run python -m src.rag_pipeline.cli --days-back 14
```

**Dependencies:** Task 3.1
**Estimated effort:** 1 hour

---

### Phase 4: Testing (6-8 hours)

#### Task 4.1: Unit Tests for Services
**Description:** Write unit tests for each service.

**Files to create:**
- `tests/rag_pipeline/test_youtube_service.py`
- `tests/rag_pipeline/test_chunking_service.py`
- `tests/rag_pipeline/test_embedding_service.py`
- `tests/rag_pipeline/test_storage_service.py`

**Example:** `tests/rag_pipeline/test_chunking_service.py`
```python
import pytest
from src.rag_pipeline.chunking_service import ChunkingService
from src.rag_pipeline.schemas import Transcript, TranscriptSegment, VideoMetadata
from src.rag_pipeline.config import YouTubeRAGConfig

@pytest.fixture
def chunking_service():
    config = YouTubeRAGConfig(
        min_tokens=100,
        max_tokens=300,
        embedding_model="all-MiniLM-L6-v2"
    )
    return ChunkingService(config)

@pytest.fixture
def sample_transcript():
    segments = [
        TranscriptSegment(text="First segment text.", offset_ms=0, duration_ms=2000),
        TranscriptSegment(text="Second segment text.", offset_ms=2000, duration_ms=2000),
        TranscriptSegment(text="Third segment text.", offset_ms=4000, duration_ms=2000),
    ]
    return Transcript(
        video_id="test123",
        segments=segments,
        lang="en",
        available_langs=["en"]
    )

@pytest.fixture
def sample_video():
    return VideoMetadata(
        id="test123",
        channel_id="channel123",
        title="Test Video",
        url="https://youtube.com/watch?v=test123"
    )

@pytest.mark.unit
def test_chunk_transcript_respects_max_tokens(chunking_service, sample_transcript, sample_video):
    """Test that chunks don't exceed max token limit."""
    chunks = chunking_service.chunk_transcript(sample_transcript, sample_video)

    for chunk in chunks:
        assert chunk.token_count <= chunking_service.config.max_tokens

@pytest.mark.unit
def test_chunk_transcript_preserves_timestamps(chunking_service, sample_transcript, sample_video):
    """Test that timestamp information is preserved."""
    chunks = chunking_service.chunk_transcript(sample_transcript, sample_video)

    assert chunks[0].start_offset_ms == 0
    assert all(chunk.start_offset_ms >= 0 for chunk in chunks)
    assert all(chunk.end_offset_ms > chunk.start_offset_ms for chunk in chunks)

@pytest.mark.unit
def test_chunk_metadata_includes_video_info(chunking_service, sample_transcript, sample_video):
    """Test that chunk metadata includes video information."""
    chunks = chunking_service.chunk_transcript(sample_transcript, sample_video)

    for chunk in chunks:
        assert chunk.metadata["video_title"] == sample_video.title
        assert chunk.metadata["video_url"] == sample_video.url
        assert chunk.metadata["channel_id"] == sample_video.channel_id
        assert "timestamp_url" in chunk.metadata
```

**Dependencies:** Phase 2 tasks
**Estimated effort:** 3-4 hours

---

#### Task 4.2: Integration Tests
**Description:** Test full pipeline flow.

**Files to create:**
- `tests/integration/test_rag_pipeline.py`

**Example:**
```python
import pytest
from src.rag_pipeline.pipeline import YouTubeRAGPipeline
from src.rag_pipeline.config import YouTubeRAGConfig

@pytest.mark.integration
@pytest.mark.asyncio
async def test_pipeline_end_to_end(test_supabase_client, test_embedding_client):
    """Test full pipeline from video fetch to storage."""
    config = YouTubeRAGConfig(
        supadata_api_key="test_key",
        youtube_channel_id="test_channel",
        days_back=7,
        max_retries=1,
        min_tokens=100,
        max_tokens=500
    )

    pipeline = YouTubeRAGPipeline(config)

    # Mock external API calls
    with patch.object(pipeline.youtube_service, 'get_recent_videos') as mock_videos:
        mock_videos.return_value = [/* test video data */]

        result = await pipeline.process_channel()

        assert result.total_videos > 0
        # Add more assertions
```

**Dependencies:** Phase 3 tasks
**Estimated effort:** 2-3 hours

---

#### Task 4.3: Validation & Linting
**Description:** Ensure code passes all quality checks.

**Commands:**
```bash
# Type checking
uv run mypy src/rag_pipeline/ src/tools/rag_tools/

# Linting
uv run ruff check src/rag_pipeline/ src/tools/rag_tools/

# Auto-fix
uv run ruff check --fix src/

# Run tests
uv run pytest tests/ -v -m unit
uv run pytest tests/ -m integration
```

**Dependencies:** All previous tasks
**Estimated effort:** 1 hour

---

### Phase 5: Documentation & Deployment (2-3 hours)

#### Task 5.1: Update Dependencies
**Description:** Add new dependencies to project.

**File to update:** `pyproject.toml` or `requirements.txt`

**Dependencies to add:**
```txt
# YouTube RAG Pipeline
supadata>=1.3.1
docling>=1.0.0
transformers>=4.36.0
sentence-transformers>=2.2.2

# Already included (verify versions)
supabase>=2.15.1
openai>=1.79.0
pydantic>=2.11.4
structlog>=24.0.0
```

**Estimated effort:** 15 minutes

---

#### Task 5.2: Usage Documentation
**Description:** Create user-facing documentation.

**File to create:** `docs/youtube-rag-pipeline.md`

**Content:**
```markdown
# YouTube RAG Pipeline

## Overview
Automated pipeline to fetch YouTube transcripts, chunk them intelligently, and store in vector database for AI coach queries.

## Setup

### 1. Environment Variables
Add to `.env`:
\`\`\`bash
# Supadata
SUPADATA_API_KEY=your_key_here
YOUTUBE_CHANNEL_ID=UCxxxxxxxxxxxxx

# Processing
YOUTUBE_DAYS_BACK=7
YOUTUBE_MAX_RETRIES=1
YOUTUBE_BATCH_SIZE=5

# Chunking
MIN_CHUNK_TOKENS=400
MAX_CHUNK_TOKENS=1000

# Embeddings (reuse existing)
EMBEDDING_PROVIDER=openai
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_API_KEY=sk-...
EMBEDDING_MODEL_CHOICE=text-embedding-3-small

# Database (reuse existing)
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_SERVICE_KEY=eyJ...
\`\`\`

### 2. Database Setup
Run migration in Supabase Dashboard > SQL Editor:
\`\`\`bash
# migrations/001_youtube_rag_schema.sql
\`\`\`

### 3. Install Dependencies
\`\`\`bash
uv sync
\`\`\`

## Usage

### CLI
\`\`\`bash
# Process channel videos from last 7 days
uv run python -m src.rag_pipeline.cli

# Custom parameters
uv run python -m src.rag_pipeline.cli --channel-id UCxxx --days-back 14
\`\`\`

### Programmatic
\`\`\`python
from src.rag_pipeline.pipeline import YouTubeRAGPipeline

pipeline = YouTubeRAGPipeline()
result = await pipeline.process_channel()
print(f"Processed {result.processed} videos, created {result.chunks_created} chunks")
\`\`\`

### Agent Tool
\`\`\`python
# Agent automatically has access to search_youtube_transcripts tool
response = await agent.run("What are the best productivity tips from my videos?")
\`\`\`

## Scheduling

### Cron (Linux/Mac)
\`\`\`bash
# Daily at 2 AM
0 2 * * * cd /path/to/project && uv run python -m src.rag_pipeline.cli >> logs/rag-pipeline.log 2>&1
\`\`\`

### Task Scheduler (Windows)
Create scheduled task to run:
\`\`\`
C:\\path\\to\\uv.exe run python -m src.rag_pipeline.cli
\`\`\`

## Monitoring

Check logs for errors:
\`\`\`bash
grep "error" logs/rag-pipeline.log
grep "failed" logs/rag-pipeline.log
\`\`\`

Query processing status:
\`\`\`sql
SELECT status, COUNT(*) FROM videos GROUP BY status;
\`\`\`

## Troubleshooting

### Transcript Unavailable
Some videos don't have transcripts. Pipeline marks these as "failed" with error message.

### Rate Limit Exceeded
Reduce `YOUTUBE_BATCH_SIZE` or add delays between API calls.

### Token Limit Errors
Adjust `MAX_CHUNK_TOKENS` based on your embedding model's limits.
```

**Estimated effort:** 1 hour

---

#### Task 5.3: Update Project README
**Description:** Add YouTube RAG pipeline section to main README.

**File to update:** `README.md`

**Content to add:**
```markdown
## YouTube RAG Pipeline

Automatically fetches YouTube video transcripts, chunks them intelligently, and stores in vector database for semantic search by the AI coach.

**Features:**
- Fetch transcripts from YouTube channel with timestamps
- Token-aware hybrid chunking (400-1000 tokens)
- Configurable embedding provider (OpenAI/Ollama)
- Vector storage in Supabase with pgvector
- CLI and programmatic interfaces
- Retry logic and error handling

**Quick Start:**
\`\`\`bash
# Setup environment variables (see .env.example)
# Run database migration (see docs/youtube-rag-pipeline.md)
# Process channel videos
uv run python -m src.rag_pipeline.cli
\`\`\`

See [docs/youtube-rag-pipeline.md](docs/youtube-rag-pipeline.md) for full documentation.
```

**Estimated effort:** 30 minutes

---

## Dependencies and Libraries

### New Dependencies
```txt
supadata>=1.3.1              # YouTube transcript API
docling>=1.0.0               # Hybrid chunking
transformers>=4.36.0         # Tokenizers
sentence-transformers>=2.2.2 # Embedding models (optional)
```

### Existing Dependencies (verify)
```txt
supabase>=2.15.1             # Vector database
openai>=1.79.0               # Embeddings API
pydantic>=2.11.4             # Data validation
pydantic-ai>=0.2.4           # Agent framework
structlog>=24.0.0            # Structured logging
pytest>=8.3.5                # Testing
pytest-asyncio>=0.26.0       # Async testing
mypy                         # Type checking
ruff                         # Linting
```

---

## Testing Strategy

### Unit Tests
- **YouTube Service:** Mock Supadata API calls, test error handling
- **Chunking Service:** Test token limits, timestamp preservation, merge logic
- **Embedding Service:** Mock OpenAI calls, test batch processing
- **Storage Service:** Mock Supabase calls, test deduplication logic

### Integration Tests
- **End-to-End Pipeline:** Test full flow from video fetch to storage
- **Vector Search:** Test RAG tool with real database (test environment)
- **Error Recovery:** Test retry logic and failure scenarios

### Test Markers
```python
@pytest.mark.unit           # Fast, isolated tests
@pytest.mark.integration    # Slower, requires external services
```

### Coverage Goal
- Aim for >80% code coverage on core services
- 100% coverage on critical paths (chunking, embedding, storage)

---

## Success Criteria

- [ ] Pipeline successfully fetches videos from YouTube channel
- [ ] Transcripts chunked respecting 400-1000 token limits
- [ ] Embeddings generated using configured provider
- [ ] Chunks stored in Supabase with vector index
- [ ] RAG tool returns relevant results for queries
- [ ] Deduplication prevents reprocessing videos
- [ ] Retry logic handles transient failures
- [ ] All tests pass (unit + integration)
- [ ] Type checking passes (mypy strict mode)
- [ ] Linting passes (ruff)
- [ ] Documentation complete and accurate
- [ ] CLI interface functional
- [ ] Logging provides debugging context
- [ ] Performance: Process 10 videos in <5 minutes

---

## Notes and Considerations

### Rate Limiting
- Supadata has plan-based rate limits
- Implement exponential backoff for `limit-exceeded` errors
- Consider batching video processing to stay within limits

### Embedding Model Selection
- **Local/Free:** `sentence-transformers/all-MiniLM-L6-v2` (384 dim)
- **Production/Quality:** OpenAI `text-embedding-3-small` (1536 dim)
- **Must match:** Vector dimension in database schema

### Chunking Strategy
- Min 400 tokens ensures meaningful context
- Max 1000 tokens fits most embedding model limits
- Merge small chunks to avoid fragmentation
- Preserve timestamps for video navigation

### Database Optimization
- IVFFlat index: `lists = 100` is good for <1M rows
- Consider HNSW index for >1M rows (better recall, slower inserts)
- Adjust `match_count` in vector search based on use case

### Future Enhancements
- Support for multiple channels
- Scheduled background processing (FastAPI background tasks)
- Web dashboard for monitoring
- Support for other video platforms (Vimeo, etc.)
- Incremental processing (only new videos since last run)
- Webhook triggers for new video uploads

### Security
- Store API keys in environment variables (never commit)
- Use Supabase RLS policies for access control
- Validate user inputs in agent tools
- Rate limit agent tool calls to prevent abuse

---

## Timeline Estimate

**Total Estimated Time: 22-31 hours**

- **Phase 1:** Foundation & Configuration (4-6 hours)
- **Phase 2:** Core Services (8-12 hours)
- **Phase 3:** Pipeline Orchestration (4-6 hours)
- **Phase 4:** Testing (6-8 hours)
- **Phase 5:** Documentation & Deployment (2-3 hours)

**Recommended Sprint:**
- **Week 1:** Phases 1-2 (Foundation + Core Services)
- **Week 2:** Phase 3 (Pipeline Orchestration)
- **Week 3:** Phases 4-5 (Testing + Documentation)

**Note:** Agent RAG tools for querying the knowledge base will be implemented in a separate phase after the pipeline is operational.

---

*This plan is ready for execution with `/execute-plan PRPs/requests/youtube-rag-pipeline.md`*
