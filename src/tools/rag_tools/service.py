"""RAG tools service layer implementation.

Contains helper functions and tool implementation functions for searching
YouTube transcript chunks and retrieving full video transcripts.
"""

import os

from openai import AsyncOpenAI
from supabase import Client

from src.utils.logging import get_logger

logger = get_logger(__name__)


# ==============================================================================
# Helper Functions (Deterministic, not exposed as LLM tools)
# ==============================================================================


def format_video_url(video_id: str, timestamp_ms: int | None = None) -> str:
    """Format a YouTube video URL with optional timestamp.

    Args:
        video_id: YouTube video ID (e.g., "dQw4w9WgXcQ").
        timestamp_ms: Optional start time in milliseconds.
            If provided, converts to seconds and adds &t=XXs parameter.

    Returns:
        YouTube video URL string with optional timestamp.

    Examples:
        >>> format_video_url("dQw4w9WgXcQ")
        "https://youtube.com/watch?v=dQw4w9WgXcQ"
        >>> format_video_url("dQw4w9WgXcQ", 120000)
        "https://youtube.com/watch?v=dQw4w9WgXcQ&t=120s"
    """
    base_url = f"https://youtube.com/watch?v={video_id}"

    if timestamp_ms is not None and timestamp_ms > 0:
        timestamp_seconds = int(timestamp_ms / 1000)
        return f"{base_url}&t={timestamp_seconds}s"

    return base_url


def format_timestamp_display(milliseconds: int) -> str:
    """Format milliseconds as [MM:SS] or [HH:MM:SS] for display.

    Args:
        milliseconds: Time in milliseconds.

    Returns:
        Formatted timestamp string like [MM:SS] or [HH:MM:SS].

    Examples:
        >>> format_timestamp_display(125000)
        "[02:05]"
        >>> format_timestamp_display(3725000)
        "[01:02:05]"
    """
    total_seconds = int(milliseconds / 1000)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60

    if hours > 0:
        return f"[{hours:02d}:{minutes:02d}:{seconds:02d}]"
    else:
        return f"[{minutes:02d}:{seconds:02d}]"


def format_duration(seconds: int) -> str:
    """Format seconds as HH:MM:SS or MM:SS for video duration display.

    Args:
        seconds: Duration in seconds.

    Returns:
        Formatted duration string like MM:SS or HH:MM:SS.

    Examples:
        >>> format_duration(125)
        "02:05"
        >>> format_duration(3725)
        "01:02:05"
    """
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"


async def get_embedding(text: str, client: AsyncOpenAI) -> list[float]:
    """Generate embedding vector for text using OpenAI-compatible API.

    Args:
        text: Text to embed.
        client: AsyncOpenAI client configured with API credentials.

    Returns:
        List of floats representing the embedding vector.

    Raises:
        Exception: If embedding generation fails.

    Examples:
        >>> embedding = await get_embedding("coaching advice", client)
        >>> len(embedding)
        1536  # For text-embedding-3-small
    """
    logger.info("embedding_generation_started", text_length=len(text))

    try:
        embedding_model = os.getenv("EMBEDDING_MODEL_CHOICE", "text-embedding-3-small")

        response = await client.embeddings.create(
            input=[text],
            model=embedding_model,
        )

        embedding = response.data[0].embedding

        logger.info(
            "embedding_generation_completed",
            model=embedding_model,
            embedding_dim=len(embedding),
        )

        return embedding

    except Exception as e:
        logger.exception(
            "embedding_generation_failed",
            text_length=len(text),
            error_type=type(e).__name__,
        )
        raise


# ==============================================================================
# Tool Implementation Functions
# ==============================================================================


async def search_transcript_chunks(
    supabase: Client,
    embedding_client: AsyncOpenAI,
    query: str,
    match_count: int = 5,
) -> str:
    """Search for relevant transcript chunks using vector similarity.

    This function:
    1. Generates an embedding for the query
    2. Calls the match_transcript_chunks RPC function
    3. Formats results with video metadata and citations

    Args:
        supabase: Supabase client for database operations.
        embedding_client: AsyncOpenAI client for generating embeddings.
        query: User's search query.
        match_count: Maximum number of results to return (default: 5).

    Returns:
        Formatted string containing search results with:
        - Video title, URL, and duration
        - Timestamp with clickable YouTube link
        - Transcript chunk text
        - Similarity score

    Raises:
        Exception: If search fails (embedding generation or database query).

    Examples:
        >>> results = await search_transcript_chunks(
        ...     supabase, client, "how to set goals", match_count=3
        ... )
    """
    logger.info(
        "transcript_search_started",
        query=query,
        match_count=match_count,
    )

    try:
        # Generate embedding for the query
        query_embedding = await get_embedding(query, embedding_client)

        # Call Supabase RPC function for vector search
        response = supabase.rpc(
            "match_transcript_chunks",
            {
                "query_embedding": query_embedding,
                "match_count": match_count,
                "filter": {},
            },
        ).execute()

        chunks = response.data if response.data else []

        logger.info(
            "transcript_search_completed",
            query=query,
            results_found=len(chunks),
        )

        if not chunks:
            return "No relevant coaching content found for your query. Try rephrasing or using different keywords."

        # Fetch video metadata for all unique video IDs
        video_ids = list(set(chunk["video_id"] for chunk in chunks))

        videos_response = (
            supabase.table("videos")
            .select("id, title, url, duration_seconds")
            .in_("id", video_ids)
            .execute()
        )

        # Create video lookup dict
        videos = {v["id"]: v for v in videos_response.data} if videos_response.data else {}

        # Format results with citations
        formatted_results = []

        for i, chunk in enumerate(chunks, 1):
            video_id = chunk["video_id"]
            video = videos.get(video_id, {})

            video_title = video.get("title", "Unknown Video")
            duration_seconds = video.get("duration_seconds", 0)
            duration_display = format_duration(duration_seconds) if duration_seconds else "Unknown"

            start_ms = chunk.get("start_offset_ms", 0)
            timestamp_display = format_timestamp_display(start_ms)
            video_url_with_timestamp = format_video_url(video_id, start_ms)

            similarity = chunk.get("similarity", 0)
            text_content = chunk.get("text_content", "")

            result = f"""
**Result {i}** (Similarity: {similarity:.2%})
**Video:** {video_title} (Duration: {duration_display})
**Timestamp:** {timestamp_display} - {video_url_with_timestamp}

{text_content}
"""
            formatted_results.append(result.strip())

        final_output = "\n\n---\n\n".join(formatted_results)

        logger.info(
            "transcript_search_formatted",
            results_count=len(formatted_results),
            total_chars=len(final_output),
        )

        return final_output

    except Exception as e:
        logger.exception(
            "transcript_search_failed",
            query=query,
            error_type=type(e).__name__,
        )
        raise


async def get_full_video_transcript(
    supabase: Client,
    video_id: str,
    max_chars: int = 20000,
) -> str:
    """Retrieve the full transcript for a video by combining all chunks.

    This function:
    1. Fetches all transcript chunks for the video
    2. Orders them by chunk_index
    3. Combines them into a single transcript
    4. Truncates if exceeding max_chars limit

    Args:
        supabase: Supabase client for database operations.
        video_id: YouTube video ID.
        max_chars: Maximum characters to return (default: 20000).

    Returns:
        Formatted string containing:
        - Video title, URL, and duration
        - Full transcript text (or truncated with warning)

    Raises:
        Exception: If video not found or transcript retrieval fails.

    Examples:
        >>> transcript = await get_full_video_transcript(
        ...     supabase, "dQw4w9WgXcQ", max_chars=15000
        ... )
    """
    logger.info(
        "full_transcript_retrieval_started",
        video_id=video_id,
        max_chars=max_chars,
    )

    try:
        # Fetch video metadata
        video_response = (
            supabase.table("videos")
            .select("id, title, url, duration_seconds, transcript_available")
            .eq("id", video_id)
            .single()
            .execute()
        )

        if not video_response.data:
            logger.warning("video_not_found", video_id=video_id)
            return f"Video with ID '{video_id}' not found in the knowledge base."

        video = video_response.data

        if not video.get("transcript_available", False):
            logger.warning(
                "transcript_not_available",
                video_id=video_id,
                video_title=video.get("title", "Unknown"),
            )
            return f"Transcript is not available for video: {video.get('title', 'Unknown')}"

        # Fetch all transcript chunks for this video
        chunks_response = (
            supabase.table("transcript_chunks")
            .select("chunk_index, text_content, start_offset_ms, end_offset_ms")
            .eq("video_id", video_id)
            .order("chunk_index")
            .execute()
        )

        chunks = chunks_response.data if chunks_response.data else []

        if not chunks:
            logger.warning("no_transcript_chunks_found", video_id=video_id)
            return f"No transcript chunks found for video: {video.get('title', 'Unknown')}"

        # Combine all chunks in order
        full_text = "\n\n".join(chunk["text_content"] for chunk in chunks)

        # Truncate if necessary
        was_truncated = False
        if len(full_text) > max_chars:
            full_text = full_text[:max_chars]
            was_truncated = True

        # Format output
        video_title = video.get("title", "Unknown Video")
        video_url = video.get("url", format_video_url(video_id))
        duration_seconds = video.get("duration_seconds", 0)
        duration_display = format_duration(duration_seconds) if duration_seconds else "Unknown"

        output = f"""
**Video:** {video_title}
**URL:** {video_url}
**Duration:** {duration_display}
**Transcript Chunks:** {len(chunks)}

---

{full_text}
"""

        if was_truncated:
            output += f"\n\n[TRUNCATED: Transcript exceeded {max_chars} characters limit]"

        logger.info(
            "full_transcript_retrieval_completed",
            video_id=video_id,
            chunks_count=len(chunks),
            total_chars=len(full_text),
            was_truncated=was_truncated,
        )

        return output.strip()

    except Exception as e:
        logger.exception(
            "full_transcript_retrieval_failed",
            video_id=video_id,
            error_type=type(e).__name__,
        )
        raise
