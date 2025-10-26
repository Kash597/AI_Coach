"""YouTube service for fetching videos and transcripts via Supadata API."""

from supadata import Supadata

from src.utils.logging import get_logger

from .config import YouTubeRAGConfig
from .schemas import Transcript, TranscriptSegment, VideoMetadata

logger = get_logger(__name__)


class YouTubeService:
    """Service for fetching YouTube data via Supadata API.

    This service provides methods to fetch recent videos from a YouTube channel
    and retrieve transcripts with timestamps. It handles API errors gracefully
    and provides structured logging for debugging.
    """

    def __init__(self, config: YouTubeRAGConfig):
        """Initialize YouTube service with configuration.

        Args:
            config: Configuration object with Supadata API key and settings.
        """
        self.config = config
        self.client = Supadata(api_key=config.supadata_api_key)
        logger.info("youtube_service_initialized", api_key_present=bool(config.supadata_api_key))

    async def get_recent_videos(
        self, channel_id: str, days_back: int = 7
    ) -> list[VideoMetadata]:
        """Fetch recent videos from a YouTube channel.

        Since Supadata doesn't provide publish dates in the bulk endpoint,
        we estimate the number of videos to fetch based on days_back.

        Args:
            channel_id: YouTube channel ID, URL, or handle.
            days_back: Number of days to look back (used to calculate video limit).

        Returns:
            List of VideoMetadata objects for recent videos.

        Raises:
            Exception: If the API request fails.
        """
        # Estimate: assume ~2 videos per day max, with min 5 and max 50
        video_limit = max(5, min(days_back * 2, 50))

        logger.info(
            "fetching_channel_videos",
            channel_id=channel_id,
            days_back=days_back,
            video_limit=video_limit,
        )

        try:
            # Fetch video IDs from channel
            response = self.client.youtube.channel.videos(
                id=channel_id,
                type="video",  # Exclude shorts and live streams
                limit=video_limit,
            )

            video_ids = response.video_ids
            logger.info("videos_fetched", count=len(video_ids))

            # Create VideoMetadata objects
            recent_videos = []
            for video_id in video_ids:
                video = VideoMetadata(
                    id=video_id,
                    channel_id=channel_id,
                    title="",  # Will be enriched later from transcript metadata
                    url=f"https://youtube.com/watch?v={video_id}",
                )
                recent_videos.append(video)

            logger.info("videos_processed", total=len(recent_videos))
            return recent_videos

        except Exception as e:
            logger.exception(
                "channel_fetch_failed",
                channel_id=channel_id,
                error_type=type(e).__name__,
            )
            raise

    async def get_transcript(
        self, video_id: str, retry: bool = False
    ) -> Transcript | None:
        """Fetch transcript for a video with timestamps.

        This method retrieves the transcript segments with timing information
        for the specified video. Returns None if transcript is unavailable.

        Args:
            video_id: YouTube video ID.
            retry: Whether this is a retry attempt (for logging purposes).

        Returns:
            Transcript object with segments and language info, or None if unavailable.

        Raises:
            Exception: If API request fails (non-transcript errors).
        """
        logger.info("fetching_transcript", video_id=video_id, retry=retry)

        try:
            response = self.client.youtube.transcript(
                video_id=video_id,
                text=False,  # Get segments with timestamps instead of plain text
            )

            # Convert API response to TranscriptSegment objects
            segments = [
                TranscriptSegment(
                    text=seg.text,
                    offset_ms=int(seg.offset),
                    duration_ms=int(seg.duration),
                    lang=getattr(seg, "lang", "en"),
                )
                for seg in response.content
            ]

            transcript = Transcript(
                video_id=video_id,
                segments=segments,
                lang=response.lang,
                available_langs=response.available_langs,
            )

            logger.info(
                "transcript_fetched",
                video_id=video_id,
                segments=len(segments),
                lang=response.lang,
            )
            return transcript

        except Exception as e:
            # Check if transcript is unavailable (common case, not an error)
            error_str = str(e).lower()
            if "transcript-unavailable" in error_str or "206" in error_str:
                logger.warning("transcript_unavailable", video_id=video_id)
                return None

            # Real error - log and raise
            logger.exception(
                "transcript_fetch_error",
                video_id=video_id,
                error_type=type(e).__name__,
            )
            raise
