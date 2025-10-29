"""Script to clear existing data and force reprocessing with correct format.

This script:
1. Deletes all transcript chunks
2. Resets video statuses to 'pending'
3. Allows the pipeline to reprocess with the fixed storage format
"""

import asyncio
import os

from dotenv import load_dotenv
from supabase import create_client

load_dotenv()


async def clear_and_reset() -> None:
    """Clear transcript data and reset video statuses."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")

    if not url or not key:
        raise ValueError("Missing SUPABASE_URL or SUPABASE_SERVICE_KEY")

    client = create_client(url, key)

    # Get current stats
    videos_response = client.table("videos").select("id, title, status").execute()
    chunks_response = client.table("transcript_chunks").select("id").execute()

    print(f"Current state:")
    print(f"  Videos: {len(videos_response.data) if videos_response.data else 0}")
    print(f"  Chunks: {len(chunks_response.data) if chunks_response.data else 0}")

    # Ask for confirmation
    print(f"\nThis will:")
    print(f"  1. DELETE all transcript chunks")
    print(f"  2. RESET all video statuses to 'pending'")
    print(f"  3. Allow you to re-run the pipeline")

    confirm = input("\nAre you sure? Type 'yes' to continue: ")

    if confirm.lower() != "yes":
        print("Aborted")
        return

    # Delete all transcript chunks
    print("\nDeleting transcript chunks...")
    client.table("transcript_chunks").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()

    # Reset video statuses
    print("Resetting video statuses...")
    client.table("videos").update(
        {
            "status": "pending",
            "processed_at": None,
            "transcript_available": False,
            "error_message": None,
            "retry_count": 0,
        }
    ).neq("id", "dummy").execute()

    print("\nDone! You can now run the pipeline:")
    print("  uv run python -m src.rag_pipeline.cli --days-back 7")


if __name__ == "__main__":
    asyncio.run(clear_and_reset())
