"""Script to fix embedding storage format in the database.

This script converts embeddings stored as TEXT strings to proper pgvector format.
Run this once to fix existing data.
"""

import asyncio
import json
import os

from dotenv import load_dotenv
from supabase import create_client

load_dotenv()


async def fix_embeddings() -> None:
    """Convert string embeddings to proper vector format."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")

    if not url or not key:
        raise ValueError("Missing SUPABASE_URL or SUPABASE_SERVICE_KEY")

    client = create_client(url, key)

    print("Fetching all transcript chunks...")
    response = client.table("transcript_chunks").select("id, embedding").execute()

    if not response.data:
        print("No chunks found")
        return

    print(f"Found {len(response.data)} chunks to process")

    fixed_count = 0
    error_count = 0

    for chunk in response.data:
        chunk_id = chunk["id"]
        embedding_str = chunk["embedding"]

        # Check if it's stored as string
        if isinstance(embedding_str, str):
            try:
                # Parse the string representation: "[0.1, 0.2, ...]"
                embedding_list = json.loads(embedding_str)

                # Convert to pgvector format (string without spaces)
                # PostgreSQL vector format: '[x,y,z]' (no spaces)
                vector_str = f"[{','.join(str(x) for x in embedding_list)}]"

                # Update using raw SQL to ensure proper type casting
                # We need to explicitly cast to vector type
                client.rpc(
                    "sql",
                    {"query": f"UPDATE transcript_chunks SET embedding = '{vector_str}'::vector WHERE id = '{chunk_id}'"},
                ).execute()

                fixed_count += 1

                if fixed_count % 10 == 0:
                    print(f"Processed {fixed_count} chunks...")

            except Exception as e:
                print(f"Error processing chunk {chunk_id}: {e}")
                error_count += 1
        else:
            print(f"Chunk {chunk_id} is already in correct format (type: {type(embedding_str)})")

    print(f"\nComplete!")
    print(f"Fixed: {fixed_count}")
    print(f"Errors: {error_count}")


if __name__ == "__main__":
    asyncio.run(fix_embeddings())
