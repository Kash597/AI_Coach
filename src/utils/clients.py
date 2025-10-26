"""Client initialization utilities.

Provides functions for initializing external service clients
(Supabase, OpenAI) used by the agent.
"""

import os

from openai import AsyncOpenAI
from supabase import Client


def get_agent_clients() -> tuple[AsyncOpenAI, Client]:
    """Initialize and return embedding and Supabase clients.

    Reads configuration from environment variables:
    - EMBEDDING_BASE_URL: OpenAI-compatible API base URL
    - EMBEDDING_API_KEY: API key for embeddings
    - SUPABASE_URL: Supabase project URL
    - SUPABASE_SERVICE_KEY: Supabase service role key

    Returns:
        Tuple of (AsyncOpenAI embedding client, Supabase client).

    Raises:
        ValueError: If required environment variables are missing.

    Examples:
        >>> embedding_client, supabase = get_agent_clients()
        >>> # Use clients for RAG operations
    """
    # Embedding client setup
    base_url = os.getenv("EMBEDDING_BASE_URL", "https://api.openai.com/v1")
    api_key = os.getenv("EMBEDDING_API_KEY")

    if not api_key:
        raise ValueError("EMBEDDING_API_KEY environment variable is required")

    embedding_client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    # Supabase client setup
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_KEY")

    if not supabase_url or not supabase_key:
        raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY environment variables are required")

    supabase = Client(supabase_url, supabase_key)

    return embedding_client, supabase
