"""Agent configuration utilities.

Provides functions for loading LLM configuration and agent settings from
environment variables.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

# Check if we're in production
is_production = os.getenv("ENVIRONMENT") == "production"

if not is_production:
    # Development: prioritize .env file
    project_root = Path(__file__).resolve().parent.parent.parent
    dotenv_path = project_root / ".env"
    load_dotenv(dotenv_path, override=True)
else:
    # Production: use cloud platform env vars only
    load_dotenv()


def get_model() -> OpenAIModel:
    """Get the configured LLM model for the agent.

    Reads configuration from environment variables:
    - LLM_CHOICE: Model name (default: gpt-4o-mini)
    - LLM_BASE_URL: API base URL (default: https://api.openai.com/v1)
    - LLM_API_KEY: API key (default: ollama for local testing)

    Returns:
        OpenAIModel configured with environment settings.

    Examples:
        >>> model = get_model()
        >>> # Uses gpt-4o-mini by default
    """
    llm = os.getenv("LLM_CHOICE") or "gpt-4o-mini"
    base_url = os.getenv("LLM_BASE_URL") or "https://api.openai.com/v1"
    api_key = os.getenv("LLM_API_KEY") or "ollama"

    return OpenAIModel(llm, provider=OpenAIProvider(base_url=base_url, api_key=api_key))


def get_max_transcript_chars() -> int:
    """Get the maximum transcript character limit.

    Reads MAX_TRANSCRIPT_CHARS from environment (default: 20000).
    This limit prevents excessive token usage when retrieving full transcripts.

    Returns:
        Maximum number of characters to return in full transcript retrieval.

    Examples:
        >>> max_chars = get_max_transcript_chars()
        >>> # Returns 20000 by default
    """
    return int(os.getenv("MAX_TRANSCRIPT_CHARS", "20000"))


def get_rate_limit() -> int:
    """Get the rate limit for API requests.

    Reads RATE_LIMIT_REQUESTS from environment (default: 5).
    This determines how many requests a user can make per minute.

    Returns:
        Maximum number of requests per minute per user.

    Examples:
        >>> rate_limit = get_rate_limit()
        >>> # Returns 5 by default
    """
    return int(os.getenv("RATE_LIMIT_REQUESTS", "5"))
