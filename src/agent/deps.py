"""Agent dependency definitions.

Defines the AgentDeps dataclass that holds runtime dependencies
passed to agent tools during execution.
"""

from dataclasses import dataclass

from httpx import AsyncClient
from openai import AsyncOpenAI
from supabase import Client


@dataclass
class AgentDeps:
    """Runtime dependencies for agent tools.

    These dependencies are injected into agent tools via RunContext
    and provide access to external services needed for tool execution.

    Attributes:
        supabase: Supabase client for database operations (RAG search, history).
        embedding_client: AsyncOpenAI client for generating embeddings.
        http_client: AsyncClient for making HTTP requests (future use).
    """

    supabase: Client
    embedding_client: AsyncOpenAI
    http_client: AsyncClient
