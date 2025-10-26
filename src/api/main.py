"""FastAPI application for the AI Coach Agent.

Provides streaming agent endpoint with JWT authentication, rate limiting,
conversation management, and title generation.
"""

import asyncio
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from httpx import AsyncClient
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.messages import PartDeltaEvent, PartStartEvent, TextPartDelta

from src.agent.agent import agent
from src.agent.config import get_model, get_rate_limit
from src.agent.deps import AgentDeps

# Missing import - need to add update_conversation_title
from src.api.db_utils import (
    check_rate_limit,
    convert_history_to_pydantic_format,
    create_conversation,
    fetch_conversation_history,
    generate_conversation_title,
    generate_session_id,
    store_message,
    store_request,
    update_conversation_title,  # noqa: E402
)
from src.utils.clients import get_agent_clients
from src.utils.logging import get_logger

logger = get_logger(__name__)

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

# Global clients initialized in lifespan
embedding_client = None
supabase = None
http_client = None
title_agent = None


# ==============================================================================
# Lifespan Management
# ==============================================================================


async def lifespan(app: FastAPI):  # type: ignore[misc]
    """Lifecycle manager for the FastAPI application.

    Handles initialization and cleanup of resources.
    """
    global embedding_client, supabase, http_client, title_agent

    logger.info("application_startup_started")

    try:
        # Initialize clients
        embedding_client, supabase = get_agent_clients()
        http_client = AsyncClient()
        title_agent = Agent(model=get_model())

        logger.info(
            "application_startup_completed",
            clients=["embedding", "supabase", "http", "title_agent"],
        )

    except Exception:
        logger.exception("application_startup_failed")
        raise

    yield  # Application runs here

    # Shutdown: Clean up resources
    logger.info("application_shutdown_started")

    if http_client:
        await http_client.aclose()

    logger.info("application_shutdown_completed")


# ==============================================================================
# FastAPI Application Setup
# ==============================================================================

app = FastAPI(
    title="AI Coach Agent API",
    description="YouTube coaching knowledge base RAG agent with streaming responses",
    version="1.0.0",
    lifespan=lifespan,
)

security = HTTPBearer()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# Authentication
# ==============================================================================


async def verify_token(
    credentials: HTTPAuthorizationCredentials = Security(security),
) -> dict[str, Any]:
    """Verify the JWT token from Supabase and return the user information.

    Args:
        credentials: The HTTP Authorization credentials containing the bearer token.

    Returns:
        User information from Supabase.

    Raises:
        HTTPException: If the token is invalid or the user cannot be verified.
    """
    logger.info("auth_verification_started")

    try:
        # Get the token from the Authorization header
        token = credentials.credentials

        # Access the global HTTP client
        global http_client  # noqa: PLW0602
        if not http_client:
            logger.error("auth_verification_failed", reason="http_client_not_initialized")
            raise HTTPException(status_code=500, detail="HTTP client not initialized")

        # Get the Supabase URL and service key from environment
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_KEY")

        # Make request to Supabase auth API to get user info
        response = await http_client.get(
            f"{supabase_url}/auth/v1/user",
            headers={"Authorization": f"Bearer {token}", "apikey": supabase_key},
        )

        # Check if the request was successful
        if response.status_code != 200:
            logger.warning(
                "auth_verification_failed",
                status_code=response.status_code,
                response_text=response.text[:200],
            )
            raise HTTPException(status_code=401, detail="Invalid authentication token")

        # Return the user information
        user_data = response.json()

        logger.info("auth_verification_completed", user_id=user_data.get("id"))

        return user_data

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("auth_verification_error")
        raise HTTPException(status_code=401, detail=f"Authentication error: {str(e)}")


# ==============================================================================
# Request/Response Models
# ==============================================================================


class AgentRequest(BaseModel):
    """Request model for the agent endpoint."""

    query: str
    user_id: str
    request_id: str
    session_id: str


# ==============================================================================
# Helper Functions
# ==============================================================================


async def stream_error_response(error_message: str, session_id: str):
    """Create a streaming response for error messages.

    Args:
        error_message: The error message to display to the user.
        session_id: The current session ID.

    Yields:
        Encoded JSON chunks for the streaming response.
    """
    # First yield the error message as text
    yield json.dumps({"text": error_message}).encode("utf-8") + b"\n"

    # Then yield a final chunk with complete flag
    final_data = {
        "text": error_message,
        "session_id": session_id,
        "error": error_message,
        "complete": True,
    }
    yield json.dumps(final_data).encode("utf-8") + b"\n"


# ==============================================================================
# API Endpoints
# ==============================================================================


@app.get("/health")
async def health_check():
    """Health check endpoint.

    Returns:
        Health status and timestamp.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now(UTC).isoformat(),
        "services": {
            "embedding_client": embedding_client is not None,
            "supabase": supabase is not None,
            "http_client": http_client is not None,
            "title_agent": title_agent is not None,
        },
    }


@app.post("/api/pydantic-agent")
async def pydantic_agent_endpoint(
    request: AgentRequest,
    user: dict[str, Any] = Depends(verify_token),
):
    """Main agent endpoint with streaming response.

    Args:
        request: Agent request containing query and session info.
        user: Authenticated user info from JWT token.

    Returns:
        StreamingResponse with agent output.
    """
    logger.info(
        "agent_request_started",
        user_id=request.user_id,
        session_id=request.session_id,
        query_length=len(request.query),
    )

    # Verify that the user ID in the request matches the user ID from the token
    if request.user_id != user.get("id"):
        logger.warning(
            "agent_request_rejected",
            reason="user_id_mismatch",
            request_user_id=request.user_id,
            token_user_id=user.get("id"),
        )
        return StreamingResponse(
            stream_error_response(
                "User ID in request does not match authenticated user",
                request.session_id,
            ),
            media_type="text/plain",
        )

    try:
        # Check rate limit
        rate_limit = get_rate_limit()
        rate_limit_ok = await check_rate_limit(supabase, request.user_id, rate_limit)

        if not rate_limit_ok:
            logger.warning(
                "agent_request_rejected",
                reason="rate_limit_exceeded",
                user_id=request.user_id,
                rate_limit=rate_limit,
            )
            return StreamingResponse(
                stream_error_response(
                    "Rate limit exceeded. Please try again later.",
                    request.session_id,
                ),
                media_type="text/plain",
            )

        # Start request tracking in parallel
        request_tracking_task = asyncio.create_task(
            store_request(supabase, request.request_id, request.user_id, request.query)
        )

        session_id = request.session_id
        conversation_record = None

        # Check if session_id is empty, create a new conversation if needed
        if not session_id:
            session_id = generate_session_id(request.user_id)
            # Create a new conversation record
            conversation_record = await create_conversation(
                supabase, request.user_id, session_id
            )

            logger.info(
                "new_conversation_created",
                session_id=session_id,
                user_id=request.user_id,
            )

        # Store user's query immediately
        await store_message(
            supabase=supabase,
            session_id=session_id,
            message_type="human",
            content=request.query,
        )

        # Fetch conversation history from the DB
        conversation_history = await fetch_conversation_history(supabase, session_id)

        # Convert conversation history to Pydantic AI format
        pydantic_messages = await convert_history_to_pydantic_format(conversation_history)

        logger.info(
            "conversation_history_loaded",
            session_id=session_id,
            message_count=len(pydantic_messages),
        )

        # Start title generation in parallel if this is a new conversation
        title_task = None
        if conversation_record:
            title_task = asyncio.create_task(
                generate_conversation_title(title_agent, request.query)
            )

        async def stream_response():
            """Inner async generator for streaming agent response."""
            nonlocal conversation_record

            agent_deps = AgentDeps(
                embedding_client=embedding_client,
                supabase=supabase,
                http_client=http_client,
            )

            full_response = ""

            logger.info("agent_execution_started", session_id=session_id)

            # Run the agent with the user prompt and chat history
            async with agent.iter(
                request.query,
                deps=agent_deps,
                message_history=pydantic_messages,
            ) as run:
                async for node in run:
                    if Agent.is_model_request_node(node):
                        # Stream tokens from the model's request
                        async with node.stream(run.ctx) as request_stream:
                            async for event in request_stream:
                                if (
                                    isinstance(event, PartStartEvent)
                                    and event.part.part_kind == "text"
                                ):
                                    yield (
                                        json.dumps({"text": event.part.content}).encode(
                                            "utf-8"
                                        )
                                        + b"\n"
                                    )
                                    full_response += event.part.content
                                elif isinstance(event, PartDeltaEvent) and isinstance(
                                    event.delta, TextPartDelta
                                ):
                                    delta = event.delta.content_delta
                                    yield (
                                        json.dumps({"text": full_response}).encode("utf-8")
                                        + b"\n"
                                    )
                                    full_response += delta

            logger.info(
                "agent_execution_completed",
                session_id=session_id,
                response_length=len(full_response),
            )

            # After streaming is complete, store the full response in the database
            message_data = run.result.new_messages_json()

            # Store agent's response
            await store_message(
                supabase=supabase,
                session_id=session_id,
                message_type="ai",
                content=full_response,
                message_data=message_data,
                data={"request_id": request.request_id},
            )

            # Wait for title generation to complete if it's running
            if title_task:
                try:
                    conversation_title = await asyncio.wait_for(title_task, timeout=10.0)
                    await update_conversation_title(
                        supabase, session_id, conversation_title
                    )
                    logger.info(
                        "conversation_title_generated",
                        session_id=session_id,
                        title=conversation_title,
                    )
                except TimeoutError:
                    logger.warning(
                        "conversation_title_generation_timeout",
                        session_id=session_id,
                    )
                except Exception:
                    logger.exception(
                        "conversation_title_generation_failed",
                        session_id=session_id,
                    )

            # Wait for request tracking to complete
            try:
                await request_tracking_task
            except Exception:
                logger.exception("request_tracking_failed")

            # Send final chunk with complete flag
            final_data = {
                "text": full_response,
                "session_id": session_id,
                "complete": True,
            }

            yield json.dumps(final_data).encode("utf-8") + b"\n"

        return StreamingResponse(stream_response(), media_type="text/plain")

    except Exception as e:
        logger.exception(
            "agent_request_failed",
            user_id=request.user_id,
            session_id=request.session_id,
        )
        return StreamingResponse(
            stream_error_response(
                f"An error occurred: {str(e)}",
                request.session_id,
            ),
            media_type="text/plain",
        )
