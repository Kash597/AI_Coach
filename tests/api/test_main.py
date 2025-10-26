"""Unit tests for FastAPI application main endpoints.

Basic tests for health check and authentication requirements.
Note: Full integration testing of the agent endpoint requires more complex setup.
"""

from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.testclient import TestClient

from src.api.main import app, verify_token


@pytest.mark.unit
class TestHealthEndpoint:
    """Test /health endpoint."""

    def test_health_check_returns_healthy_status(self) -> None:
        """Test that health endpoint returns healthy status."""
        # Use FastAPI TestClient (synchronous)
        client = TestClient(app)
        response = client.get("/health")

        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "services" in data

    def test_health_check_includes_timestamp(self) -> None:
        """Test that health endpoint includes valid timestamp."""
        client = TestClient(app)
        response = client.get("/health")

        data = response.json()
        timestamp_str = data["timestamp"]

        # Verify timestamp is in ISO format
        timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        assert isinstance(timestamp, datetime)

    def test_health_check_includes_services_status(self) -> None:
        """Test that health endpoint includes services status."""
        client = TestClient(app)
        response = client.get("/health")

        data = response.json()
        services = data["services"]

        # Check expected service keys exist
        assert "embedding_client" in services
        assert "supabase" in services
        assert "http_client" in services
        assert "title_agent" in services

        # All values should be boolean
        for service_name, status in services.items():
            assert isinstance(status, bool), f"{service_name} status should be boolean"


@pytest.mark.unit
class TestVerifyToken:
    """Test authentication token verification."""

    @pytest.mark.asyncio
    async def test_verify_token_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test successful token verification."""
        # Mock environment variables
        monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
        monkeypatch.setenv("SUPABASE_SERVICE_KEY", "test-service-key")

        # Mock HTTP client
        mock_http_client = AsyncMock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json = Mock(return_value={"id": "user123", "email": "test@example.com"})
        mock_http_client.get = AsyncMock(return_value=mock_response)

        # Create mock credentials
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="valid-token")

        # Patch the global http_client
        with patch("src.api.main.http_client", mock_http_client):
            result = await verify_token(credentials)

        # Assertions
        assert result["id"] == "user123"
        assert result["email"] == "test@example.com"

    @pytest.mark.asyncio
    async def test_verify_token_invalid_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test token verification with invalid token."""
        # Mock environment variables
        monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
        monkeypatch.setenv("SUPABASE_SERVICE_KEY", "test-service-key")

        # Mock HTTP client with 401 response
        mock_http_client = AsyncMock()
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.text = "Invalid token"
        mock_http_client.get = AsyncMock(return_value=mock_response)

        # Create mock credentials
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="invalid-token")

        # Patch the global http_client
        with patch("src.api.main.http_client", mock_http_client):
            with pytest.raises(HTTPException) as exc_info:
                await verify_token(credentials)

        # Assertions
        assert exc_info.value.status_code == 401
        assert "Invalid authentication token" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_verify_token_http_client_not_initialized(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test token verification when HTTP client is not initialized."""
        # Create mock credentials
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="test-token")

        # Patch the global http_client to None
        with patch("src.api.main.http_client", None):
            with pytest.raises(HTTPException) as exc_info:
                await verify_token(credentials)

        # Assertions
        assert exc_info.value.status_code == 500
        assert "HTTP client not initialized" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_verify_token_exception_handling(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test token verification exception handling."""
        # Mock environment variables
        monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
        monkeypatch.setenv("SUPABASE_SERVICE_KEY", "test-service-key")

        # Mock HTTP client that raises exception
        mock_http_client = AsyncMock()
        mock_http_client.get = AsyncMock(side_effect=Exception("Network error"))

        # Create mock credentials
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="test-token")

        # Patch the global http_client
        with patch("src.api.main.http_client", mock_http_client):
            with pytest.raises(HTTPException) as exc_info:
                await verify_token(credentials)

        # Assertions
        assert exc_info.value.status_code == 401
        assert "Authentication error" in str(exc_info.value.detail)


@pytest.mark.unit
class TestAgentEndpointBasics:
    """Basic tests for /api/pydantic-agent endpoint."""

    def test_agent_endpoint_requires_authentication(self) -> None:
        """Test that agent endpoint requires authentication token."""
        client = TestClient(app)
        response = client.post(
            "/api/pydantic-agent",
            json={
                "query": "test query",
                "user_id": "user123",
                "request_id": "req123",
                "session_id": "session123",
            },
        )

        # Should return 403 Forbidden (no token provided)
        assert response.status_code == 403

    def test_agent_endpoint_rejects_invalid_token(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that agent endpoint rejects invalid authentication token."""
        # Mock environment variables
        monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
        monkeypatch.setenv("SUPABASE_SERVICE_KEY", "test-service-key")

        # Mock HTTP client with 401 response for token verification
        mock_http_client = AsyncMock()
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.text = "Invalid token"
        mock_http_client.get = AsyncMock(return_value=mock_response)

        with patch("src.api.main.http_client", mock_http_client):
            client = TestClient(app)
            response = client.post(
                "/api/pydantic-agent",
                json={
                    "query": "test query",
                    "user_id": "user123",
                    "request_id": "req123",
                    "session_id": "session123",
                },
                headers={"Authorization": "Bearer invalid-token"},
            )

        # Should return 401 Unauthorized
        assert response.status_code == 401
