"""Unit tests for agent configuration utilities.

Tests functions for loading LLM configuration and agent settings from
environment variables.
"""

import pytest
from pydantic_ai.models.openai import OpenAIModel

from src.agent.config import get_max_transcript_chars, get_model, get_rate_limit


@pytest.mark.unit
class TestGetModel:
    """Test get_model configuration function."""

    def test_get_model_returns_openai_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that get_model returns an OpenAIModel instance."""
        # Set environment variables
        monkeypatch.setenv("LLM_CHOICE", "gpt-4o-mini")
        monkeypatch.setenv("LLM_BASE_URL", "https://api.openai.com/v1")
        monkeypatch.setenv("LLM_API_KEY", "test-key")

        result = get_model()

        assert isinstance(result, OpenAIModel)

    def test_get_model_uses_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that get_model uses default values when env vars not set."""
        # Remove environment variables
        monkeypatch.delenv("LLM_CHOICE", raising=False)
        monkeypatch.delenv("LLM_BASE_URL", raising=False)
        monkeypatch.delenv("LLM_API_KEY", raising=False)

        result = get_model()

        # Should return an OpenAIModel with defaults
        assert isinstance(result, OpenAIModel)

    def test_get_model_with_custom_llm_choice(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that get_model respects custom LLM_CHOICE."""
        monkeypatch.setenv("LLM_CHOICE", "gpt-4-turbo")
        monkeypatch.setenv("LLM_BASE_URL", "https://api.openai.com/v1")
        monkeypatch.setenv("LLM_API_KEY", "test-key")

        result = get_model()

        assert isinstance(result, OpenAIModel)

    def test_get_model_with_custom_base_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that get_model respects custom LLM_BASE_URL."""
        monkeypatch.setenv("LLM_CHOICE", "gpt-4o-mini")
        monkeypatch.setenv("LLM_BASE_URL", "https://custom.api.com/v1")
        monkeypatch.setenv("LLM_API_KEY", "test-key")

        result = get_model()

        assert isinstance(result, OpenAIModel)


@pytest.mark.unit
class TestGetMaxTranscriptChars:
    """Test get_max_transcript_chars configuration function."""

    def test_get_max_transcript_chars_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that get_max_transcript_chars returns default value."""
        # Remove environment variable
        monkeypatch.delenv("MAX_TRANSCRIPT_CHARS", raising=False)

        result = get_max_transcript_chars()

        assert result == 20000
        assert isinstance(result, int)

    def test_get_max_transcript_chars_custom(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that get_max_transcript_chars respects custom value."""
        monkeypatch.setenv("MAX_TRANSCRIPT_CHARS", "15000")

        result = get_max_transcript_chars()

        assert result == 15000
        assert isinstance(result, int)

    def test_get_max_transcript_chars_returns_int(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that get_max_transcript_chars always returns an integer."""
        monkeypatch.setenv("MAX_TRANSCRIPT_CHARS", "5000")

        result = get_max_transcript_chars()

        assert isinstance(result, int)
        assert result == 5000


@pytest.mark.unit
class TestGetRateLimit:
    """Test get_rate_limit configuration function."""

    def test_get_rate_limit_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that get_rate_limit returns default value."""
        # Remove environment variable
        monkeypatch.delenv("RATE_LIMIT_REQUESTS", raising=False)

        result = get_rate_limit()

        assert result == 5
        assert isinstance(result, int)

    def test_get_rate_limit_custom(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that get_rate_limit respects custom value."""
        monkeypatch.setenv("RATE_LIMIT_REQUESTS", "10")

        result = get_rate_limit()

        assert result == 10
        assert isinstance(result, int)

    def test_get_rate_limit_returns_int(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that get_rate_limit always returns an integer."""
        monkeypatch.setenv("RATE_LIMIT_REQUESTS", "3")

        result = get_rate_limit()

        assert isinstance(result, int)
        assert result == 3
