"""Shared logging utilities for structured logging across the application.

This module provides a centralized logging configuration using structlog for
structured, JSON-formatted logs that are optimized for AI agent consumption
and easy debugging.
"""

import logging
import sys

import structlog


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a configured structured logger instance.

    This function returns a structlog logger configured with JSON output,
    timestamps, log levels, and exception formatting. All logs are written
    in structured format for easy parsing and analysis.

    Args:
        name: Logger name (typically __name__ from the calling module).
            This helps identify the source of log messages.

    Returns:
        Configured structlog logger instance ready for use.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("user_created", user_id="123", role="admin")
        >>> logger.exception("operation_failed", operation="fetch_data")
    """
    # Configure structlog if not already configured
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard library logging to use INFO level
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.INFO,
    )

    return structlog.get_logger(name)
