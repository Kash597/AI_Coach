# Dynamous AI Coach

RAG-powered AI coaching assistant with YouTube transcript processing pipeline.

> NOTE: This code isn't fully human vetted yet since it was created as a part of [my livestream](https://youtube.com/live/ZHcXavLTA5s). I will be refining this heavily soon!

## Features

### YouTube RAG Pipeline
- **Automatic transcript processing**: Fetch, chunk, and index video transcripts
- **Token-aware chunking**: Intelligent transcript segmentation (400-1000 tokens)
- **Vector search**: Semantic search powered by Supabase + pgvector
- **Flexible embedding providers**: OpenAI, Ollama, or OpenRouter
- **Timestamp preservation**: Navigate directly to relevant video sections

### AI Coach Agent
- **Pydantic AI agent**: Supportive coaching assistant with RAG capabilities
- **Semantic search**: Find relevant coaching insights across video transcripts
- **Full transcript retrieval**: Get complete video transcripts with citations
- **FastAPI streaming**: Real-time streaming responses via Server-Sent Events
- **JWT authentication**: Secure access via Supabase Auth
- **Rate limiting**: 5 requests per minute (configurable)
- **Conversation management**: Auto-generated titles and message history

## Quick Start

**📖 See [SETUP_GUIDE.md](SETUP_GUIDE.md) for detailed step-by-step instructions.**

### 1. Install Dependencies

```bash
uv sync
```

### 2. Set Up Supabase Database

Run the migrations in Supabase SQL Editor:
```bash
# 1. RAG Pipeline tables (channels, videos, transcript_chunks)
# Copy contents of migrations/001_youtube_rag_schema.sql
# Paste into Supabase Dashboard > SQL Editor > Run

# 2. AI Agent tables (user_profiles, conversations, messages, requests)
# Copy contents of migrations/002_agent_tables.sql
# Paste into Supabase Dashboard > SQL Editor > Run
```

### 3. Configure Environment

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 4. Run Pipeline

```bash
# Process videos from last 7 days
uv run python -m src.rag_pipeline.cli

# Custom parameters
uv run python -m src.rag_pipeline.cli --channel-id UCxxxxx --days-back 14
```

### 5. Run AI Coach Agent (Optional)

```bash
# Start the FastAPI server (default port 8030)
uv run uvicorn src.main:app --host 127.0.0.1 --port 8030 --reload

# Custom port
uv run uvicorn src.main:app --host 127.0.0.1 --port 8080 --reload

# Or use python -m to run
uv run python -m src.main

# For containers/production (listen on all interfaces)
uv run uvicorn src.main:app --host 0.0.0.0 --port 8030
```

**Endpoints:**
- `GET /health` - Health check
- `POST /api/pydantic-agent` - Streaming agent endpoint (requires JWT auth)

## Project Structure

```
src/
├── agent/                   # AI Coach Agent core
│   ├── config.py           # Model & environment config
│   ├── deps.py             # Runtime dependencies
│   └── agent.py            # Agent definition with system prompt
├── tools/                   # Agent tools
│   └── rag_tools/          # RAG search and retrieval
│       ├── service.py      # Tool implementation + helpers
│       └── tool.py         # Agent tool decorators
├── api/                     # FastAPI application
│   ├── main.py             # Streaming endpoint, auth, rate limiting
│   └── db_utils.py         # Conversation & message management
├── rag_pipeline/            # YouTube transcript pipeline
│   ├── config.py           # Configuration management
│   ├── schemas.py          # Pydantic data models
│   ├── youtube_service.py  # Supadata API client
│   ├── chunking_service.py # Token-aware chunking
│   ├── embedding_service.py # Embedding generation
│   ├── storage_service.py  # Supabase vector storage
│   ├── pipeline.py         # Main orchestration
│   └── cli.py              # Command-line interface
└── utils/                   # Shared utilities
    ├── logging.py          # Structured logging
    └── clients.py          # Client initialization

tests/
├── agent/                   # Agent config tests
├── tools/rag_tools/        # RAG tools unit tests
├── api/                     # API endpoint tests
├── rag_pipeline/           # Pipeline unit tests
└── integration/            # Integration tests
```

## Development

### Lint and Type Check

```bash
# Run linter
uv run ruff check src/

# Auto-fix
uv run ruff check --fix src/

# Type check
uv run mypy src/
```

### Run Tests

```bash
# All tests
uv run pytest tests/ -v

# Unit tests only
uv run pytest tests/ -v -m unit

# Integration tests
uv run pytest tests/ -m integration
```

## Architecture

This project follows the **vertical slice architecture** with strict type safety:

- Each feature is a self-contained slice
- 100% type annotations (strict mypy)
- Google-style docstrings
- Structured logging for AI debugging

## License

MIT
