# Implementation Plan: YouTube AI Coach Agent

## Overview

Build a Pydantic AI agent that queries YouTube transcript knowledge base to provide coaching advice. The agent uses RAG to search transcript chunks and retrieve full transcripts with automatic URL/timestamp citations.

**Core Features:**
- 2 RAG tools: search_coaching_content, get_full_transcript
- Citation formatting as helper functions (not LLM tools)
- FastAPI streaming endpoint compatible with existing frontend
- Conversation history management
- No Mem0, no web search (focused MVP)

## Requirements Summary

### Functional
- Semantic search across transcript chunks using pgvector
- Full transcript retrieval with size limits (MAX_TRANSCRIPT_CHARS)
- Automatic citations: video URLs with timestamps embedded in search results
- Conversation management: store/retrieve history, auto-generate titles
- Streaming responses via FastAPI SSE
- Rate limiting (5 req/min default)
- JWT authentication via Supabase

### Technical
- 100% type annotations (strict mypy)
- Structured JSON logging (existing src/utils/logging.py)
- Google-style docstrings for code, agent-optimized for tools
- Unit + integration tests
- LLM flexibility: OpenAI/Ollama/OpenRouter via env config
- API structure matches PRPs/examples/backend_agent_api/agent_api.py

### Excluded
- ❌ Mem0, web search, image analysis, SQL/code execution

## Research Findings

### Pydantic AI Patterns
- Use `@dataclass` for AgentDeps to pass runtime dependencies
- Tools access deps via `RunContext[AgentDeps]`
- Separate tool registration (`@agent.tool`) from implementation (service layer)
- Combine static + dynamic system prompts
- Stream via `agent.iter()` → `node.stream()` for token-by-token delivery
- Store message history in DB, convert via `ModelMessagesTypeAdapter`

### Reference Implementations
- **agent.py**: Model config via `get_model()`, AgentDeps dataclass, tool registration
- **tools.py**: Thin decorators delegate to service layer, helper functions not exposed
- **agent_api.py**: Lifespan for clients, JWT auth, streaming with parallel tasks
- **db_utils.py**: Conversation CRUD, message storage, history conversion, rate limiting

### Existing Codebase Patterns
- **Service layer**: Class-based with `__init__(config)`, async methods, structured logging
- **Configuration**: Pydantic models, `os.getenv()` with defaults
- **Schemas**: Pydantic models with full type annotations
- **Logging**: `logger.info("event_name", key=value, ...)` structured JSON

## Implementation Tasks

### Phase 1: Database Foundation (Day 1)

**Task 1.1: Create Agent Tables Migration**
- Create `migrations/002_agent_tables.sql`
- Copy from `PRPs/examples/rag_agent_tables.sql`:
  - Tables: user_profiles, conversations, messages, requests
  - Indexes: idx_conversations_user, idx_messages_session, idx_messages_computed_session
  - Functions: handle_new_user(), is_admin()
  - Triggers: on_auth_user_created
  - RLS policies for all tables
- Exclude: document_metadata, document_rows, documents (we use videos/transcript_chunks)
- **Estimated effort**: 1 hour

**Task 1.2: Update Environment Configuration**
- Update `.env.example` with:
  - LLM_PROVIDER, LLM_BASE_URL, LLM_API_KEY, LLM_CHOICE
  - MAX_TRANSCRIPT_CHARS (default 20000)
  - API_PORT (default 8030)
  - RATE_LIMIT_REQUESTS (default 5)
  - ENVIRONMENT (development/production)
- **Estimated effort**: 15 minutes

---

### Phase 2: Agent Core Infrastructure (Days 2-3)

**Task 2.1: Create Agent Configuration Module**
- `src/agent/__init__.py`, `src/agent/config.py`
- Functions:
  - `get_model()` - Returns OpenAIModel with env config
  - `get_max_transcript_chars()` - Returns int from env
  - `get_rate_limit()` - Returns int from env
- **Estimated effort**: 30 minutes

**Task 2.2: Create Agent Dependencies**
- `src/agent/deps.py`
- Define AgentDeps dataclass:
  - supabase: Client
  - embedding_client: AsyncOpenAI
  - http_client: AsyncClient
- **Estimated effort**: 15 minutes

**Task 2.3: Create Utility Clients Module**
- `src/utils/clients.py`
- Function: `get_agent_clients()` returns (AsyncOpenAI, Client)
- Initialize embedding client and Supabase from env vars
- **Estimated effort**: 20 minutes

**Task 2.4: Copy Database Utilities**
- Copy `PRPs/examples/backend_agent_api/db_utils.py` → `src/api/db_utils.py`
- No modifications needed
- Functions: fetch_conversation_history, create_conversation, update_conversation_title,
  generate_session_id, generate_conversation_title, store_message,
  convert_history_to_pydantic_format, check_rate_limit, store_request
- **Estimated effort**: 5 minutes

---

### Phase 3: RAG Tools Implementation (Days 3-4)

**Task 3.1: Create RAG Tools Service Layer**
- `src/tools/__init__.py`, `src/tools/rag_tools/__init__.py`, `src/tools/rag_tools/service.py`
- Helper functions (deterministic, not LLM tools):
  - `format_video_url(video_id, timestamp_ms)` - Returns YouTube URL with &t=XXs
  - `format_timestamp_display(ms)` - Returns [MM:SS] or [HH:MM:SS]
  - `format_duration(seconds)` - Returns HH:MM:SS or MM:SS
  - `get_embedding(text, client)` - Returns embedding vector
- Tool implementation functions:
  - `search_transcript_chunks(supabase, embedding_client, query, match_count)`
    - Generate embedding → call match_transcript_chunks RPC → format with citations
  - `get_full_video_transcript(supabase, video_id, max_chars)`
    - Query chunks → combine in order → format with metadata → truncate if needed
- **Estimated effort**: 3 hours

**Task 3.2: Create RAG Tools Decorators**
- `src/tools/rag_tools/tool.py`
- Functions:
  - `search_coaching_content_tool(ctx, query, match_count=5)` - Agent-optimized docstring
  - `get_full_transcript_tool(ctx, video_id)` - Agent-optimized docstring
- Docstrings must include: "Use this when", "Do NOT use this for", Args, Returns, Performance Notes, Examples
- Reference `PRPs/ai_docs/tool_guide.md` for docstring format
- **Estimated effort**: 2 hours

**Task 3.3: Create RAG Tools Unit Tests**
- `tests/tools/__init__.py`, `tests/tools/rag_tools/__init__.py`, `tests/tools/rag_tools/test_service.py`
- Test classes:
  - TestHelperFunctions: format_video_url, format_timestamp_display, format_duration
  - TestSearchTranscriptChunks: successful_search, no_results
  - TestGetFullVideoTranscript: successful_retrieval, video_not_found, transcript_truncation
- Mock Supabase and embedding client
- **Estimated effort**: 2 hours

---

### Phase 4: Agent Definition (Day 5)

**Task 4.1: Create Agent with System Prompt**
- `src/agent/agent.py`
- System prompt personality: Supportive coaching assistant
- Tool usage strategy: Search first, retrieve full transcript if needed, always cite sources
- Response format: Direct answer → actionable steps with citations → optional follow-up question
- Register tools: `agent.tool(search_coaching_content_tool)`, `agent.tool(get_full_transcript_tool)`
- **Estimated effort**: 1.5 hours

---

### Phase 5: FastAPI Application (Days 6-7)

**Task 5.1: Create API Main Module**
- `src/api/__init__.py`, `src/api/main.py`
- Based on `PRPs/examples/backend_agent_api/agent_api.py` with modifications:
  - Remove Mem0 client initialization (lifespan)
  - Remove Mem0 memory retrieval/storage (in /api/pydantic-agent endpoint)
  - Remove Mem0 from AgentDeps
  - Keep: Authentication (verify_token), rate limiting, conversation management,
    title generation, streaming response, error handling
- Endpoints:
  - `POST /api/pydantic-agent` - Main agent endpoint with streaming
  - `GET /health` - Health check
- **Estimated effort**: 4 hours

**Task 5.2: Create API Integration Tests**
- `tests/api/__init__.py`, `tests/api/test_main.py`
- Test health endpoint, authentication requirement
- Mock full request flow (complex, mostly manual testing)
- **Estimated effort**: 1.5 hours

---

### Phase 6: End-to-End Testing (Day 8)

**Task 6.1: Manual E2E Testing**
- **Checklist:**
  - Database: Run migration, verify tables, create test user
  - Environment: Verify all env vars set
  - Server: Start API, test health endpoint
  - Authentication: Test with invalid/valid JWT tokens
  - Conversations: Create new, continue existing, verify title generation
  - RAG Tools: Search content, verify citations format, retrieve full transcript, test no results
  - Rate Limiting: Send 6 requests in 1 min, verify 6th blocked
  - Streaming: Verify incremental tokens, check final chunk has "complete": true
  - Errors: Invalid query, invalid video_id, simulate DB/LLM failures
- **Estimated effort**: 3 hours

**Task 6.2: Create Integration Test Suite**
- `tests/integration/__init__.py`, `tests/integration/test_agent_flow.py`
- Mark with `@pytest.mark.integration`
- Tests: new_conversation_flow, rag_search_flow (requires test database)
- **Estimated effort**: 2 hours

---

### Phase 7: Documentation (Day 9)

**Task 7.1: Update Project Documentation**
- Update `README.md`:
  - Add "AI Coach Agent" section to Features
  - Add "Run the Agent API" to Quick Start
  - Add agent structure to Architecture
- Create `docs/AGENT_GUIDE.md`:
  - Tools overview (when to use each)
  - System prompt strategy
  - Configuration (env vars)
  - API reference
  - Troubleshooting
- **Estimated effort**: 2 hours

**Task 7.2: Create Deployment Guide**
- Create `docs/DEPLOYMENT.md`:
  - Prerequisites
  - Local development setup
  - Production deployment (env vars, Docker optional)
  - Platform-specific guides (Railway, Render, Fly.io)
  - Health checks and monitoring
- **Estimated effort**: 1 hour

---

## Technical Architecture

### Component Structure
```
src/
├── agent/              # Agent configuration, deps, agent definition
│   ├── config.py      # get_model(), get_max_transcript_chars(), get_rate_limit()
│   ├── deps.py        # AgentDeps dataclass
│   └── agent.py       # Agent with system prompt + tools
├── tools/
│   └── rag_tools/     # RAG tools (search, full transcript)
│       ├── service.py # Implementation + helpers
│       └── tool.py    # @agent.tool decorators
├── api/               # FastAPI application
│   ├── main.py        # Streaming endpoint, auth, rate limiting
│   └── db_utils.py    # Conversation/message management
└── utils/
    ├── logging.py     # (existing) Structured logging
    └── clients.py     # Client initialization
```

### Data Flow
1. User message → POST /api/pydantic-agent
2. JWT auth → rate limit check → store user message
3. Fetch conversation history → convert to Pydantic AI format
4. Pass to agent.iter() with deps (supabase, embedding_client, http_client)
5. Agent calls tools:
   - search_coaching_content: embedding → match_transcript_chunks RPC → format with citations
   - get_full_transcript: query chunks → combine → format with metadata
6. Stream response tokens incrementally
7. Store AI response in messages table
8. Generate title if new conversation (parallel task)

### Database Schema
**Existing (RAG Pipeline):**
- videos, channels, transcript_chunks (with pgvector)

**New (Agent):**
- user_profiles (auth integration)
- conversations (session_id, user_id, title, timestamps)
- messages (session_id, message JSONB, message_data TEXT)
- requests (rate limiting)

### API Endpoints

**POST /api/pydantic-agent**
- Request: { query, user_id, request_id, session_id, files? }
- Response: Streaming JSON chunks
- Auth: Bearer JWT token (Supabase)

**GET /health**
- Response: { status, timestamp, services: {...} }

---

## Integration Points

### Files to Create (19 files)
1. migrations/002_agent_tables.sql
2. src/agent/__init__.py, config.py, deps.py, agent.py
3. src/tools/__init__.py, rag_tools/__init__.py, rag_tools/service.py, rag_tools/tool.py
4. src/api/__init__.py, main.py, db_utils.py
5. src/utils/clients.py
6. tests/tools/rag_tools/test_service.py
7. tests/api/test_main.py
8. tests/integration/test_agent_flow.py
9. docs/AGENT_GUIDE.md, docs/DEPLOYMENT.md

### Files to Modify (2 files)
1. .env.example - Add LLM/agent config
2. README.md - Add agent features

### Patterns to Follow
- Service layer: Class-based with config, async methods, structured logging
- Configuration: Pydantic models with env defaults
- Schemas: Full type annotations, Google-style docstrings
- Tools: Thin decorators → service layer implementation
- API: Lifespan for clients, dependency injection for auth

---

## Dependencies

**All dependencies already in pyproject.toml:**
- pydantic, pydantic-ai, fastapi, uvicorn
- openai, supabase, httpx
- python-dotenv, structlog

**No new dependencies required.**

---

## Testing Strategy

### Unit Tests
- RAG tools service: helpers, search, full transcript (mock Supabase)
- Target: 80%+ coverage of service layer

### Integration Tests
- Agent flow: new conversation, history loading, RAG with real DB
- Marked with `@pytest.mark.integration`

### Manual Testing
- 9-point checklist covering database, auth, conversations, RAG tools, rate limiting, streaming, errors

---

## Success Criteria

- [ ] Migration runs successfully, all tables created
- [ ] Agent initializes with correct LLM config
- [ ] Search returns results with formatted citations (title, timestamp, URL)
- [ ] Full transcript retrieval respects character limit
- [ ] API server starts on configured port
- [ ] Authentication validates JWT tokens
- [ ] Rate limiting blocks after limit exceeded
- [ ] Conversations created with auto-generated titles
- [ ] Message history loaded and passed to agent
- [ ] Streaming delivers tokens incrementally
- [ ] Unit tests pass (80%+ coverage)
- [ ] Manual E2E checklist completed
- [ ] Documentation updated (README, guides)
- [ ] API compatible with existing frontend

---

## Key Design Decisions

1. **2 Tools, Not 3**: Citation formatting is deterministic (helper function), not an LLM tool
2. **No Mem0**: Removed from example architecture as requested
3. **New Migration File**: 002_agent_tables.sql preserves history vs modifying existing
4. **Citations in Search Results**: URLs with timestamps embedded automatically (better UX, fewer tokens)
5. **Exact API Structure**: Match agent_api.py for frontend compatibility

## Potential Challenges

1. **Vector dimension mismatch**: Migration defaults to 1536 (text-embedding-3-small). If switching to Ollama (768), update migration.
2. **Transcript truncation**: Long videos truncated at MAX_TRANSCRIPT_CHARS. Clear message needed.
3. **Rate limiting**: Default 5 req/min may be restrictive for testing. Configurable via env.
4. **LLM tool support**: Not all Ollama models support tools. Document compatible models.

## Future Enhancements (Out of Scope)

- Web search tool (Brave/SearXNG) for hybrid responses
- Image analysis tool for coaching diagrams
- Multi-channel support with channel filtering
- Conversation analytics (popular topics, videos)
- Advanced citations (quote extraction, video highlights)

---

**Total Estimated Time: 8-9 days**

*This plan is ready for execution with `/execute-plan PRPs/requests/youtube-ai-coach-agent.md`*
