"""AI Coach Agent definition.

Defines the Pydantic AI agent with system prompt and registered tools
for the YouTube coaching knowledge base.
"""

from pydantic_ai import Agent, RunContext

from src.agent.config import get_model
from src.agent.deps import AgentDeps
from src.tools.rag_tools.tool import (
    get_full_transcript_tool,
    search_coaching_content_tool,
)

# ==============================================================================
# System Prompt
# ==============================================================================

AGENT_SYSTEM_PROMPT = """You are a supportive AI coaching assistant with access to a comprehensive knowledge base of YouTube coaching transcripts.

Your role is to help users find relevant coaching insights, strategies, and advice by searching through and analyzing video transcripts.

## Core Personality Traits
- **Supportive**: Encourage users and validate their challenges
- **Direct**: Provide clear, actionable answers without excessive elaboration
- **Curious**: Ask clarifying questions when user intent is unclear
- **Practical**: Focus on actionable steps and real-world application

## Tool Usage Strategy

### Search First
Always start with `search_coaching_content` to:
- Find relevant coaching insights across videos
- Identify which videos address the user's question
- Get focused excerpts with timestamps

Use focused, specific queries. Good examples:
- "strategies for overcoming fear of failure"
- "morning routine for productivity"
- "communication skills for difficult conversations"

### Retrieve Full Transcript When Needed
Use `get_full_transcript` only when:
- User specifically asks for a full video transcript
- You need complete context after search identified a highly relevant video
- User wants detailed step-by-step instructions from a specific video
- Search results indicate a video contains comprehensive coverage

WARNING: Full transcripts are token-heavy (2000-8000 tokens). Use sparingly.

### Always Cite Sources
- Include video titles in your responses
- Provide clickable YouTube URLs with timestamps
- Reference specific moments: "At [02:35], the coach explains..."
- Help users jump directly to relevant video segments

## Response Format

For most queries, follow this structure:

1. **Direct Answer**: Start with a clear, concise answer to the user's question
2. **Supporting Evidence**: Share 2-3 relevant insights from the transcripts with citations
3. **Actionable Steps**: Provide 2-4 concrete next steps the user can take
4. **Optional Follow-up**: Ask a clarifying question if helpful

Keep responses focused and scannable. Use:
- Short paragraphs (2-3 sentences max)
- Bullet points for lists
- Bold for emphasis on key concepts
- Clear section breaks

## Example Interaction

User: "How can I build better habits?"

You:
Building lasting habits requires three key elements: clarity, consistency, and environment design.

Based on the coaching transcripts:

**Start with tiny habits** - At [03:45] in "Atomic Habits Framework", the coach emphasizes starting so small it feels too easy. Example: If you want to exercise daily, start with just 2 push-ups.
https://youtube.com/watch?v=example&t=225s

**Stack habits onto existing routines** - In "Morning Routine Secrets" [07:12], habit stacking is explained: attach your new habit to something you already do automatically.
https://youtube.com/watch?v=example2&t=432s

**Actionable Steps:**
1. Choose ONE habit to build (not multiple)
2. Make it ridiculously small (2 minutes or less)
3. Stack it onto an existing daily routine
4. Track it with a simple check mark each day

What specific habit are you trying to build? I can search for more targeted advice.

---

Remember:
- Search is your primary tool (fast, focused, efficient)
- Full transcripts are for deep dives only
- Always include clickable citations
- Focus on helping users take action
"""

# ==============================================================================
# Agent Definition
# ==============================================================================

agent = Agent(
    get_model(),
    system_prompt=AGENT_SYSTEM_PROMPT,
    deps_type=AgentDeps,
    retries=2,
)


# ==============================================================================
# Tool Registration
# ==============================================================================


@agent.tool
async def search_coaching_content(
    ctx: RunContext[AgentDeps],
    query: str,
    match_count: int = 5,
) -> str:
    """Search YouTube coaching transcripts for relevant content using semantic search.

    Use this when you need to:
    - Find coaching advice or insights related to a specific topic
    - Search for examples, strategies, or frameworks mentioned in videos
    - Locate relevant quotes or discussions from the coach
    - Explore what the coach has said about a particular subject
    - Get a broad overview of topics before diving into specific videos

    Do NOT use this for:
    - Reading the complete transcript of a specific video (use get_full_transcript instead)
    - Getting detailed step-by-step instructions from one video (use get_full_transcript)
    - When you already know the exact video ID and need the full context
    - Searching for very generic terms without context (be specific in your query)

    Args:
        ctx: RunContext containing agent dependencies (Supabase, embedding client).
        query: Search query describing the coaching topic or question.
            Be specific and focused. Good queries include context and intent.
            Examples: "goal setting strategies for entrepreneurs",
                     "overcoming fear of failure",
                     "morning routine for productivity"
        match_count: Number of results to return (1-10). Default: 5.
            - Small (1-3): Use when you need highly focused, top results only
            - Medium (4-6): Good balance for most queries (DEFAULT)
            - Large (7-10): Use when you want comprehensive coverage of a topic,
                           but be aware this increases token usage significantly

    Returns:
        Formatted string containing search results with:
        - Video title, URL, and duration
        - Timestamp with clickable YouTube link (opens at exact moment)
        - Relevant transcript excerpt
        - Similarity score showing relevance
        Returns empty state message if no results found.

    Performance Notes:
        - Typical execution time: 200-800ms depending on match_count
        - Token usage scales with match_count and content length:
          * 1-3 results: ~300-800 tokens
          * 4-6 results: ~800-1500 tokens
          * 7-10 results: ~1500-2500 tokens
        - Each result includes video metadata + transcript chunk (~150-400 tokens)
        - Embedding generation adds ~50-100ms overhead
        - Results are automatically ranked by semantic similarity

    Examples:
        # Find focused advice on a specific topic (3 results)
        search_coaching_content(
            query="how to overcome procrastination on important tasks",
            match_count=3
        )

        # Get comprehensive overview of a topic (default 5 results)
        search_coaching_content(
            query="building habits that stick long-term"
        )

        # Explore what coach has said about a broad area (8 results)
        search_coaching_content(
            query="communication skills for leaders",
            match_count=8
        )
    """
    return await search_coaching_content_tool(ctx, query, match_count)


@agent.tool
async def get_full_transcript(
    ctx: RunContext[AgentDeps],
    video_id: str,
) -> str:
    """Retrieve the complete transcript of a YouTube video.

    Use this when you need to:
    - Read the full transcript of a specific video to understand complete context
    - Get detailed step-by-step instructions that span an entire video
    - Analyze the structure and flow of a complete coaching session
    - Find specific details after search_coaching_content identified a relevant video
    - Provide comprehensive summaries of a video's content

    Do NOT use this for:
    - Searching for topics across videos (use search_coaching_content instead)
    - When you don't have a specific video_id (search first to find videos)
    - Quick lookups of specific concepts (search is more token-efficient)
    - Exploring what content exists (use search to discover first)

    Args:
        ctx: RunContext containing agent dependencies (Supabase client).
        video_id: YouTube video ID (11-character string).
            You can get this from:
            - search_coaching_content results (in the URL)
            - User-provided YouTube URLs (extract the ID)
            - Previous conversation context
            Example: For URL "youtube.com/watch?v=dQw4w9WgXcQ", use "dQw4w9WgXcQ"

    Returns:
        Formatted string containing:
        - Video title, URL, and duration
        - Number of transcript chunks
        - Complete transcript text (all chunks combined)
        - Truncation warning if transcript exceeds MAX_TRANSCRIPT_CHARS limit

        Returns error message if:
        - Video ID not found in knowledge base
        - Transcript is not available for the video

    Performance Notes:
        - Typical execution time: 150-500ms depending on transcript length
        - Token usage WARNING: Can be VERY HIGH (2000-8000+ tokens)
          * Short videos (5-10 min): ~2000-3000 tokens
          * Medium videos (15-30 min): ~4000-6000 tokens
          * Long videos (45+ min): ~8000+ tokens (may hit truncation limit)
        - MAX_TRANSCRIPT_CHARS default: 20,000 characters (~5000 tokens)
        - Transcripts exceeding limit are truncated with clear warning
        - Use sparingly - prefer search_coaching_content when possible
        - Good for: Deep dives into specific videos after search narrows focus

    Examples:
        # Get full transcript after finding relevant video via search
        # (User asked about video found in search results)
        get_full_transcript(video_id="dQw4w9WgXcQ")

        # Read complete video when user provides YouTube URL
        # Extract ID: youtube.com/watch?v=abc123 -> "abc123"
        get_full_transcript(video_id="abc123")

        # Deep dive into specific coaching session
        # (After search identified this as THE most relevant video)
        get_full_transcript(video_id="xYz789")
    """
    return await get_full_transcript_tool(ctx, video_id)
