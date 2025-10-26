## Logging Rules

**Philosophy:** Logs are optimized for AI agent consumption. Include enough context for an LLM to understand and fix issues without human intervention.

### Required (MUST)

1. **Import shared logger:** `from src.shared.logging import get_logger; logger = get_logger(__name__)`

2. **Use appropriate levels:** `debug` (diagnostics), `info` (operations), `warning` (recoverable), `error` (non-fatal), `exception` (in except blocks with stack traces)

3. **Use structured logging:** Always use keyword arguments, never string formatting
   ```python
   logger.info("user_created", user_id="123", role="admin")  # ✅
   logger.info(f"User {user_id} created")  # ❌ NO
   ```

4. **Descriptive event names:** Use `snake_case` that answers "what happened?"
   - Good: `database_connection_established`, `tool_execution_started`, `api_request_completed`
   - Bad: `connected`, `done`, `success`

5. **Use logger.exception() in except blocks:** Captures full stack trace automatically
   ```python
   try:
       result = await operation()
   except ValueError:
       logger.exception("operation_failed", expected="int", received=type(value).__name__)
       raise
   ```

6. **Include debugging context:** IDs (user_id, request_id, session_id), input values, expected vs actual, external responses, performance metrics (duration_ms)

### Recommended (SHOULD)

- Log entry/exit for complex operations with relevant metadata
- Log performance metrics for bottlenecks (timing, counts)
- Log state transitions (old_state, new_state)
- Log external system interactions (API calls, database queries, tool executions)
- Use `bind_context()` for request-scoped data (middleware handles this automatically)

### DO NOT

- **DO NOT log sensitive data:** No passwords, API keys, tokens (mask: `api_key[:8] + "..."`)
- **DO NOT use string formatting:** Always use structured kwargs
- **DO NOT spam logs in loops:** Log batch summaries instead
- **DO NOT silently catch exceptions:** Always log with `logger.exception()` or re-raise
- **DO NOT use vague event names:** Be specific about what happened

### Common Patterns

**Tool execution:**
```python
logger.info("tool_execution_started", tool=name, params=params)
try:
    result = await tool.execute(params)
    logger.info("tool_execution_completed", tool=name, duration_ms=duration)
except ToolError:
    logger.exception("tool_execution_failed", tool=name, retry_count=count)
    raise
```

**External API calls:**
```python
logger.info("api_call", provider="openai", endpoint="/v1/chat", status=200,
            duration_ms=1245.5, tokens={"prompt": 245, "completion": 128})
```

### Debugging

Logs include: `correlation_id` (links request logs), `source` (file:function:line), `duration_ms` (performance), `exc_type/exc_message` (errors). Use `grep "correlation_id=abc-123"` to trace requests.