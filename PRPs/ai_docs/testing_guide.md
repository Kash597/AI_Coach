## Testing

**Tests mirror the source directory structure.** Every file in `src/` MUST have a corresponding test file.

**Structure:**
```
src/shared/logging.py  →  tests/shared/test_logging.py
src/shared/config.py   →  tests/shared/test_config.py
src/agent/agent.py     →  tests/agent/test_agent.py
```

**Requirements:**
- **Unit tests:** Test individual components in isolation. Mark with `@pytest.mark.unit`
- **Integration tests:** Test multiple components together. Mark with `@pytest.mark.integration`
- Place integration tests in `tests/integration/` when testing full application stack

**Run tests:** `uv run pytest tests/ -v`

**Run specific types:** `uv run pytest tests/ -m unit` or `uv run pytest tests/ -m integration`