# AI Coach Web UI Guide

## Overview

A clean, modern web interface for your AI Coach that provides:
- Real-time streaming chat responses
- Beautiful, responsive design
- No authentication required for testing
- Works with your RAG-powered coaching agent

## Quick Start

### 1. Start the Server

In your terminal (at project root):

```bash
uv run uvicorn src.main:app --host 127.0.0.1 --port 8030 --reload
```

### 2. Open the UI

Open your browser and go to:

```
http://127.0.0.1:8030
```

That's it! You should see the AI Coach chat interface.

## Features

### Chat Interface
- **Clean Design**: Modern, professional look with smooth animations
- **Real-time Streaming**: See the AI's response as it's being generated
- **Auto-scroll**: Messages automatically scroll into view
- **Responsive**: Works on desktop, tablet, and mobile

### Status Indicator
- **Green dot**: Connected to server and ready
- **Yellow dot**: Connecting or server issues
- **Red dot**: Cannot connect to server

### Input Features
- **Enter to send**: Press Enter to send your message
- **Shift+Enter**: Add a new line in your message
- **Auto-resize**: Input box grows as you type
- **Character count**: See how long your message is

## Example Questions to Ask

- "What advice do you have about building AI agents?"
- "How can I use AI coding assistants effectively?"
- "What are best practices for RAG systems?"
- "Tell me about Pydantic AI framework"
- "How do I implement streaming in agents?"

## API Endpoints

The UI uses these endpoints:

### `GET /`
Serves the HTML interface (what you see in the browser)

### `GET /health`
Check if the server is running and healthy

### `POST /api/chat`
Send a message and get streaming response (no auth required)

**Request body:**
```json
{
  "message": "Your question here",
  "conversation_id": null
}
```

**Response:**
Server-Sent Events (SSE) stream with JSON data:
- `{"type": "token", "content": "word"}` - Streaming text
- `{"type": "complete", "full_response": "..."}` - Done
- `{"type": "error", "error": "message"}` - Error occurred

### `GET /docs`
Swagger UI for full API documentation

## Project Structure

```
frontend/
├── index.html              # Main HTML file
├── static/
│   ├── css/
│   │   └── styles.css     # All styling
│   └── js/
│       └── app.js         # Frontend logic
```

## Customization

### Change Colors

Edit `frontend/static/css/styles.css` and modify the `:root` variables:

```css
:root {
    --primary-color: #2563eb;     /* Main blue color */
    --primary-hover: #1d4ed8;     /* Hover state */
    --background: #f9fafb;        /* Page background */
    --user-bg: #2563eb;           /* User message bubble */
    /* ... more variables ... */
}
```

### Change Welcome Message

Edit `frontend/index.html`, find the `.welcome-message` section:

```html
<div class="message assistant-message welcome-message">
    <div class="message-content">
        <p>Your custom welcome message here!</p>
    </div>
</div>
```

### Add Logo or Branding

Edit `frontend/index.html` in the `<header>` section:

```html
<header class="header">
    <img src="/static/logo.png" alt="Logo" style="height: 40px;">
    <h1>AI Coach</h1>
    <!-- ... -->
</header>
```

## Troubleshooting

### UI doesn't load
1. Check if server is running: `http://127.0.0.1:8030/health`
2. Make sure you're at the correct URL: `http://127.0.0.1:8030`
3. Check browser console for errors (F12)

### Status shows "Cannot connect"
- Server might not be running
- Check if port 8030 is available
- Try restarting the server

### No response from AI
1. Check server logs for errors
2. Verify vector database has data: see [FIX_EMBEDDING_STORAGE.md](FIX_EMBEDDING_STORAGE.md)
3. Check API keys in `.env` file
4. Test health endpoint: `curl http://127.0.0.1:8030/health`

### Streaming is slow
- This is normal for large responses
- The AI generates tokens one at a time
- Depends on your OpenAI API response time

## Advanced: Adding Authentication

The current UI uses the `/api/chat` endpoint which has no authentication for testing.

For production, you should use the `/api/pydantic-agent` endpoint which requires:
1. Supabase JWT token
2. User authentication
3. Rate limiting

See `src/api/main.py` for the authenticated endpoint implementation.

## Development

### File Editing Workflow

1. **Edit HTML**: `frontend/index.html`
2. **Edit CSS**: `frontend/static/css/styles.css`
3. **Edit JS**: `frontend/static/js/app.js`
4. **Reload browser** - Changes take effect immediately (no build step!)

### Browser DevTools

Press **F12** to open developer tools:
- **Console**: See logs and errors
- **Network**: Monitor API requests
- **Elements**: Inspect and modify HTML/CSS live

## Next Steps

1. **Test the UI**: Open it and ask questions!
2. **Customize**: Change colors, text, layout
3. **Add Features**: Conversation history, export chats, etc.
4. **Deploy**: Use a cloud platform (Railway, Render, etc.)

## Support

If you encounter issues:
1. Check server logs
2. Check browser console (F12)
3. Verify all dependencies are installed: `uv sync`
4. Check that vector database has data

Enjoy your AI Coach! 🚀
