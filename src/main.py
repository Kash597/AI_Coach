"""Main FastAPI application entry point for the AI Coach Agent.

This is the main application file that starts the FastAPI server.
It imports from src.api.main to keep the structure organized.
"""

from src.api.main import app

# This file is kept minimal for clean deployment
# All application logic is in src.api.main.py

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.main:app", host="127.0.0.1", port=8030, reload=True)
