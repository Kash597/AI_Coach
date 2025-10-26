# YouTube RAG Pipeline Setup Guide

This guide will walk you through setting up the YouTube RAG Pipeline from scratch.

## Prerequisites

- Python 3.11 or higher
- UV package manager installed
- Supabase account (free tier works)
- Supadata API key ([get one here](https://supadata.com))
- OpenAI API key (or Ollama for local embeddings)

## Step 1: Install Dependencies

```bash
# Install all dependencies using UV
uv sync
```

This will install all required packages including:
- supadata (YouTube transcript API)
- transformers (tokenizers)
- supabase (vector database client)
- openai (embeddings)
- and all other dependencies

## Step 2: Set Up Supabase Database

### 2.1 Create a Supabase Project

1. Go to [supabase.com](https://supabase.com)
2. Click "New Project"
3. Choose a name, password, and region
4. Wait for the project to be created

### 2.2 Run the Database Migration

1. In your Supabase dashboard, go to **SQL Editor**
2. Click "New Query"
3. Copy the contents of `migrations/001_youtube_rag_schema.sql`
4. Paste into the SQL Editor
5. **IMPORTANT:** Check the vector dimension setting on line 47:
   - If using `text-embedding-3-small` (default): keep `vector(1536)`
   - If using `all-MiniLM-L6-v2`: change to `vector(384)`
   - If using `nomic-embed-text`: change to `vector(768)`
6. Also update line 75 to match the same dimension
7. Click "Run" to execute the migration

### 2.3 Get Your Supabase Credentials

1. Go to **Project Settings** > **API**
2. Copy the following:
   - **Project URL** (e.g., `https://abc123.supabase.co`)
   - **Service Role Key** (anon key won't work - you need service_role)

## Step 3: Configure Environment Variables

### 3.1 Create .env File

```bash
# Copy the example file
cp .env.example .env
```

### 3.2 Fill in Your Credentials

Edit `.env` and add your credentials:

```bash
# Supadata API (YouTube transcript fetching)
SUPADATA_API_KEY=your_supadata_key_here
YOUTUBE_CHANNEL_ID=UCxxxxxxxxxxxxx  # The channel you want to process

# Processing Settings
YOUTUBE_DAYS_BACK=7
YOUTUBE_MAX_RETRIES=1
YOUTUBE_BATCH_SIZE=5

# Chunking Settings
MIN_CHUNK_TOKENS=400
MAX_CHUNK_TOKENS=1000

# Embedding Provider Configuration
EMBEDDING_PROVIDER=openai  # or "ollama" for local
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_API_KEY=sk-your-openai-api-key-here
EMBEDDING_MODEL_CHOICE=text-embedding-3-small

# Supabase Configuration
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your-service-role-key-here
```

### 3.3 Find Your YouTube Channel ID

If you don't know your channel ID:

1. Go to the YouTube channel
2. Look at the URL:
   - If it's `youtube.com/channel/UCxxxxx` → the ID is `UCxxxxx`
   - If it's `youtube.com/@username` → use `@username`
   - If it's `youtube.com/c/customname` → use the custom name

## Step 4: Verify Installation

Run a quick test to ensure everything is configured correctly:

```bash
# This should show the help message
uv run python -m src.rag_pipeline.cli --help
```

You should see output like:

```
usage: cli.py [-h] [--channel-id CHANNEL_ID] [--days-back DAYS_BACK] [--dry-run]

YouTube RAG Pipeline - Extract and index video transcripts
...
```

## Step 5: Run the Pipeline

### Option 1: Use Default Settings (from .env)

```bash
uv run python -m src.rag_pipeline.cli
```

### Option 2: Override Settings

```bash
# Process a specific channel
uv run python -m src.rag_pipeline.cli --channel-id UC_x5XG1OV2P6uZZ5FSM9Ttw

# Look back 30 days instead of 7
uv run python -m src.rag_pipeline.cli --days-back 30

# Dry run (test without writing to database)
uv run python -m src.rag_pipeline.cli --dry-run
```

### Expected Output

```
============================================================
YouTube RAG Pipeline
============================================================
Channel ID: UCxxxxxxxxxxxxx
Days back: 7
Embedding provider: openai
Embedding model: text-embedding-3-small
Chunk size: 400-1000 tokens
============================================================

[Processing videos...]

============================================================
Pipeline Results
============================================================
Total videos found: 15
Successfully processed: 12
Failed: 2
Skipped (already processed): 1
Total chunks created: 347

Errors encountered:
  ❌ video123: Transcript unavailable
  ❌ video456: API rate limit exceeded
============================================================
```

## Step 6: Verify Data in Supabase

1. Go to Supabase Dashboard > **Table Editor**
2. Check the `videos` table - you should see your processed videos
3. Check the `transcript_chunks` table - you should see the chunked transcripts with embeddings
4. Run a test query in SQL Editor:

```sql
-- Check how many chunks were created
SELECT COUNT(*) FROM transcript_chunks;

-- View recent videos
SELECT id, title, status, chunks_created
FROM videos
ORDER BY created_at DESC
LIMIT 10;

-- Test vector search (replace with actual embedding)
SELECT * FROM match_transcript_chunks(
    '[0.1, 0.2, ...]'::vector,  -- Your query embedding
    5  -- Number of results
);
```

## Troubleshooting

### Error: "supabase_url is required"

**Cause:** Environment variables not loaded from `.env`

**Solution:**
1. Make sure you have a `.env` file in the project root
2. Verify the file contains `SUPABASE_URL=...` and `SUPABASE_SERVICE_KEY=...`
3. Check for typos in the variable names

### Error: "Transcript unavailable"

**Cause:** Some YouTube videos don't have transcripts (closed captions)

**Solution:** This is expected. The pipeline will mark these videos as "failed" and move on.

### Error: "API rate limit exceeded"

**Cause:** You've hit Supadata's rate limit

**Solutions:**
1. Reduce `YOUTUBE_BATCH_SIZE` in `.env`
2. Wait a few minutes and try again
3. Upgrade your Supadata plan

### Error: "vector dimension mismatch"

**Cause:** The vector dimension in your database doesn't match your embedding model

**Solution:**
1. Check your `EMBEDDING_MODEL_CHOICE` in `.env`
2. Update the database migration (lines 47 and 75) to match:
   - `text-embedding-3-small` → `vector(1536)`
   - `all-MiniLM-L6-v2` → `vector(384)`
   - `nomic-embed-text` → `vector(768)`
3. Re-run the migration (you may need to drop the tables first if they exist)

### Error: "Not enough data for ivfflat index"

**Cause:** IVFFlat index requires training data

**Solution:**
1. Process at least 100-200 videos first
2. Then the index will work properly
3. Alternatively, comment out the index creation in the migration and add it later

## Using with Ollama (Local Embeddings)

To use local embeddings instead of OpenAI:

1. Install Ollama from [ollama.ai](https://ollama.ai)
2. Pull the embedding model:
   ```bash
   ollama pull nomic-embed-text
   ```
3. Update `.env`:
   ```bash
   EMBEDDING_PROVIDER=ollama
   EMBEDDING_BASE_URL=http://localhost:11434/v1
   EMBEDDING_API_KEY=ollama
   EMBEDDING_MODEL_CHOICE=nomic-embed-text
   ```
4. Update database dimension to `vector(768)`
5. Run the pipeline as normal

## Next Steps

Once your pipeline is running successfully:

1. **Schedule regular runs** using cron (Linux/Mac) or Task Scheduler (Windows)
2. **Build agent tools** to query the knowledge base (future phase)
3. **Monitor processing** by querying the `videos` table for failed videos
4. **Optimize chunk size** based on your use case (adjust `MIN_CHUNK_TOKENS` and `MAX_CHUNK_TOKENS`)

## Support

If you encounter issues not covered here:

1. Check the logs for detailed error messages (structured JSON logs)
2. Verify all environment variables are set correctly
3. Test each component individually (YouTube service, embeddings, database)
4. Review the implementation plan: `PRPs/requests/youtube-rag-pipeline.md`

Happy embedding! 🚀
