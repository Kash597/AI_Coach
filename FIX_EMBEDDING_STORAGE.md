# Fix Embedding Storage Issue

## Problem Diagnosed

The vector search is returning 0 results because embeddings are being stored as **TEXT strings** instead of proper **pgvector VECTOR type** in the database.

**Root Cause**: The Supabase Python client serializes Python lists as JSON strings when inserting data. The `embedding` column accepts this as TEXT instead of properly casting to VECTOR type.

**Evidence**:
- Embeddings stored as: `"[0.021360122,0.025763005,...]"` (string)
- Should be stored as: PostgreSQL `vector(1536)` type
- Vector similarity operations only work on proper VECTOR type columns

## Solution

You need to:
1. Fix the database column type
2. Clear existing (incorrectly formatted) data
3. Reprocess videos with the corrected schema

### Step 1: Fix the Database Column

Go to your **Supabase Dashboard → SQL Editor** and run this SQL:

```sql
-- migrations/fix_embedding_column.sql

-- Check current type (should show if it's wrong)
SELECT column_name, data_type, udt_name
FROM information_schema.columns
WHERE table_schema = 'public'
  AND table_name = 'transcript_chunks'
  AND column_name = 'embedding';

-- Delete existing data (it's in wrong format anyway)
DELETE FROM transcript_chunks;

-- Fix the column type
ALTER TABLE transcript_chunks
ALTER COLUMN embedding TYPE vector(1536) USING embedding::vector(1536);

-- Reset videos so they can be reprocessed
UPDATE videos
SET status = 'pending',
    processed_at = NULL,
    transcript_available = FALSE,
    error_message = NULL,
    retry_count = 0;

-- Verify the fix
SELECT column_name, data_type, udt_name
FROM information_schema.columns
WHERE table_schema = 'public'
  AND table_name = 'transcript_chunks'
  AND column_name = 'embedding';
```

You should see:
- `column_name`: embedding
- `data_type`: USER-DEFINED
- `udt_name`: **vector**

### Step 2: Reprocess Videos

Now run the RAG pipeline again to reprocess videos with the correct format:

```bash
uv run python -m src.rag_pipeline.cli --days-back 7
```

The pipeline will:
1. Fetch videos (status is now 'pending')
2. Generate embeddings
3. Store them in proper VECTOR format this time

### Step 3: Verify the Fix

Test that vector search works:

```bash
uv run python -c "
import asyncio
from dotenv import load_dotenv
import os

load_dotenv()

from openai import AsyncOpenAI
from supabase import create_client

async def test():
    url = os.getenv('SUPABASE_URL')
    key = os.getenv('SUPABASE_SERVICE_KEY')
    openai_key = os.getenv('LLM_API_KEY') or os.getenv('EMBEDDING_API_KEY')

    client = create_client(url, key)
    openai_client = AsyncOpenAI(api_key=openai_key)

    # Generate embedding
    response = await openai_client.embeddings.create(
        input=['coaching advice'],
        model='text-embedding-3-small'
    )
    embedding = response.data[0].embedding

    # Search
    result = client.rpc(
        'match_transcript_chunks',
        {
            'query_embedding': embedding,
            'match_count': 3,
            'filter': {}
        }
    ).execute()

    print(f'Found {len(result.data) if result.data else 0} results')
    if result.data:
        print('✅ Vector search is working!')
        for r in result.data[:2]:
            print(f'  - Video: {r[\"video_id\"]}, Similarity: {r[\"similarity\"]:.3f}')
    else:
        print('❌ Still not working - check the migration')

asyncio.run(test())
"
```

Expected output:
```
Found 3 results
✅ Vector search is working!
  - Video: pjF-0dliYhg, Similarity: 0.847
  - Video: pjF-0dliYhg, Similarity: 0.823
```

## Alternative: Python Script

If you prefer to use Python instead of SQL Editor:

```bash
python scripts/clear_and_reprocess.py
```

This will:
1. Delete all transcript chunks
2. Reset video statuses to 'pending'
3. Prompt you to run the pipeline

**Note**: This doesn't fix the column type. You still need to run the SQL migration if the column is wrong type.

## Why This Happened

The original migration (`001_youtube_rag_schema.sql`) correctly defined:

```sql
embedding vector(1536)
```

However, if the migration was run but then the column was somehow modified or if there was an issue during creation, it might have fallen back to TEXT type. Alternatively, Supabase's schema might not have properly recognized the pgvector type.

The fix explicitly casts the column to VECTOR type using `ALTER COLUMN`.

## Next Steps After Fix

Once the fix is complete and verified:

1. **Start the API server**:
   ```bash
   uv run uvicorn src.main:app --host 127.0.0.1 --port 8030 --reload
   ```

2. **Test the AI coach**:
   ```bash
   curl -X POST http://127.0.0.1:8030/api/pydantic-agent \
     -H "Content-Type: application/json" \
     -d '{"message": "What advice do you have about goal setting?"}'
   ```

3. The agent should now be able to search and retrieve relevant coaching content!

## Questions?

If the fix doesn't work:
1. Check that pgvector extension is installed: `SELECT * FROM pg_extension WHERE extname = 'vector';`
2. Verify column type after migration
3. Check Supabase logs for any insertion errors
4. Ensure you're using the correct API keys in `.env`
