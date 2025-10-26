-- YouTube RAG Pipeline Database Schema
-- Execute this in Supabase Dashboard > SQL Editor

-- ============================================================================
-- STEP 1: Enable pgvector extension
-- ============================================================================
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================================
-- STEP 2: Create tables
-- ============================================================================

-- Channels table
CREATE TABLE IF NOT EXISTS channels (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    url TEXT,
    last_processed_at TIMESTAMPTZ,
    active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Videos table
CREATE TABLE IF NOT EXISTS videos (
    id TEXT PRIMARY KEY,
    channel_id TEXT REFERENCES channels(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    url TEXT NOT NULL,
    published_at TIMESTAMPTZ,
    duration_seconds INTEGER,
    status TEXT DEFAULT 'pending', -- pending, processing, completed, failed
    transcript_available BOOLEAN DEFAULT FALSE,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    processed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Transcript chunks with embeddings
-- NOTE: Adjust vector dimension based on your embedding model:
--   - all-MiniLM-L6-v2: 384 dimensions
--   - text-embedding-3-small: 1536 dimensions
--   - nomic-embed-text: 768 dimensions
CREATE TABLE IF NOT EXISTS transcript_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id TEXT REFERENCES videos(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    text_content TEXT NOT NULL,
    start_offset_ms INTEGER,
    end_offset_ms INTEGER,
    duration_ms INTEGER,
    token_count INTEGER,
    embedding vector(1536), -- Adjust dimension based on EMBEDDING_MODEL_CHOICE
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(video_id, chunk_index)
);

-- ============================================================================
-- STEP 3: Create indexes for performance
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_videos_channel_id ON videos(channel_id);
CREATE INDEX IF NOT EXISTS idx_videos_status ON videos(status);
CREATE INDEX IF NOT EXISTS idx_videos_published_at ON videos(published_at DESC);
CREATE INDEX IF NOT EXISTS idx_chunks_video_id ON transcript_chunks(video_id);

-- Vector similarity index (IVFFlat)
-- NOTE: The 'lists' parameter should be adjusted based on your data size:
--   - For < 1M rows: lists = 100 is good
--   - For > 1M rows: lists = sqrt(rows) is recommended
CREATE INDEX IF NOT EXISTS idx_chunks_embedding
ON transcript_chunks USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- ============================================================================
-- STEP 4: Create RPC function for vector search
-- ============================================================================

CREATE OR REPLACE FUNCTION match_transcript_chunks(
    query_embedding vector(1536), -- Match dimension with table
    match_count INT DEFAULT 5,
    filter JSONB DEFAULT '{}'::jsonb
)
RETURNS TABLE (
    id UUID,
    video_id TEXT,
    text_content TEXT,
    start_offset_ms INTEGER,
    similarity FLOAT,
    metadata JSONB
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        tc.id,
        tc.video_id,
        tc.text_content,
        tc.start_offset_ms,
        1 - (tc.embedding <=> query_embedding) AS similarity,
        tc.metadata
    FROM transcript_chunks tc
    WHERE
        CASE
            WHEN filter != '{}'::jsonb THEN tc.metadata @> filter
            ELSE TRUE
        END
    ORDER BY tc.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- ============================================================================
-- STEP 5: Enable Row Level Security (RLS) - OPTIONAL
-- ============================================================================

-- Enable RLS on all tables
ALTER TABLE channels ENABLE ROW LEVEL SECURITY;
ALTER TABLE videos ENABLE ROW LEVEL SECURITY;
ALTER TABLE transcript_chunks ENABLE ROW LEVEL SECURITY;

-- Public read access policy (adjust based on your auth strategy)
CREATE POLICY "Public read access on channels" ON channels
    FOR SELECT USING (true);

CREATE POLICY "Public read access on videos" ON videos
    FOR SELECT USING (true);

CREATE POLICY "Public read access on transcript_chunks" ON transcript_chunks
    FOR SELECT USING (true);

-- If you need write access from service role (for the pipeline):
-- These policies allow service role to insert/update/delete
CREATE POLICY "Service role full access on channels" ON channels
    FOR ALL USING (auth.jwt() ->> 'role' = 'service_role');

CREATE POLICY "Service role full access on videos" ON videos
    FOR ALL USING (auth.jwt() ->> 'role' = 'service_role');

CREATE POLICY "Service role full access on transcript_chunks" ON transcript_chunks
    FOR ALL USING (auth.jwt() ->> 'role' = 'service_role');

-- ============================================================================
-- STEP 6: Verification queries
-- ============================================================================

-- Verify tables were created
SELECT table_name
FROM information_schema.tables
WHERE table_schema = 'public'
AND table_name IN ('channels', 'videos', 'transcript_chunks');

-- Verify pgvector extension is enabled
SELECT * FROM pg_extension WHERE extname = 'vector';

-- Verify indexes were created
SELECT indexname
FROM pg_indexes
WHERE tablename IN ('channels', 'videos', 'transcript_chunks');

-- ============================================================================
-- NOTES
-- ============================================================================

-- 1. IMPORTANT: Adjust vector dimensions in the following places if not using
--    text-embedding-3-small (1536 dimensions):
--    - Line 47: CREATE TABLE transcript_chunks (embedding vector(1536))
--    - Line 75: CREATE OR REPLACE FUNCTION match_transcript_chunks(query_embedding vector(1536))
--
-- 2. Common embedding model dimensions:
--    - all-MiniLM-L6-v2: 384
--    - text-embedding-3-small: 1536
--    - text-embedding-3-large: 3072
--    - nomic-embed-text: 768
--
-- 3. If you change the dimension after data is inserted, you'll need to:
--    - Drop and recreate the table (losing data), OR
--    - Create a new table and migrate data
--
-- 4. The IVFFlat index requires training data. If you get errors about
--    "not enough data", insert at least a few hundred rows first, then
--    create the index.
--
-- 5. For better search quality with large datasets (>100k chunks),
--    consider using HNSW index instead of IVFFlat:
--    CREATE INDEX idx_chunks_embedding ON transcript_chunks
--    USING hnsw (embedding vector_cosine_ops);
--
-- 6. RLS policies can be customized based on your authentication needs.
--    The service_role policies assume you're using Supabase service key.
