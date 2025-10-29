-- Check the actual data type of the embedding column
SELECT
    column_name,
    data_type,
    udt_name,
    character_maximum_length
FROM information_schema.columns
WHERE table_schema = 'public'
  AND table_name = 'transcript_chunks'
  AND column_name = 'embedding';

-- Also check if pgvector extension is enabled
SELECT * FROM pg_extension WHERE extname = 'vector';

-- Try to see actual stored data type
SELECT
    id,
    video_id,
    pg_typeof(embedding) as embedding_type,
    substring(embedding::text, 1, 50) as embedding_preview
FROM transcript_chunks
LIMIT 2;
