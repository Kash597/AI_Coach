-- Fix embedding column type if it's incorrectly set as TEXT
-- Run this in Supabase SQL Editor

-- First, check current type
SELECT column_name, data_type, udt_name
FROM information_schema.columns
WHERE table_schema = 'public'
  AND table_name = 'transcript_chunks'
  AND column_name = 'embedding';

-- If the above shows data_type as 'text' or 'USER-DEFINED' but udt_name is not 'vector', run:

-- Drop the existing data (we'll reprocess)
DELETE FROM transcript_chunks;

-- Alter the column to proper vector type
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
