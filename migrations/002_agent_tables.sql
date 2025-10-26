-- ==============================================================================
-- Dynamous AI Coach Agent Database Schema
-- ==============================================================================
-- This migration adds agent-specific tables for user management, conversations,
-- messages, and rate limiting. It works alongside the existing YouTube RAG
-- pipeline tables (channels, videos, transcript_chunks) from migration 001.
--
-- Tables added:
-- - user_profiles: User authentication and profile data
-- - conversations: Chat sessions with titles and metadata
-- - messages: Individual messages within conversations
-- - requests: Rate limiting tracking
--
-- Note: We exclude document_metadata, document_rows, and documents tables
-- from the original rag_agent_tables.sql since we use videos/transcript_chunks
-- for our knowledge base.

-- ============================================================================
-- STEP 1: Drop existing policies, triggers, and functions if they exist
-- ============================================================================

DO $$
DECLARE
    rec RECORD;
BEGIN
    -- Drop policies safely
    FOR rec IN
        SELECT schemaname, tablename, policyname
        FROM pg_policies
        WHERE schemaname = 'public'
        AND policyname IN (
            'Deny delete for messages',
            'Admins can insert messages',
            'Admins can view all messages',
            'Users can insert messages in their conversations',
            'Users can view their own messages',
            'Deny delete for conversations',
            'Admins can insert conversations',
            'Admins can update all conversations',
            'Admins can view all conversations',
            'Users can update their own conversations',
            'Users can insert their own conversations',
            'Users can view their own conversations',
            'Deny delete for requests',
            'Admins can insert requests',
            'Admins can view all requests',
            'Users can view their own requests',
            'Admins can update all profiles',
            'Admins can view all profiles',
            'Only admins can change admin status',
            'Users can update their own profile',
            'Users can view their own profile',
            'Deny delete for user_profiles'
        )
    LOOP
        EXECUTE format('DROP POLICY IF EXISTS %I ON %I.%I',
                      rec.policyname, rec.schemaname, rec.tablename);
    END LOOP;

    -- Drop triggers from public schema
    FOR rec IN
        SELECT t.tgname, c.relname
        FROM pg_trigger t
        JOIN pg_class c ON t.tgrelid = c.oid
        JOIN pg_namespace n ON c.relnamespace = n.oid
        WHERE n.nspname = 'public'
        AND t.tgname = 'on_auth_user_created'
        AND NOT t.tgisinternal
    LOOP
        EXECUTE format('DROP TRIGGER IF EXISTS %I ON %I', rec.tgname, rec.relname);
    END LOOP;

    -- Drop trigger from auth schema if it exists
    BEGIN
        DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
    EXCEPTION
        WHEN undefined_table THEN
            NULL; -- auth.users table doesn't exist, ignore
        WHEN undefined_object THEN
            NULL; -- trigger doesn't exist, ignore
    END;

END $$;

-- Drop functions
DROP FUNCTION IF EXISTS public.handle_new_user();
DROP FUNCTION IF EXISTS public.is_admin();

-- Drop tables (in reverse dependency order) - CASCADE will handle dependencies
DROP TABLE IF EXISTS messages CASCADE;
DROP TABLE IF EXISTS conversations CASCADE;
DROP TABLE IF EXISTS requests CASCADE;
DROP TABLE IF EXISTS user_profiles CASCADE;

-- ============================================================================
-- STEP 2: Create tables
-- ============================================================================

-- 1. User Profiles Table
-- Linked to Supabase auth.users, auto-populated on user signup
CREATE TABLE user_profiles (
    id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    email TEXT NOT NULL,
    full_name TEXT,
    is_admin BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL
);

-- 2. Requests Table
-- Tracks requests for rate limiting
CREATE TABLE requests (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    user_query TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES user_profiles(id) ON DELETE CASCADE
);

-- 3. Conversations Table
-- Chat sessions with auto-generated titles
CREATE TABLE conversations (
    session_id VARCHAR PRIMARY KEY NOT NULL,
    user_id UUID NOT NULL,
    title VARCHAR,  -- Auto-generated from first message
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_message_at TIMESTAMPTZ DEFAULT NOW(),
    is_archived BOOLEAN DEFAULT FALSE,
    metadata JSONB DEFAULT '{}'::jsonb,

    UNIQUE(session_id),
    FOREIGN KEY (user_id) REFERENCES user_profiles(id) ON DELETE CASCADE
);

-- 4. Messages Table
-- Individual messages within conversations
-- Stores Pydantic AI message format as JSONB
CREATE TABLE messages (
    id INT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    computed_session_user_id UUID GENERATED ALWAYS AS (
        CAST(SPLIT_PART(session_id, '~', 1) AS UUID)
    ) STORED,
    session_id VARCHAR NOT NULL,
    message JSONB NOT NULL,
    message_data TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    FOREIGN KEY (session_id) REFERENCES conversations(session_id)
);

-- ============================================================================
-- STEP 3: Create indexes for performance
-- ============================================================================

-- Conversation and message indexes
CREATE INDEX idx_conversations_user ON conversations(user_id);
CREATE INDEX idx_messages_session ON messages(session_id);
CREATE INDEX idx_messages_computed_session ON messages(computed_session_user_id);
CREATE INDEX idx_requests_user_timestamp ON requests(user_id, timestamp);

-- ============================================================================
-- STEP 4: Create functions
-- ============================================================================

-- 1. Handle New User Function
-- Auto-creates user profile when new user signs up via Supabase Auth
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO public.user_profiles (id, email)
    VALUES (new.id, new.email);
    RETURN new;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- 2. Admin Check Function
-- Helper function to check if current user is admin
CREATE OR REPLACE FUNCTION public.is_admin()
RETURNS BOOLEAN AS $$
DECLARE
  is_admin_user BOOLEAN;
BEGIN
  SELECT COALESCE(up.is_admin, FALSE) INTO is_admin_user
  FROM user_profiles up
  WHERE up.id = auth.uid();

  RETURN is_admin_user;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- ============================================================================
-- STEP 5: Create triggers
-- ============================================================================

-- Auto-create user profile on signup
CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE PROCEDURE public.handle_new_user();

-- ============================================================================
-- STEP 6: Enable Row Level Security
-- ============================================================================

ALTER TABLE user_profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE requests ENABLE ROW LEVEL SECURITY;
ALTER TABLE conversations ENABLE ROW LEVEL SECURITY;
ALTER TABLE messages ENABLE ROW LEVEL SECURITY;

-- ============================================================================
-- STEP 7: Create Row Level Security policies
-- ============================================================================

-- User Profiles Policies
CREATE POLICY "Users can view their own profile"
ON user_profiles
FOR SELECT
USING (auth.uid() = id);

CREATE POLICY "Users can update their own profile"
ON user_profiles
FOR UPDATE
USING (auth.uid() = id)
WITH CHECK (auth.uid() = id AND is_admin IS NOT DISTINCT FROM FALSE);

CREATE POLICY "Only admins can change admin status"
ON user_profiles
FOR UPDATE
TO authenticated
USING (is_admin())
WITH CHECK (is_admin());

CREATE POLICY "Admins can view all profiles"
ON user_profiles
FOR SELECT
USING (is_admin());

CREATE POLICY "Admins can update all profiles"
ON user_profiles
FOR UPDATE
USING (is_admin());

CREATE POLICY "Deny delete for user_profiles" ON user_profiles FOR DELETE USING (false);

-- Requests Policies
CREATE POLICY "Users can view their own requests"
ON requests
FOR SELECT
USING (auth.uid() = user_id);

CREATE POLICY "Admins can view all requests"
ON requests
FOR SELECT
USING (is_admin());

CREATE POLICY "Admins can insert requests"
ON requests
FOR INSERT
WITH CHECK (is_admin());

CREATE POLICY "Deny delete for requests" ON requests FOR DELETE USING (false);

-- Conversations Policies
CREATE POLICY "Users can view their own conversations"
ON conversations
FOR SELECT
USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own conversations"
ON conversations
FOR INSERT
WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own conversations"
ON conversations
FOR UPDATE
USING (auth.uid() = user_id);

CREATE POLICY "Admins can view all conversations"
ON conversations
FOR SELECT
USING (is_admin());

CREATE POLICY "Admins can update all conversations"
ON conversations
FOR UPDATE
USING (is_admin());

CREATE POLICY "Admins can insert conversations"
ON conversations
FOR INSERT
WITH CHECK (is_admin());

CREATE POLICY "Deny delete for conversations" ON conversations FOR DELETE USING (false);

-- Messages Policies
CREATE POLICY "Users can view their own messages"
ON messages
FOR SELECT
USING (
  auth.uid() = computed_session_user_id
);

CREATE POLICY "Users can insert messages in their conversations"
ON messages
FOR INSERT
WITH CHECK (
  auth.uid() = computed_session_user_id
);

CREATE POLICY "Admins can view all messages"
ON messages
FOR SELECT
USING (is_admin());

CREATE POLICY "Admins can insert messages"
ON messages
FOR INSERT
WITH CHECK (is_admin());

CREATE POLICY "Deny delete for messages" ON messages FOR DELETE USING (false);

-- ============================================================================
-- STEP 8: Verification queries
-- ============================================================================

-- Verify tables were created
SELECT table_name
FROM information_schema.tables
WHERE table_schema = 'public'
AND table_name IN ('user_profiles', 'conversations', 'messages', 'requests');

-- Verify triggers were created
SELECT trigger_name, event_object_table
FROM information_schema.triggers
WHERE trigger_schema = 'public'
AND trigger_name = 'on_auth_user_created';

-- Verify RLS is enabled
SELECT tablename, rowsecurity
FROM pg_tables
WHERE schemaname = 'public'
AND tablename IN ('user_profiles', 'conversations', 'messages', 'requests');

-- ============================================================================
-- SETUP COMPLETE
-- ============================================================================

-- The agent database schema is now fully configured with:
-- ✅ User profiles table with auth integration
-- ✅ Conversations table for chat sessions
-- ✅ Messages table for Pydantic AI message history
-- ✅ Requests table for rate limiting
-- ✅ Indexes for performance optimization
-- ✅ Functions for user management and admin checks
-- ✅ Trigger for automated user profile creation
-- ✅ Row Level Security enabled with comprehensive policies
--
-- Next steps:
-- 1. Configure your application environment variables
-- 2. Run the AI agent API server
-- 3. Test authentication and conversation management
