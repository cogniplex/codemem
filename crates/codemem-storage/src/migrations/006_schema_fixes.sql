-- Replace non-unique content_hash index with UNIQUE constraint.
DROP INDEX IF EXISTS idx_memories_hash;
CREATE UNIQUE INDEX IF NOT EXISTS idx_memories_hash_unique ON memories(content_hash);

-- Composite index for session activity queries by tool.
CREATE INDEX IF NOT EXISTS idx_session_activity_tool
    ON session_activity (session_id, tool_name);
