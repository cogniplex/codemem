-- Expression indexes for JSON metadata fields (L3)
CREATE INDEX IF NOT EXISTS idx_memories_metadata_tool
    ON memories(json_extract(metadata, '$.tool'));
CREATE INDEX IF NOT EXISTS idx_memories_metadata_file_path
    ON memories(json_extract(metadata, '$.file_path'));

-- Content hash prefix index for faster dedup lookups (M6)
CREATE INDEX IF NOT EXISTS idx_memories_hash_prefix
    ON memories(substr(content_hash, 1, 16));

-- Session activity composite index (M8)
CREATE INDEX IF NOT EXISTS idx_session_activity_composite
    ON session_activity(session_id, tool_name, file_path);
