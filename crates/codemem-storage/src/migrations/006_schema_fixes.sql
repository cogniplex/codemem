-- Replace non-unique content_hash index with UNIQUE constraint.
-- First, deduplicate existing rows that share the same content_hash
-- (keeps the most recently updated row per hash; ignores NULLs which
-- SQLite already treats as distinct in UNIQUE indexes).
DELETE FROM memories
WHERE content_hash IS NOT NULL
  AND id NOT IN (
    SELECT id FROM (
      SELECT id, ROW_NUMBER() OVER (
        PARTITION BY content_hash ORDER BY updated_at DESC, created_at DESC, id DESC
      ) AS rn
      FROM memories
      WHERE content_hash IS NOT NULL
    )
    WHERE rn = 1
  );

-- Clean up orphaned memory_embeddings left by the dedup above.
-- (memory_embeddings has no FK, so cascade does not apply.)
DELETE FROM memory_embeddings
WHERE memory_id NOT IN (SELECT id FROM memories);

DROP INDEX IF EXISTS idx_memories_hash;
CREATE UNIQUE INDEX IF NOT EXISTS idx_memories_hash_unique ON memories(content_hash);

-- Composite index for session activity queries by tool.
CREATE INDEX IF NOT EXISTS idx_session_activity_tool
    ON session_activity (session_id, tool_name);
