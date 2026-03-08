-- Make content_hash dedup namespace-scoped instead of global.
-- The same content can now exist in different namespaces.
-- Use COALESCE so NULL namespaces are treated as equal (SQLite treats NULLs as distinct in UNIQUE).
DROP INDEX IF EXISTS idx_memories_hash_unique;
CREATE UNIQUE INDEX IF NOT EXISTS idx_memories_hash_ns
    ON memories(content_hash, COALESCE(namespace, ''));
