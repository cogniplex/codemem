-- Add optional expiration timestamp to memories.
-- NULL means the memory never expires.
ALTER TABLE memories ADD COLUMN expires_at INTEGER;

CREATE INDEX IF NOT EXISTS idx_memories_expires ON memories(expires_at)
    WHERE expires_at IS NOT NULL;
