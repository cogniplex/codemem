-- v0.3.0: Compound indexes for common query patterns + file hash tracking
CREATE INDEX IF NOT EXISTS idx_memories_ns_type ON memories(namespace, memory_type);
CREATE INDEX IF NOT EXISTS idx_memories_imp_accessed ON memories(importance, last_accessed_at);
CREATE INDEX IF NOT EXISTS idx_graph_edges_src_rel ON graph_edges(src, relationship);

-- File hash tracking for incremental indexing
CREATE TABLE IF NOT EXISTS file_hashes (
    file_path TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL
);
