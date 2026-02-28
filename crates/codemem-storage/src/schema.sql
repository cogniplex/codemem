-- Codemem SQLite Schema
-- WAL mode + FK enforcement set programmatically

-- Core memory storage
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    memory_type TEXT NOT NULL,
    importance REAL DEFAULT 0.5,
    confidence REAL DEFAULT 1.0,
    access_count INTEGER DEFAULT 0,
    content_hash TEXT,
    tags TEXT DEFAULT '[]',
    metadata TEXT DEFAULT '{}',
    namespace TEXT,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    last_accessed_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);
CREATE INDEX IF NOT EXISTS idx_memories_hash ON memories(content_hash);
CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);
CREATE INDEX IF NOT EXISTS idx_memories_namespace ON memories(namespace);

-- Vector embeddings (stored as BLOB). No FK — also stores symbol embeddings (sym:* IDs).
CREATE TABLE IF NOT EXISTS memory_embeddings (
    memory_id TEXT PRIMARY KEY,
    embedding BLOB NOT NULL,
    model TEXT DEFAULT 'all-MiniLM-L6-v2'
);

-- Graph nodes
CREATE TABLE IF NOT EXISTS graph_nodes (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    label TEXT,
    payload TEXT DEFAULT '{}',
    centrality REAL DEFAULT 0.0,
    memory_id TEXT,
    namespace TEXT,
    FOREIGN KEY(memory_id) REFERENCES memories(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_graph_nodes_kind ON graph_nodes(kind);
CREATE INDEX IF NOT EXISTS idx_graph_nodes_memory ON graph_nodes(memory_id);
CREATE INDEX IF NOT EXISTS idx_graph_nodes_namespace ON graph_nodes(namespace);

-- Graph edges
CREATE TABLE IF NOT EXISTS graph_edges (
    id TEXT PRIMARY KEY,
    src TEXT NOT NULL,
    dst TEXT NOT NULL,
    relationship TEXT NOT NULL,
    weight REAL DEFAULT 1.0,
    properties TEXT DEFAULT '{}',
    created_at INTEGER NOT NULL,
    FOREIGN KEY(src) REFERENCES graph_nodes(id) ON DELETE CASCADE,
    FOREIGN KEY(dst) REFERENCES graph_nodes(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_graph_edges_src ON graph_edges(src);
CREATE INDEX IF NOT EXISTS idx_graph_edges_dst ON graph_edges(dst);
CREATE INDEX IF NOT EXISTS idx_graph_edges_relationship ON graph_edges(relationship);

-- Consolidation run log
CREATE TABLE IF NOT EXISTS consolidation_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_type TEXT NOT NULL,
    run_at INTEGER NOT NULL,
    affected_count INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_consolidation_log_cycle ON consolidation_log(cycle_type, run_at);

-- Sessions: track interaction periods with AI assistants
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    namespace TEXT,
    started_at INTEGER NOT NULL,
    ended_at INTEGER,
    memory_count INTEGER DEFAULT 0,
    summary TEXT
);

CREATE INDEX IF NOT EXISTS idx_sessions_namespace ON sessions(namespace);
CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at);

-- Compound indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_memories_ns_type ON memories(namespace, memory_type);
CREATE INDEX IF NOT EXISTS idx_memories_imp_accessed ON memories(importance, last_accessed_at);
CREATE INDEX IF NOT EXISTS idx_graph_edges_src_rel ON graph_edges(src, relationship);

-- File hash tracking for incremental indexing
CREATE TABLE IF NOT EXISTS file_hashes (
    file_path TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL
);

-- Note: session_id column on memories is added dynamically via
-- Storage::ensure_session_column() since SQLite lacks
-- ALTER TABLE ADD COLUMN IF NOT EXISTS.
