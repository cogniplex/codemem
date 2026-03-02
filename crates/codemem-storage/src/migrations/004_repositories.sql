CREATE TABLE IF NOT EXISTS repositories (
    id TEXT PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    name TEXT,
    namespace TEXT,
    created_at TEXT NOT NULL,
    last_indexed_at TEXT,
    status TEXT DEFAULT 'idle'
);
