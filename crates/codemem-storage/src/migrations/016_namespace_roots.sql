-- Map namespace names to their absolute root paths on disk.
-- Populated by `codemem analyze` and used by the UI file content endpoint.
CREATE TABLE IF NOT EXISTS namespace_roots (
    namespace TEXT PRIMARY KEY,
    root_path TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
