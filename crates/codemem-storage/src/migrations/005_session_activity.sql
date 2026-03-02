-- Session activity tracking for trigger-based auto-insights.
CREATE TABLE IF NOT EXISTS session_activity (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    file_path TEXT,
    directory TEXT,
    pattern TEXT,
    created_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_session_activity_session
    ON session_activity (session_id);

CREATE INDEX IF NOT EXISTS idx_session_activity_session_dir
    ON session_activity (session_id, directory);
