-- Separate table for API client calls (previously overloaded into api_endpoints)
CREATE TABLE IF NOT EXISTS api_client_calls (
    id        TEXT PRIMARY KEY,
    namespace TEXT NOT NULL,
    method    TEXT,
    target    TEXT NOT NULL,
    caller    TEXT NOT NULL,
    library   TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_api_client_calls_ns ON api_client_calls(namespace);
CREATE INDEX IF NOT EXISTS idx_api_client_calls_target ON api_client_calls(target);

-- Migrate any existing client calls from api_endpoints to the new table
INSERT OR IGNORE INTO api_client_calls (id, namespace, method, target, caller, library)
    SELECT id, namespace, method, path, handler, ''
    FROM api_endpoints
    WHERE schema = '{"type":"client_call"}';

-- Remove migrated client calls from api_endpoints
DELETE FROM api_endpoints WHERE schema = '{"type":"client_call"}';
