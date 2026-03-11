-- Cross-repo package registry
CREATE TABLE IF NOT EXISTS package_registry (
    package_name TEXT NOT NULL,
    namespace    TEXT NOT NULL,
    version      TEXT DEFAULT '',
    manifest     TEXT DEFAULT '',
    PRIMARY KEY (package_name, namespace)
);

-- Unresolved references for deferred cross-repo linking
CREATE TABLE IF NOT EXISTS unresolved_refs (
    id           TEXT PRIMARY KEY,
    namespace    TEXT NOT NULL,
    source_node  TEXT NOT NULL,
    target_name  TEXT NOT NULL,
    package_hint TEXT,
    ref_kind     TEXT NOT NULL,
    file_path    TEXT,
    line         INTEGER,
    created_at   INTEGER NOT NULL
);

-- API endpoint registry for service-to-service linking
CREATE TABLE IF NOT EXISTS api_endpoints (
    id        TEXT PRIMARY KEY,
    namespace TEXT NOT NULL,
    method    TEXT,
    path      TEXT NOT NULL,
    handler   TEXT,
    schema    TEXT DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_unresolved_refs_pkg ON unresolved_refs(package_hint);
CREATE INDEX IF NOT EXISTS idx_unresolved_refs_ns ON unresolved_refs(namespace);
CREATE INDEX IF NOT EXISTS idx_package_registry_ns ON package_registry(namespace);
CREATE INDEX IF NOT EXISTS idx_api_endpoints_path ON api_endpoints(path);
CREATE INDEX IF NOT EXISTS idx_api_endpoints_ns ON api_endpoints(namespace);
