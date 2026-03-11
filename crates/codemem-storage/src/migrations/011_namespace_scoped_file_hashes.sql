-- Scope file_hashes by namespace so different repos don't collide.
-- Recreate the table with (namespace, file_path) as the composite primary key.

CREATE TABLE IF NOT EXISTS file_hashes_new (
    namespace TEXT NOT NULL DEFAULT '',
    file_path TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    PRIMARY KEY (namespace, file_path)
);

-- Migrate existing data with empty namespace (preserves hashes for single-repo setups)
INSERT OR IGNORE INTO file_hashes_new (namespace, file_path, content_hash)
    SELECT '', file_path, content_hash FROM file_hashes;

DROP TABLE IF EXISTS file_hashes;
ALTER TABLE file_hashes_new RENAME TO file_hashes;
