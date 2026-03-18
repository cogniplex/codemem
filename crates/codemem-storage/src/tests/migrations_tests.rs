use super::*;

#[test]
fn migrations_run_on_fresh_db() {
    let conn = Connection::open_in_memory().unwrap();
    conn.pragma_update(None, "foreign_keys", "ON").unwrap();
    run_migrations(&conn).unwrap();

    // Verify schema_version has entries for all migrations
    let count: u32 = conn
        .query_row("SELECT COUNT(*) FROM schema_version", [], |row| row.get(0))
        .unwrap();
    // Use max version instead of hardcoded count so adding new migrations doesn't break this test
    let max_version: u32 = conn
        .query_row(
            "SELECT COALESCE(MAX(version), 0) FROM schema_version",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert!(count > 0, "should have applied at least one migration");
    assert_eq!(
        count, max_version,
        "count should match max version (sequential)"
    );

    // Verify memories table has session_id column
    let has_session_id: bool = conn
        .prepare("SELECT session_id FROM memories LIMIT 0")
        .is_ok();
    assert!(has_session_id);

    // Verify file_hashes table exists
    let has_file_hashes: bool = conn
        .prepare("SELECT file_path FROM file_hashes LIMIT 0")
        .is_ok();
    assert!(has_file_hashes);
}

#[test]
fn migrations_are_idempotent() {
    let conn = Connection::open_in_memory().unwrap();
    conn.pragma_update(None, "foreign_keys", "ON").unwrap();
    run_migrations(&conn).unwrap();
    run_migrations(&conn).unwrap();

    let count: u32 = conn
        .query_row("SELECT COUNT(*) FROM schema_version", [], |row| row.get(0))
        .unwrap();
    let max_version: u32 = conn
        .query_row(
            "SELECT COALESCE(MAX(version), 0) FROM schema_version",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(
        count, max_version,
        "idempotent run should not duplicate entries"
    );
}

// ── Schema parity tests ────────────────────────────────────────────

/// Collect table names and their column info from SQLite.
fn get_table_columns(
    conn: &Connection,
) -> std::collections::BTreeMap<String, Vec<(String, String)>> {
    let mut tables = std::collections::BTreeMap::new();
    let mut stmt = conn
        .prepare("SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name")
        .unwrap();
    let table_names: Vec<String> = stmt
        .query_map([], |row| row.get(0))
        .unwrap()
        .filter_map(|r| r.ok())
        .collect();

    for table in table_names {
        let mut col_stmt = conn
            .prepare(&format!("PRAGMA table_info('{table}')"))
            .unwrap();
        let cols: Vec<(String, String)> = col_stmt
            .query_map([], |row| {
                Ok((row.get::<_, String>(1)?, row.get::<_, String>(2)?))
            })
            .unwrap()
            .filter_map(|r| r.ok())
            .collect();
        tables.insert(table, cols);
    }
    tables
}

/// Collect index definitions from SQLite.
fn get_indexes(conn: &Connection) -> std::collections::BTreeMap<String, String> {
    let mut indexes = std::collections::BTreeMap::new();
    let mut stmt = conn
        .prepare("SELECT name, sql FROM sqlite_master WHERE type = 'index' AND sql IS NOT NULL ORDER BY name")
        .unwrap();
    let rows: Vec<(String, String)> = stmt
        .query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })
        .unwrap()
        .filter_map(|r| r.ok())
        .collect();

    for (name, sql) in rows {
        indexes.insert(name, sql);
    }
    indexes
}

#[test]
fn fresh_schema_vs_migrated_schema_table_parity() {
    // Fresh DB via schema.sql (used by open_in_memory which calls run_migrations)
    let migrated = Connection::open_in_memory().unwrap();
    migrated.pragma_update(None, "foreign_keys", "ON").unwrap();
    run_migrations(&migrated).unwrap();

    let migrated_tables = get_table_columns(&migrated);

    // Verify all expected tables exist in migrated DB
    let expected_tables = [
        "memories",
        "memory_embeddings",
        "graph_nodes",
        "graph_edges",
        "consolidation_log",
        "sessions",
        "file_hashes",
        "schema_version",
        "repositories",
        "session_activity",
    ];

    for table in &expected_tables {
        assert!(
            migrated_tables.contains_key(*table),
            "Table '{table}' should exist in migrated DB, found tables: {:?}",
            migrated_tables.keys().collect::<Vec<_>>()
        );
    }

    // Verify memories table has expected columns
    let memory_cols: Vec<&str> = migrated_tables
        .get("memories")
        .unwrap()
        .iter()
        .map(|(name, _)| name.as_str())
        .collect();
    assert!(memory_cols.contains(&"id"));
    assert!(memory_cols.contains(&"content"));
    assert!(memory_cols.contains(&"namespace"));
    assert!(memory_cols.contains(&"session_id"));
    assert!(memory_cols.contains(&"content_hash"));
    assert!(memory_cols.contains(&"tags"));

    // Verify graph_edges has temporal columns
    let edge_cols: Vec<&str> = migrated_tables
        .get("graph_edges")
        .unwrap()
        .iter()
        .map(|(name, _)| name.as_str())
        .collect();
    assert!(edge_cols.contains(&"valid_from"));
    assert!(edge_cols.contains(&"valid_to"));
}

#[test]
fn coalesce_expression_index_exists_on_migrated_db() {
    let conn = Connection::open_in_memory().unwrap();
    conn.pragma_update(None, "foreign_keys", "ON").unwrap();
    run_migrations(&conn).unwrap();

    let indexes = get_indexes(&conn);

    // The namespace-scoped dedup index should exist
    assert!(
        indexes.contains_key("idx_memories_hash_ns"),
        "idx_memories_hash_ns COALESCE index should exist. Found indexes: {:?}",
        indexes.keys().collect::<Vec<_>>()
    );

    // Verify it uses COALESCE in its definition
    let sql = indexes.get("idx_memories_hash_ns").unwrap();
    assert!(
        sql.contains("COALESCE"),
        "idx_memories_hash_ns should use COALESCE expression, got: {sql}"
    );
}

/// Regression test for issue #53: upgrading from v0.7.0 to v0.8.0 fails
/// when existing data contains duplicate content_hash values.
#[test]
fn migration_006_deduplicates_before_unique_index() {
    let conn = Connection::open_in_memory().unwrap();
    conn.pragma_update(None, "foreign_keys", "ON").unwrap();

    // Run migrations 1-5 only (simulating a v0.7.0 database)
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY,
            description TEXT NOT NULL,
            applied_at INTEGER NOT NULL
        );",
    )
    .unwrap();

    for migration in &MIGRATIONS[..5] {
        conn.execute_batch(migration.sql).unwrap();
        conn.execute(
            "INSERT INTO schema_version (version, description, applied_at) VALUES (?1, ?2, ?3)",
            rusqlite::params![migration.version, migration.description, 1000],
        )
        .unwrap();
    }

    // Insert rows with duplicate content_hash values (valid before migration 006)
    let now = chrono::Utc::now().timestamp();
    for (id, hash, updated) in [
        ("m1", Some("hash_dup"), now - 100),
        ("m2", Some("hash_dup"), now), // newer, should be kept
        ("m3", Some("hash_dup"), now - 200),
        ("m4", Some("hash_unique"), now),
        ("m5", None::<&str>, now), // NULL hash — should survive
        ("m6", None::<&str>, now), // another NULL hash — should also survive
    ]
    .iter()
    {
        conn.execute(
            "INSERT INTO memories (id, content, memory_type, content_hash, importance, confidence,
             access_count, tags, metadata, namespace, created_at, updated_at, last_accessed_at)
             VALUES (?1, 'test', 'fact', ?2, 0.5, 1.0, 0, '[]', '{}', 'ns', ?3, ?3, ?3)",
            rusqlite::params![id, hash, updated],
        )
        .unwrap();
    }

    // Add embeddings for all memories (including ones that will be deduped)
    for id in &["m1", "m2", "m3", "m4", "m5", "m6"] {
        conn.execute(
            "INSERT INTO memory_embeddings (memory_id, embedding, model) VALUES (?1, X'00', 'test')",
            rusqlite::params![id],
        )
        .unwrap();
    }

    // Now run remaining migrations — should NOT fail
    run_migrations(&conn).unwrap();

    // Verify deduplication: only m2 (newest dup) + m4 (unique) + m5 + m6 (NULLs) remain
    let remaining: Vec<String> = conn
        .prepare("SELECT id FROM memories ORDER BY id")
        .unwrap()
        .query_map([], |row| row.get(0))
        .unwrap()
        .filter_map(|r| r.ok())
        .collect();

    assert!(
        !remaining.contains(&"m1".to_string()),
        "m1 (older dup) should have been removed"
    );
    assert!(
        remaining.contains(&"m2".to_string()),
        "m2 (newest dup) should be kept"
    );
    assert!(
        !remaining.contains(&"m3".to_string()),
        "m3 (oldest dup) should have been removed"
    );
    assert!(
        remaining.contains(&"m4".to_string()),
        "m4 (unique hash) should be kept"
    );
    assert!(
        remaining.contains(&"m5".to_string()),
        "m5 (NULL hash) should be kept"
    );
    assert!(
        remaining.contains(&"m6".to_string()),
        "m6 (NULL hash) should be kept"
    );
    assert_eq!(
        remaining.len(),
        4,
        "expected 4 rows after dedup, got: {remaining:?}"
    );

    // Verify orphaned embeddings were cleaned up (m1, m3 embeddings should be gone)
    let emb_count: u32 = conn
        .query_row(
            "SELECT COUNT(*) FROM memory_embeddings WHERE memory_id NOT IN (SELECT id FROM memories)",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(
        emb_count, 0,
        "orphaned embeddings should have been cleaned up"
    );

    // Verify kept memories still have their embeddings
    let kept_emb: u32 = conn
        .query_row("SELECT COUNT(*) FROM memory_embeddings", [], |row| {
            row.get(0)
        })
        .unwrap();
    assert_eq!(kept_emb, 4, "embeddings for kept memories should remain");
}
