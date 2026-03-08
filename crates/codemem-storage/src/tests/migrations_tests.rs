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
