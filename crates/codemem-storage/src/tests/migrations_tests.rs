use super::*;

#[test]
fn migrations_run_on_fresh_db() {
    let conn = Connection::open_in_memory().unwrap();
    conn.pragma_update(None, "foreign_keys", "ON").unwrap();
    run_migrations(&conn).unwrap();

    // Verify schema_version has 3 entries
    let count: u32 = conn
        .query_row("SELECT COUNT(*) FROM schema_version", [], |row| row.get(0))
        .unwrap();
    assert_eq!(count, 5);

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
    assert_eq!(count, 5);
}
