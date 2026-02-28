use codemem_core::CodememError;
use rusqlite::Connection;

struct Migration {
    version: u32,
    description: &'static str,
    sql: &'static str,
}

const MIGRATIONS: &[Migration] = &[
    Migration {
        version: 1,
        description: "Initial schema",
        sql: include_str!("migrations/001_initial.sql"),
    },
    Migration {
        version: 2,
        description: "Compound indexes and file hashes",
        sql: include_str!("migrations/002_compound_indexes.sql"),
    },
    Migration {
        version: 3,
        description: "Temporal edges (valid_from, valid_to)",
        sql: include_str!("migrations/003_temporal_edges.sql"),
    },
];

/// Run all pending migrations on the given connection.
pub(crate) fn run_migrations(conn: &Connection) -> Result<(), CodememError> {
    // Create schema_version table if it doesn't exist
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY,
            description TEXT NOT NULL,
            applied_at INTEGER NOT NULL
        );",
    )
    .map_err(|e| CodememError::Storage(e.to_string()))?;

    // Get current version
    let current_version: u32 = conn
        .query_row(
            "SELECT COALESCE(MAX(version), 0) FROM schema_version",
            [],
            |row| row.get(0),
        )
        .map_err(|e| CodememError::Storage(e.to_string()))?;

    // Run unapplied migrations
    for migration in MIGRATIONS {
        if migration.version > current_version {
            tracing::info!(
                "Applying migration {}: {}",
                migration.version,
                migration.description
            );
            conn.execute_batch(migration.sql).map_err(|e| {
                CodememError::Storage(format!(
                    "Migration {} ({}) failed: {}",
                    migration.version, migration.description, e
                ))
            })?;
            conn.execute(
                "INSERT INTO schema_version (version, description, applied_at) VALUES (?1, ?2, ?3)",
                rusqlite::params![
                    migration.version,
                    migration.description,
                    chrono::Utc::now().timestamp()
                ],
            )
            .map_err(|e| CodememError::Storage(e.to_string()))?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
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
        assert_eq!(count, 3);

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
        assert_eq!(count, 3);
    }
}
