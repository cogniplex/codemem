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
    Migration {
        version: 4,
        description: "Repository tracking",
        sql: include_str!("migrations/004_repositories.sql"),
    },
    Migration {
        version: 5,
        description: "Session activity tracking",
        sql: include_str!("migrations/005_session_activity.sql"),
    },
    Migration {
        version: 6,
        description: "UNIQUE content_hash and session_activity tool index",
        sql: include_str!("migrations/006_schema_fixes.sql"),
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

    // Run unapplied migrations, each wrapped in an EXCLUSIVE transaction
    // so migration SQL + version INSERT are atomic.
    for migration in MIGRATIONS {
        if migration.version > current_version {
            tracing::info!(
                "Applying migration {}: {}",
                migration.version,
                migration.description
            );
            let tx = conn
                .unchecked_transaction()
                .map_err(|e| CodememError::Storage(e.to_string()))?;

            tx.execute_batch(migration.sql).map_err(|e| {
                CodememError::Storage(format!(
                    "Migration {} ({}) failed: {}",
                    migration.version, migration.description, e
                ))
            })?;
            tx.execute(
                "INSERT INTO schema_version (version, description, applied_at) VALUES (?1, ?2, ?3)",
                rusqlite::params![
                    migration.version,
                    migration.description,
                    chrono::Utc::now().timestamp()
                ],
            )
            .map_err(|e| CodememError::Storage(e.to_string()))?;

            tx.commit()
                .map_err(|e| CodememError::Storage(e.to_string()))?;
        }
    }

    Ok(())
}

#[cfg(test)]
#[path = "tests/migrations_tests.rs"]
mod tests;
