use crate::MapStorageErr;
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
    Migration {
        version: 7,
        description: "JSON expression indexes, hash prefix index, session activity composite index",
        sql: include_str!("migrations/007_expression_indexes.sql"),
    },
    Migration {
        version: 8,
        description: "Namespace-scoped content_hash dedup",
        sql: include_str!("migrations/008_namespace_scoped_dedup.sql"),
    },
    Migration {
        version: 9,
        description: "Cross-repo linking tables",
        sql: include_str!("migrations/009_cross_repo_linking.sql"),
    },
    Migration {
        version: 10,
        description: "Separate api_client_calls table",
        sql: include_str!("migrations/010_api_client_calls.sql"),
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
    .storage_err()?;

    // Get current version
    let current_version: u32 = conn
        .query_row(
            "SELECT COALESCE(MAX(version), 0) FROM schema_version",
            [],
            |row| row.get(0),
        )
        .storage_err()?;

    // Run unapplied migrations, each wrapped in an EXCLUSIVE transaction
    // so migration SQL + version INSERT are atomic.
    for migration in MIGRATIONS {
        if migration.version > current_version {
            tracing::info!(
                "Applying migration {}: {}",
                migration.version,
                migration.description
            );
            let tx = conn.unchecked_transaction().storage_err()?;

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
            .storage_err()?;

            tx.commit().storage_err()?;
        }
    }

    Ok(())
}

#[cfg(test)]
#[path = "tests/migrations_tests.rs"]
mod tests;
