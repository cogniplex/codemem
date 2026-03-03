//! codemem-storage: SQLite persistence layer for Codemem.
//!
//! Uses rusqlite with bundled SQLite, WAL mode, and embedded schema.

use codemem_core::{CodememError, MemoryNode, MemoryType};
use rusqlite::Connection;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;

mod backend;
pub mod graph;
mod graph_persistence;
mod memory;
mod migrations;
mod queries;
pub mod vector;

pub use graph::GraphEngine;
pub use graph::RawGraphMetrics;
pub use vector::HnswIndex;

/// SQLite-backed storage for Codemem memories, embeddings, and graph data.
///
/// Wraps `rusqlite::Connection` in a `Mutex` to satisfy `Send + Sync` bounds
/// required by the `StorageBackend` trait.
pub struct Storage {
    conn: Mutex<Connection>,
}

impl Storage {
    /// Get a lock on the underlying connection.
    pub(crate) fn conn(&self) -> Result<std::sync::MutexGuard<'_, Connection>, CodememError> {
        self.conn
            .lock()
            .map_err(|e| CodememError::LockPoisoned(format!("Storage mutex: {e}")))
    }

    /// Apply standard pragmas to a connection.
    fn apply_pragmas(conn: &Connection) -> Result<(), CodememError> {
        // WAL mode for concurrent reads
        conn.pragma_update(None, "journal_mode", "WAL")
            .map_err(|e| CodememError::Storage(e.to_string()))?;
        // 64MB cache
        conn.pragma_update(None, "cache_size", -64000i64)
            .map_err(|e| CodememError::Storage(e.to_string()))?;
        // Foreign keys ON
        conn.pragma_update(None, "foreign_keys", "ON")
            .map_err(|e| CodememError::Storage(e.to_string()))?;
        // NORMAL sync (good balance of safety vs speed)
        conn.pragma_update(None, "synchronous", "NORMAL")
            .map_err(|e| CodememError::Storage(e.to_string()))?;
        // 256MB mmap for faster reads
        conn.pragma_update(None, "mmap_size", 268435456i64)
            .map_err(|e| CodememError::Storage(e.to_string()))?;
        // Temp tables in memory
        conn.pragma_update(None, "temp_store", "MEMORY")
            .map_err(|e| CodememError::Storage(e.to_string()))?;
        // 5s busy timeout
        conn.busy_timeout(std::time::Duration::from_secs(5))
            .map_err(|e| CodememError::Storage(e.to_string()))?;
        Ok(())
    }

    /// Open (or create) a Codemem database at the given path.
    pub fn open(path: &Path) -> Result<Self, CodememError> {
        let conn = Connection::open(path).map_err(|e| CodememError::Storage(e.to_string()))?;
        Self::apply_pragmas(&conn)?;
        migrations::run_migrations(&conn)?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    /// Open an existing database without running migrations.
    ///
    /// Use this in lifecycle hooks (context, prompt, summarize) where the
    /// database has already been migrated by `codemem init` or `codemem serve`,
    /// to avoid SQLITE_BUSY race conditions with the concurrent MCP server.
    pub fn open_without_migrations(path: &Path) -> Result<Self, CodememError> {
        let conn = Connection::open(path).map_err(|e| CodememError::Storage(e.to_string()))?;
        Self::apply_pragmas(&conn)?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    /// Open an in-memory database (for testing).
    pub fn open_in_memory() -> Result<Self, CodememError> {
        let conn =
            Connection::open_in_memory().map_err(|e| CodememError::Storage(e.to_string()))?;
        conn.pragma_update(None, "foreign_keys", "ON")
            .map_err(|e| CodememError::Storage(e.to_string()))?;
        migrations::run_migrations(&conn)?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    /// Compute SHA-256 hash of content for deduplication.
    pub fn content_hash(content: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        format!("{:x}", hasher.finalize())
    }
}

/// Internal row struct for memory deserialization.
pub(crate) struct MemoryRow {
    pub(crate) id: String,
    pub(crate) content: String,
    pub(crate) memory_type: String,
    pub(crate) importance: f64,
    pub(crate) confidence: f64,
    pub(crate) access_count: i64,
    pub(crate) content_hash: String,
    pub(crate) tags: String,
    pub(crate) metadata: String,
    pub(crate) namespace: Option<String>,
    pub(crate) created_at: i64,
    pub(crate) updated_at: i64,
    pub(crate) last_accessed_at: i64,
}

impl MemoryRow {
    pub(crate) fn into_memory_node(self) -> Result<MemoryNode, CodememError> {
        let memory_type: MemoryType = self.memory_type.parse()?;
        let tags: Vec<String> = serde_json::from_str(&self.tags).unwrap_or_else(|e| {
            tracing::warn!(id = %self.id, error = %e, "Malformed tags JSON for memory");
            Vec::new()
        });
        let metadata: HashMap<String, serde_json::Value> = serde_json::from_str(&self.metadata)
            .unwrap_or_else(|e| {
                tracing::warn!(id = %self.id, error = %e, "Malformed metadata JSON for memory");
                HashMap::new()
            });

        let created_at = chrono::DateTime::from_timestamp(self.created_at, 0)
            .unwrap_or_else(|| {
                tracing::warn!(id = %self.id, ts = self.created_at, "Invalid created_at timestamp");
                chrono::DateTime::<chrono::Utc>::default()
            })
            .with_timezone(&chrono::Utc);
        let updated_at = chrono::DateTime::from_timestamp(self.updated_at, 0)
            .unwrap_or_else(|| {
                tracing::warn!(id = %self.id, ts = self.updated_at, "Invalid updated_at timestamp");
                chrono::DateTime::<chrono::Utc>::default()
            })
            .with_timezone(&chrono::Utc);
        let last_accessed_at = chrono::DateTime::from_timestamp(self.last_accessed_at, 0)
            .unwrap_or_else(|| {
                tracing::warn!(id = %self.id, ts = self.last_accessed_at, "Invalid last_accessed_at timestamp");
                chrono::DateTime::<chrono::Utc>::default()
            })
            .with_timezone(&chrono::Utc);

        Ok(MemoryNode {
            id: self.id,
            content: self.content,
            memory_type,
            importance: self.importance,
            confidence: self.confidence,
            access_count: u32::try_from(self.access_count).unwrap_or(u32::MAX),
            content_hash: self.content_hash,
            tags,
            metadata,
            namespace: self.namespace,
            created_at,
            updated_at,
            last_accessed_at,
        })
    }
}
