//! codemem-storage: SQLite persistence layer for Codemem.
//!
//! Uses rusqlite with bundled SQLite, WAL mode, and embedded schema.

use codemem_core::{CodememError, MemoryNode, MemoryType};
use rusqlite::Connection;
use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
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
    /// Whether an explicit outer transaction is active (via `begin_transaction`).
    /// When set, individual methods like `insert_memory` skip starting their own
    /// transaction so that all operations participate in the outer one.
    in_transaction: AtomicBool,
}

impl Storage {
    /// Get a lock on the underlying connection.
    pub(crate) fn conn(&self) -> Result<std::sync::MutexGuard<'_, Connection>, CodememError> {
        self.conn
            .lock()
            .map_err(|e| CodememError::LockPoisoned(format!("Storage mutex: {e}")))
    }

    /// Apply standard pragmas to a connection.
    ///
    /// `cache_size_mb` and `busy_timeout_secs` override the defaults (64 MB / 5 s)
    /// when provided — typically sourced from `StorageConfig`.
    fn apply_pragmas(
        conn: &Connection,
        cache_size_mb: Option<u32>,
        busy_timeout_secs: Option<u64>,
    ) -> Result<(), CodememError> {
        // WAL mode for concurrent reads
        conn.pragma_update(None, "journal_mode", "WAL")
            .map_err(|e| CodememError::Storage(e.to_string()))?;
        // Cache size (negative value = KiB in SQLite)
        let cache_kb = i64::from(cache_size_mb.unwrap_or(64)) * 1000;
        conn.pragma_update(None, "cache_size", -cache_kb)
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
        // Busy timeout
        let timeout = busy_timeout_secs.unwrap_or(5);
        conn.busy_timeout(std::time::Duration::from_secs(timeout))
            .map_err(|e| CodememError::Storage(e.to_string()))?;
        Ok(())
    }

    /// Open (or create) a Codemem database at the given path.
    pub fn open(path: &Path) -> Result<Self, CodememError> {
        Self::open_with_config(path, None, None)
    }

    /// Open a database with explicit storage configuration overrides.
    pub fn open_with_config(
        path: &Path,
        cache_size_mb: Option<u32>,
        busy_timeout_secs: Option<u64>,
    ) -> Result<Self, CodememError> {
        let conn = Connection::open(path).map_err(|e| CodememError::Storage(e.to_string()))?;
        Self::apply_pragmas(&conn, cache_size_mb, busy_timeout_secs)?;
        migrations::run_migrations(&conn)?;
        Ok(Self {
            conn: Mutex::new(conn),
            in_transaction: AtomicBool::new(false),
        })
    }

    /// Open an existing database without running migrations.
    ///
    /// Use this in lifecycle hooks (context, prompt, summarize) where the
    /// database has already been migrated by `codemem init` or `codemem serve`,
    /// to avoid SQLITE_BUSY race conditions with the concurrent MCP server.
    pub fn open_without_migrations(path: &Path) -> Result<Self, CodememError> {
        Self::open_without_migrations_with_config(path, None, None)
    }

    /// Open without migrations, with explicit storage configuration overrides.
    pub fn open_without_migrations_with_config(
        path: &Path,
        cache_size_mb: Option<u32>,
        busy_timeout_secs: Option<u64>,
    ) -> Result<Self, CodememError> {
        let conn = Connection::open(path).map_err(|e| CodememError::Storage(e.to_string()))?;
        Self::apply_pragmas(&conn, cache_size_mb, busy_timeout_secs)?;
        Ok(Self {
            conn: Mutex::new(conn),
            in_transaction: AtomicBool::new(false),
        })
    }

    /// Open an in-memory database (for testing).
    pub fn open_in_memory() -> Result<Self, CodememError> {
        let conn =
            Connection::open_in_memory().map_err(|e| CodememError::Storage(e.to_string()))?;
        Self::apply_pragmas(&conn, None, None)?;
        migrations::run_migrations(&conn)?;
        Ok(Self {
            conn: Mutex::new(conn),
            in_transaction: AtomicBool::new(false),
        })
    }

    /// Compute SHA-256 hash of content for deduplication.
    pub fn content_hash(content: &str) -> String {
        codemem_core::content_hash(content)
    }

    /// Check whether an outer transaction is currently active.
    pub(crate) fn has_outer_transaction(&self) -> bool {
        self.in_transaction.load(Ordering::Acquire)
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
    pub(crate) session_id: Option<String>,
    pub(crate) created_at: i64,
    pub(crate) updated_at: i64,
    pub(crate) last_accessed_at: i64,
}

impl MemoryRow {
    pub(crate) fn from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<Self> {
        Ok(Self {
            id: row.get(0)?,
            content: row.get(1)?,
            memory_type: row.get(2)?,
            importance: row.get(3)?,
            confidence: row.get(4)?,
            access_count: row.get(5)?,
            content_hash: row.get(6)?,
            tags: row.get(7)?,
            metadata: row.get(8)?,
            namespace: row.get(9)?,
            session_id: row.get(10)?,
            created_at: row.get(11)?,
            updated_at: row.get(12)?,
            last_accessed_at: row.get(13)?,
        })
    }

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
            session_id: self.session_id,
            created_at,
            updated_at,
            last_accessed_at,
        })
    }
}
