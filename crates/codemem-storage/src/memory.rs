//! Memory CRUD operations on Storage.

use crate::{MapStorageErr, MemoryRow, Storage};
use codemem_core::{CodememError, MemoryNode, Repository};
use rusqlite::{params, OptionalExtension};

/// Shared dedup-check + INSERT logic used by both transactional and
/// non-transactional insert paths.  Accepts `&rusqlite::Connection` because
/// both `Connection` and `Transaction` deref to it.
fn insert_memory_inner(
    conn: &rusqlite::Connection,
    memory: &MemoryNode,
) -> Result<(), CodememError> {
    // Check dedup (namespace-scoped)
    let existing: Option<String> = conn
        .query_row(
            "SELECT id FROM memories WHERE content_hash = ?1 AND namespace IS ?2",
            params![memory.content_hash, memory.namespace],
            |row| row.get(0),
        )
        .optional()
        .storage_err()?;

    if existing.is_some() {
        return Err(CodememError::Duplicate(memory.content_hash.clone()));
    }

    let tags_json = serde_json::to_string(&memory.tags)?;
    let metadata_json = serde_json::to_string(&memory.metadata)?;

    conn.execute(
        "INSERT OR IGNORE INTO memories (id, content, memory_type, importance, confidence, access_count, content_hash, tags, metadata, namespace, session_id, repo, git_ref, expires_at, created_at, updated_at, last_accessed_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17)",
        params![
            memory.id,
            memory.content,
            memory.memory_type.to_string(),
            memory.importance,
            memory.confidence,
            memory.access_count,
            memory.content_hash,
            tags_json,
            metadata_json,
            memory.namespace,
            memory.session_id,
            memory.repo,
            memory.git_ref,
            memory.expires_at.map(|dt| dt.timestamp()),
            memory.created_at.timestamp(),
            memory.updated_at.timestamp(),
            memory.last_accessed_at.timestamp(),
        ],
    )
    .storage_err()?;

    Ok(())
}

impl Storage {
    /// Insert a new memory. Returns Err(Duplicate) if content hash already exists.
    ///
    /// Uses BEGIN IMMEDIATE to acquire a write lock before the dedup check,
    /// ensuring the SELECT + INSERT are atomic. INSERT OR IGNORE is a safety
    /// net against the UNIQUE constraint on content_hash.
    ///
    /// When an outer transaction is active (via `begin_transaction`), the
    /// method skips starting its own transaction and executes directly on the
    /// connection, so that all operations participate in the outer transaction.
    pub fn insert_memory(&self, memory: &MemoryNode) -> Result<(), CodememError> {
        // Check inside conn lock to avoid TOCTOU race with begin_transaction.
        // The conn() mutex serializes all access, so the flag check + INSERT
        // are atomic with respect to other callers.
        let mut conn = self.conn()?;
        if self.has_outer_transaction() {
            drop(conn);
            return self.insert_memory_no_tx(memory);
        }

        let tx = conn
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
            .storage_err()?;

        // Dedup check + INSERT inside the transaction; on Duplicate error the
        // transaction is rolled back automatically when `tx` is dropped.
        insert_memory_inner(&tx, memory)?;

        tx.commit().storage_err()?;

        Ok(())
    }

    /// Insert a memory without starting a new transaction.
    /// Used when an outer transaction is already active.
    fn insert_memory_no_tx(&self, memory: &MemoryNode) -> Result<(), CodememError> {
        let conn = self.conn()?;
        insert_memory_inner(&conn, memory)
    }

    /// Get a memory by ID. Updates access_count and last_accessed_at.
    ///
    /// Note: returns expired memories intentionally — direct ID lookups should
    /// always succeed for debugging and internal use. Expiry filtering happens
    /// in `list_memories_filtered` (bulk queries) and the opportunistic sweep
    /// in `delete_expired_memories` handles actual cleanup.
    pub fn get_memory(&self, id: &str) -> Result<Option<MemoryNode>, CodememError> {
        let conn = self.conn()?;

        // Bump access count first
        let updated = conn
            .execute(
                "UPDATE memories SET access_count = access_count + 1, last_accessed_at = ?1 WHERE id = ?2",
                params![chrono::Utc::now().timestamp(), id],
            )
            .storage_err()?;

        if updated == 0 {
            return Ok(None);
        }

        let result = conn
            .query_row(
                "SELECT id, content, memory_type, importance, confidence, access_count, content_hash, tags, metadata, namespace, session_id, repo, git_ref, expires_at, created_at, updated_at, last_accessed_at FROM memories WHERE id = ?1",
                params![id],
                MemoryRow::from_row,
            )
            .optional()
            .storage_err()?;

        match result {
            Some(row) => Ok(Some(row.into_memory_node()?)),
            None => Ok(None),
        }
    }

    /// Get a memory by ID without updating access_count or last_accessed_at.
    /// Use for internal/system reads (consolidation, stats, batch processing).
    pub fn get_memory_no_touch(&self, id: &str) -> Result<Option<MemoryNode>, CodememError> {
        let conn = self.conn()?;

        let result = conn
            .query_row(
                "SELECT id, content, memory_type, importance, confidence, access_count, content_hash, tags, metadata, namespace, session_id, repo, git_ref, expires_at, created_at, updated_at, last_accessed_at FROM memories WHERE id = ?1",
                params![id],
                MemoryRow::from_row,
            )
            .optional()
            .storage_err()?;

        match result {
            Some(row) => Ok(Some(row.into_memory_node()?)),
            None => Ok(None),
        }
    }

    /// Update a memory's content and re-hash. Returns `Err(NotFound)` if the ID doesn't exist.
    pub fn update_memory(
        &self,
        id: &str,
        content: &str,
        importance: Option<f64>,
    ) -> Result<(), CodememError> {
        let conn = self.conn()?;
        let hash = Self::content_hash(content);
        let now = chrono::Utc::now().timestamp();

        let rows_affected = if let Some(imp) = importance {
            conn.execute(
                "UPDATE memories SET content = ?1, content_hash = ?2, updated_at = ?3, importance = ?4 WHERE id = ?5",
                params![content, hash, now, imp, id],
            )
            .storage_err()?
        } else {
            conn.execute(
                "UPDATE memories SET content = ?1, content_hash = ?2, updated_at = ?3 WHERE id = ?4",
                params![content, hash, now, id],
            )
            .storage_err()?
        };

        if rows_affected == 0 {
            return Err(CodememError::NotFound(format!("Memory not found: {id}")));
        }

        Ok(())
    }

    /// Delete a memory by ID.
    pub fn delete_memory(&self, id: &str) -> Result<bool, CodememError> {
        let conn = self.conn()?;
        let rows = conn
            .execute("DELETE FROM memories WHERE id = ?1", params![id])
            .storage_err()?;
        Ok(rows > 0)
    }

    /// Delete a memory and all related data (embeddings, graph nodes/edges) atomically.
    /// Returns true if the memory existed and was deleted.
    pub fn delete_memory_cascade(&self, id: &str) -> Result<bool, CodememError> {
        let mut conn = self.conn()?;
        // L2: Use IMMEDIATE transaction to acquire write lock upfront,
        // avoiding potential deadlock with DEFERRED transaction.
        let tx = conn
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
            .storage_err()?;

        // Delete edges that reference graph nodes linked to this memory
        tx.execute(
            "DELETE FROM graph_edges WHERE src IN (SELECT id FROM graph_nodes WHERE memory_id = ?1)
             OR dst IN (SELECT id FROM graph_nodes WHERE memory_id = ?1)",
            params![id],
        )
        .storage_err()?;

        // Delete graph nodes linked to this memory
        tx.execute("DELETE FROM graph_nodes WHERE memory_id = ?1", params![id])
            .storage_err()?;

        // Delete embedding
        tx.execute(
            "DELETE FROM memory_embeddings WHERE memory_id = ?1",
            params![id],
        )
        .storage_err()?;

        // Delete the memory itself
        let rows = tx
            .execute("DELETE FROM memories WHERE id = ?1", params![id])
            .storage_err()?;

        tx.commit().storage_err()?;

        Ok(rows > 0)
    }

    /// Delete multiple memories and all related data (embeddings, graph nodes/edges) atomically.
    /// Returns the number of memories that were actually deleted.
    pub fn delete_memories_batch_cascade(&self, ids: &[&str]) -> Result<usize, CodememError> {
        if ids.is_empty() {
            return Ok(0);
        }

        let mut conn = self.conn()?;
        let tx = conn
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
            .storage_err()?;

        let placeholders: String = (1..=ids.len())
            .map(|i| format!("?{i}"))
            .collect::<Vec<_>>()
            .join(",");
        let params: Vec<&dyn rusqlite::types::ToSql> = ids
            .iter()
            .map(|id| id as &dyn rusqlite::types::ToSql)
            .collect();

        // Delete edges referencing graph nodes linked to these memories.
        // Uses ?N numbered params which SQLite allows to be reused in the same statement.
        let edge_sql = format!(
            "DELETE FROM graph_edges WHERE \
             src IN (SELECT id FROM graph_nodes WHERE memory_id IN ({placeholders})) \
             OR dst IN (SELECT id FROM graph_nodes WHERE memory_id IN ({placeholders})) \
             OR src IN ({placeholders}) OR dst IN ({placeholders})"
        );
        tx.execute(&edge_sql, params.as_slice()).storage_err()?;

        // Delete graph nodes linked to these memories (by memory_id column or by id)
        let node_sql = format!(
            "DELETE FROM graph_nodes WHERE memory_id IN ({placeholders}) OR id IN ({placeholders})"
        );
        tx.execute(&node_sql, params.as_slice()).storage_err()?;

        // Delete embeddings
        let emb_sql = format!("DELETE FROM memory_embeddings WHERE memory_id IN ({placeholders})");
        tx.execute(&emb_sql, params.as_slice()).storage_err()?;

        // Delete the memories themselves
        let mem_sql = format!("DELETE FROM memories WHERE id IN ({placeholders})");
        let deleted = tx.execute(&mem_sql, params.as_slice()).storage_err()?;

        tx.commit().storage_err()?;

        Ok(deleted)
    }

    /// List all memory IDs with an optional limit.
    pub fn list_memory_ids(&self) -> Result<Vec<String>, CodememError> {
        self.list_memory_ids_limited(None)
    }

    /// List memory IDs with an optional limit.
    pub fn list_memory_ids_limited(
        &self,
        limit: Option<usize>,
    ) -> Result<Vec<String>, CodememError> {
        let conn = self.conn()?;
        let (sql, params_vec): (&str, Vec<Box<dyn rusqlite::types::ToSql>>) =
            if let Some(lim) = limit {
                (
                    "SELECT id FROM memories ORDER BY created_at DESC LIMIT ?1",
                    vec![Box::new(lim as i64) as Box<dyn rusqlite::types::ToSql>],
                )
            } else {
                ("SELECT id FROM memories ORDER BY created_at DESC", vec![])
            };

        let mut stmt = conn.prepare(sql).storage_err()?;

        let refs: Vec<&dyn rusqlite::types::ToSql> =
            params_vec.iter().map(|p| p.as_ref()).collect();

        let ids = stmt
            .query_map(refs.as_slice(), |row| row.get(0))
            .storage_err()?
            .collect::<Result<Vec<String>, _>>()
            .storage_err()?;

        Ok(ids)
    }

    /// List memory IDs scoped to a specific namespace.
    pub fn list_memory_ids_for_namespace(
        &self,
        namespace: &str,
    ) -> Result<Vec<String>, CodememError> {
        let conn = self.conn()?;
        let mut stmt = conn
            .prepare("SELECT id FROM memories WHERE namespace = ?1 ORDER BY created_at DESC")
            .storage_err()?;

        let ids = stmt
            .query_map(params![namespace], |row| row.get(0))
            .storage_err()?
            .collect::<Result<Vec<String>, _>>()
            .storage_err()?;

        Ok(ids)
    }

    /// List all distinct namespaces.
    pub fn list_namespaces(&self) -> Result<Vec<String>, CodememError> {
        let conn = self.conn()?;
        let mut stmt = conn
            .prepare(
                "SELECT DISTINCT namespace FROM (
                    SELECT namespace FROM memories WHERE namespace IS NOT NULL
                    UNION
                    SELECT namespace FROM graph_nodes WHERE namespace IS NOT NULL
                ) ORDER BY namespace",
            )
            .storage_err()?;

        let namespaces = stmt
            .query_map([], |row| row.get(0))
            .storage_err()?
            .collect::<Result<Vec<String>, _>>()
            .storage_err()?;

        Ok(namespaces)
    }

    /// Get memory count.
    pub fn memory_count(&self) -> Result<usize, CodememError> {
        let conn = self.conn()?;
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM memories", [], |row| row.get(0))
            .storage_err()?;
        Ok(count as usize)
    }

    // ── Repository CRUD ─────────────────────────────────────────────────────

    /// List all registered repositories.
    pub fn list_repos(&self) -> Result<Vec<Repository>, CodememError> {
        let conn = self.conn()?;
        let mut stmt = conn
            .prepare(
                "SELECT id, path, name, namespace, created_at, last_indexed_at, status FROM repositories ORDER BY created_at DESC",
            )
            .storage_err()?;

        let repos = stmt
            .query_map([], |row| {
                let created_ts: String = row.get(4)?;
                let indexed_ts: Option<String> = row.get(5)?;
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, Option<String>>(2)?,
                    row.get::<_, Option<String>>(3)?,
                    created_ts,
                    indexed_ts,
                    row.get::<_, Option<String>>(6)?
                        .unwrap_or_else(|| "idle".to_string()),
                ))
            })
            .storage_err()?
            .collect::<Result<Vec<_>, _>>()
            .storage_err()?;

        let mut result = Vec::new();
        for (id, path, name, namespace, created_at, last_indexed_at, status) in repos {
            result.push(Repository {
                id,
                path,
                name,
                namespace,
                created_at,
                last_indexed_at,
                status,
            });
        }

        Ok(result)
    }

    /// Add a new repository.
    pub fn add_repo(&self, repo: &Repository) -> Result<(), CodememError> {
        let conn = self.conn()?;
        conn.execute(
            "INSERT INTO repositories (id, path, name, namespace, created_at, last_indexed_at, status) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                repo.id,
                repo.path,
                repo.name,
                repo.namespace,
                repo.created_at,
                repo.last_indexed_at,
                repo.status,
            ],
        )
        .storage_err()?;
        Ok(())
    }

    /// Remove a repository by ID.
    pub fn remove_repo(&self, id: &str) -> Result<bool, CodememError> {
        let conn = self.conn()?;
        let rows = conn
            .execute("DELETE FROM repositories WHERE id = ?1", params![id])
            .storage_err()?;
        Ok(rows > 0)
    }

    /// Get a repository by ID.
    pub fn get_repo(&self, id: &str) -> Result<Option<Repository>, CodememError> {
        let conn = self.conn()?;
        let result = conn
            .query_row(
                "SELECT id, path, name, namespace, created_at, last_indexed_at, status FROM repositories WHERE id = ?1",
                params![id],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, Option<String>>(2)?,
                        row.get::<_, Option<String>>(3)?,
                        row.get::<_, String>(4)?,
                        row.get::<_, Option<String>>(5)?,
                        row.get::<_, Option<String>>(6)?.unwrap_or_else(|| "idle".to_string()),
                    ))
                },
            )
            .optional()
            .storage_err()?;

        match result {
            Some((id, path, name, namespace, created_at, last_indexed_at, status)) => {
                Ok(Some(Repository {
                    id,
                    path,
                    name,
                    namespace,
                    created_at,
                    last_indexed_at,
                    status,
                }))
            }
            None => Ok(None),
        }
    }

    /// Update a repository's status and optionally last_indexed_at.
    pub fn update_repo_status(
        &self,
        id: &str,
        status: &str,
        indexed_at: Option<&str>,
    ) -> Result<(), CodememError> {
        let conn = self.conn()?;
        if let Some(ts) = indexed_at {
            conn.execute(
                "UPDATE repositories SET status = ?1, last_indexed_at = ?2 WHERE id = ?3",
                params![status, ts, id],
            )
            .storage_err()?;
        } else {
            conn.execute(
                "UPDATE repositories SET status = ?1 WHERE id = ?2",
                params![status, id],
            )
            .storage_err()?;
        }
        Ok(())
    }

    /// Delete all expired memories (where `expires_at <= now`).
    /// Returns the number of memories deleted.
    pub fn delete_expired_memories(&self) -> Result<usize, CodememError> {
        let conn = self.conn()?;
        let now = chrono::Utc::now().timestamp();

        // Collect IDs of expired memories first, then delete embeddings for exactly
        // those IDs (avoids O(all-embeddings) NOT IN subquery).
        let mut stmt = conn
            .prepare("SELECT id FROM memories WHERE expires_at IS NOT NULL AND expires_at <= ?1")
            .storage_err()?;
        let expired_ids: Vec<String> = stmt
            .query_map(params![now], |row| row.get(0))
            .storage_err()?
            .collect::<Result<Vec<String>, _>>()
            .storage_err()?;

        if expired_ids.is_empty() {
            return Ok(0);
        }

        // Batch in chunks of 999 to respect SQLite's parameter limit.
        let mut total_deleted = 0usize;
        for chunk in expired_ids.chunks(999) {
            let placeholders: String = (1..=chunk.len())
                .map(|i| format!("?{i}"))
                .collect::<Vec<_>>()
                .join(",");
            let params: Vec<&dyn rusqlite::types::ToSql> = chunk
                .iter()
                .map(|id| id as &dyn rusqlite::types::ToSql)
                .collect();

            let emb_sql =
                format!("DELETE FROM memory_embeddings WHERE memory_id IN ({placeholders})");
            conn.execute(&emb_sql, params.as_slice()).storage_err()?;

            let mem_sql = format!("DELETE FROM memories WHERE id IN ({placeholders})");
            total_deleted += conn.execute(&mem_sql, params.as_slice()).storage_err()?;
        }

        Ok(total_deleted)
    }

    /// Mark memories as expired for symbols in files whose content hash changed.
    /// Targets memories with `static-analysis` tag linked (via graph) to symbols
    /// in the given file — found via `graph_nodes.memory_id` (primary link) or
    /// `RELATES_TO` edges in `graph_edges` (secondary link for SCIP doc memories).
    /// Sets `expires_at` to now so they'll be cleaned up on next opportunistic sweep.
    pub fn expire_memories_for_file(&self, file_path: &str) -> Result<usize, CodememError> {
        let conn = self.conn()?;
        let now = chrono::Utc::now().timestamp();
        let expired = conn
            .execute(
                "UPDATE memories SET expires_at = ?1
                 WHERE expires_at IS NULL
                   AND id IN (SELECT m2.id FROM memories m2, json_each(m2.tags) jt
                              WHERE jt.value = 'static-analysis')
                   AND (
                     id IN (
                       SELECT gn.memory_id FROM graph_nodes gn
                       WHERE gn.memory_id IS NOT NULL
                         AND json_extract(gn.payload, '$.file_path') = ?2
                     )
                     OR id IN (
                       SELECT REPLACE(ge.dst, 'mem:', '')
                       FROM graph_edges ge
                       JOIN graph_nodes gn ON ge.src = gn.id
                       WHERE ge.relationship = 'RELATES_TO'
                         AND ge.dst LIKE 'mem:%'
                         AND json_extract(gn.payload, '$.file_path') = ?2
                     )
                   )",
                params![now, file_path],
            )
            .storage_err()?;
        Ok(expired)
    }
}

#[cfg(test)]
#[path = "tests/memory_tests.rs"]
mod tests;
