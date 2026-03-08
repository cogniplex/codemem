//! Memory CRUD operations on Storage.

use crate::{MemoryRow, Storage};
use codemem_core::{CodememError, MemoryNode, Repository};
use rusqlite::{params, OptionalExtension};

impl Storage {
    /// Insert a new memory. Returns Err(Duplicate) if content hash already exists.
    ///
    /// Uses BEGIN IMMEDIATE to acquire a write lock before the dedup check,
    /// ensuring the SELECT + INSERT are atomic. INSERT OR IGNORE is a safety
    /// net against the UNIQUE constraint on content_hash.
    pub fn insert_memory(&self, memory: &MemoryNode) -> Result<(), CodememError> {
        let mut conn = self.conn()?;

        let tx = conn
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
            .map_err(|e| CodememError::Storage(e.to_string()))?;

        // Check dedup inside the transaction (namespace-scoped)
        let existing: Option<String> = tx
            .query_row(
                "SELECT id FROM memories WHERE content_hash = ?1 AND namespace IS ?2",
                params![memory.content_hash, memory.namespace],
                |row| row.get(0),
            )
            .optional()
            .map_err(|e| CodememError::Storage(e.to_string()))?;

        if existing.is_some() {
            tx.rollback()
                .map_err(|e| CodememError::Storage(e.to_string()))?;
            return Err(CodememError::Duplicate(memory.content_hash.clone()));
        }

        let tags_json = serde_json::to_string(&memory.tags)?;
        let metadata_json = serde_json::to_string(&memory.metadata)?;

        tx.execute(
            "INSERT OR IGNORE INTO memories (id, content, memory_type, importance, confidence, access_count, content_hash, tags, metadata, namespace, session_id, created_at, updated_at, last_accessed_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)",
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
                memory.created_at.timestamp(),
                memory.updated_at.timestamp(),
                memory.last_accessed_at.timestamp(),
            ],
        )
        .map_err(|e| CodememError::Storage(e.to_string()))?;

        tx.commit()
            .map_err(|e| CodememError::Storage(e.to_string()))?;

        Ok(())
    }

    /// Get a memory by ID. Updates access_count and last_accessed_at.
    pub fn get_memory(&self, id: &str) -> Result<Option<MemoryNode>, CodememError> {
        let conn = self.conn()?;

        // Bump access count first
        let updated = conn
            .execute(
                "UPDATE memories SET access_count = access_count + 1, last_accessed_at = ?1 WHERE id = ?2",
                params![chrono::Utc::now().timestamp(), id],
            )
            .map_err(|e| CodememError::Storage(e.to_string()))?;

        if updated == 0 {
            return Ok(None);
        }

        let result = conn
            .query_row(
                "SELECT id, content, memory_type, importance, confidence, access_count, content_hash, tags, metadata, namespace, session_id, created_at, updated_at, last_accessed_at FROM memories WHERE id = ?1",
                params![id],
                |row| {
                    Ok(MemoryRow {
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
                },
            )
            .optional()
            .map_err(|e| CodememError::Storage(e.to_string()))?;

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
                "SELECT id, content, memory_type, importance, confidence, access_count, content_hash, tags, metadata, namespace, session_id, created_at, updated_at, last_accessed_at FROM memories WHERE id = ?1",
                params![id],
                |row| {
                    Ok(MemoryRow {
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
                },
            )
            .optional()
            .map_err(|e| CodememError::Storage(e.to_string()))?;

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
            .map_err(|e| CodememError::Storage(e.to_string()))?
        } else {
            conn.execute(
                "UPDATE memories SET content = ?1, content_hash = ?2, updated_at = ?3 WHERE id = ?4",
                params![content, hash, now, id],
            )
            .map_err(|e| CodememError::Storage(e.to_string()))?
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
            .map_err(|e| CodememError::Storage(e.to_string()))?;
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
            .map_err(|e| CodememError::Storage(e.to_string()))?;

        // Delete edges that reference graph nodes linked to this memory
        tx.execute(
            "DELETE FROM graph_edges WHERE src IN (SELECT id FROM graph_nodes WHERE memory_id = ?1)
             OR dst IN (SELECT id FROM graph_nodes WHERE memory_id = ?1)",
            params![id],
        )
        .map_err(|e| CodememError::Storage(e.to_string()))?;

        // Delete graph nodes linked to this memory
        tx.execute("DELETE FROM graph_nodes WHERE memory_id = ?1", params![id])
            .map_err(|e| CodememError::Storage(e.to_string()))?;

        // Delete embedding
        tx.execute(
            "DELETE FROM memory_embeddings WHERE memory_id = ?1",
            params![id],
        )
        .map_err(|e| CodememError::Storage(e.to_string()))?;

        // Delete the memory itself
        let rows = tx
            .execute("DELETE FROM memories WHERE id = ?1", params![id])
            .map_err(|e| CodememError::Storage(e.to_string()))?;

        tx.commit()
            .map_err(|e| CodememError::Storage(e.to_string()))?;

        Ok(rows > 0)
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

        let mut stmt = conn
            .prepare(sql)
            .map_err(|e| CodememError::Storage(e.to_string()))?;

        let refs: Vec<&dyn rusqlite::types::ToSql> =
            params_vec.iter().map(|p| p.as_ref()).collect();

        let ids = stmt
            .query_map(refs.as_slice(), |row| row.get(0))
            .map_err(|e| CodememError::Storage(e.to_string()))?
            .collect::<Result<Vec<String>, _>>()
            .map_err(|e| CodememError::Storage(e.to_string()))?;

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
            .map_err(|e| CodememError::Storage(e.to_string()))?;

        let ids = stmt
            .query_map(params![namespace], |row| row.get(0))
            .map_err(|e| CodememError::Storage(e.to_string()))?
            .collect::<Result<Vec<String>, _>>()
            .map_err(|e| CodememError::Storage(e.to_string()))?;

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
            .map_err(|e| CodememError::Storage(e.to_string()))?;

        let namespaces = stmt
            .query_map([], |row| row.get(0))
            .map_err(|e| CodememError::Storage(e.to_string()))?
            .collect::<Result<Vec<String>, _>>()
            .map_err(|e| CodememError::Storage(e.to_string()))?;

        Ok(namespaces)
    }

    /// Get memory count.
    pub fn memory_count(&self) -> Result<usize, CodememError> {
        let conn = self.conn()?;
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM memories", [], |row| row.get(0))
            .map_err(|e| CodememError::Storage(e.to_string()))?;
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
            .map_err(|e| CodememError::Storage(e.to_string()))?;

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
            .map_err(|e| CodememError::Storage(e.to_string()))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| CodememError::Storage(e.to_string()))?;

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
        .map_err(|e| CodememError::Storage(e.to_string()))?;
        Ok(())
    }

    /// Remove a repository by ID.
    pub fn remove_repo(&self, id: &str) -> Result<bool, CodememError> {
        let conn = self.conn()?;
        let rows = conn
            .execute("DELETE FROM repositories WHERE id = ?1", params![id])
            .map_err(|e| CodememError::Storage(e.to_string()))?;
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
            .map_err(|e| CodememError::Storage(e.to_string()))?;

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
            .map_err(|e| CodememError::Storage(e.to_string()))?;
        } else {
            conn.execute(
                "UPDATE repositories SET status = ?1 WHERE id = ?2",
                params![status, id],
            )
            .map_err(|e| CodememError::Storage(e.to_string()))?;
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "tests/memory_tests.rs"]
mod tests;
