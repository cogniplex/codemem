//! Memory CRUD operations on Storage.

use crate::{MemoryRow, Storage};
use codemem_core::{CodememError, MemoryNode, Repository};
use rusqlite::{params, OptionalExtension};

impl Storage {
    /// Insert a new memory. Returns Err(Duplicate) if content hash already exists.
    pub fn insert_memory(&self, memory: &MemoryNode) -> Result<(), CodememError> {
        let conn = self.conn()?;

        // Check dedup
        let existing: Option<String> = conn
            .query_row(
                "SELECT id FROM memories WHERE content_hash = ?1",
                params![memory.content_hash],
                |row| row.get(0),
            )
            .optional()
            .map_err(|e| CodememError::Storage(e.to_string()))?;

        if let Some(_existing_id) = existing {
            return Err(CodememError::Duplicate(memory.content_hash.clone()));
        }

        let tags_json = serde_json::to_string(&memory.tags)?;
        let metadata_json = serde_json::to_string(&memory.metadata)?;

        conn.execute(
            "INSERT INTO memories (id, content, memory_type, importance, confidence, access_count, content_hash, tags, metadata, namespace, created_at, updated_at, last_accessed_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
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
                memory.created_at.timestamp(),
                memory.updated_at.timestamp(),
                memory.last_accessed_at.timestamp(),
            ],
        )
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
                "SELECT id, content, memory_type, importance, confidence, access_count, content_hash, tags, metadata, namespace, created_at, updated_at, last_accessed_at FROM memories WHERE id = ?1",
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
                        created_at: row.get(10)?,
                        updated_at: row.get(11)?,
                        last_accessed_at: row.get(12)?,
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

    /// Update a memory's content and re-hash.
    pub fn update_memory(
        &self,
        id: &str,
        content: &str,
        importance: Option<f64>,
    ) -> Result<(), CodememError> {
        let conn = self.conn()?;
        let hash = Self::content_hash(content);
        let now = chrono::Utc::now().timestamp();

        if let Some(imp) = importance {
            conn.execute(
                "UPDATE memories SET content = ?1, content_hash = ?2, updated_at = ?3, importance = ?4 WHERE id = ?5",
                params![content, hash, now, imp, id],
            )
            .map_err(|e| CodememError::Storage(e.to_string()))?;
        } else {
            conn.execute(
                "UPDATE memories SET content = ?1, content_hash = ?2, updated_at = ?3 WHERE id = ?4",
                params![content, hash, now, id],
            )
            .map_err(|e| CodememError::Storage(e.to_string()))?;
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

    /// List all memory IDs.
    pub fn list_memory_ids(&self) -> Result<Vec<String>, CodememError> {
        let conn = self.conn()?;
        let mut stmt = conn
            .prepare("SELECT id FROM memories ORDER BY created_at DESC")
            .map_err(|e| CodememError::Storage(e.to_string()))?;

        let ids = stmt
            .query_map([], |row| row.get(0))
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
                Ok(Repository {
                    id: row.get(0)?,
                    path: row.get(1)?,
                    name: row.get(2)?,
                    namespace: row.get(3)?,
                    created_at: row.get(4)?,
                    last_indexed_at: row.get(5)?,
                    status: row
                        .get::<_, Option<String>>(6)?
                        .unwrap_or_else(|| "idle".to_string()),
                })
            })
            .map_err(|e| CodememError::Storage(e.to_string()))?
            .collect::<Result<Vec<Repository>, _>>()
            .map_err(|e| CodememError::Storage(e.to_string()))?;

        Ok(repos)
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
                    Ok(Repository {
                        id: row.get(0)?,
                        path: row.get(1)?,
                        name: row.get(2)?,
                        namespace: row.get(3)?,
                        created_at: row.get(4)?,
                        last_indexed_at: row.get(5)?,
                        status: row.get::<_, Option<String>>(6)?.unwrap_or_else(|| "idle".to_string()),
                    })
                },
            )
            .optional()
            .map_err(|e| CodememError::Storage(e.to_string()))?;
        Ok(result)
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
