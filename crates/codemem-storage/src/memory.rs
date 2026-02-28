//! Memory CRUD operations on Storage.

use crate::{MemoryRow, Storage};
use codemem_core::{CodememError, MemoryNode};
use rusqlite::{params, OptionalExtension};

impl Storage {
    /// Insert a new memory. Returns Err(Duplicate) if content hash already exists.
    pub fn insert_memory(&self, memory: &MemoryNode) -> Result<(), CodememError> {
        let conn = self.conn();

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
        let conn = self.conn();

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
        let conn = self.conn();
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
        let conn = self.conn();
        let rows = conn
            .execute("DELETE FROM memories WHERE id = ?1", params![id])
            .map_err(|e| CodememError::Storage(e.to_string()))?;
        Ok(rows > 0)
    }

    /// List all memory IDs.
    pub fn list_memory_ids(&self) -> Result<Vec<String>, CodememError> {
        let conn = self.conn();
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
        let conn = self.conn();
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
        let conn = self.conn();
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
        let conn = self.conn();
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM memories", [], |row| row.get(0))
            .map_err(|e| CodememError::Storage(e.to_string()))?;
        Ok(count as usize)
    }
}

#[cfg(test)]
mod tests {
    use crate::Storage;
    use codemem_core::{CodememError, MemoryNode, MemoryType};
    use std::collections::HashMap;

    fn test_memory() -> MemoryNode {
        let now = chrono::Utc::now();
        let content = "Test memory content";
        MemoryNode {
            id: uuid::Uuid::new_v4().to_string(),
            content: content.to_string(),
            memory_type: MemoryType::Context,
            importance: 0.7,
            confidence: 1.0,
            access_count: 0,
            content_hash: Storage::content_hash(content),
            tags: vec!["test".to_string()],
            metadata: HashMap::new(),
            namespace: None,
            created_at: now,
            updated_at: now,
            last_accessed_at: now,
        }
    }

    #[test]
    fn insert_and_get_memory() {
        let storage = Storage::open_in_memory().unwrap();
        let memory = test_memory();
        storage.insert_memory(&memory).unwrap();

        let retrieved = storage.get_memory(&memory.id).unwrap().unwrap();
        assert_eq!(retrieved.id, memory.id);
        assert_eq!(retrieved.content, memory.content);
        assert_eq!(retrieved.access_count, 1); // bumped on get
    }

    #[test]
    fn dedup_by_content_hash() {
        let storage = Storage::open_in_memory().unwrap();
        let m1 = test_memory();
        storage.insert_memory(&m1).unwrap();

        let mut m2 = test_memory();
        m2.id = uuid::Uuid::new_v4().to_string();
        m2.content_hash = m1.content_hash.clone(); // same hash

        assert!(matches!(
            storage.insert_memory(&m2),
            Err(CodememError::Duplicate(_))
        ));
    }

    #[test]
    fn delete_memory() {
        let storage = Storage::open_in_memory().unwrap();
        let memory = test_memory();
        storage.insert_memory(&memory).unwrap();
        assert!(storage.delete_memory(&memory.id).unwrap());
        assert!(storage.get_memory(&memory.id).unwrap().is_none());
    }
}
