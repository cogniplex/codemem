//! `StorageBackend` trait implementation for Storage.

use crate::{MemoryRow, Storage};
use codemem_core::{
    CodememError, ConsolidationLogEntry, Edge, GraphNode, MemoryNode, NodeKind, Session,
    StorageBackend, StorageStats,
};
use rusqlite::params;
use std::collections::HashMap;

impl StorageBackend for Storage {
    fn insert_memory(&self, memory: &MemoryNode) -> Result<(), CodememError> {
        Storage::insert_memory(self, memory)
    }

    fn get_memory(&self, id: &str) -> Result<Option<MemoryNode>, CodememError> {
        Storage::get_memory(self, id)
    }

    fn get_memories_batch(&self, ids: &[&str]) -> Result<Vec<MemoryNode>, CodememError> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let conn = self.conn();

        let placeholders: Vec<String> = (1..=ids.len()).map(|i| format!("?{i}")).collect();
        let sql = format!(
            "SELECT id, content, memory_type, importance, confidence, access_count, content_hash, tags, metadata, namespace, created_at, updated_at, last_accessed_at FROM memories WHERE id IN ({})",
            placeholders.join(",")
        );

        let mut stmt = conn
            .prepare(&sql)
            .map_err(|e| CodememError::Storage(e.to_string()))?;

        let params: Vec<&dyn rusqlite::types::ToSql> = ids
            .iter()
            .map(|id| id as &dyn rusqlite::types::ToSql)
            .collect();

        let rows = stmt
            .query_map(params.as_slice(), |row| {
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
            })
            .map_err(|e| CodememError::Storage(e.to_string()))?;

        let mut memories = Vec::new();
        for row in rows {
            let row = row.map_err(|e| CodememError::Storage(e.to_string()))?;
            memories.push(row.into_memory_node()?);
        }
        Ok(memories)
    }

    fn update_memory(
        &self,
        id: &str,
        content: &str,
        importance: Option<f64>,
    ) -> Result<(), CodememError> {
        Storage::update_memory(self, id, content, importance)
    }

    fn delete_memory(&self, id: &str) -> Result<bool, CodememError> {
        Storage::delete_memory(self, id)
    }

    fn list_memory_ids(&self) -> Result<Vec<String>, CodememError> {
        Storage::list_memory_ids(self)
    }

    fn list_memory_ids_for_namespace(&self, namespace: &str) -> Result<Vec<String>, CodememError> {
        Storage::list_memory_ids_for_namespace(self, namespace)
    }

    fn list_namespaces(&self) -> Result<Vec<String>, CodememError> {
        Storage::list_namespaces(self)
    }

    fn memory_count(&self) -> Result<usize, CodememError> {
        Storage::memory_count(self)
    }

    fn store_embedding(&self, memory_id: &str, embedding: &[f32]) -> Result<(), CodememError> {
        Storage::store_embedding(self, memory_id, embedding)
    }

    fn get_embedding(&self, memory_id: &str) -> Result<Option<Vec<f32>>, CodememError> {
        Storage::get_embedding(self, memory_id)
    }

    fn delete_embedding(&self, memory_id: &str) -> Result<bool, CodememError> {
        let conn = self.conn();
        let deleted = conn
            .execute(
                "DELETE FROM memory_embeddings WHERE memory_id = ?1",
                [memory_id],
            )
            .map_err(|e| CodememError::Storage(e.to_string()))?;
        Ok(deleted > 0)
    }

    fn list_all_embeddings(&self) -> Result<Vec<(String, Vec<f32>)>, CodememError> {
        let conn = self.conn();
        let mut stmt = conn
            .prepare("SELECT memory_id, embedding FROM memory_embeddings")
            .map_err(|e| CodememError::Storage(e.to_string()))?;
        let rows = stmt
            .query_map([], |row| {
                let id: String = row.get(0)?;
                let blob: Vec<u8> = row.get(1)?;
                Ok((id, blob))
            })
            .map_err(|e| CodememError::Storage(e.to_string()))?;
        let mut result = Vec::new();
        for row in rows {
            let (id, blob) = row.map_err(|e| CodememError::Storage(e.to_string()))?;
            let floats: Vec<f32> = blob
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();
            result.push((id, floats));
        }
        Ok(result)
    }

    fn insert_graph_node(&self, node: &GraphNode) -> Result<(), CodememError> {
        Storage::insert_graph_node(self, node)
    }

    fn get_graph_node(&self, id: &str) -> Result<Option<GraphNode>, CodememError> {
        Storage::get_graph_node(self, id)
    }

    fn delete_graph_node(&self, id: &str) -> Result<bool, CodememError> {
        Storage::delete_graph_node(self, id)
    }

    fn all_graph_nodes(&self) -> Result<Vec<GraphNode>, CodememError> {
        Storage::all_graph_nodes(self)
    }

    fn insert_graph_edge(&self, edge: &Edge) -> Result<(), CodememError> {
        Storage::insert_graph_edge(self, edge)
    }

    fn get_edges_for_node(&self, node_id: &str) -> Result<Vec<Edge>, CodememError> {
        Storage::get_edges_for_node(self, node_id)
    }

    fn all_graph_edges(&self) -> Result<Vec<Edge>, CodememError> {
        Storage::all_graph_edges(self)
    }

    fn delete_graph_edges_for_node(&self, node_id: &str) -> Result<usize, CodememError> {
        Storage::delete_graph_edges_for_node(self, node_id)
    }

    fn start_session(&self, id: &str, namespace: Option<&str>) -> Result<(), CodememError> {
        Storage::start_session(self, id, namespace)
    }

    fn end_session(&self, id: &str, summary: Option<&str>) -> Result<(), CodememError> {
        Storage::end_session(self, id, summary)
    }

    fn list_sessions(
        &self,
        namespace: Option<&str>,
        limit: usize,
    ) -> Result<Vec<Session>, CodememError> {
        self.list_sessions_with_limit(namespace, limit)
    }

    fn insert_consolidation_log(
        &self,
        cycle_type: &str,
        affected_count: usize,
    ) -> Result<(), CodememError> {
        Storage::insert_consolidation_log(self, cycle_type, affected_count)
    }

    fn last_consolidation_runs(&self) -> Result<Vec<ConsolidationLogEntry>, CodememError> {
        Storage::last_consolidation_runs(self)
    }

    fn get_repeated_searches(
        &self,
        min_count: usize,
        namespace: Option<&str>,
    ) -> Result<Vec<(String, usize, Vec<String>)>, CodememError> {
        Storage::get_repeated_searches(self, min_count, namespace)
    }

    fn get_file_hotspots(
        &self,
        min_count: usize,
        namespace: Option<&str>,
    ) -> Result<Vec<(String, usize, Vec<String>)>, CodememError> {
        Storage::get_file_hotspots(self, min_count, namespace)
    }

    fn get_tool_usage_stats(
        &self,
        namespace: Option<&str>,
    ) -> Result<Vec<(String, usize)>, CodememError> {
        let map = Storage::get_tool_usage_stats(self, namespace)?;
        let mut vec: Vec<(String, usize)> = map.into_iter().collect();
        vec.sort_by(|a, b| b.1.cmp(&a.1));
        Ok(vec)
    }

    fn get_decision_chains(
        &self,
        min_count: usize,
        namespace: Option<&str>,
    ) -> Result<Vec<(String, usize, Vec<String>)>, CodememError> {
        Storage::get_decision_chains(self, min_count, namespace)
    }

    fn decay_stale_memories(
        &self,
        threshold_ts: i64,
        decay_factor: f64,
    ) -> Result<usize, CodememError> {
        let conn = self.conn();
        let rows = conn
            .execute(
                "UPDATE memories SET importance = importance * ?1 WHERE last_accessed_at < ?2",
                params![decay_factor, threshold_ts],
            )
            .map_err(|e| CodememError::Storage(e.to_string()))?;
        Ok(rows)
    }

    fn list_memories_for_creative(
        &self,
    ) -> Result<Vec<(String, String, Vec<String>)>, CodememError> {
        let conn = self.conn();
        let mut stmt = conn
            .prepare("SELECT id, memory_type, tags FROM memories ORDER BY created_at DESC")
            .map_err(|e| CodememError::Storage(e.to_string()))?;

        let rows = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                ))
            })
            .map_err(|e| CodememError::Storage(e.to_string()))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| CodememError::Storage(e.to_string()))?;

        Ok(rows
            .into_iter()
            .map(|(id, mtype, tags_json)| {
                let tags: Vec<String> = serde_json::from_str(&tags_json).unwrap_or_default();
                (id, mtype, tags)
            })
            .collect())
    }

    fn find_cluster_duplicates(&self) -> Result<Vec<(String, String, f64)>, CodememError> {
        let conn = self.conn();
        let mut stmt = conn
            .prepare(
                "SELECT a.id, b.id, 1.0 as similarity
                 FROM memories a
                 INNER JOIN memories b ON substr(a.content_hash, 1, 16) = substr(b.content_hash, 1, 16)
                 WHERE a.id < b.id",
            )
            .map_err(|e| CodememError::Storage(e.to_string()))?;

        let rows = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, f64>(2)?,
                ))
            })
            .map_err(|e| CodememError::Storage(e.to_string()))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| CodememError::Storage(e.to_string()))?;

        Ok(rows)
    }

    fn find_forgettable(&self, importance_threshold: f64) -> Result<Vec<String>, CodememError> {
        let conn = self.conn();
        let mut stmt = conn
            .prepare(
                "SELECT id FROM memories WHERE importance < ?1 AND access_count = 0 ORDER BY importance ASC, last_accessed_at ASC",
            )
            .map_err(|e| CodememError::Storage(e.to_string()))?;

        let ids = stmt
            .query_map(params![importance_threshold], |row| row.get(0))
            .map_err(|e| CodememError::Storage(e.to_string()))?
            .collect::<Result<Vec<String>, _>>()
            .map_err(|e| CodememError::Storage(e.to_string()))?;

        Ok(ids)
    }

    fn insert_memories_batch(&self, memories: &[MemoryNode]) -> Result<(), CodememError> {
        let conn = self.conn();
        let tx = conn
            .unchecked_transaction()
            .map_err(|e| CodememError::Storage(e.to_string()))?;

        for memory in memories {
            let tags_json = serde_json::to_string(&memory.tags)?;
            let metadata_json = serde_json::to_string(&memory.metadata)?;

            tx.execute(
                "INSERT OR IGNORE INTO memories (id, content, memory_type, importance, confidence, access_count, content_hash, tags, metadata, namespace, created_at, updated_at, last_accessed_at)
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
        }

        tx.commit()
            .map_err(|e| CodememError::Storage(e.to_string()))?;
        Ok(())
    }

    fn store_embeddings_batch(&self, items: &[(&str, &[f32])]) -> Result<(), CodememError> {
        let conn = self.conn();
        let tx = conn
            .unchecked_transaction()
            .map_err(|e| CodememError::Storage(e.to_string()))?;

        for (id, embedding) in items {
            let blob: Vec<u8> = embedding.iter().flat_map(|f| f.to_le_bytes()).collect();
            tx.execute(
                "INSERT OR REPLACE INTO memory_embeddings (memory_id, embedding) VALUES (?1, ?2)",
                params![id, blob],
            )
            .map_err(|e| CodememError::Storage(e.to_string()))?;
        }

        tx.commit()
            .map_err(|e| CodememError::Storage(e.to_string()))?;
        Ok(())
    }

    fn load_file_hashes(&self) -> Result<HashMap<String, String>, CodememError> {
        let conn = self.conn();
        let mut stmt = conn
            .prepare("SELECT file_path, content_hash FROM file_hashes")
            .map_err(|e| CodememError::Storage(e.to_string()))?;

        let rows = stmt
            .query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })
            .map_err(|e| CodememError::Storage(e.to_string()))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| CodememError::Storage(e.to_string()))?;

        Ok(rows.into_iter().collect())
    }

    fn save_file_hashes(&self, hashes: &HashMap<String, String>) -> Result<(), CodememError> {
        let conn = self.conn();
        let tx = conn
            .unchecked_transaction()
            .map_err(|e| CodememError::Storage(e.to_string()))?;

        tx.execute("DELETE FROM file_hashes", [])
            .map_err(|e| CodememError::Storage(e.to_string()))?;

        for (path, hash) in hashes {
            tx.execute(
                "INSERT INTO file_hashes (file_path, content_hash) VALUES (?1, ?2)",
                params![path, hash],
            )
            .map_err(|e| CodememError::Storage(e.to_string()))?;
        }

        tx.commit()
            .map_err(|e| CodememError::Storage(e.to_string()))?;
        Ok(())
    }

    fn insert_graph_nodes_batch(&self, nodes: &[GraphNode]) -> Result<(), CodememError> {
        let conn = self.conn();
        let tx = conn
            .unchecked_transaction()
            .map_err(|e| CodememError::Storage(e.to_string()))?;

        for node in nodes {
            let payload_json =
                serde_json::to_string(&node.payload).unwrap_or_else(|_| "{}".to_string());
            tx.execute(
                "INSERT OR REPLACE INTO graph_nodes (id, kind, label, payload, centrality, memory_id, namespace)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                params![
                    node.id,
                    node.kind.to_string(),
                    node.label,
                    payload_json,
                    node.centrality,
                    node.memory_id,
                    node.namespace,
                ],
            )
            .map_err(|e| CodememError::Storage(e.to_string()))?;
        }

        tx.commit()
            .map_err(|e| CodememError::Storage(e.to_string()))?;
        Ok(())
    }

    fn insert_graph_edges_batch(&self, edges: &[Edge]) -> Result<(), CodememError> {
        let conn = self.conn();
        let tx = conn
            .unchecked_transaction()
            .map_err(|e| CodememError::Storage(e.to_string()))?;

        for edge in edges {
            let props_json =
                serde_json::to_string(&edge.properties).unwrap_or_else(|_| "{}".to_string());
            tx.execute(
                "INSERT OR REPLACE INTO graph_edges (id, src, dst, relationship, weight, properties, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                params![
                    edge.id,
                    edge.src,
                    edge.dst,
                    edge.relationship.to_string(),
                    edge.weight,
                    props_json,
                    edge.created_at.timestamp(),
                ],
            )
            .map_err(|e| CodememError::Storage(e.to_string()))?;
        }

        tx.commit()
            .map_err(|e| CodememError::Storage(e.to_string()))?;
        Ok(())
    }

    fn find_unembedded_memories(&self) -> Result<Vec<(String, String)>, CodememError> {
        let conn = self.conn();
        let mut stmt = conn
            .prepare(
                "SELECT m.id, m.content FROM memories m
                 LEFT JOIN memory_embeddings me ON m.id = me.memory_id
                 WHERE me.memory_id IS NULL",
            )
            .map_err(|e| CodememError::Storage(e.to_string()))?;

        let rows = stmt
            .query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })
            .map_err(|e| CodememError::Storage(e.to_string()))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| CodememError::Storage(e.to_string()))?;

        Ok(rows)
    }

    fn search_graph_nodes(
        &self,
        query: &str,
        namespace: Option<&str>,
        limit: usize,
    ) -> Result<Vec<GraphNode>, CodememError> {
        let conn = self.conn();
        let pattern = format!("%{}%", query.to_lowercase());

        let (sql, params_vec): (String, Vec<Box<dyn rusqlite::types::ToSql>>) =
            if let Some(ns) = namespace {
                (
                    "SELECT id, kind, label, payload, centrality, memory_id, namespace \
                 FROM graph_nodes WHERE LOWER(label) LIKE ?1 AND namespace = ?2 \
                 ORDER BY centrality DESC LIMIT ?3"
                        .to_string(),
                    vec![
                        Box::new(pattern) as Box<dyn rusqlite::types::ToSql>,
                        Box::new(ns.to_string()),
                        Box::new(limit as i64),
                    ],
                )
            } else {
                (
                    "SELECT id, kind, label, payload, centrality, memory_id, namespace \
                 FROM graph_nodes WHERE LOWER(label) LIKE ?1 \
                 ORDER BY centrality DESC LIMIT ?2"
                        .to_string(),
                    vec![
                        Box::new(pattern) as Box<dyn rusqlite::types::ToSql>,
                        Box::new(limit as i64),
                    ],
                )
            };

        let refs: Vec<&dyn rusqlite::types::ToSql> =
            params_vec.iter().map(|p| p.as_ref()).collect();
        let mut stmt = conn
            .prepare(&sql)
            .map_err(|e| CodememError::Storage(e.to_string()))?;

        let rows = stmt
            .query_map(refs.as_slice(), |row| {
                let kind_str: String = row.get(1)?;
                let payload_str: String = row.get(3)?;
                Ok(GraphNode {
                    id: row.get(0)?,
                    kind: kind_str.parse().unwrap_or(NodeKind::Memory),
                    label: row.get(2)?,
                    payload: serde_json::from_str(&payload_str).unwrap_or_default(),
                    centrality: row.get(4)?,
                    memory_id: row.get(5)?,
                    namespace: row.get(6)?,
                })
            })
            .map_err(|e| CodememError::Storage(e.to_string()))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| CodememError::Storage(e.to_string()))?;

        Ok(rows)
    }

    fn list_memories_filtered(
        &self,
        namespace: Option<&str>,
        memory_type: Option<&str>,
    ) -> Result<Vec<MemoryNode>, CodememError> {
        let conn = self.conn();
        let mut sql = "SELECT id, content, memory_type, importance, confidence, access_count, \
                        content_hash, tags, metadata, namespace, created_at, updated_at, \
                        last_accessed_at FROM memories WHERE 1=1"
            .to_string();
        let mut param_values: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();

        if let Some(ns) = namespace {
            param_values.push(Box::new(ns.to_string()));
            sql.push_str(&format!(" AND namespace = ?{}", param_values.len()));
        }
        if let Some(mt) = memory_type {
            param_values.push(Box::new(mt.to_string()));
            sql.push_str(&format!(" AND memory_type = ?{}", param_values.len()));
        }
        sql.push_str(" ORDER BY created_at DESC");

        let refs: Vec<&dyn rusqlite::types::ToSql> =
            param_values.iter().map(|p| p.as_ref()).collect();
        let mut stmt = conn
            .prepare(&sql)
            .map_err(|e| CodememError::Storage(e.to_string()))?;

        let rows = stmt
            .query_map(refs.as_slice(), |row| {
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
            })
            .map_err(|e| CodememError::Storage(e.to_string()))?;

        let mut result = Vec::new();
        for row in rows {
            let mr = row.map_err(|e| CodememError::Storage(e.to_string()))?;
            result.push(mr.into_memory_node()?);
        }

        Ok(result)
    }

    fn graph_edges_for_namespace(&self, namespace: &str) -> Result<Vec<Edge>, CodememError> {
        Storage::graph_edges_for_namespace(self, namespace)
    }

    fn stats(&self) -> Result<StorageStats, CodememError> {
        Storage::stats(self)
    }
}

#[cfg(test)]
mod tests {
    use crate::Storage;
    use codemem_core::{MemoryNode, MemoryType, StorageBackend};
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
    fn get_memories_batch_returns_multiple() {
        let storage = Storage::open_in_memory().unwrap();
        let m1 = test_memory();
        let mut m2 = test_memory();
        m2.id = uuid::Uuid::new_v4().to_string();
        m2.content = "Different content".to_string();
        m2.content_hash = Storage::content_hash(&m2.content);

        storage.insert_memory(&m1).unwrap();
        storage.insert_memory(&m2).unwrap();

        let backend: &dyn StorageBackend = &storage;
        let batch = backend.get_memories_batch(&[&m1.id, &m2.id]).unwrap();
        assert_eq!(batch.len(), 2);
    }

    #[test]
    fn get_memories_batch_empty() {
        let storage = Storage::open_in_memory().unwrap();
        let backend: &dyn StorageBackend = &storage;
        let batch = backend.get_memories_batch(&[]).unwrap();
        assert!(batch.is_empty());
    }

    #[test]
    fn storage_backend_trait_object() {
        let storage = Storage::open_in_memory().unwrap();
        let backend: Box<dyn StorageBackend> = Box::new(storage);

        let m = test_memory();
        backend.insert_memory(&m).unwrap();
        let retrieved = backend.get_memory(&m.id).unwrap().unwrap();
        assert_eq!(retrieved.id, m.id);
    }

    #[test]
    fn file_hashes_roundtrip() {
        let storage = Storage::open_in_memory().unwrap();
        let backend: &dyn StorageBackend = &storage;

        let mut hashes = HashMap::new();
        hashes.insert("src/main.rs".to_string(), "abc123".to_string());
        hashes.insert("src/lib.rs".to_string(), "def456".to_string());

        backend.save_file_hashes(&hashes).unwrap();
        let loaded = backend.load_file_hashes().unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded.get("src/main.rs"), Some(&"abc123".to_string()));
    }

    #[test]
    fn decay_stale_memories_updates() {
        let storage = Storage::open_in_memory().unwrap();
        let backend: &dyn StorageBackend = &storage;

        let m = test_memory();
        backend.insert_memory(&m).unwrap();

        // Decay memories older than far future = none affected
        let count = backend.decay_stale_memories(0, 0.5).unwrap();
        assert_eq!(count, 0);

        // Decay all memories (threshold in the future)
        let count = backend.decay_stale_memories(i64::MAX, 0.5).unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn find_forgettable_returns_low_importance() {
        let storage = Storage::open_in_memory().unwrap();
        let backend: &dyn StorageBackend = &storage;

        let mut m = test_memory();
        m.importance = 0.1;
        backend.insert_memory(&m).unwrap();

        let forgettable = backend.find_forgettable(0.5).unwrap();
        assert_eq!(forgettable.len(), 1);
        assert_eq!(forgettable[0], m.id);

        let forgettable = backend.find_forgettable(0.05).unwrap();
        assert!(forgettable.is_empty());
    }
}
