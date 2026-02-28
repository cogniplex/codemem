//! Graph node/edge CRUD and embedding storage on Storage.

use crate::Storage;
use codemem_core::{CodememError, Edge, GraphNode, NodeKind, RelationshipType};
use rusqlite::{params, OptionalExtension};
use std::collections::HashMap;

impl Storage {
    // ── Embedding Storage ───────────────────────────────────────────────

    /// Store an embedding for a memory.
    pub fn store_embedding(&self, memory_id: &str, embedding: &[f32]) -> Result<(), CodememError> {
        let conn = self.conn();
        let blob: Vec<u8> = embedding.iter().flat_map(|f| f.to_le_bytes()).collect();

        conn.execute(
            "INSERT OR REPLACE INTO memory_embeddings (memory_id, embedding) VALUES (?1, ?2)",
            params![memory_id, blob],
        )
        .map_err(|e| CodememError::Storage(e.to_string()))?;

        Ok(())
    }

    /// Get an embedding by memory ID.
    pub fn get_embedding(&self, memory_id: &str) -> Result<Option<Vec<f32>>, CodememError> {
        let conn = self.conn();
        let blob: Option<Vec<u8>> = conn
            .query_row(
                "SELECT embedding FROM memory_embeddings WHERE memory_id = ?1",
                params![memory_id],
                |row| row.get(0),
            )
            .optional()
            .map_err(|e| CodememError::Storage(e.to_string()))?;

        match blob {
            Some(bytes) => {
                let floats: Vec<f32> = bytes
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                Ok(Some(floats))
            }
            None => Ok(None),
        }
    }

    // ── Graph Node Storage ──────────────────────────────────────────────

    /// Insert a graph node.
    pub fn insert_graph_node(&self, node: &GraphNode) -> Result<(), CodememError> {
        let conn = self.conn();
        let payload_json = serde_json::to_string(&node.payload)?;

        conn.execute(
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

        Ok(())
    }

    /// Get a graph node by ID.
    pub fn get_graph_node(&self, id: &str) -> Result<Option<GraphNode>, CodememError> {
        let conn = self.conn();
        conn.query_row(
            "SELECT id, kind, label, payload, centrality, memory_id, namespace FROM graph_nodes WHERE id = ?1",
            params![id],
            |row| {
                let kind_str: String = row.get(1)?;
                let payload_str: String = row.get(3)?;
                Ok((
                    row.get::<_, String>(0)?,
                    kind_str,
                    row.get::<_, String>(2)?,
                    payload_str,
                    row.get::<_, f64>(4)?,
                    row.get::<_, Option<String>>(5)?,
                    row.get::<_, Option<String>>(6)?,
                ))
            },
        )
        .optional()
        .map_err(|e| CodememError::Storage(e.to_string()))?
        .map(|(id, kind_str, label, payload_str, centrality, memory_id, namespace)| {
            let kind: NodeKind = kind_str.parse().map_err(|e: CodememError| CodememError::Storage(e.to_string()))?;
            let payload: HashMap<String, serde_json::Value> =
                serde_json::from_str(&payload_str).unwrap_or_default();
            Ok(GraphNode {
                id,
                kind,
                label,
                payload,
                centrality,
                memory_id,
                namespace,
            })
        })
        .transpose()
    }

    /// Delete a graph node by ID.
    pub fn delete_graph_node(&self, id: &str) -> Result<bool, CodememError> {
        let conn = self.conn();
        let rows = conn
            .execute("DELETE FROM graph_nodes WHERE id = ?1", params![id])
            .map_err(|e| CodememError::Storage(e.to_string()))?;
        Ok(rows > 0)
    }

    /// Get all graph nodes.
    pub fn all_graph_nodes(&self) -> Result<Vec<GraphNode>, CodememError> {
        let conn = self.conn();
        let mut stmt = conn
            .prepare("SELECT id, kind, label, payload, centrality, memory_id, namespace FROM graph_nodes")
            .map_err(|e| CodememError::Storage(e.to_string()))?;

        let nodes = stmt
            .query_map([], |row| {
                let kind_str: String = row.get(1)?;
                let payload_str: String = row.get(3)?;
                Ok((
                    row.get::<_, String>(0)?,
                    kind_str,
                    row.get::<_, String>(2)?,
                    payload_str,
                    row.get::<_, f64>(4)?,
                    row.get::<_, Option<String>>(5)?,
                    row.get::<_, Option<String>>(6)?,
                ))
            })
            .map_err(|e| CodememError::Storage(e.to_string()))?
            .filter_map(|r| r.ok())
            .filter_map(
                |(id, kind_str, label, payload_str, centrality, memory_id, namespace)| {
                    let kind: NodeKind = kind_str.parse().ok()?;
                    let payload: HashMap<String, serde_json::Value> =
                        serde_json::from_str(&payload_str).unwrap_or_default();
                    Some(GraphNode {
                        id,
                        kind,
                        label,
                        payload,
                        centrality,
                        memory_id,
                        namespace,
                    })
                },
            )
            .collect();

        Ok(nodes)
    }

    // ── Graph Edge Storage ──────────────────────────────────────────────

    /// Insert a graph edge.
    pub fn insert_graph_edge(&self, edge: &Edge) -> Result<(), CodememError> {
        let conn = self.conn();
        let props_json = serde_json::to_string(&edge.properties)?;

        conn.execute(
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

        Ok(())
    }

    /// Get all edges from or to a node.
    pub fn get_edges_for_node(&self, node_id: &str) -> Result<Vec<Edge>, CodememError> {
        let conn = self.conn();
        let mut stmt = conn
            .prepare(
                "SELECT id, src, dst, relationship, weight, properties, created_at FROM graph_edges WHERE src = ?1 OR dst = ?1",
            )
            .map_err(|e| CodememError::Storage(e.to_string()))?;

        let edges = stmt
            .query_map(params![node_id], |row| {
                let rel_str: String = row.get(3)?;
                let props_str: String = row.get(5)?;
                let created_ts: i64 = row.get(6)?;
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    rel_str,
                    row.get::<_, f64>(4)?,
                    props_str,
                    created_ts,
                ))
            })
            .map_err(|e| CodememError::Storage(e.to_string()))?
            .filter_map(|r| r.ok())
            .filter_map(|(id, src, dst, rel_str, weight, props_str, created_ts)| {
                let relationship: RelationshipType = rel_str.parse().ok()?;
                let properties: HashMap<String, serde_json::Value> =
                    serde_json::from_str(&props_str).unwrap_or_default();
                let created_at =
                    chrono::DateTime::from_timestamp(created_ts, 0)?.with_timezone(&chrono::Utc);
                Some(Edge {
                    id,
                    src,
                    dst,
                    relationship,
                    weight,
                    properties,
                    created_at,
                })
            })
            .collect();

        Ok(edges)
    }

    /// Get all graph edges.
    pub fn all_graph_edges(&self) -> Result<Vec<Edge>, CodememError> {
        let conn = self.conn();
        let mut stmt = conn
            .prepare("SELECT id, src, dst, relationship, weight, properties, created_at FROM graph_edges")
            .map_err(|e| CodememError::Storage(e.to_string()))?;

        let edges = stmt
            .query_map([], |row| {
                let rel_str: String = row.get(3)?;
                let props_str: String = row.get(5)?;
                let created_ts: i64 = row.get(6)?;
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    rel_str,
                    row.get::<_, f64>(4)?,
                    props_str,
                    created_ts,
                ))
            })
            .map_err(|e| CodememError::Storage(e.to_string()))?
            .filter_map(|r| r.ok())
            .filter_map(|(id, src, dst, rel_str, weight, props_str, created_ts)| {
                let relationship: RelationshipType = rel_str.parse().ok()?;
                let properties: HashMap<String, serde_json::Value> =
                    serde_json::from_str(&props_str).unwrap_or_default();
                let created_at =
                    chrono::DateTime::from_timestamp(created_ts, 0)?.with_timezone(&chrono::Utc);
                Some(Edge {
                    id,
                    src,
                    dst,
                    relationship,
                    weight,
                    properties,
                    created_at,
                })
            })
            .collect();

        Ok(edges)
    }

    /// Delete all graph edges connected to a node (as src or dst).
    pub fn delete_graph_edges_for_node(&self, node_id: &str) -> Result<usize, CodememError> {
        let conn = self.conn();
        let rows = conn
            .execute(
                "DELETE FROM graph_edges WHERE src = ?1 OR dst = ?1",
                params![node_id],
            )
            .map_err(|e| CodememError::Storage(e.to_string()))?;
        Ok(rows)
    }

    /// Get all graph edges where both src and dst nodes belong to the given namespace.
    pub fn graph_edges_for_namespace(&self, namespace: &str) -> Result<Vec<Edge>, CodememError> {
        let conn = self.conn();
        let mut stmt = conn
            .prepare(
                "SELECT e.id, e.src, e.dst, e.relationship, e.weight, e.properties, e.created_at
                 FROM graph_edges e
                 INNER JOIN graph_nodes gs ON e.src = gs.id
                 INNER JOIN graph_nodes gd ON e.dst = gd.id
                 WHERE gs.namespace = ?1 AND gd.namespace = ?1",
            )
            .map_err(|e| CodememError::Storage(e.to_string()))?;

        let edges = stmt
            .query_map(params![namespace], |row| {
                let rel_str: String = row.get(3)?;
                let props_str: String = row.get(5)?;
                let created_ts: i64 = row.get(6)?;
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    rel_str,
                    row.get::<_, f64>(4)?,
                    props_str,
                    created_ts,
                ))
            })
            .map_err(|e| CodememError::Storage(e.to_string()))?
            .filter_map(|r| r.ok())
            .filter_map(|(id, src, dst, rel_str, weight, props_str, created_ts)| {
                let relationship: RelationshipType = rel_str.parse().ok()?;
                let properties: HashMap<String, serde_json::Value> =
                    serde_json::from_str(&props_str).unwrap_or_default();
                let created_at =
                    chrono::DateTime::from_timestamp(created_ts, 0)?.with_timezone(&chrono::Utc);
                Some(Edge {
                    id,
                    src,
                    dst,
                    relationship,
                    weight,
                    properties,
                    created_at,
                })
            })
            .collect();

        Ok(edges)
    }

    /// Delete a graph edge by ID.
    pub fn delete_graph_edge(&self, id: &str) -> Result<bool, CodememError> {
        let conn = self.conn();
        let rows = conn
            .execute("DELETE FROM graph_edges WHERE id = ?1", params![id])
            .map_err(|e| CodememError::Storage(e.to_string()))?;
        Ok(rows > 0)
    }
}

#[cfg(test)]
mod tests {
    use crate::Storage;
    use codemem_core::{GraphNode, MemoryNode, MemoryType, NodeKind};
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
    fn store_and_get_embedding() {
        let storage = Storage::open_in_memory().unwrap();
        let memory = test_memory();
        storage.insert_memory(&memory).unwrap();

        let embedding: Vec<f32> = (0..768).map(|i| i as f32 / 768.0).collect();
        storage.store_embedding(&memory.id, &embedding).unwrap();

        let retrieved = storage.get_embedding(&memory.id).unwrap().unwrap();
        assert_eq!(retrieved.len(), 768);
        assert!((retrieved[0] - embedding[0]).abs() < f32::EPSILON);
    }

    #[test]
    fn graph_node_crud() {
        let storage = Storage::open_in_memory().unwrap();
        let node = GraphNode {
            id: "file:src/main.rs".to_string(),
            kind: NodeKind::File,
            label: "src/main.rs".to_string(),
            payload: HashMap::new(),
            centrality: 0.0,
            memory_id: None,
            namespace: None,
        };

        storage.insert_graph_node(&node).unwrap();
        let retrieved = storage.get_graph_node(&node.id).unwrap().unwrap();
        assert_eq!(retrieved.kind, NodeKind::File);
        assert!(storage.delete_graph_node(&node.id).unwrap());
    }
}
