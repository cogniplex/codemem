//! codemem-storage: SQLite persistence layer for Codemem.
//!
//! Uses rusqlite with bundled SQLite, WAL mode, and embedded schema.

use codemem_core::{
    CodememError, Edge, GraphNode, MemoryNode, MemoryType, NodeKind, RelationshipType,
};
use rusqlite::{params, Connection, OptionalExtension};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::Path;

const SCHEMA: &str = include_str!("schema.sql");

/// SQLite-backed storage for Codemem memories, embeddings, and graph data.
pub struct Storage {
    conn: Connection,
}

impl Storage {
    /// Open (or create) an Codemem database at the given path.
    pub fn open(path: &Path) -> Result<Self, CodememError> {
        let conn = Connection::open(path).map_err(|e| CodememError::Storage(e.to_string()))?;

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

        // Apply schema
        conn.execute_batch(SCHEMA)
            .map_err(|e| CodememError::Storage(e.to_string()))?;

        Ok(Self { conn })
    }

    /// Open an in-memory database (for testing).
    pub fn open_in_memory() -> Result<Self, CodememError> {
        let conn =
            Connection::open_in_memory().map_err(|e| CodememError::Storage(e.to_string()))?;
        conn.pragma_update(None, "foreign_keys", "ON")
            .map_err(|e| CodememError::Storage(e.to_string()))?;
        conn.execute_batch(SCHEMA)
            .map_err(|e| CodememError::Storage(e.to_string()))?;
        Ok(Self { conn })
    }

    /// Compute SHA-256 hash of content for deduplication.
    pub fn content_hash(content: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    // ── Memory CRUD ─────────────────────────────────────────────────────

    /// Insert a new memory. Returns Err(Duplicate) if content hash already exists.
    pub fn insert_memory(&self, memory: &MemoryNode) -> Result<(), CodememError> {
        // Check dedup
        let existing: Option<String> = self
            .conn
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

        self.conn
            .execute(
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
        // Bump access count first
        let updated = self
            .conn
            .execute(
                "UPDATE memories SET access_count = access_count + 1, last_accessed_at = ?1 WHERE id = ?2",
                params![chrono::Utc::now().timestamp(), id],
            )
            .map_err(|e| CodememError::Storage(e.to_string()))?;

        if updated == 0 {
            return Ok(None);
        }

        let result = self
            .conn
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
        let hash = Self::content_hash(content);
        let now = chrono::Utc::now().timestamp();

        let mut sql =
            "UPDATE memories SET content = ?1, content_hash = ?2, updated_at = ?3".to_string();
        if importance.is_some() {
            sql.push_str(", importance = ?4");
        }
        sql.push_str(" WHERE id = ?5");

        if let Some(imp) = importance {
            self.conn
                .execute(&sql, params![content, hash, now, imp, id])
                .map_err(|e| CodememError::Storage(e.to_string()))?;
        } else {
            // Re-bind without importance param
            self.conn
                .execute(
                    "UPDATE memories SET content = ?1, content_hash = ?2, updated_at = ?3 WHERE id = ?4",
                    params![content, hash, now, id],
                )
                .map_err(|e| CodememError::Storage(e.to_string()))?;
        }

        Ok(())
    }

    /// Delete a memory by ID.
    pub fn delete_memory(&self, id: &str) -> Result<bool, CodememError> {
        let rows = self
            .conn
            .execute("DELETE FROM memories WHERE id = ?1", params![id])
            .map_err(|e| CodememError::Storage(e.to_string()))?;
        Ok(rows > 0)
    }

    /// List all memory IDs.
    pub fn list_memory_ids(&self) -> Result<Vec<String>, CodememError> {
        let mut stmt = self
            .conn
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
        let mut stmt = self
            .conn
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
        let mut stmt = self
            .conn
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
        let count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM memories", [], |row| row.get(0))
            .map_err(|e| CodememError::Storage(e.to_string()))?;
        Ok(count as usize)
    }

    // ── Embedding Storage ───────────────────────────────────────────────

    /// Store an embedding for a memory.
    pub fn store_embedding(&self, memory_id: &str, embedding: &[f32]) -> Result<(), CodememError> {
        let blob: Vec<u8> = embedding.iter().flat_map(|f| f.to_le_bytes()).collect();

        self.conn
            .execute(
                "INSERT OR REPLACE INTO memory_embeddings (memory_id, embedding) VALUES (?1, ?2)",
                params![memory_id, blob],
            )
            .map_err(|e| CodememError::Storage(e.to_string()))?;

        Ok(())
    }

    /// Get an embedding by memory ID.
    pub fn get_embedding(&self, memory_id: &str) -> Result<Option<Vec<f32>>, CodememError> {
        let blob: Option<Vec<u8>> = self
            .conn
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
        let payload_json = serde_json::to_string(&node.payload)?;

        self.conn
            .execute(
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
        self.conn
            .query_row(
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
        let rows = self
            .conn
            .execute("DELETE FROM graph_nodes WHERE id = ?1", params![id])
            .map_err(|e| CodememError::Storage(e.to_string()))?;
        Ok(rows > 0)
    }

    /// Get all graph nodes.
    pub fn all_graph_nodes(&self) -> Result<Vec<GraphNode>, CodememError> {
        let mut stmt = self
            .conn
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
        let props_json = serde_json::to_string(&edge.properties)?;

        self.conn
            .execute(
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
        let mut stmt = self
            .conn
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
        let mut stmt = self
            .conn
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
        let rows = self
            .conn
            .execute(
                "DELETE FROM graph_edges WHERE src = ?1 OR dst = ?1",
                params![node_id],
            )
            .map_err(|e| CodememError::Storage(e.to_string()))?;
        Ok(rows)
    }

    /// Get all graph edges where both src and dst nodes belong to the given namespace.
    pub fn graph_edges_for_namespace(&self, namespace: &str) -> Result<Vec<Edge>, CodememError> {
        let mut stmt = self
            .conn
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
        let rows = self
            .conn
            .execute("DELETE FROM graph_edges WHERE id = ?1", params![id])
            .map_err(|e| CodememError::Storage(e.to_string()))?;
        Ok(rows > 0)
    }

    // ── Stats ───────────────────────────────────────────────────────────

    /// Get database statistics.
    pub fn stats(&self) -> Result<StorageStats, CodememError> {
        let memory_count = self.memory_count()?;

        let embedding_count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM memory_embeddings", [], |row| {
                row.get(0)
            })
            .map_err(|e| CodememError::Storage(e.to_string()))?;

        let node_count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM graph_nodes", [], |row| row.get(0))
            .map_err(|e| CodememError::Storage(e.to_string()))?;

        let edge_count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM graph_edges", [], |row| row.get(0))
            .map_err(|e| CodememError::Storage(e.to_string()))?;

        Ok(StorageStats {
            memory_count,
            embedding_count: embedding_count as usize,
            node_count: node_count as usize,
            edge_count: edge_count as usize,
        })
    }

    // ── Consolidation Log ──────────────────────────────────────────────

    /// Record a consolidation run.
    pub fn insert_consolidation_log(
        &self,
        cycle_type: &str,
        affected_count: usize,
    ) -> Result<(), CodememError> {
        let now = chrono::Utc::now().timestamp();
        self.conn
            .execute(
                "INSERT INTO consolidation_log (cycle_type, run_at, affected_count) VALUES (?1, ?2, ?3)",
                params![cycle_type, now, affected_count as i64],
            )
            .map_err(|e| CodememError::Storage(e.to_string()))?;
        Ok(())
    }

    /// Get the last consolidation run for each cycle type.
    pub fn last_consolidation_runs(&self) -> Result<Vec<ConsolidationLogEntry>, CodememError> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT cycle_type, run_at, affected_count FROM consolidation_log
                 WHERE id IN (
                     SELECT id FROM consolidation_log c2
                     WHERE c2.cycle_type = consolidation_log.cycle_type
                     ORDER BY run_at DESC LIMIT 1
                 )
                 GROUP BY cycle_type
                 ORDER BY cycle_type",
            )
            .map_err(|e| CodememError::Storage(e.to_string()))?;

        let entries = stmt
            .query_map([], |row| {
                Ok(ConsolidationLogEntry {
                    cycle_type: row.get(0)?,
                    run_at: row.get(1)?,
                    affected_count: row.get::<_, i64>(2)? as usize,
                })
            })
            .map_err(|e| CodememError::Storage(e.to_string()))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| CodememError::Storage(e.to_string()))?;

        Ok(entries)
    }

    /// Get a reference to the underlying connection (for advanced use).
    pub fn connection(&self) -> &Connection {
        &self.conn
    }

    // ── Pattern Detection Queries ───────────────────────────────────────

    /// Find repeated search patterns (Grep/Glob) by extracting the "pattern" field
    /// from memory metadata JSON. Returns (pattern, count, memory_ids) tuples where
    /// count >= min_count, ordered by count descending.
    pub fn get_repeated_searches(
        &self,
        min_count: usize,
        namespace: Option<&str>,
    ) -> Result<Vec<(String, usize, Vec<String>)>, CodememError> {
        // Use json_extract to pull the "pattern" field from the metadata JSON column.
        // Filter to memories whose metadata contains a "tool" of "Grep" or "Glob".
        let sql = if namespace.is_some() {
            "SELECT json_extract(metadata, '$.pattern') AS pat,
                    COUNT(*) AS cnt,
                    GROUP_CONCAT(id, ',') AS ids
             FROM memories
             WHERE json_extract(metadata, '$.tool') IN ('Grep', 'Glob')
               AND pat IS NOT NULL
               AND namespace = ?1
             GROUP BY pat
             HAVING cnt >= ?2
             ORDER BY cnt DESC"
        } else {
            "SELECT json_extract(metadata, '$.pattern') AS pat,
                    COUNT(*) AS cnt,
                    GROUP_CONCAT(id, ',') AS ids
             FROM memories
             WHERE json_extract(metadata, '$.tool') IN ('Grep', 'Glob')
               AND pat IS NOT NULL
             GROUP BY pat
             HAVING cnt >= ?1
             ORDER BY cnt DESC"
        };

        let mut stmt = self
            .conn
            .prepare(sql)
            .map_err(|e| CodememError::Storage(e.to_string()))?;

        let rows = if let Some(ns) = namespace {
            stmt.query_map(params![ns, min_count as i64], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, String>(2)?,
                ))
            })
            .map_err(|e| CodememError::Storage(e.to_string()))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| CodememError::Storage(e.to_string()))?
        } else {
            stmt.query_map(params![min_count as i64], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, String>(2)?,
                ))
            })
            .map_err(|e| CodememError::Storage(e.to_string()))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| CodememError::Storage(e.to_string()))?
        };

        Ok(rows
            .into_iter()
            .map(|(pat, cnt, ids_str)| {
                let ids: Vec<String> = ids_str.split(',').map(String::from).collect();
                (pat, cnt as usize, ids)
            })
            .collect())
    }

    /// Find file hotspots by extracting the "file_path" field from memory metadata.
    /// Returns (file_path, count, memory_ids) tuples where count >= min_count,
    /// ordered by count descending.
    pub fn get_file_hotspots(
        &self,
        min_count: usize,
        namespace: Option<&str>,
    ) -> Result<Vec<(String, usize, Vec<String>)>, CodememError> {
        let sql = if namespace.is_some() {
            "SELECT json_extract(metadata, '$.file_path') AS fp,
                    COUNT(*) AS cnt,
                    GROUP_CONCAT(id, ',') AS ids
             FROM memories
             WHERE fp IS NOT NULL
               AND namespace = ?1
             GROUP BY fp
             HAVING cnt >= ?2
             ORDER BY cnt DESC"
        } else {
            "SELECT json_extract(metadata, '$.file_path') AS fp,
                    COUNT(*) AS cnt,
                    GROUP_CONCAT(id, ',') AS ids
             FROM memories
             WHERE fp IS NOT NULL
             GROUP BY fp
             HAVING cnt >= ?1
             ORDER BY cnt DESC"
        };

        let mut stmt = self
            .conn
            .prepare(sql)
            .map_err(|e| CodememError::Storage(e.to_string()))?;

        let rows = if let Some(ns) = namespace {
            stmt.query_map(params![ns, min_count as i64], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, String>(2)?,
                ))
            })
            .map_err(|e| CodememError::Storage(e.to_string()))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| CodememError::Storage(e.to_string()))?
        } else {
            stmt.query_map(params![min_count as i64], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, String>(2)?,
                ))
            })
            .map_err(|e| CodememError::Storage(e.to_string()))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| CodememError::Storage(e.to_string()))?
        };

        Ok(rows
            .into_iter()
            .map(|(fp, cnt, ids_str)| {
                let ids: Vec<String> = ids_str.split(',').map(String::from).collect();
                (fp, cnt as usize, ids)
            })
            .collect())
    }

    /// Get tool usage statistics from memory metadata.
    /// Returns a map of tool_name -> count, ordered by count descending.
    pub fn get_tool_usage_stats(
        &self,
        namespace: Option<&str>,
    ) -> Result<HashMap<String, usize>, CodememError> {
        let sql = if namespace.is_some() {
            "SELECT json_extract(metadata, '$.tool') AS tool,
                    COUNT(*) AS cnt
             FROM memories
             WHERE tool IS NOT NULL
               AND namespace = ?1
             GROUP BY tool
             ORDER BY cnt DESC"
        } else {
            "SELECT json_extract(metadata, '$.tool') AS tool,
                    COUNT(*) AS cnt
             FROM memories
             WHERE tool IS NOT NULL
             GROUP BY tool
             ORDER BY cnt DESC"
        };

        let mut stmt = self
            .conn
            .prepare(sql)
            .map_err(|e| CodememError::Storage(e.to_string()))?;

        let rows = if let Some(ns) = namespace {
            stmt.query_map(params![ns], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
            })
            .map_err(|e| CodememError::Storage(e.to_string()))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| CodememError::Storage(e.to_string()))?
        } else {
            stmt.query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
            })
            .map_err(|e| CodememError::Storage(e.to_string()))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| CodememError::Storage(e.to_string()))?
        };

        Ok(rows
            .into_iter()
            .map(|(tool, cnt)| (tool, cnt as usize))
            .collect())
    }

    /// Find decision chains: files with multiple Edit/Write memories over time.
    /// Returns (file_path, count, memory_ids) tuples ordered by count descending.
    pub fn get_decision_chains(
        &self,
        min_count: usize,
        namespace: Option<&str>,
    ) -> Result<Vec<(String, usize, Vec<String>)>, CodememError> {
        let sql = if namespace.is_some() {
            "SELECT json_extract(metadata, '$.file_path') AS fp,
                    COUNT(*) AS cnt,
                    GROUP_CONCAT(id, ',') AS ids
             FROM memories
             WHERE json_extract(metadata, '$.tool') IN ('Edit', 'Write')
               AND fp IS NOT NULL
               AND namespace = ?1
             GROUP BY fp
             HAVING cnt >= ?2
             ORDER BY cnt DESC"
        } else {
            "SELECT json_extract(metadata, '$.file_path') AS fp,
                    COUNT(*) AS cnt,
                    GROUP_CONCAT(id, ',') AS ids
             FROM memories
             WHERE json_extract(metadata, '$.tool') IN ('Edit', 'Write')
               AND fp IS NOT NULL
             GROUP BY fp
             HAVING cnt >= ?1
             ORDER BY cnt DESC"
        };

        let mut stmt = self
            .conn
            .prepare(sql)
            .map_err(|e| CodememError::Storage(e.to_string()))?;

        let rows = if let Some(ns) = namespace {
            stmt.query_map(params![ns, min_count as i64], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, String>(2)?,
                ))
            })
            .map_err(|e| CodememError::Storage(e.to_string()))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| CodememError::Storage(e.to_string()))?
        } else {
            stmt.query_map(params![min_count as i64], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, String>(2)?,
                ))
            })
            .map_err(|e| CodememError::Storage(e.to_string()))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| CodememError::Storage(e.to_string()))?
        };

        Ok(rows
            .into_iter()
            .map(|(fp, cnt, ids_str)| {
                let ids: Vec<String> = ids_str.split(',').map(String::from).collect();
                (fp, cnt as usize, ids)
            })
            .collect())
    }

    // ── Session Management ─────────────────────────────────────────────

    /// Ensure session_id column exists on memories table.
    pub fn ensure_session_column(&self) -> Result<(), CodememError> {
        // Check if column exists by attempting a query that references it
        let has_col: bool = self
            .conn
            .prepare("SELECT session_id FROM memories LIMIT 0")
            .is_ok();
        if !has_col {
            self.conn
                .execute_batch("ALTER TABLE memories ADD COLUMN session_id TEXT;")
                .map_err(|e| CodememError::Storage(e.to_string()))?;
        }
        Ok(())
    }

    /// Start a new session. Inserts a row into the sessions table.
    pub fn start_session(&self, id: &str, namespace: Option<&str>) -> Result<(), CodememError> {
        let now = chrono::Utc::now().timestamp();
        self.conn
            .execute(
                "INSERT OR IGNORE INTO sessions (id, namespace, started_at) VALUES (?1, ?2, ?3)",
                params![id, namespace, now],
            )
            .map_err(|e| CodememError::Storage(e.to_string()))?;
        Ok(())
    }

    /// End a session by setting ended_at and optionally a summary.
    pub fn end_session(&self, id: &str, summary: Option<&str>) -> Result<(), CodememError> {
        let now = chrono::Utc::now().timestamp();
        self.conn
            .execute(
                "UPDATE sessions SET ended_at = ?1, summary = ?2 WHERE id = ?3",
                params![now, summary, id],
            )
            .map_err(|e| CodememError::Storage(e.to_string()))?;
        Ok(())
    }

    /// List sessions, optionally filtered by namespace.
    pub fn list_sessions(
        &self,
        namespace: Option<&str>,
    ) -> Result<Vec<codemem_core::Session>, CodememError> {
        let sql_with_ns = "SELECT id, namespace, started_at, ended_at, memory_count, summary FROM sessions WHERE namespace = ?1 ORDER BY started_at DESC";
        let sql_all = "SELECT id, namespace, started_at, ended_at, memory_count, summary FROM sessions ORDER BY started_at DESC";

        let map_row = |row: &rusqlite::Row<'_>| -> rusqlite::Result<codemem_core::Session> {
            let started_ts: i64 = row.get(2)?;
            let ended_ts: Option<i64> = row.get(3)?;
            Ok(codemem_core::Session {
                id: row.get(0)?,
                namespace: row.get(1)?,
                started_at: chrono::DateTime::from_timestamp(started_ts, 0)
                    .unwrap_or_default()
                    .with_timezone(&chrono::Utc),
                ended_at: ended_ts.and_then(|ts| {
                    chrono::DateTime::from_timestamp(ts, 0).map(|dt| dt.with_timezone(&chrono::Utc))
                }),
                memory_count: row.get::<_, i64>(4).unwrap_or(0) as u32,
                summary: row.get(5)?,
            })
        };

        if let Some(ns) = namespace {
            let mut stmt = self
                .conn
                .prepare(sql_with_ns)
                .map_err(|e| CodememError::Storage(e.to_string()))?;
            let rows = stmt
                .query_map(params![ns], map_row)
                .map_err(|e| CodememError::Storage(e.to_string()))?;
            rows.collect::<Result<Vec<_>, _>>()
                .map_err(|e| CodememError::Storage(e.to_string()))
        } else {
            let mut stmt = self
                .conn
                .prepare(sql_all)
                .map_err(|e| CodememError::Storage(e.to_string()))?;
            let rows = stmt
                .query_map([], map_row)
                .map_err(|e| CodememError::Storage(e.to_string()))?;
            rows.collect::<Result<Vec<_>, _>>()
                .map_err(|e| CodememError::Storage(e.to_string()))
        }
    }
}

/// Database statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageStats {
    pub memory_count: usize,
    pub embedding_count: usize,
    pub node_count: usize,
    pub edge_count: usize,
}

/// A single consolidation log entry.
#[derive(Debug, Clone)]
pub struct ConsolidationLogEntry {
    pub cycle_type: String,
    pub run_at: i64,
    pub affected_count: usize,
}

use serde::{Deserialize, Serialize};

/// Internal row struct for memory deserialization.
struct MemoryRow {
    id: String,
    content: String,
    memory_type: String,
    importance: f64,
    confidence: f64,
    access_count: i64,
    content_hash: String,
    tags: String,
    metadata: String,
    namespace: Option<String>,
    created_at: i64,
    updated_at: i64,
    last_accessed_at: i64,
}

impl MemoryRow {
    fn into_memory_node(self) -> Result<MemoryNode, CodememError> {
        let memory_type: MemoryType = self.memory_type.parse()?;
        let tags: Vec<String> = serde_json::from_str(&self.tags).unwrap_or_default();
        let metadata: HashMap<String, serde_json::Value> =
            serde_json::from_str(&self.metadata).unwrap_or_default();

        let created_at = chrono::DateTime::from_timestamp(self.created_at, 0)
            .unwrap_or_default()
            .with_timezone(&chrono::Utc);
        let updated_at = chrono::DateTime::from_timestamp(self.updated_at, 0)
            .unwrap_or_default()
            .with_timezone(&chrono::Utc);
        let last_accessed_at = chrono::DateTime::from_timestamp(self.last_accessed_at, 0)
            .unwrap_or_default()
            .with_timezone(&chrono::Utc);

        Ok(MemoryNode {
            id: self.id,
            content: self.content,
            memory_type,
            importance: self.importance,
            confidence: self.confidence,
            access_count: self.access_count as u32,
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

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn test_memory() -> MemoryNode {
        let now = Utc::now();
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

    fn test_memory_with_metadata(
        content: &str,
        tool: &str,
        extra: HashMap<String, serde_json::Value>,
    ) -> MemoryNode {
        let now = Utc::now();
        let mut metadata = extra;
        metadata.insert(
            "tool".to_string(),
            serde_json::Value::String(tool.to_string()),
        );
        MemoryNode {
            id: uuid::Uuid::new_v4().to_string(),
            content: content.to_string(),
            memory_type: MemoryType::Context,
            importance: 0.5,
            confidence: 1.0,
            access_count: 0,
            content_hash: Storage::content_hash(content),
            tags: vec![],
            metadata,
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

    #[test]
    fn stats() {
        let storage = Storage::open_in_memory().unwrap();
        let stats = storage.stats().unwrap();
        assert_eq!(stats.memory_count, 0);
    }

    // ── Pattern Detection Query Tests ───────────────────────────────────

    #[test]
    fn get_repeated_searches_groups_by_pattern() {
        let storage = Storage::open_in_memory().unwrap();

        // Insert 3 Grep memories with pattern "error"
        for i in 0..3 {
            let mut extra = HashMap::new();
            extra.insert(
                "pattern".to_string(),
                serde_json::Value::String("error".to_string()),
            );
            let mem =
                test_memory_with_metadata(&format!("grep search {i} for error"), "Grep", extra);
            storage.insert_memory(&mem).unwrap();
        }

        // Insert 1 Glob memory with pattern "*.rs"
        let mut extra = HashMap::new();
        extra.insert(
            "pattern".to_string(),
            serde_json::Value::String("*.rs".to_string()),
        );
        let mem = test_memory_with_metadata("glob search for rs files", "Glob", extra);
        storage.insert_memory(&mem).unwrap();

        // min_count=2: only "error" should appear
        let results = storage.get_repeated_searches(2, None).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "error");
        assert_eq!(results[0].1, 3);
        assert_eq!(results[0].2.len(), 3);

        // min_count=1: both should appear
        let results = storage.get_repeated_searches(1, None).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn get_file_hotspots_groups_by_file_path() {
        let storage = Storage::open_in_memory().unwrap();

        // Insert 4 memories referencing src/main.rs
        for i in 0..4 {
            let mut extra = HashMap::new();
            extra.insert(
                "file_path".to_string(),
                serde_json::Value::String("src/main.rs".to_string()),
            );
            let mem =
                test_memory_with_metadata(&format!("read main.rs attempt {i}"), "Read", extra);
            storage.insert_memory(&mem).unwrap();
        }

        // Insert 1 memory for a different file
        let mut extra = HashMap::new();
        extra.insert(
            "file_path".to_string(),
            serde_json::Value::String("src/lib.rs".to_string()),
        );
        let mem = test_memory_with_metadata("read lib.rs", "Read", extra);
        storage.insert_memory(&mem).unwrap();

        let results = storage.get_file_hotspots(3, None).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "src/main.rs");
        assert_eq!(results[0].1, 4);
    }

    #[test]
    fn get_tool_usage_stats_counts_by_tool() {
        let storage = Storage::open_in_memory().unwrap();

        // Insert various tool memories
        for i in 0..5 {
            let mem = test_memory_with_metadata(&format!("read file {i}"), "Read", HashMap::new());
            storage.insert_memory(&mem).unwrap();
        }
        for i in 0..3 {
            let mem =
                test_memory_with_metadata(&format!("grep search {i}"), "Grep", HashMap::new());
            storage.insert_memory(&mem).unwrap();
        }
        let mem = test_memory_with_metadata("edit file", "Edit", HashMap::new());
        storage.insert_memory(&mem).unwrap();

        let stats = storage.get_tool_usage_stats(None).unwrap();
        assert_eq!(stats.get("Read"), Some(&5));
        assert_eq!(stats.get("Grep"), Some(&3));
        assert_eq!(stats.get("Edit"), Some(&1));
    }

    #[test]
    fn get_decision_chains_groups_edits_by_file() {
        let storage = Storage::open_in_memory().unwrap();

        // 3 edits to the same file
        for i in 0..3 {
            let mut extra = HashMap::new();
            extra.insert(
                "file_path".to_string(),
                serde_json::Value::String("src/main.rs".to_string()),
            );
            let mem = test_memory_with_metadata(&format!("edit main.rs {i}"), "Edit", extra);
            storage.insert_memory(&mem).unwrap();
        }

        // 1 Write to a different file
        let mut extra = HashMap::new();
        extra.insert(
            "file_path".to_string(),
            serde_json::Value::String("src/new.rs".to_string()),
        );
        let mem = test_memory_with_metadata("write new.rs", "Write", extra);
        storage.insert_memory(&mem).unwrap();

        let results = storage.get_decision_chains(2, None).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "src/main.rs");
        assert_eq!(results[0].1, 3);
    }

    #[test]
    fn pattern_queries_empty_db() {
        let storage = Storage::open_in_memory().unwrap();

        let searches = storage.get_repeated_searches(1, None).unwrap();
        assert!(searches.is_empty());

        let hotspots = storage.get_file_hotspots(1, None).unwrap();
        assert!(hotspots.is_empty());

        let stats = storage.get_tool_usage_stats(None).unwrap();
        assert!(stats.is_empty());

        let chains = storage.get_decision_chains(1, None).unwrap();
        assert!(chains.is_empty());
    }

    #[test]
    fn pattern_queries_with_namespace_filter() {
        let storage = Storage::open_in_memory().unwrap();

        // Insert memories in namespace "project-a"
        for i in 0..3 {
            let mut extra = HashMap::new();
            extra.insert(
                "pattern".to_string(),
                serde_json::Value::String("error".to_string()),
            );
            let mut mem = test_memory_with_metadata(&format!("ns-a grep {i}"), "Grep", extra);
            mem.namespace = Some("project-a".to_string());
            storage.insert_memory(&mem).unwrap();
        }

        // Insert memories in namespace "project-b"
        for i in 0..2 {
            let mut extra = HashMap::new();
            extra.insert(
                "pattern".to_string(),
                serde_json::Value::String("error".to_string()),
            );
            let mut mem = test_memory_with_metadata(&format!("ns-b grep {i}"), "Grep", extra);
            mem.namespace = Some("project-b".to_string());
            storage.insert_memory(&mem).unwrap();
        }

        // Without namespace: 5 total
        let results = storage.get_repeated_searches(1, None).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].1, 5);

        // With namespace "project-a": only 3
        let results = storage.get_repeated_searches(1, Some("project-a")).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].1, 3);
    }

    // ── Session Management Tests ────────────────────────────────────────

    #[test]
    fn session_lifecycle() {
        let storage = Storage::open_in_memory().unwrap();

        // Start a session
        storage.start_session("sess-1", Some("my-project")).unwrap();

        // List sessions
        let sessions = storage.list_sessions(Some("my-project")).unwrap();
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].id, "sess-1");
        assert_eq!(sessions[0].namespace, Some("my-project".to_string()));
        assert!(sessions[0].ended_at.is_none());

        // End the session
        storage
            .end_session("sess-1", Some("Explored the codebase"))
            .unwrap();

        let sessions = storage.list_sessions(None).unwrap();
        assert_eq!(sessions.len(), 1);
        assert!(sessions[0].ended_at.is_some());
        assert_eq!(
            sessions[0].summary,
            Some("Explored the codebase".to_string())
        );
    }

    #[test]
    fn ensure_session_column_idempotent() {
        let storage = Storage::open_in_memory().unwrap();
        // Should succeed even when called multiple times
        storage.ensure_session_column().unwrap();
        storage.ensure_session_column().unwrap();
    }

    #[test]
    fn list_sessions_filters_by_namespace() {
        let storage = Storage::open_in_memory().unwrap();

        storage.start_session("sess-a", Some("project-a")).unwrap();
        storage.start_session("sess-b", Some("project-b")).unwrap();
        storage.start_session("sess-c", None).unwrap();

        let all = storage.list_sessions(None).unwrap();
        assert_eq!(all.len(), 3);

        let proj_a = storage.list_sessions(Some("project-a")).unwrap();
        assert_eq!(proj_a.len(), 1);
        assert_eq!(proj_a[0].id, "sess-a");
    }

    #[test]
    fn start_session_ignores_duplicate() {
        let storage = Storage::open_in_memory().unwrap();
        storage.start_session("sess-1", Some("ns")).unwrap();
        // Second call with same ID should be ignored (INSERT OR IGNORE)
        storage.start_session("sess-1", Some("ns")).unwrap();

        let sessions = storage.list_sessions(None).unwrap();
        assert_eq!(sessions.len(), 1);
    }
}
