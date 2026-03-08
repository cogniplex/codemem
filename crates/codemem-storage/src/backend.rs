//! `StorageBackend` trait implementation for Storage.

use crate::{MapStorageErr, MemoryRow, Storage};
use codemem_core::{
    CodememError, ConsolidationLogEntry, Edge, GraphNode, MemoryNode, NodeKind, Session,
    StorageBackend, StorageStats,
};
use rusqlite::params;
use std::collections::HashMap;

/// Generate SQL placeholders for multi-row INSERT: "(?1,?2,...),(?3,?4,...)"
fn multi_row_placeholders(cols: usize, rows: usize) -> String {
    let mut s = String::new();
    for r in 0..rows {
        if r > 0 {
            s.push(',');
        }
        s.push('(');
        for c in 0..cols {
            if c > 0 {
                s.push(',');
            }
            s.push('?');
            s.push_str(&(r * cols + c + 1).to_string());
        }
        s.push(')');
    }
    s
}

/// Macro to delegate pure-forwarding trait methods to `Storage` inherent methods.
macro_rules! delegate_storage {
    // &self, no args
    ($method:ident(&self) -> $ret:ty) => {
        fn $method(&self) -> $ret {
            Storage::$method(self)
        }
    };
    // &self, one arg
    ($method:ident(&self, $a1:ident: $t1:ty) -> $ret:ty) => {
        fn $method(&self, $a1: $t1) -> $ret {
            Storage::$method(self, $a1)
        }
    };
    // &self, two args
    ($method:ident(&self, $a1:ident: $t1:ty, $a2:ident: $t2:ty) -> $ret:ty) => {
        fn $method(&self, $a1: $t1, $a2: $t2) -> $ret {
            Storage::$method(self, $a1, $a2)
        }
    };
    // &self, three args
    ($method:ident(&self, $a1:ident: $t1:ty, $a2:ident: $t2:ty, $a3:ident: $t3:ty) -> $ret:ty) => {
        fn $method(&self, $a1: $t1, $a2: $t2, $a3: $t3) -> $ret {
            Storage::$method(self, $a1, $a2, $a3)
        }
    };
    // &self, five args
    ($method:ident(&self, $a1:ident: $t1:ty, $a2:ident: $t2:ty, $a3:ident: $t3:ty, $a4:ident: $t4:ty, $a5:ident: $t5:ty) -> $ret:ty) => {
        fn $method(&self, $a1: $t1, $a2: $t2, $a3: $t3, $a4: $t4, $a5: $t5) -> $ret {
            Storage::$method(self, $a1, $a2, $a3, $a4, $a5)
        }
    };
}

impl StorageBackend for Storage {
    // ── Memory CRUD (delegated) ───────────────────────────────────────

    delegate_storage!(insert_memory(&self, memory: &MemoryNode) -> Result<(), CodememError>);
    delegate_storage!(get_memory(&self, id: &str) -> Result<Option<MemoryNode>, CodememError>);
    delegate_storage!(get_memory_no_touch(&self, id: &str) -> Result<Option<MemoryNode>, CodememError>);
    delegate_storage!(update_memory(&self, id: &str, content: &str, importance: Option<f64>) -> Result<(), CodememError>);
    delegate_storage!(delete_memory(&self, id: &str) -> Result<bool, CodememError>);

    /// M1: Override with transactional cascade delete.
    fn delete_memory_cascade(&self, id: &str) -> Result<bool, CodememError> {
        // Delegates to Storage::delete_memory_cascade which wraps all
        // deletes (memory + graph nodes/edges + embedding) in a single transaction.
        Storage::delete_memory_cascade(self, id)
    }

    /// Override with transactional batch cascade delete.
    fn delete_memories_batch_cascade(&self, ids: &[&str]) -> Result<usize, CodememError> {
        Storage::delete_memories_batch_cascade(self, ids)
    }

    delegate_storage!(list_memory_ids(&self) -> Result<Vec<String>, CodememError>);
    delegate_storage!(list_memory_ids_for_namespace(&self, namespace: &str) -> Result<Vec<String>, CodememError>);
    delegate_storage!(find_memory_ids_by_tag(&self, tag: &str, namespace: Option<&str>, exclude_id: &str) -> Result<Vec<String>, CodememError>);
    delegate_storage!(list_namespaces(&self) -> Result<Vec<String>, CodememError>);
    delegate_storage!(memory_count(&self) -> Result<usize, CodememError>);

    fn get_memories_batch(&self, ids: &[&str]) -> Result<Vec<MemoryNode>, CodememError> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let conn = self.conn()?;

        let placeholders: Vec<String> = (1..=ids.len()).map(|i| format!("?{i}")).collect();
        let sql = format!(
            "SELECT id, content, memory_type, importance, confidence, access_count, content_hash, tags, metadata, namespace, session_id, created_at, updated_at, last_accessed_at FROM memories WHERE id IN ({})",
            placeholders.join(",")
        );

        let mut stmt = conn.prepare(&sql).storage_err()?;

        let params: Vec<&dyn rusqlite::types::ToSql> = ids
            .iter()
            .map(|id| id as &dyn rusqlite::types::ToSql)
            .collect();

        let rows = stmt
            .query_map(params.as_slice(), MemoryRow::from_row)
            .storage_err()?;

        let mut memories = Vec::new();
        for row in rows {
            let row = row.storage_err()?;
            memories.push(row.into_memory_node()?);
        }
        Ok(memories)
    }

    // ── Embedding Persistence (delegated where possible) ──────────────

    delegate_storage!(store_embedding(&self, memory_id: &str, embedding: &[f32]) -> Result<(), CodememError>);
    delegate_storage!(get_embedding(&self, memory_id: &str) -> Result<Option<Vec<f32>>, CodememError>);

    fn delete_embedding(&self, memory_id: &str) -> Result<bool, CodememError> {
        let conn = self.conn()?;
        let deleted = conn
            .execute(
                "DELETE FROM memory_embeddings WHERE memory_id = ?1",
                [memory_id],
            )
            .storage_err()?;
        Ok(deleted > 0)
    }

    fn list_all_embeddings(&self) -> Result<Vec<(String, Vec<f32>)>, CodememError> {
        let conn = self.conn()?;
        let mut stmt = conn
            .prepare("SELECT memory_id, embedding FROM memory_embeddings")
            .storage_err()?;
        let rows = stmt
            .query_map([], |row| {
                let id: String = row.get(0)?;
                let blob: Vec<u8> = row.get(1)?;
                Ok((id, blob))
            })
            .storage_err()?;
        let mut result = Vec::new();
        for row in rows {
            let (id, blob) = row.storage_err()?;
            let floats: Vec<f32> = blob
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();
            result.push((id, floats));
        }
        Ok(result)
    }

    // ── Graph Node/Edge Persistence (delegated) ───────────────────────

    delegate_storage!(insert_graph_node(&self, node: &GraphNode) -> Result<(), CodememError>);
    delegate_storage!(get_graph_node(&self, id: &str) -> Result<Option<GraphNode>, CodememError>);
    delegate_storage!(delete_graph_node(&self, id: &str) -> Result<bool, CodememError>);
    delegate_storage!(all_graph_nodes(&self) -> Result<Vec<GraphNode>, CodememError>);
    delegate_storage!(insert_graph_edge(&self, edge: &Edge) -> Result<(), CodememError>);
    delegate_storage!(get_edges_for_node(&self, node_id: &str) -> Result<Vec<Edge>, CodememError>);
    delegate_storage!(all_graph_edges(&self) -> Result<Vec<Edge>, CodememError>);
    delegate_storage!(delete_graph_edges_for_node(&self, node_id: &str) -> Result<usize, CodememError>);
    delegate_storage!(delete_graph_nodes_by_prefix(&self, prefix: &str) -> Result<usize, CodememError>);

    // ── Sessions (delegated where possible) ───────────────────────────

    delegate_storage!(start_session(&self, id: &str, namespace: Option<&str>) -> Result<(), CodememError>);
    delegate_storage!(end_session(&self, id: &str, summary: Option<&str>) -> Result<(), CodememError>);

    fn list_sessions(
        &self,
        namespace: Option<&str>,
        limit: usize,
    ) -> Result<Vec<Session>, CodememError> {
        self.list_sessions_with_limit(namespace, limit)
    }

    // ── Consolidation (delegated) ─────────────────────────────────────

    delegate_storage!(insert_consolidation_log(&self, cycle_type: &str, affected_count: usize) -> Result<(), CodememError>);
    delegate_storage!(last_consolidation_runs(&self) -> Result<Vec<ConsolidationLogEntry>, CodememError>);

    // ── Pattern Detection (delegated) ─────────────────────────────────

    delegate_storage!(get_repeated_searches(&self, min_count: usize, namespace: Option<&str>) -> Result<Vec<(String, usize, Vec<String>)>, CodememError>);
    delegate_storage!(get_file_hotspots(&self, min_count: usize, namespace: Option<&str>) -> Result<Vec<(String, usize, Vec<String>)>, CodememError>);
    delegate_storage!(get_tool_usage_stats(&self, namespace: Option<&str>) -> Result<Vec<(String, usize)>, CodememError>);
    delegate_storage!(get_decision_chains(&self, min_count: usize, namespace: Option<&str>) -> Result<Vec<(String, usize, Vec<String>)>, CodememError>);

    // ── Bulk Operations ───────────────────────────────────────────────

    fn decay_stale_memories(
        &self,
        threshold_ts: i64,
        decay_factor: f64,
    ) -> Result<usize, CodememError> {
        let conn = self.conn()?;
        let rows = conn
            .execute(
                "UPDATE memories SET importance = importance * ?1 WHERE last_accessed_at < ?2",
                params![decay_factor, threshold_ts],
            )
            .storage_err()?;
        Ok(rows)
    }

    fn list_memories_for_creative(
        &self,
    ) -> Result<Vec<(String, String, Vec<String>)>, CodememError> {
        let conn = self.conn()?;
        let mut stmt = conn
            .prepare("SELECT id, memory_type, tags FROM memories ORDER BY created_at DESC")
            .storage_err()?;

        let rows = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                ))
            })
            .storage_err()?
            .collect::<Result<Vec<_>, _>>()
            .storage_err()?;

        Ok(rows
            .into_iter()
            .map(|(id, mtype, tags_json)| {
                let tags: Vec<String> = serde_json::from_str(&tags_json).unwrap_or_default();
                (id, mtype, tags)
            })
            .collect())
    }

    fn find_hash_duplicates(&self) -> Result<Vec<(String, String, f64)>, CodememError> {
        let conn = self.conn()?;
        let mut stmt = conn
            .prepare(
                "SELECT a.id, b.id, 1.0 as similarity
                 FROM memories a
                 INNER JOIN memories b ON substr(a.content_hash, 1, 16) = substr(b.content_hash, 1, 16)
                 WHERE a.id < b.id",
            )
            .storage_err()?;

        let rows = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, f64>(2)?,
                ))
            })
            .storage_err()?
            .collect::<Result<Vec<_>, _>>()
            .storage_err()?;

        Ok(rows)
    }

    fn find_forgettable(&self, importance_threshold: f64) -> Result<Vec<String>, CodememError> {
        let conn = self.conn()?;
        let mut stmt = conn
            .prepare(
                "SELECT id FROM memories WHERE importance < ?1 AND access_count = 0 ORDER BY importance ASC, last_accessed_at ASC",
            )
            .storage_err()?;

        let ids = stmt
            .query_map(params![importance_threshold], |row| row.get(0))
            .storage_err()?
            .collect::<Result<Vec<String>, _>>()
            .storage_err()?;

        Ok(ids)
    }

    // ── Batch Operations ──────────────────────────────────────────────

    fn insert_memories_batch(&self, memories: &[MemoryNode]) -> Result<(), CodememError> {
        if memories.is_empty() {
            return Ok(());
        }
        let conn = self.conn()?;
        let tx = conn.unchecked_transaction().storage_err()?;

        const COLS: usize = 14;
        const BATCH: usize = 999 / COLS; // 71

        for chunk in memories.chunks(BATCH) {
            let placeholders = multi_row_placeholders(COLS, chunk.len());
            let sql = format!(
                "INSERT OR IGNORE INTO memories (id, content, memory_type, importance, confidence, access_count, content_hash, tags, metadata, namespace, session_id, created_at, updated_at, last_accessed_at) VALUES {placeholders}"
            );

            let mut param_values: Vec<Box<dyn rusqlite::types::ToSql>> =
                Vec::with_capacity(chunk.len() * COLS);
            for memory in chunk {
                let tags_json = serde_json::to_string(&memory.tags)?;
                let metadata_json = serde_json::to_string(&memory.metadata)?;
                param_values.push(Box::new(memory.id.clone()));
                param_values.push(Box::new(memory.content.clone()));
                param_values.push(Box::new(memory.memory_type.to_string()));
                param_values.push(Box::new(memory.importance));
                param_values.push(Box::new(memory.confidence));
                param_values.push(Box::new(memory.access_count as i64));
                param_values.push(Box::new(memory.content_hash.clone()));
                param_values.push(Box::new(tags_json));
                param_values.push(Box::new(metadata_json));
                param_values.push(Box::new(memory.namespace.clone()));
                param_values.push(Box::new(memory.session_id.clone()));
                param_values.push(Box::new(memory.created_at.timestamp()));
                param_values.push(Box::new(memory.updated_at.timestamp()));
                param_values.push(Box::new(memory.last_accessed_at.timestamp()));
            }

            let refs: Vec<&dyn rusqlite::types::ToSql> =
                param_values.iter().map(|p| p.as_ref()).collect();
            tx.execute(&sql, refs.as_slice()).storage_err()?;
        }

        tx.commit().storage_err()?;
        Ok(())
    }

    fn store_embeddings_batch(&self, items: &[(&str, &[f32])]) -> Result<(), CodememError> {
        if items.is_empty() {
            return Ok(());
        }
        let conn = self.conn()?;
        let tx = conn.unchecked_transaction().storage_err()?;

        const COLS: usize = 2;
        const BATCH: usize = 999 / COLS; // 499

        for chunk in items.chunks(BATCH) {
            let placeholders = multi_row_placeholders(COLS, chunk.len());
            let sql = format!(
                "INSERT OR REPLACE INTO memory_embeddings (memory_id, embedding) VALUES {placeholders}"
            );

            let mut param_values: Vec<Box<dyn rusqlite::types::ToSql>> =
                Vec::with_capacity(chunk.len() * COLS);
            for (id, embedding) in chunk {
                let blob: Vec<u8> = embedding.iter().flat_map(|f| f.to_le_bytes()).collect();
                param_values.push(Box::new(id.to_string()));
                param_values.push(Box::new(blob));
            }

            let refs: Vec<&dyn rusqlite::types::ToSql> =
                param_values.iter().map(|p| p.as_ref()).collect();
            tx.execute(&sql, refs.as_slice()).storage_err()?;
        }

        tx.commit().storage_err()?;
        Ok(())
    }

    fn load_file_hashes(&self) -> Result<HashMap<String, String>, CodememError> {
        let conn = self.conn()?;
        let mut stmt = conn
            .prepare("SELECT file_path, content_hash FROM file_hashes")
            .storage_err()?;

        let rows = stmt
            .query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })
            .storage_err()?
            .collect::<Result<Vec<_>, _>>()
            .storage_err()?;

        Ok(rows.into_iter().collect())
    }

    fn save_file_hashes(&self, hashes: &HashMap<String, String>) -> Result<(), CodememError> {
        let conn = self.conn()?;
        let tx = conn.unchecked_transaction().storage_err()?;

        tx.execute("DELETE FROM file_hashes", []).storage_err()?;

        for (path, hash) in hashes {
            tx.execute(
                "INSERT INTO file_hashes (file_path, content_hash) VALUES (?1, ?2)",
                params![path, hash],
            )
            .storage_err()?;
        }

        tx.commit().storage_err()?;
        Ok(())
    }

    fn insert_graph_nodes_batch(&self, nodes: &[GraphNode]) -> Result<(), CodememError> {
        if nodes.is_empty() {
            return Ok(());
        }
        let conn = self.conn()?;
        let tx = conn.unchecked_transaction().storage_err()?;

        const COLS: usize = 7;
        const BATCH: usize = 999 / COLS; // 142

        for chunk in nodes.chunks(BATCH) {
            let placeholders = multi_row_placeholders(COLS, chunk.len());
            let sql = format!(
                "INSERT OR REPLACE INTO graph_nodes (id, kind, label, payload, centrality, memory_id, namespace) VALUES {placeholders}"
            );

            let mut param_values: Vec<Box<dyn rusqlite::types::ToSql>> =
                Vec::with_capacity(chunk.len() * COLS);
            for node in chunk {
                let payload_json =
                    serde_json::to_string(&node.payload).unwrap_or_else(|_| "{}".to_string());
                param_values.push(Box::new(node.id.clone()));
                param_values.push(Box::new(node.kind.to_string()));
                param_values.push(Box::new(node.label.clone()));
                param_values.push(Box::new(payload_json));
                param_values.push(Box::new(node.centrality));
                param_values.push(Box::new(node.memory_id.clone()));
                param_values.push(Box::new(node.namespace.clone()));
            }

            let refs: Vec<&dyn rusqlite::types::ToSql> =
                param_values.iter().map(|p| p.as_ref()).collect();
            tx.execute(&sql, refs.as_slice()).storage_err()?;
        }

        tx.commit().storage_err()?;
        Ok(())
    }

    fn insert_graph_edges_batch(&self, edges: &[Edge]) -> Result<(), CodememError> {
        if edges.is_empty() {
            return Ok(());
        }
        let conn = self.conn()?;
        let tx = conn.unchecked_transaction().storage_err()?;

        const COLS: usize = 9;
        const BATCH: usize = 999 / COLS; // 111

        for chunk in edges.chunks(BATCH) {
            let placeholders = multi_row_placeholders(COLS, chunk.len());
            let sql = format!(
                "INSERT OR REPLACE INTO graph_edges (id, src, dst, relationship, weight, properties, created_at, valid_from, valid_to) VALUES {placeholders}"
            );

            let mut param_values: Vec<Box<dyn rusqlite::types::ToSql>> =
                Vec::with_capacity(chunk.len() * COLS);
            for edge in chunk {
                let props_json =
                    serde_json::to_string(&edge.properties).unwrap_or_else(|_| "{}".to_string());
                param_values.push(Box::new(edge.id.clone()));
                param_values.push(Box::new(edge.src.clone()));
                param_values.push(Box::new(edge.dst.clone()));
                param_values.push(Box::new(edge.relationship.to_string()));
                param_values.push(Box::new(edge.weight));
                param_values.push(Box::new(props_json));
                param_values.push(Box::new(edge.created_at.timestamp()));
                param_values.push(Box::new(edge.valid_from.map(|dt| dt.timestamp())));
                param_values.push(Box::new(edge.valid_to.map(|dt| dt.timestamp())));
            }

            let refs: Vec<&dyn rusqlite::types::ToSql> =
                param_values.iter().map(|p| p.as_ref()).collect();
            tx.execute(&sql, refs.as_slice()).storage_err()?;
        }

        tx.commit().storage_err()?;
        Ok(())
    }

    fn get_stale_memories_for_decay(
        &self,
        threshold_ts: i64,
    ) -> Result<Vec<(String, f64, u32, i64)>, CodememError> {
        let conn = self.conn()?;
        let mut stmt = conn
            .prepare(
                "SELECT id, importance, access_count, last_accessed_at FROM memories WHERE last_accessed_at < ?1",
            )
            .storage_err()?;

        let rows = stmt
            .query_map(params![threshold_ts], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, f64>(1)?,
                    row.get::<_, u32>(2)?,
                    row.get::<_, i64>(3)?,
                ))
            })
            .storage_err()?
            .collect::<Result<Vec<_>, _>>()
            .storage_err()?;

        Ok(rows)
    }

    fn batch_update_importance(&self, updates: &[(String, f64)]) -> Result<usize, CodememError> {
        if updates.is_empty() {
            return Ok(0);
        }
        let conn = self.conn()?;
        let tx = conn.unchecked_transaction().storage_err()?;

        let mut count = 0usize;
        for (id, importance) in updates {
            let rows = tx
                .execute(
                    "UPDATE memories SET importance = ?1 WHERE id = ?2",
                    params![importance, id],
                )
                .storage_err()?;
            count += rows;
        }

        tx.commit().storage_err()?;
        Ok(count)
    }

    fn session_count(&self, namespace: Option<&str>) -> Result<usize, CodememError> {
        let conn = self.conn()?;
        let count: i64 = if let Some(ns) = namespace {
            conn.query_row(
                "SELECT COUNT(*) FROM sessions WHERE namespace = ?1",
                params![ns],
                |row| row.get(0),
            )
            .storage_err()?
        } else {
            conn.query_row("SELECT COUNT(*) FROM sessions", [], |row| row.get(0))
                .storage_err()?
        };
        Ok(count as usize)
    }

    // ── Query Helpers ─────────────────────────────────────────────────

    fn find_unembedded_memories(&self) -> Result<Vec<(String, String)>, CodememError> {
        let conn = self.conn()?;
        let mut stmt = conn
            .prepare(
                "SELECT m.id, m.content FROM memories m
                 LEFT JOIN memory_embeddings me ON m.id = me.memory_id
                 WHERE me.memory_id IS NULL",
            )
            .storage_err()?;

        let rows = stmt
            .query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })
            .storage_err()?
            .collect::<Result<Vec<_>, _>>()
            .storage_err()?;

        Ok(rows)
    }

    fn search_graph_nodes(
        &self,
        query: &str,
        namespace: Option<&str>,
        limit: usize,
    ) -> Result<Vec<GraphNode>, CodememError> {
        let conn = self.conn()?;
        let escaped = query
            .to_lowercase()
            .replace('\\', "\\\\")
            .replace('%', "\\%")
            .replace('_', "\\_");
        let pattern = format!("%{escaped}%");

        let (sql, params_vec): (String, Vec<Box<dyn rusqlite::types::ToSql>>) =
            if let Some(ns) = namespace {
                (
                    "SELECT id, kind, label, payload, centrality, memory_id, namespace \
                 FROM graph_nodes WHERE LOWER(label) LIKE ?1 ESCAPE '\\' AND namespace = ?2 \
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
                 FROM graph_nodes WHERE LOWER(label) LIKE ?1 ESCAPE '\\' \
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
        let mut stmt = conn.prepare(&sql).storage_err()?;

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
            .storage_err()?
            .collect::<Result<Vec<_>, _>>()
            .storage_err()?;

        Ok(rows)
    }

    fn list_memories_filtered(
        &self,
        namespace: Option<&str>,
        memory_type: Option<&str>,
    ) -> Result<Vec<MemoryNode>, CodememError> {
        let conn = self.conn()?;
        let mut sql = "SELECT id, content, memory_type, importance, confidence, access_count, \
                        content_hash, tags, metadata, namespace, session_id, created_at, updated_at, \
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
        let mut stmt = conn.prepare(&sql).storage_err()?;

        let rows = stmt
            .query_map(refs.as_slice(), MemoryRow::from_row)
            .storage_err()?;

        let mut result = Vec::new();
        for row in rows {
            let mr = row.storage_err()?;
            result.push(mr.into_memory_node()?);
        }

        Ok(result)
    }

    // ── Session Activity (delegated) ──────────────────────────────────

    delegate_storage!(record_session_activity(&self, session_id: &str, tool_name: &str, file_path: Option<&str>, directory: Option<&str>, pattern: Option<&str>) -> Result<(), CodememError>);
    delegate_storage!(get_session_activity_summary(&self, session_id: &str) -> Result<codemem_core::SessionActivitySummary, CodememError>);
    delegate_storage!(get_session_hot_directories(&self, session_id: &str, limit: usize) -> Result<Vec<(String, usize)>, CodememError>);
    delegate_storage!(has_auto_insight(&self, session_id: &str, dedup_tag: &str) -> Result<bool, CodememError>);
    delegate_storage!(count_directory_reads(&self, session_id: &str, directory: &str) -> Result<usize, CodememError>);
    delegate_storage!(was_file_read_in_session(&self, session_id: &str, file_path: &str) -> Result<bool, CodememError>);
    delegate_storage!(count_search_pattern_in_session(&self, session_id: &str, pattern: &str) -> Result<usize, CodememError>);

    // ── Stats (delegated) ─────────────────────────────────────────────

    delegate_storage!(stats(&self) -> Result<StorageStats, CodememError>);

    // ── Transaction Control ──────────────────────────────────────────

    fn begin_transaction(&self) -> Result<(), CodememError> {
        let conn = self.conn()?;
        conn.execute_batch("BEGIN IMMEDIATE").storage_err()?;
        self.in_transaction
            .store(true, std::sync::atomic::Ordering::Release);
        Ok(())
    }

    fn commit_transaction(&self) -> Result<(), CodememError> {
        let conn = self.conn()?;
        conn.execute_batch("COMMIT").storage_err()?;
        // Clear flag after COMMIT succeeds — if COMMIT fails, the flag
        // stays set so callers know a transaction is still active.
        self.in_transaction
            .store(false, std::sync::atomic::Ordering::Release);
        Ok(())
    }

    fn rollback_transaction(&self) -> Result<(), CodememError> {
        self.in_transaction
            .store(false, std::sync::atomic::Ordering::Release);
        let conn = self.conn()?;
        conn.execute_batch("ROLLBACK").storage_err()?;
        Ok(())
    }
}

#[cfg(test)]
#[path = "tests/backend_tests.rs"]
mod tests;
