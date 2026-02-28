//! Stats, consolidation, pattern queries, and session management on Storage.

use crate::Storage;
use codemem_core::{CodememError, ConsolidationLogEntry, Session, StorageStats};
use rusqlite::params;
use std::collections::HashMap;

impl Storage {
    // ── Stats ───────────────────────────────────────────────────────────

    /// Get database statistics.
    pub fn stats(&self) -> Result<StorageStats, CodememError> {
        let memory_count = self.memory_count()?;
        let conn = self.conn();

        let embedding_count: i64 = conn
            .query_row("SELECT COUNT(*) FROM memory_embeddings", [], |row| {
                row.get(0)
            })
            .map_err(|e| CodememError::Storage(e.to_string()))?;

        let node_count: i64 = conn
            .query_row("SELECT COUNT(*) FROM graph_nodes", [], |row| row.get(0))
            .map_err(|e| CodememError::Storage(e.to_string()))?;

        let edge_count: i64 = conn
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
        let conn = self.conn();
        let now = chrono::Utc::now().timestamp();
        conn.execute(
            "INSERT INTO consolidation_log (cycle_type, run_at, affected_count) VALUES (?1, ?2, ?3)",
            params![cycle_type, now, affected_count as i64],
        )
        .map_err(|e| CodememError::Storage(e.to_string()))?;
        Ok(())
    }

    /// Get the last consolidation run for each cycle type.
    pub fn last_consolidation_runs(&self) -> Result<Vec<ConsolidationLogEntry>, CodememError> {
        let conn = self.conn();
        let mut stmt = conn
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

    // ── Pattern Detection Queries ───────────────────────────────────────

    /// Find repeated search patterns (Grep/Glob) by extracting the "pattern" field
    /// from memory metadata JSON. Returns (pattern, count, memory_ids) tuples where
    /// count >= min_count, ordered by count descending.
    pub fn get_repeated_searches(
        &self,
        min_count: usize,
        namespace: Option<&str>,
    ) -> Result<Vec<(String, usize, Vec<String>)>, CodememError> {
        let conn = self.conn();
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

        let mut stmt = conn
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
    pub fn get_file_hotspots(
        &self,
        min_count: usize,
        namespace: Option<&str>,
    ) -> Result<Vec<(String, usize, Vec<String>)>, CodememError> {
        let conn = self.conn();
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

        let mut stmt = conn
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
    pub fn get_tool_usage_stats(
        &self,
        namespace: Option<&str>,
    ) -> Result<HashMap<String, usize>, CodememError> {
        let conn = self.conn();
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

        let mut stmt = conn
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
    pub fn get_decision_chains(
        &self,
        min_count: usize,
        namespace: Option<&str>,
    ) -> Result<Vec<(String, usize, Vec<String>)>, CodememError> {
        let conn = self.conn();
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

        let mut stmt = conn
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
        let conn = self.conn();
        let has_col: bool = conn
            .prepare("SELECT session_id FROM memories LIMIT 0")
            .is_ok();
        if !has_col {
            conn.execute_batch("ALTER TABLE memories ADD COLUMN session_id TEXT;")
                .map_err(|e| CodememError::Storage(e.to_string()))?;
        }
        Ok(())
    }

    /// Start a new session.
    pub fn start_session(&self, id: &str, namespace: Option<&str>) -> Result<(), CodememError> {
        let conn = self.conn();
        let now = chrono::Utc::now().timestamp();
        conn.execute(
            "INSERT OR IGNORE INTO sessions (id, namespace, started_at) VALUES (?1, ?2, ?3)",
            params![id, namespace, now],
        )
        .map_err(|e| CodememError::Storage(e.to_string()))?;
        Ok(())
    }

    /// End a session by setting ended_at and optionally a summary.
    pub fn end_session(&self, id: &str, summary: Option<&str>) -> Result<(), CodememError> {
        let conn = self.conn();
        let now = chrono::Utc::now().timestamp();
        conn.execute(
            "UPDATE sessions SET ended_at = ?1, summary = ?2 WHERE id = ?3",
            params![now, summary, id],
        )
        .map_err(|e| CodememError::Storage(e.to_string()))?;
        Ok(())
    }

    /// List sessions, optionally filtered by namespace.
    pub fn list_sessions(&self, namespace: Option<&str>) -> Result<Vec<Session>, CodememError> {
        self.list_sessions_with_limit(namespace, usize::MAX)
    }

    /// List sessions with a limit.
    pub(crate) fn list_sessions_with_limit(
        &self,
        namespace: Option<&str>,
        limit: usize,
    ) -> Result<Vec<Session>, CodememError> {
        let conn = self.conn();
        let sql_with_ns = "SELECT id, namespace, started_at, ended_at, memory_count, summary FROM sessions WHERE namespace = ?1 ORDER BY started_at DESC LIMIT ?2";
        let sql_all = "SELECT id, namespace, started_at, ended_at, memory_count, summary FROM sessions ORDER BY started_at DESC LIMIT ?1";

        let map_row = |row: &rusqlite::Row<'_>| -> rusqlite::Result<Session> {
            let started_ts: i64 = row.get(2)?;
            let ended_ts: Option<i64> = row.get(3)?;
            Ok(Session {
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
            let mut stmt = conn
                .prepare(sql_with_ns)
                .map_err(|e| CodememError::Storage(e.to_string()))?;
            let rows = stmt
                .query_map(params![ns, limit as i64], map_row)
                .map_err(|e| CodememError::Storage(e.to_string()))?;
            rows.collect::<Result<Vec<_>, _>>()
                .map_err(|e| CodememError::Storage(e.to_string()))
        } else {
            let mut stmt = conn
                .prepare(sql_all)
                .map_err(|e| CodememError::Storage(e.to_string()))?;
            let rows = stmt
                .query_map(params![limit as i64], map_row)
                .map_err(|e| CodememError::Storage(e.to_string()))?;
            rows.collect::<Result<Vec<_>, _>>()
                .map_err(|e| CodememError::Storage(e.to_string()))
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::Storage;
    use codemem_core::{MemoryNode, MemoryType};
    use std::collections::HashMap;

    fn test_memory_with_metadata(
        content: &str,
        tool: &str,
        extra: HashMap<String, serde_json::Value>,
    ) -> MemoryNode {
        let now = chrono::Utc::now();
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
    fn stats() {
        let storage = Storage::open_in_memory().unwrap();
        let stats = storage.stats().unwrap();
        assert_eq!(stats.memory_count, 0);
    }

    #[test]
    fn get_repeated_searches_groups_by_pattern() {
        let storage = Storage::open_in_memory().unwrap();

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

        let mut extra = HashMap::new();
        extra.insert(
            "pattern".to_string(),
            serde_json::Value::String("*.rs".to_string()),
        );
        let mem = test_memory_with_metadata("glob search for rs files", "Glob", extra);
        storage.insert_memory(&mem).unwrap();

        let results = storage.get_repeated_searches(2, None).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "error");
        assert_eq!(results[0].1, 3);
        assert_eq!(results[0].2.len(), 3);

        let results = storage.get_repeated_searches(1, None).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn get_file_hotspots_groups_by_file_path() {
        let storage = Storage::open_in_memory().unwrap();

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

        for i in 0..3 {
            let mut extra = HashMap::new();
            extra.insert(
                "file_path".to_string(),
                serde_json::Value::String("src/main.rs".to_string()),
            );
            let mem = test_memory_with_metadata(&format!("edit main.rs {i}"), "Edit", extra);
            storage.insert_memory(&mem).unwrap();
        }

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

        let results = storage.get_repeated_searches(1, None).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].1, 5);

        let results = storage.get_repeated_searches(1, Some("project-a")).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].1, 3);
    }

    // ── Session Management Tests ────────────────────────────────────────

    #[test]
    fn session_lifecycle() {
        let storage = Storage::open_in_memory().unwrap();

        storage.start_session("sess-1", Some("my-project")).unwrap();

        let sessions = storage.list_sessions(Some("my-project")).unwrap();
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].id, "sess-1");
        assert_eq!(sessions[0].namespace, Some("my-project".to_string()));
        assert!(sessions[0].ended_at.is_none());

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
        storage.start_session("sess-1", Some("ns")).unwrap();

        let sessions = storage.list_sessions(None).unwrap();
        assert_eq!(sessions.len(), 1);
    }
}
