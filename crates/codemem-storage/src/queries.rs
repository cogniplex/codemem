//! Stats, consolidation, pattern queries, and session management on Storage.

use crate::Storage;
use codemem_core::{CodememError, ConsolidationLogEntry, Session, StorageStats};
use rusqlite::params;
use std::collections::HashMap;

impl Storage {
    // ── Health / Diagnostics ────────────────────────────────────────────

    /// Run SQLite `PRAGMA integrity_check`. Returns `true` if the database is OK.
    pub fn integrity_check(&self) -> Result<bool, CodememError> {
        let conn = self.conn()?;
        let result: String = conn
            .query_row("PRAGMA integrity_check", [], |row| row.get(0))
            .map_err(|e| CodememError::Storage(e.to_string()))?;
        Ok(result == "ok")
    }

    /// Return the current schema version (max applied migration number).
    pub fn schema_version(&self) -> Result<u32, CodememError> {
        let conn = self.conn()?;
        let version: u32 = conn
            .query_row(
                "SELECT COALESCE(MAX(version), 0) FROM schema_version",
                [],
                |row| row.get(0),
            )
            .map_err(|e| CodememError::Storage(e.to_string()))?;
        Ok(version)
    }

    // ── Stats ───────────────────────────────────────────────────────────

    /// Get database statistics.
    pub fn stats(&self) -> Result<StorageStats, CodememError> {
        let memory_count = self.memory_count()?;
        let conn = self.conn()?;

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
        let conn = self.conn()?;
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
        let conn = self.conn()?;
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
        let conn = self.conn()?;
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
        let conn = self.conn()?;
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
        let conn = self.conn()?;
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
        let conn = self.conn()?;
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

    // ── Insight / Tag Queries ──────────────────────────────────────────

    /// Count memories whose content matches any of the given keywords (SQL LIKE).
    pub fn count_memories_matching_keywords(
        &self,
        keywords: &[&str],
        namespace: Option<&str>,
    ) -> Result<usize, CodememError> {
        if keywords.is_empty() {
            return Ok(0);
        }
        let conn = self.conn()?;
        let like_clauses: Vec<String> = keywords
            .iter()
            .enumerate()
            .map(|(i, _)| format!("content LIKE ?{}", i + 1))
            .collect();
        let where_likes = like_clauses.join(" OR ");

        let (sql, params_vec): (String, Vec<Box<dyn rusqlite::types::ToSql>>) =
            if let Some(ns) = namespace {
                let sql = format!(
                    "SELECT COUNT(*) FROM memories WHERE ({}) AND namespace = ?{}",
                    where_likes,
                    keywords.len() + 1,
                );
                let mut p: Vec<Box<dyn rusqlite::types::ToSql>> = keywords
                    .iter()
                    .map(|k| Box::new(format!("%{k}%")) as Box<dyn rusqlite::types::ToSql>)
                    .collect();
                p.push(Box::new(ns.to_string()));
                (sql, p)
            } else {
                let sql = format!("SELECT COUNT(*) FROM memories WHERE ({})", where_likes);
                let p: Vec<Box<dyn rusqlite::types::ToSql>> = keywords
                    .iter()
                    .map(|k| Box::new(format!("%{k}%")) as Box<dyn rusqlite::types::ToSql>)
                    .collect();
                (sql, p)
            };

        let params_refs: Vec<&dyn rusqlite::types::ToSql> =
            params_vec.iter().map(|b| &**b).collect();

        let count: i64 = conn
            .query_row(&sql, params_refs.as_slice(), |row| row.get(0))
            .map_err(|e| CodememError::Storage(e.to_string()))?;
        Ok(count as usize)
    }

    /// List memories that contain a specific tag, optionally scoped to a namespace.
    pub fn list_memories_by_tag(
        &self,
        tag: &str,
        namespace: Option<&str>,
        limit: usize,
    ) -> Result<Vec<codemem_core::MemoryNode>, CodememError> {
        let conn = self.conn()?;
        let like_pattern = format!("%\"{tag}\"%");

        let (sql, params_vec): (String, Vec<Box<dyn rusqlite::types::ToSql>>) = if let Some(ns) =
            namespace
        {
            (
                    "SELECT id, content, memory_type, importance, confidence, access_count, \
                     content_hash, tags, metadata, namespace, created_at, updated_at, last_accessed_at \
                     FROM memories WHERE tags LIKE ?1 AND namespace = ?2 \
                     ORDER BY created_at DESC LIMIT ?3"
                        .to_string(),
                    vec![
                        Box::new(like_pattern) as Box<dyn rusqlite::types::ToSql>,
                        Box::new(ns.to_string()),
                        Box::new(limit as i64),
                    ],
                )
        } else {
            (
                    "SELECT id, content, memory_type, importance, confidence, access_count, \
                     content_hash, tags, metadata, namespace, created_at, updated_at, last_accessed_at \
                     FROM memories WHERE tags LIKE ?1 \
                     ORDER BY created_at DESC LIMIT ?2"
                        .to_string(),
                    vec![
                        Box::new(like_pattern) as Box<dyn rusqlite::types::ToSql>,
                        Box::new(limit as i64),
                    ],
                )
        };

        let params_refs: Vec<&dyn rusqlite::types::ToSql> =
            params_vec.iter().map(|b| &**b).collect();

        let mut stmt = conn
            .prepare(&sql)
            .map_err(|e| CodememError::Storage(e.to_string()))?;

        let rows = stmt
            .query_map(params_refs.as_slice(), |row| {
                let created_ts: i64 = row.get(10)?;
                let updated_ts: i64 = row.get(11)?;
                let accessed_ts: i64 = row.get(12)?;
                let tags_json: String = row.get(7)?;
                let metadata_json: String = row.get(8)?;
                let memory_type_str: String = row.get(2)?;

                Ok(codemem_core::MemoryNode {
                    id: row.get(0)?,
                    content: row.get(1)?,
                    memory_type: memory_type_str
                        .parse()
                        .unwrap_or(codemem_core::MemoryType::Context),
                    importance: row.get(3)?,
                    confidence: row.get(4)?,
                    access_count: row.get::<_, i64>(5).unwrap_or(0) as u32,
                    content_hash: row.get(6)?,
                    tags: serde_json::from_str(&tags_json).unwrap_or_default(),
                    metadata: serde_json::from_str(&metadata_json).unwrap_or_default(),
                    namespace: row.get(9)?,
                    created_at: chrono::DateTime::from_timestamp(created_ts, 0)
                        .unwrap_or_default()
                        .with_timezone(&chrono::Utc),
                    updated_at: chrono::DateTime::from_timestamp(updated_ts, 0)
                        .unwrap_or_default()
                        .with_timezone(&chrono::Utc),
                    last_accessed_at: chrono::DateTime::from_timestamp(accessed_ts, 0)
                        .unwrap_or_default()
                        .with_timezone(&chrono::Utc),
                })
            })
            .map_err(|e| CodememError::Storage(e.to_string()))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| CodememError::Storage(e.to_string()))?;

        Ok(rows)
    }

    // ── Session Management ─────────────────────────────────────────────

    /// Start a new session.
    pub fn start_session(&self, id: &str, namespace: Option<&str>) -> Result<(), CodememError> {
        let conn = self.conn()?;
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
        let conn = self.conn()?;
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

    // ── Session Activity Tracking ─────────────────────────────────

    /// Record a session activity event.
    pub fn record_session_activity(
        &self,
        session_id: &str,
        tool_name: &str,
        file_path: Option<&str>,
        directory: Option<&str>,
        pattern: Option<&str>,
    ) -> Result<(), CodememError> {
        let conn = self.conn()?;
        let now = chrono::Utc::now().timestamp();
        conn.execute(
            "INSERT INTO session_activity (session_id, tool_name, file_path, directory, pattern, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![session_id, tool_name, file_path, directory, pattern, now],
        )
        .map_err(|e| CodememError::Storage(e.to_string()))?;
        Ok(())
    }

    /// Get a summary of session activity counts.
    pub fn get_session_activity_summary(
        &self,
        session_id: &str,
    ) -> Result<codemem_core::SessionActivitySummary, CodememError> {
        let conn = self.conn()?;

        let files_read: i64 = conn
            .query_row(
                "SELECT COUNT(DISTINCT file_path) FROM session_activity
                 WHERE session_id = ?1 AND tool_name = 'Read' AND file_path IS NOT NULL",
                params![session_id],
                |row| row.get(0),
            )
            .map_err(|e| CodememError::Storage(e.to_string()))?;

        let files_edited: i64 = conn
            .query_row(
                "SELECT COUNT(DISTINCT file_path) FROM session_activity
                 WHERE session_id = ?1 AND tool_name IN ('Edit', 'Write') AND file_path IS NOT NULL",
                params![session_id],
                |row| row.get(0),
            )
            .map_err(|e| CodememError::Storage(e.to_string()))?;

        let searches: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM session_activity
                 WHERE session_id = ?1 AND tool_name IN ('Grep', 'Glob')",
                params![session_id],
                |row| row.get(0),
            )
            .map_err(|e| CodememError::Storage(e.to_string()))?;

        let total_actions: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM session_activity WHERE session_id = ?1",
                params![session_id],
                |row| row.get(0),
            )
            .map_err(|e| CodememError::Storage(e.to_string()))?;

        Ok(codemem_core::SessionActivitySummary {
            files_read: files_read as usize,
            files_edited: files_edited as usize,
            searches: searches as usize,
            total_actions: total_actions as usize,
        })
    }

    /// Get the most active directories in a session.
    pub fn get_session_hot_directories(
        &self,
        session_id: &str,
        limit: usize,
    ) -> Result<Vec<(String, usize)>, CodememError> {
        let conn = self.conn()?;
        let mut stmt = conn
            .prepare(
                "SELECT directory, COUNT(*) AS cnt FROM session_activity
                 WHERE session_id = ?1 AND directory IS NOT NULL
                 GROUP BY directory ORDER BY cnt DESC LIMIT ?2",
            )
            .map_err(|e| CodememError::Storage(e.to_string()))?;

        let rows = stmt
            .query_map(params![session_id, limit as i64], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
            })
            .map_err(|e| CodememError::Storage(e.to_string()))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| CodememError::Storage(e.to_string()))?;

        Ok(rows
            .into_iter()
            .map(|(dir, cnt)| (dir, cnt as usize))
            .collect())
    }

    /// Check whether an auto-insight dedup tag exists for a session.
    pub fn has_auto_insight(
        &self,
        session_id: &str,
        dedup_tag: &str,
    ) -> Result<bool, CodememError> {
        let conn = self.conn()?;
        let like_session = format!("%\"session_id\":\"{session_id}\"%");
        let like_dedup = format!("%\"auto_insight_tag\":\"{dedup_tag}\"%");
        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM memories
                 WHERE metadata LIKE ?1 AND metadata LIKE ?2",
                params![like_session, like_dedup],
                |row| row.get(0),
            )
            .map_err(|e| CodememError::Storage(e.to_string()))?;
        Ok(count > 0)
    }

    /// Count Read events in a directory during a session.
    pub fn count_directory_reads(
        &self,
        session_id: &str,
        directory: &str,
    ) -> Result<usize, CodememError> {
        let conn = self.conn()?;
        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM session_activity
                 WHERE session_id = ?1 AND tool_name = 'Read' AND directory = ?2",
                params![session_id, directory],
                |row| row.get(0),
            )
            .map_err(|e| CodememError::Storage(e.to_string()))?;
        Ok(count as usize)
    }

    /// Check if a file was read in the current session.
    pub fn was_file_read_in_session(
        &self,
        session_id: &str,
        file_path: &str,
    ) -> Result<bool, CodememError> {
        let conn = self.conn()?;
        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM session_activity
                 WHERE session_id = ?1 AND tool_name = 'Read' AND file_path = ?2",
                params![session_id, file_path],
                |row| row.get(0),
            )
            .map_err(|e| CodememError::Storage(e.to_string()))?;
        Ok(count > 0)
    }

    /// Count how many times a search pattern was used in a session.
    pub fn count_search_pattern_in_session(
        &self,
        session_id: &str,
        pattern: &str,
    ) -> Result<usize, CodememError> {
        let conn = self.conn()?;
        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM session_activity
                 WHERE session_id = ?1 AND tool_name IN ('Grep', 'Glob') AND pattern = ?2",
                params![session_id, pattern],
                |row| row.get(0),
            )
            .map_err(|e| CodememError::Storage(e.to_string()))?;
        Ok(count as usize)
    }

    /// List sessions with a limit.
    pub(crate) fn list_sessions_with_limit(
        &self,
        namespace: Option<&str>,
        limit: usize,
    ) -> Result<Vec<Session>, CodememError> {
        let conn = self.conn()?;
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
    // ── Graph Cleanup ───────────────────────────────────────────────

    /// Delete all graph nodes, their edges, and their embeddings where the
    /// node ID starts with the given prefix. Returns count of nodes deleted.
    pub fn delete_graph_nodes_by_prefix(&self, prefix: &str) -> Result<usize, CodememError> {
        let conn = self.conn()?;
        let like_pattern = format!("{prefix}%");

        // Delete edges where src or dst matches prefix
        conn.execute(
            "DELETE FROM graph_edges WHERE src LIKE ?1 OR dst LIKE ?1",
            params![like_pattern],
        )
        .map_err(|e| CodememError::Storage(e.to_string()))?;

        // Delete embeddings for matching nodes
        conn.execute(
            "DELETE FROM memory_embeddings WHERE memory_id LIKE ?1",
            params![like_pattern],
        )
        .map_err(|e| CodememError::Storage(e.to_string()))?;

        // Delete the nodes themselves
        let rows = conn
            .execute(
                "DELETE FROM graph_nodes WHERE id LIKE ?1",
                params![like_pattern],
            )
            .map_err(|e| CodememError::Storage(e.to_string()))?;

        Ok(rows)
    }
}

#[cfg(test)]
#[path = "tests/queries_tests.rs"]
mod tests;
