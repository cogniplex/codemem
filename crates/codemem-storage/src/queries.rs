//! Stats, consolidation, pattern queries, and session management on Storage.

use crate::{MapStorageErr, Storage};
use codemem_core::{CodememError, ConsolidationLogEntry, Session, StorageStats};
use rusqlite::params;

impl Storage {
    // ── Health / Diagnostics ────────────────────────────────────────────

    /// Run SQLite `PRAGMA integrity_check`. Returns `true` if the database is OK.
    pub fn integrity_check(&self) -> Result<bool, CodememError> {
        let conn = self.conn()?;
        let result: String = conn
            .query_row("PRAGMA integrity_check", [], |row| row.get(0))
            .storage_err()?;
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
            .storage_err()?;
        Ok(version)
    }

    // ── Stats ───────────────────────────────────────────────────────────

    /// Get database statistics in a single query.
    pub fn stats(&self) -> Result<StorageStats, CodememError> {
        let conn = self.conn()?;

        let (memory_count, embedding_count, node_count, edge_count) = conn
            .query_row(
                "SELECT
                    (SELECT COUNT(*) FROM memories) AS memory_count,
                    (SELECT COUNT(*) FROM memory_embeddings) AS embedding_count,
                    (SELECT COUNT(*) FROM graph_nodes) AS node_count,
                    (SELECT COUNT(*) FROM graph_edges) AS edge_count",
                [],
                |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        row.get::<_, i64>(1)?,
                        row.get::<_, i64>(2)?,
                        row.get::<_, i64>(3)?,
                    ))
                },
            )
            .storage_err()?;

        Ok(StorageStats {
            memory_count: memory_count as usize,
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
        .storage_err()?;
        Ok(())
    }

    /// Get the last consolidation run for each cycle type.
    pub fn last_consolidation_runs(&self) -> Result<Vec<ConsolidationLogEntry>, CodememError> {
        let conn = self.conn()?;
        let mut stmt = conn
            .prepare(
                "SELECT cycle_type, run_at, affected_count FROM consolidation_log
                 WHERE (cycle_type, run_at) IN (
                     SELECT cycle_type, MAX(run_at) FROM consolidation_log GROUP BY cycle_type
                 )
                 ORDER BY cycle_type",
            )
            .storage_err()?;

        let entries = stmt
            .query_map([], |row| {
                Ok(ConsolidationLogEntry {
                    cycle_type: row.get(0)?,
                    run_at: row.get(1)?,
                    affected_count: row.get::<_, i64>(2)? as usize,
                })
            })
            .storage_err()?
            .collect::<Result<Vec<_>, _>>()
            .storage_err()?;

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

        let mut stmt = conn.prepare(sql).storage_err()?;

        let rows = if let Some(ns) = namespace {
            stmt.query_map(params![ns, min_count as i64], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, String>(2)?,
                ))
            })
            .storage_err()?
            .collect::<Result<Vec<_>, _>>()
            .storage_err()?
        } else {
            stmt.query_map(params![min_count as i64], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, String>(2)?,
                ))
            })
            .storage_err()?
            .collect::<Result<Vec<_>, _>>()
            .storage_err()?
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

        let mut stmt = conn.prepare(sql).storage_err()?;

        let rows = if let Some(ns) = namespace {
            stmt.query_map(params![ns, min_count as i64], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, String>(2)?,
                ))
            })
            .storage_err()?
            .collect::<Result<Vec<_>, _>>()
            .storage_err()?
        } else {
            stmt.query_map(params![min_count as i64], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, String>(2)?,
                ))
            })
            .storage_err()?
            .collect::<Result<Vec<_>, _>>()
            .storage_err()?
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
    /// Returns (tool_name, count) pairs sorted by count descending.
    pub fn get_tool_usage_stats(
        &self,
        namespace: Option<&str>,
    ) -> Result<Vec<(String, usize)>, CodememError> {
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

        let mut stmt = conn.prepare(sql).storage_err()?;

        let rows = if let Some(ns) = namespace {
            stmt.query_map(params![ns], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
            })
            .storage_err()?
            .collect::<Result<Vec<_>, _>>()
            .storage_err()?
        } else {
            stmt.query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
            })
            .storage_err()?
            .collect::<Result<Vec<_>, _>>()
            .storage_err()?
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

        let mut stmt = conn.prepare(sql).storage_err()?;

        let rows = if let Some(ns) = namespace {
            stmt.query_map(params![ns, min_count as i64], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, String>(2)?,
                ))
            })
            .storage_err()?
            .collect::<Result<Vec<_>, _>>()
            .storage_err()?
        } else {
            stmt.query_map(params![min_count as i64], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, String>(2)?,
                ))
            })
            .storage_err()?
            .collect::<Result<Vec<_>, _>>()
            .storage_err()?
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

    /// List memories that contain a specific tag, optionally scoped to a namespace.
    /// Uses `json_each` for proper JSON array querying instead of LIKE patterns.
    pub fn list_memories_by_tag(
        &self,
        tag: &str,
        namespace: Option<&str>,
        limit: usize,
    ) -> Result<Vec<codemem_core::MemoryNode>, CodememError> {
        let conn = self.conn()?;

        let (sql, params_vec): (String, Vec<Box<dyn rusqlite::types::ToSql>>) = if let Some(ns) =
            namespace
        {
            (
                "SELECT m.id, m.content, m.memory_type, m.importance, m.confidence, m.access_count, \
                 m.content_hash, m.tags, m.metadata, m.namespace, m.session_id, m.repo, m.git_ref, m.expires_at, m.created_at, m.updated_at, m.last_accessed_at \
                 FROM memories m, json_each(m.tags) AS jt \
                 WHERE jt.value = ?1 AND m.namespace = ?2 \
                 AND (m.expires_at IS NULL OR m.expires_at > ?3) \
                 ORDER BY m.created_at DESC LIMIT ?4"
                    .to_string(),
                vec![
                    Box::new(tag.to_string()) as Box<dyn rusqlite::types::ToSql>,
                    Box::new(ns.to_string()),
                    Box::new(chrono::Utc::now().timestamp()),
                    Box::new(limit as i64),
                ],
            )
        } else {
            (
                "SELECT m.id, m.content, m.memory_type, m.importance, m.confidence, m.access_count, \
                 m.content_hash, m.tags, m.metadata, m.namespace, m.session_id, m.repo, m.git_ref, m.expires_at, m.created_at, m.updated_at, m.last_accessed_at \
                 FROM memories m, json_each(m.tags) AS jt \
                 WHERE jt.value = ?1 \
                 AND (m.expires_at IS NULL OR m.expires_at > ?2) \
                 ORDER BY m.created_at DESC LIMIT ?3"
                    .to_string(),
                vec![
                    Box::new(tag.to_string()) as Box<dyn rusqlite::types::ToSql>,
                    Box::new(chrono::Utc::now().timestamp()),
                    Box::new(limit as i64),
                ],
            )
        };

        let params_refs: Vec<&dyn rusqlite::types::ToSql> =
            params_vec.iter().map(|b| &**b).collect();

        let mut stmt = conn.prepare(&sql).storage_err()?;

        let rows = stmt
            .query_map(params_refs.as_slice(), |row| {
                let expires_ts: Option<i64> = row.get(13)?;
                let created_ts: i64 = row.get(14)?;
                let updated_ts: i64 = row.get(15)?;
                let accessed_ts: i64 = row.get(16)?;
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
                    session_id: row.get(10)?,
                    repo: row.get(11)?,
                    git_ref: row.get(12)?,
                    expires_at: expires_ts
                        .and_then(|ts| chrono::DateTime::from_timestamp(ts, 0))
                        .map(|dt| dt.with_timezone(&chrono::Utc)),
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
            .storage_err()?
            .collect::<Result<Vec<_>, _>>()
            .storage_err()?;

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
        .storage_err()?;
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
        .storage_err()?;
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
        .storage_err()?;
        Ok(())
    }

    /// Get a summary of session activity counts using a single query with conditional aggregation.
    pub fn get_session_activity_summary(
        &self,
        session_id: &str,
    ) -> Result<codemem_core::SessionActivitySummary, CodememError> {
        let conn = self.conn()?;

        let (files_read, files_edited, searches, total_actions) = conn
            .query_row(
                "SELECT
                     COUNT(DISTINCT CASE WHEN tool_name = 'Read' AND file_path IS NOT NULL THEN file_path END),
                     COUNT(DISTINCT CASE WHEN tool_name IN ('Edit', 'Write') AND file_path IS NOT NULL THEN file_path END),
                     SUM(CASE WHEN tool_name IN ('Grep', 'Glob') THEN 1 ELSE 0 END),
                     COUNT(*)
                 FROM session_activity
                 WHERE session_id = ?1",
                params![session_id],
                |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        row.get::<_, i64>(1)?,
                        row.get::<_, i64>(2)?,
                        row.get::<_, i64>(3)?,
                    ))
                },
            )
            .storage_err()?;

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
            .storage_err()?;

        let rows = stmt
            .query_map(params![session_id, limit as i64], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
            })
            .storage_err()?
            .collect::<Result<Vec<_>, _>>()
            .storage_err()?;

        Ok(rows
            .into_iter()
            .map(|(dir, cnt)| (dir, cnt as usize))
            .collect())
    }

    /// Check whether an auto-insight dedup tag exists for a session.
    /// Uses `json_extract` with proper parameter binding instead of LIKE on JSON.
    pub fn has_auto_insight(
        &self,
        session_id: &str,
        dedup_tag: &str,
    ) -> Result<bool, CodememError> {
        let conn = self.conn()?;
        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM memories
                 WHERE json_extract(metadata, '$.session_id') = ?1
                   AND json_extract(metadata, '$.auto_insight_tag') = ?2",
                params![session_id, dedup_tag],
                |row| row.get(0),
            )
            .storage_err()?;
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
            .storage_err()?;
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
            .storage_err()?;
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
            .storage_err()?;
        Ok(count as usize)
    }

    /// List sessions with a limit.
    pub(crate) fn list_sessions_with_limit(
        &self,
        namespace: Option<&str>,
        limit: usize,
    ) -> Result<Vec<Session>, CodememError> {
        let conn = self.conn()?;
        let sql_with_ns = "SELECT s.id, s.namespace, s.started_at, s.ended_at, (SELECT COUNT(*) FROM memories m WHERE m.session_id = s.id) as memory_count, s.summary FROM sessions s WHERE s.namespace = ?1 ORDER BY s.started_at DESC LIMIT ?2";
        let sql_all = "SELECT s.id, s.namespace, s.started_at, s.ended_at, (SELECT COUNT(*) FROM memories m WHERE m.session_id = s.id) as memory_count, s.summary FROM sessions s ORDER BY s.started_at DESC LIMIT ?1";

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
            let mut stmt = conn.prepare(sql_with_ns).storage_err()?;
            let rows = stmt
                .query_map(params![ns, limit as i64], map_row)
                .storage_err()?;
            rows.collect::<Result<Vec<_>, _>>().storage_err()
        } else {
            let mut stmt = conn.prepare(sql_all).storage_err()?;
            let rows = stmt
                .query_map(params![limit as i64], map_row)
                .storage_err()?;
            rows.collect::<Result<Vec<_>, _>>().storage_err()
        }
    }

    // ── Tag-based Queries ─────────────────────────────────────────

    /// Find memory IDs whose tags JSON array contains the given tag value.
    /// Optionally scoped to a namespace. Excludes the given `exclude_id`.
    /// Returns at most 50 results ordered by creation time (most recent siblings first).
    pub fn find_memory_ids_by_tag(
        &self,
        tag: &str,
        namespace: Option<&str>,
        exclude_id: &str,
    ) -> Result<Vec<String>, CodememError> {
        let conn = self.conn()?;

        // Use json_each() for exact tag matching instead of LIKE (safe against %, _, " in tags).
        let (sql, params_vec): (String, Vec<Box<dyn rusqlite::types::ToSql>>) =
            if let Some(ns) = namespace {
                (
                    "SELECT DISTINCT m.id FROM memories m, json_each(m.tags) t \
                 WHERE t.value = ?1 AND m.namespace IS ?2 AND m.id != ?3 \
                 ORDER BY m.created_at DESC LIMIT 50"
                        .to_string(),
                    vec![
                        Box::new(tag.to_string()) as Box<dyn rusqlite::types::ToSql>,
                        Box::new(ns.to_string()),
                        Box::new(exclude_id.to_string()),
                    ],
                )
            } else {
                (
                    "SELECT DISTINCT m.id FROM memories m, json_each(m.tags) t \
                 WHERE t.value = ?1 AND m.namespace IS NULL AND m.id != ?2 \
                 ORDER BY m.created_at DESC LIMIT 50"
                        .to_string(),
                    vec![
                        Box::new(tag.to_string()) as Box<dyn rusqlite::types::ToSql>,
                        Box::new(exclude_id.to_string()),
                    ],
                )
            };

        let refs: Vec<&dyn rusqlite::types::ToSql> =
            params_vec.iter().map(|p| p.as_ref()).collect();

        let mut stmt = conn.prepare(&sql).storage_err()?;

        let ids = stmt
            .query_map(refs.as_slice(), |row| row.get(0))
            .storage_err()?
            .collect::<Result<Vec<String>, _>>()
            .storage_err()?;

        Ok(ids)
    }

    // ── Graph Cleanup ───────────────────────────────────────────────

    /// Delete all graph nodes, their edges, and their embeddings where the
    /// node ID starts with the given prefix. Returns count of nodes deleted.
    /// Wrapped in a transaction so all three DELETEs are atomic.
    pub fn delete_graph_nodes_by_prefix(&self, prefix: &str) -> Result<usize, CodememError> {
        let conn = self.conn()?;
        let like_pattern = format!("{prefix}%");

        let tx = conn.unchecked_transaction().storage_err()?;

        // Delete edges where src or dst matches prefix
        tx.execute(
            "DELETE FROM graph_edges WHERE src LIKE ?1 OR dst LIKE ?1",
            params![like_pattern],
        )
        .storage_err()?;

        // Delete embeddings for matching nodes
        tx.execute(
            "DELETE FROM memory_embeddings WHERE memory_id LIKE ?1",
            params![like_pattern],
        )
        .storage_err()?;

        // Delete the nodes themselves
        let rows = tx
            .execute(
                "DELETE FROM graph_nodes WHERE id LIKE ?1",
                params![like_pattern],
            )
            .storage_err()?;

        tx.commit().storage_err()?;

        Ok(rows)
    }
}

#[cfg(test)]
#[path = "tests/queries_tests.rs"]
mod tests;
