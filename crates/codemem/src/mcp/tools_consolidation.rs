//! Consolidation & lifecycle tools: decay, creative, cluster, forget,
//! consolidation_status, recall_with_impact, get_decision_chain,
//! detect_patterns, pattern_insights, session_checkpoint, consolidate_summarize.

use super::types::ToolResult;
use super::McpServer;
use serde_json::{json, Value};

impl McpServer {
    /// MCP tool: consolidate_decay -- power-law decay that rewards access frequency.
    pub(crate) fn tool_consolidate_decay(&self, args: &Value) -> ToolResult {
        let threshold_days = args
            .get("threshold_days")
            .and_then(|v| v.as_u64())
            .map(|v| v as i64);

        match self.engine.consolidate_decay(threshold_days) {
            Ok(result) => ToolResult::text(
                json!({
                    "cycle": result.cycle,
                    "affected": result.affected,
                    "threshold_days": threshold_days.unwrap_or(30),
                    "algorithm": "power_law",
                })
                .to_string(),
            ),
            Err(e) => ToolResult::tool_error(format!("Decay failed: {e}")),
        }
    }

    /// MCP tool: consolidate_creative -- O(n log n) semantic creative consolidation.
    pub(crate) fn tool_consolidate_creative(&self, args: &Value) -> ToolResult {
        let _ = args;

        match self.engine.consolidate_creative() {
            Ok(result) => ToolResult::text(
                json!({
                    "cycle": result.cycle,
                    "new_connections": result.affected,
                    "algorithm": "vector_knn",
                })
                .to_string(),
            ),
            Err(e) => ToolResult::tool_error(format!("Creative cycle failed: {e}")),
        }
    }

    /// MCP tool: consolidate_cluster -- semantic deduplication using cosine similarity.
    pub(crate) fn tool_consolidate_cluster(&self, args: &Value) -> ToolResult {
        let similarity_threshold = args.get("similarity_threshold").and_then(|v| v.as_f64());

        match self.engine.consolidate_cluster(similarity_threshold) {
            Ok(result) => ToolResult::text(result.details.to_string()),
            Err(e) => ToolResult::tool_error(format!("Cluster cycle failed: {e}")),
        }
    }

    /// MCP tool: consolidate_forget -- delete low-importance, never-accessed memories.
    pub(crate) fn tool_consolidate_forget(&self, args: &Value) -> ToolResult {
        let importance_threshold = args.get("importance_threshold").and_then(|v| v.as_f64());

        let target_tags: Vec<String> = args
            .get("target_tags")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        let max_access_count = args
            .get("max_access_count")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32);

        let tags_ref = if target_tags.is_empty() {
            None
        } else {
            Some(target_tags.as_slice())
        };

        match self
            .engine
            .consolidate_forget(importance_threshold, tags_ref, max_access_count)
        {
            Ok(result) => ToolResult::text(
                json!({
                    "cycle": result.cycle,
                    "deleted": result.affected,
                    "threshold": importance_threshold.unwrap_or(0.1),
                })
                .to_string(),
            ),
            Err(e) => ToolResult::tool_error(format!("Forget cycle failed: {e}")),
        }
    }

    /// MCP tool: consolidation_status -- show last run of each consolidation cycle.
    pub(crate) fn tool_consolidation_status(&self) -> ToolResult {
        match self.engine.consolidation_status() {
            Ok(entries) => {
                let mut cycles = json!({});
                for entry in &entries {
                    cycles[&entry.cycle_type] = json!({
                        "last_run": entry.last_run,
                        "affected": entry.affected_count,
                    });
                }
                ToolResult::text(json!({ "cycles": cycles }).to_string())
            }
            Err(e) => ToolResult::tool_error(format!("Failed to query status: {e}")),
        }
    }

    // ── Impact-Aware Recall & Decision Chain Tools ────────────────────────

    /// MCP tool: recall_with_impact -- recall memories with PageRank-enriched impact data.
    pub(crate) fn tool_recall_with_impact(&self, args: &Value) -> ToolResult {
        let query = match args.get("query").and_then(|v| v.as_str()) {
            Some(q) if !q.is_empty() => q,
            _ => return ToolResult::tool_error("Missing or empty 'query' parameter"),
        };

        let k = args.get("k").and_then(|v| v.as_u64()).unwrap_or(10) as usize;
        let namespace_filter: Option<&str> = args.get("namespace").and_then(|v| v.as_str());

        match self.engine.recall_with_impact(query, k, namespace_filter) {
            Ok(results) if results.is_empty() => ToolResult::text("No matching memories found."),
            Ok(results) => {
                let output: Vec<Value> = results
                    .iter()
                    .map(|r| {
                        json!({
                            "id": r.search_result.memory.id,
                            "content": r.search_result.memory.content,
                            "memory_type": r.search_result.memory.memory_type.to_string(),
                            "score": format!("{:.4}", r.search_result.score),
                            "importance": r.search_result.memory.importance,
                            "tags": r.search_result.memory.tags,
                            "access_count": r.search_result.memory.access_count,
                            "impact": {
                                "pagerank": format!("{:.6}", r.pagerank),
                                "centrality": format!("{:.6}", r.centrality),
                                "connected_decisions": r.connected_decisions,
                                "dependent_files": r.dependent_files,
                                "modification_count": r.search_result.memory.access_count,
                            }
                        })
                    })
                    .collect();
                ToolResult::text(
                    serde_json::to_string_pretty(&output).expect("JSON serialization of literal"),
                )
            }
            Err(e) => ToolResult::tool_error(format!("Recall error: {e}")),
        }
    }

    /// MCP tool: get_decision_chain -- follow decision evolution through the graph.
    pub(crate) fn tool_get_decision_chain(&self, args: &Value) -> ToolResult {
        let file_path: Option<&str> = args.get("file_path").and_then(|v| v.as_str());
        let topic: Option<&str> = args.get("topic").and_then(|v| v.as_str());

        if file_path.is_none() && topic.is_none() {
            return ToolResult::tool_error("Must provide either 'file_path' or 'topic' parameter");
        }

        match self.engine.get_decision_chain(file_path, topic) {
            Ok(chain) if chain.decisions.is_empty() => {
                ToolResult::text("No decision memories found matching the criteria.")
            }
            Ok(chain) => {
                let decisions: Vec<Value> = chain
                    .decisions
                    .iter()
                    .map(|entry| {
                        let connections: Vec<Value> = entry
                            .connections
                            .iter()
                            .map(|c| {
                                json!({
                                    "relationship": c.relationship,
                                    "source": c.source,
                                    "target": c.target,
                                })
                            })
                            .collect();
                        json!({
                            "id": entry.memory.id,
                            "content": entry.memory.content,
                            "importance": entry.memory.importance,
                            "tags": entry.memory.tags,
                            "created_at": entry.memory.created_at.to_rfc3339(),
                            "connections": connections,
                        })
                    })
                    .collect();

                let response = json!({
                    "chain_length": chain.chain_length,
                    "filter": {
                        "file_path": chain.file_path,
                        "topic": chain.topic,
                    },
                    "decisions": decisions,
                });

                ToolResult::text(
                    serde_json::to_string_pretty(&response).expect("JSON serialization of literal"),
                )
            }
            Err(e) => ToolResult::tool_error(format!("{e}")),
        }
    }

    /// MCP tool: detect_patterns -- detect cross-session patterns in stored memories.
    pub(crate) fn tool_detect_patterns(&self, args: &Value) -> ToolResult {
        let min_frequency = args
            .get("min_frequency")
            .and_then(|v| v.as_u64())
            .unwrap_or(3) as usize;
        let namespace = args.get("namespace").and_then(|v| v.as_str());

        let total_sessions = self.engine.storage.session_count(namespace).unwrap_or(10);

        match codemem_engine::patterns::detect_patterns(
            &*self.engine.storage,
            namespace,
            min_frequency,
            total_sessions,
        ) {
            Ok(detected) => {
                let json_patterns: Vec<Value> = detected
                    .iter()
                    .map(|p| {
                        json!({
                            "pattern_type": p.pattern_type.to_string(),
                            "description": p.description,
                            "frequency": p.frequency,
                            "confidence": p.confidence,
                            "related_memories": p.related_memories,
                        })
                    })
                    .collect();
                ToolResult::text(
                    serde_json::to_string_pretty(&json!({
                        "patterns": json_patterns,
                        "count": detected.len(),
                    }))
                    .expect("JSON serialization of literal"),
                )
            }
            Err(e) => ToolResult::tool_error(format!("Pattern detection error: {e}")),
        }
    }

    /// MCP tool: pattern_insights -- generate human-readable pattern insights as markdown.
    pub(crate) fn tool_pattern_insights(&self, args: &Value) -> ToolResult {
        let min_frequency = args
            .get("min_frequency")
            .and_then(|v| v.as_u64())
            .unwrap_or(2) as usize;
        let namespace = args.get("namespace").and_then(|v| v.as_str());

        let total_sessions = self.engine.storage.session_count(namespace).unwrap_or(10);

        match codemem_engine::patterns::detect_patterns(
            &*self.engine.storage,
            namespace,
            min_frequency,
            total_sessions,
        ) {
            Ok(detected) => {
                let markdown = codemem_engine::patterns::generate_insights(&detected);
                ToolResult::text(markdown)
            }
            Err(e) => ToolResult::tool_error(format!("Pattern insights error: {e}")),
        }
    }

    /// MCP tool: consolidate_summarize -- LLM-powered consolidation.
    pub(crate) fn tool_consolidate_summarize(&self, args: &Value) -> ToolResult {
        let min_cluster_size = args
            .get("cluster_size")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);

        match self.engine.consolidate_summarize(min_cluster_size) {
            Ok(result) => ToolResult::text(result.details.to_string()),
            Err(e) => ToolResult::tool_error(format!("{e}")),
        }
    }

    /// MCP tool: session_checkpoint -- mid-session progress report with pattern detection.
    pub(crate) fn tool_session_checkpoint(&self, args: &Value) -> ToolResult {
        let session_id = match args.get("session_id").and_then(|v| v.as_str()) {
            Some(sid) if !sid.is_empty() => sid,
            _ => return ToolResult::tool_error("Required parameter: session_id"),
        };
        let namespace = args.get("namespace").and_then(|v| v.as_str());

        match self.engine.session_checkpoint(session_id, namespace) {
            Ok(checkpoint) => ToolResult::text(checkpoint.report),
            Err(e) => ToolResult::tool_error(format!("Failed to get activity: {e}")),
        }
    }
}

#[cfg(test)]
#[path = "tests/tools_consolidation_tests.rs"]
mod tests;
