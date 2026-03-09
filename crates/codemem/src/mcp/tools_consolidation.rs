//! Consolidation & lifecycle tools: unified consolidate, recall_with_impact (legacy compat),
//! get_decision_chain, detect_patterns, session_checkpoint.

use super::types::ToolResult;
use super::McpServer;
use serde_json::{json, Value};

impl McpServer {
    /// Unified consolidation tool: run one or all consolidation cycles.
    ///
    /// `mode`: "auto" | "decay" | "creative" | "cluster" | "forget" | "summarize"
    /// Auto mode checks consolidation_status and runs appropriate cycles.
    pub(crate) fn tool_consolidate(&self, args: &Value) -> ToolResult {
        let mode = args.get("mode").and_then(|v| v.as_str()).unwrap_or("auto");

        match mode {
            "decay" => {
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
            "creative" => match self.engine.consolidate_creative() {
                Ok(result) => ToolResult::text(
                    json!({
                        "cycle": result.cycle,
                        "new_connections": result.affected,
                        "algorithm": "vector_knn",
                    })
                    .to_string(),
                ),
                Err(e) => ToolResult::tool_error(format!("Creative cycle failed: {e}")),
            },
            "cluster" => {
                let similarity_threshold =
                    args.get("similarity_threshold").and_then(|v| v.as_f64());

                match self.engine.consolidate_cluster(similarity_threshold) {
                    Ok(result) => ToolResult::text(result.details.to_string()),
                    Err(e) => ToolResult::tool_error(format!("Cluster cycle failed: {e}")),
                }
            }
            "forget" => {
                let importance_threshold =
                    args.get("importance_threshold").and_then(|v| v.as_f64());

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

                match self.engine.consolidate_forget(
                    importance_threshold,
                    tags_ref,
                    max_access_count,
                ) {
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
            "summarize" => {
                let min_cluster_size = args
                    .get("cluster_size")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize);

                match self.engine.consolidate_summarize(min_cluster_size) {
                    Ok(result) => ToolResult::text(result.details.to_string()),
                    Err(e) => ToolResult::tool_error(format!("{e}")),
                }
            }
            "auto" => {
                // Auto mode: get status, then run each cycle, return combined results
                let status = self.engine.consolidation_status().unwrap_or_default();
                let mut results = json!({});

                // Always run decay in auto mode
                match self.engine.consolidate_decay(None) {
                    Ok(r) => {
                        results["decay"] = json!({"cycle": r.cycle, "affected": r.affected});
                    }
                    Err(e) => {
                        results["decay"] = json!({"error": format!("{e}")});
                    }
                }

                // Run creative
                match self.engine.consolidate_creative() {
                    Ok(r) => {
                        results["creative"] =
                            json!({"cycle": r.cycle, "new_connections": r.affected});
                    }
                    Err(e) => {
                        results["creative"] = json!({"error": format!("{e}")});
                    }
                }

                // Run cluster
                match self.engine.consolidate_cluster(None) {
                    Ok(r) => {
                        results["cluster"] = json!({"cycle": r.cycle, "affected": r.affected});
                    }
                    Err(e) => {
                        results["cluster"] = json!({"error": format!("{e}")});
                    }
                }

                // Run forget with safe defaults
                match self.engine.consolidate_forget(None, None, None) {
                    Ok(r) => {
                        results["forget"] = json!({"cycle": r.cycle, "deleted": r.affected});
                    }
                    Err(e) => {
                        results["forget"] = json!({"error": format!("{e}")});
                    }
                }

                // Run summarize
                match self.engine.consolidate_summarize(None) {
                    Ok(r) => {
                        results["summarize"] = json!({"cycle": r.cycle, "affected": r.affected});
                    }
                    Err(e) => {
                        results["summarize"] = json!({"error": format!("{e}")});
                    }
                }

                // Run orphan detection
                match self.engine.detect_orphans(None) {
                    Ok((sym, edges)) => {
                        results["orphans"] =
                            json!({"symbols_cleaned": sym, "edges_cleaned": edges});
                    }
                    Err(e) => {
                        results["orphans"] = json!({"error": format!("{e}")});
                    }
                }

                // Include status
                let mut status_json = json!({});
                for entry in &status {
                    status_json[&entry.cycle_type] = json!({
                        "last_run": entry.last_run,
                        "affected": entry.affected_count,
                    });
                }
                results["status"] = status_json;

                ToolResult::text(
                    serde_json::to_string_pretty(&results).expect("JSON serialization of literal"),
                )
            }
            other => ToolResult::tool_error(format!(
                "Unknown mode: '{other}'. Use: auto, decay, creative, cluster, forget, summarize"
            )),
        }
    }

    /// MCP tool: detect_patterns -- detect cross-session patterns in stored memories.
    /// Supports `format` parameter: "json" (default), "markdown", "both".
    pub(crate) fn tool_detect_patterns(&self, args: &Value) -> ToolResult {
        let min_frequency = args
            .get("min_frequency")
            .and_then(|v| v.as_u64())
            .unwrap_or(3) as usize;
        let namespace = args.get("namespace").and_then(|v| v.as_str());
        let format = args
            .get("format")
            .and_then(|v| v.as_str())
            .unwrap_or("json");

        let total_sessions = self.engine.storage().session_count(namespace).unwrap_or(10);

        match codemem_engine::patterns::detect_patterns(
            self.engine.storage(),
            namespace,
            min_frequency,
            total_sessions,
        ) {
            Ok(detected) => match format {
                "markdown" => {
                    let markdown = codemem_engine::patterns::generate_insights(&detected);
                    ToolResult::text(markdown)
                }
                "both" => {
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
                    let markdown = codemem_engine::patterns::generate_insights(&detected);
                    ToolResult::text(
                        serde_json::to_string_pretty(&json!({
                            "patterns": json_patterns,
                            "count": detected.len(),
                            "markdown": markdown,
                        }))
                        .expect("JSON serialization of literal"),
                    )
                }
                _ => {
                    // Default: "json"
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
            },
            Err(e) => ToolResult::tool_error(format!("Pattern detection error: {e}")),
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

    /// MCP tool: session_context -- returns recent memories, pending analyses, active patterns, focus areas.
    pub(crate) fn tool_session_context(&self, args: &Value) -> ToolResult {
        let namespace = args.get("namespace").and_then(|v| v.as_str());

        let ctx = match self.engine.session_context(namespace) {
            Ok(ctx) => ctx,
            Err(e) => return ToolResult::tool_error(format!("Session context error: {e}")),
        };

        let recent_memories: Vec<Value> = ctx
            .recent_memories
            .iter()
            .map(|m| {
                json!({
                    "id": m.id,
                    "content": m.content,
                    "memory_type": m.memory_type.to_string(),
                    "importance": m.importance,
                    "tags": m.tags,
                    "created_at": m.created_at.to_rfc3339(),
                })
            })
            .collect();

        let pending: Vec<Value> = ctx
            .pending_analyses
            .iter()
            .map(|m| {
                json!({
                    "id": m.id,
                    "content": m.content,
                    "tags": m.tags,
                })
            })
            .collect();

        let patterns: Vec<Value> = ctx
            .active_patterns
            .iter()
            .take(5)
            .map(|p| {
                json!({
                    "pattern_type": p.pattern_type.to_string(),
                    "description": p.description,
                    "confidence": p.confidence,
                })
            })
            .collect();

        ToolResult::text(
            serde_json::to_string_pretty(&json!({
                "recent_memories": recent_memories,
                "pending_analyses": pending,
                "active_patterns": patterns,
            }))
            .expect("JSON serialization of literal"),
        )
    }
}

#[cfg(test)]
#[path = "tests/tools_consolidation_tests.rs"]
mod tests;
