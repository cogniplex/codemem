//! Consolidation & lifecycle tools: decay, creative, cluster, forget,
//! consolidation_status, recall_with_impact, get_decision_chain,
//! detect_patterns, pattern_insights, session_checkpoint, consolidate_summarize.

use super::types::ToolResult;
use super::McpServer;
use codemem_core::{
    GraphBackend, MemoryNode, MemoryType, NodeKind, RelationshipType, SearchResult,
};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};

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

        // Reuse the core recall logic
        let results = match self.recall(query, k, None, namespace_filter, &[], None, None) {
            Ok(r) => r,
            Err(e) => return ToolResult::tool_error(format!("Recall error: {e}")),
        };

        if results.is_empty() {
            return ToolResult::text("No matching memories found.");
        }

        // Enrich each result with impact data
        let graph = match self.lock_graph() {
            Ok(g) => g,
            Err(e) => return ToolResult::tool_error(format!("Lock error: {e}")),
        };

        let output: Vec<Value> = results
            .iter()
            .map(|r| self.build_impact_result(r, &graph))
            .collect();

        ToolResult::text(
            serde_json::to_string_pretty(&output).expect("JSON serialization of literal"),
        )
    }

    /// Build a JSON value with impact data for a single search result.
    fn build_impact_result(
        &self,
        r: &SearchResult,
        graph: &codemem_storage::graph::GraphEngine,
    ) -> Value {
        let memory_id = &r.memory.id;

        let pagerank = graph.get_pagerank(memory_id);
        let centrality = graph.get_betweenness(memory_id);

        let connected_decisions: Vec<String> = graph
            .get_edges(memory_id)
            .unwrap_or_default()
            .iter()
            .filter_map(|e| {
                let other_id = if e.src == *memory_id { &e.dst } else { &e.src };
                self.engine
                    .storage
                    .get_memory(other_id)
                    .ok()
                    .flatten()
                    .and_then(|m| {
                        if m.memory_type == MemoryType::Decision {
                            Some(m.id)
                        } else {
                            None
                        }
                    })
            })
            .collect();

        let dependent_files: Vec<String> = graph
            .get_edges(memory_id)
            .unwrap_or_default()
            .iter()
            .filter_map(|e| {
                let other_id = if e.src == *memory_id { &e.dst } else { &e.src };
                graph.get_node(other_id).ok().flatten().and_then(|n| {
                    if n.kind == NodeKind::File {
                        Some(n.label.clone())
                    } else {
                        n.payload
                            .get("file_path")
                            .and_then(|v| v.as_str().map(String::from))
                    }
                })
            })
            .collect();

        json!({
            "id": r.memory.id,
            "content": r.memory.content,
            "memory_type": r.memory.memory_type.to_string(),
            "score": format!("{:.4}", r.score),
            "importance": r.memory.importance,
            "tags": r.memory.tags,
            "access_count": r.memory.access_count,
            "impact": {
                "pagerank": format!("{:.6}", pagerank),
                "centrality": format!("{:.6}", centrality),
                "connected_decisions": connected_decisions,
                "dependent_files": dependent_files,
                "modification_count": r.memory.access_count,
            }
        })
    }

    /// MCP tool: get_decision_chain -- follow decision evolution through the graph.
    pub(crate) fn tool_get_decision_chain(&self, args: &Value) -> ToolResult {
        let file_path: Option<&str> = args.get("file_path").and_then(|v| v.as_str());
        let topic: Option<&str> = args.get("topic").and_then(|v| v.as_str());

        if file_path.is_none() && topic.is_none() {
            return ToolResult::tool_error("Must provide either 'file_path' or 'topic' parameter");
        }

        let graph = match self.lock_graph() {
            Ok(g) => g,
            Err(e) => return ToolResult::tool_error(format!("Lock error: {e}")),
        };

        // Find all Decision-type memories matching the file_path or topic
        let ids = match self.engine.storage.list_memory_ids() {
            Ok(ids) => ids,
            Err(e) => return ToolResult::tool_error(format!("Storage error: {e}")),
        };

        let decision_edge_types = [
            RelationshipType::EvolvedInto,
            RelationshipType::LeadsTo,
            RelationshipType::DerivedFrom,
        ];

        // Collect all Decision memories matching the filter
        let mut decision_memories: Vec<MemoryNode> = Vec::new();
        for id in &ids {
            if let Ok(Some(memory)) = self.engine.storage.get_memory(id) {
                if memory.memory_type != MemoryType::Decision {
                    continue;
                }

                let content_lower = memory.content.to_lowercase();
                let tags_lower: String = memory.tags.join(" ").to_lowercase();

                let matches = if let Some(fp) = file_path {
                    content_lower.contains(&fp.to_lowercase())
                        || tags_lower.contains(&fp.to_lowercase())
                        || memory
                            .metadata
                            .get("file_path")
                            .and_then(|v| v.as_str())
                            .map(|v| v.to_lowercase().contains(&fp.to_lowercase()))
                            .unwrap_or(false)
                } else if let Some(t) = topic {
                    let t_lower = t.to_lowercase();
                    content_lower.contains(&t_lower) || tags_lower.contains(&t_lower)
                } else {
                    false
                };

                if matches {
                    decision_memories.push(memory);
                }
            }
        }

        if decision_memories.is_empty() {
            return ToolResult::text("No decision memories found matching the criteria.");
        }

        // Expand through decision-related edges to find the full chain
        let mut chain_ids: HashSet<String> = HashSet::new();
        let mut to_explore: Vec<String> = decision_memories.iter().map(|m| m.id.clone()).collect();

        while let Some(current_id) = to_explore.pop() {
            if !chain_ids.insert(current_id.clone()) {
                continue;
            }

            if let Ok(edges) = graph.get_edges(&current_id) {
                for edge in &edges {
                    if decision_edge_types.contains(&edge.relationship) {
                        let other_id = if edge.src == current_id {
                            &edge.dst
                        } else {
                            &edge.src
                        };
                        if !chain_ids.contains(other_id) {
                            // Only follow to other Decision memories
                            if let Ok(Some(m)) = self.engine.storage.get_memory(other_id) {
                                if m.memory_type == MemoryType::Decision {
                                    to_explore.push(other_id.clone());
                                }
                            }
                        }
                    }
                }
            }
        }

        // Collect all chain memories and sort by created_at (temporal order)
        let mut chain: Vec<Value> = Vec::new();
        for id in &chain_ids {
            if let Ok(Some(memory)) = self.engine.storage.get_memory(id) {
                // Find edges connecting this memory within the chain
                let connections: Vec<Value> = graph
                    .get_edges(id)
                    .unwrap_or_default()
                    .iter()
                    .filter(|e| {
                        decision_edge_types.contains(&e.relationship)
                            && (chain_ids.contains(&e.src) && chain_ids.contains(&e.dst))
                    })
                    .map(|e| {
                        json!({
                            "relationship": e.relationship.to_string(),
                            "source": e.src,
                            "target": e.dst,
                        })
                    })
                    .collect();

                chain.push(json!({
                    "id": memory.id,
                    "content": memory.content,
                    "importance": memory.importance,
                    "tags": memory.tags,
                    "created_at": memory.created_at.to_rfc3339(),
                    "connections": connections,
                }));
            }
        }

        // Sort chronologically
        chain.sort_by(|a, b| {
            let a_dt = a["created_at"].as_str().unwrap_or("");
            let b_dt = b["created_at"].as_str().unwrap_or("");
            a_dt.cmp(b_dt)
        });

        let response = json!({
            "chain_length": chain.len(),
            "filter": {
                "file_path": file_path,
                "topic": topic,
            },
            "decisions": chain,
        });

        ToolResult::text(
            serde_json::to_string_pretty(&response).expect("JSON serialization of literal"),
        )
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

        // 1. Get session activity summary
        let activity = match self.engine.storage.get_session_activity_summary(session_id) {
            Ok(a) => a,
            Err(e) => return ToolResult::tool_error(format!("Failed to get activity: {e}")),
        };

        // 2. Run session-scoped pattern detection (lower thresholds for single session)
        let total_sessions = self
            .engine
            .storage
            .session_count(namespace)
            .unwrap_or(1)
            .max(1);

        let session_patterns = codemem_engine::patterns::detect_patterns(
            &*self.engine.storage,
            namespace,
            2, // session-scoped: min_frequency=2
            total_sessions,
        )
        .unwrap_or_default();

        // Cross-session patterns with higher threshold
        let cross_patterns = codemem_engine::patterns::detect_patterns(
            &*self.engine.storage,
            namespace,
            3, // cross-session: min_frequency=3
            total_sessions,
        )
        .unwrap_or_default();

        // 3. Store new session patterns as Insight memories (with dedup)
        let mut stored_patterns = 0usize;
        for pattern in &session_patterns {
            let dedup_tag = format!("checkpoint:{}:{}", session_id, pattern.description);
            let already_exists = self
                .engine
                .storage
                .has_auto_insight(session_id, &dedup_tag)
                .unwrap_or(true);
            if !already_exists && pattern.confidence > 0.3 {
                let now = chrono::Utc::now();
                let hash = codemem_storage::Storage::content_hash(&pattern.description);
                let mut metadata = HashMap::new();
                metadata.insert("session_id".to_string(), json!(session_id));
                metadata.insert("auto_insight_tag".to_string(), json!(dedup_tag));
                metadata.insert("source".to_string(), json!("session_checkpoint"));
                metadata.insert(
                    "pattern_type".to_string(),
                    json!(pattern.pattern_type.to_string()),
                );

                let mem = MemoryNode {
                    id: uuid::Uuid::new_v4().to_string(),
                    content: format!("Session pattern: {}", pattern.description),
                    memory_type: MemoryType::Insight,
                    importance: 0.6,
                    confidence: pattern.confidence,
                    access_count: 0,
                    content_hash: hash,
                    tags: vec![
                        "session-checkpoint".to_string(),
                        format!("pattern:{}", pattern.pattern_type),
                    ],
                    metadata,
                    namespace: namespace.map(|s| s.to_string()),
                    created_at: now,
                    updated_at: now,
                    last_accessed_at: now,
                };
                if self.engine.storage.insert_memory(&mem).is_ok() {
                    stored_patterns += 1;
                }
            }
        }

        // 4. Get hot directories
        let hot_dirs = self
            .engine
            .storage
            .get_session_hot_directories(session_id, 5)
            .unwrap_or_default();

        // 5. Build markdown report
        let mut report = String::from("## Session Checkpoint\n\n");

        // Activity summary
        report.push_str("### Activity Summary\n\n");
        report.push_str(&format!(
            "| Metric | Count |\n|--------|-------|\n\
             | Files read | {} |\n\
             | Files edited | {} |\n\
             | Searches | {} |\n\
             | Total actions | {} |\n\n",
            activity.files_read, activity.files_edited, activity.searches, activity.total_actions,
        ));

        // Focus areas
        if !hot_dirs.is_empty() {
            report.push_str("### Focus Areas\n\n");
            report.push_str("Directories with most activity in this session:\n\n");
            for (dir, count) in &hot_dirs {
                report.push_str(&format!("- `{}` ({} actions)\n", dir, count));
            }
            report.push('\n');
        }

        // Session-scoped patterns
        if !session_patterns.is_empty() {
            report.push_str("### Session Patterns\n\n");
            for p in session_patterns.iter().take(10) {
                report.push_str(&format!(
                    "- [{}] {} (confidence: {:.0}%)\n",
                    p.pattern_type,
                    p.description,
                    p.confidence * 100.0,
                ));
            }
            report.push('\n');
        }

        // Cross-session patterns
        let unique_cross: Vec<_> = cross_patterns
            .iter()
            .filter(|p| {
                !session_patterns
                    .iter()
                    .any(|sp| sp.description == p.description)
            })
            .take(5)
            .collect();
        if !unique_cross.is_empty() {
            report.push_str("### Cross-Session Patterns\n\n");
            for p in &unique_cross {
                report.push_str(&format!(
                    "- [{}] {} (confidence: {:.0}%)\n",
                    p.pattern_type,
                    p.description,
                    p.confidence * 100.0,
                ));
            }
            report.push('\n');
        }

        // Suggestions
        report.push_str("### Suggestions\n\n");
        if activity.files_read > 5 && activity.files_edited == 0 {
            report.push_str(
                "- You've read many files but haven't edited any yet. \
                 Consider storing a `decision` memory about what you've learned.\n",
            );
        }
        if activity.searches > 3 {
            report.push_str(
                "- Multiple searches detected. Use `store_memory` to save \
                 key findings so you don't need to search again.\n",
            );
        }
        if stored_patterns > 0 {
            report.push_str(&format!(
                "- {} new pattern insight(s) stored from this checkpoint.\n",
                stored_patterns,
            ));
        }
        if activity.total_actions == 0 {
            report.push_str("- No activity recorded yet for this session.\n");
        }

        ToolResult::text(report)
    }
}

#[cfg(test)]
#[path = "tests/tools_consolidation_tests.rs"]
mod tests;
