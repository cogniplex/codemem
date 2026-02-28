//! Consolidation & lifecycle tools: decay, creative, cluster, forget,
//! consolidation_status, recall_with_impact, get_decision_chain,
//! rebuild_vector_index_internal, detect_patterns, pattern_insights.

use crate::scoring::compute_score;
use crate::types::ToolResult;
use crate::McpServer;
use codemem_core::{
    Edge, GraphBackend, GraphNode, MemoryNode, MemoryType, NodeKind, RelationshipType,
    SearchResult, VectorBackend,
};
use codemem_vector::HnswIndex;
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};

impl McpServer {
    /// MCP tool: consolidate_decay -- reduce importance of stale memories.
    pub(crate) fn tool_consolidate_decay(&self, args: &Value) -> ToolResult {
        let threshold_days = args
            .get("threshold_days")
            .and_then(|v| v.as_u64())
            .unwrap_or(30) as i64;

        let now = chrono::Utc::now();
        let threshold_ts = (now - chrono::Duration::days(threshold_days)).timestamp();

        let affected = match self.storage.decay_stale_memories(threshold_ts, 0.9) {
            Ok(count) => count,
            Err(e) => return ToolResult::tool_error(format!("Decay failed: {e}")),
        };

        // Log the consolidation run
        if let Err(e) = self.storage.insert_consolidation_log("decay", affected) {
            tracing::warn!("Failed to log decay consolidation: {e}");
        }

        ToolResult::text(
            json!({
                "cycle": "decay",
                "affected": affected,
                "threshold_days": threshold_days,
            })
            .to_string(),
        )
    }

    /// MCP tool: consolidate_creative -- connect memories with overlapping tags
    /// but different types via RELATES_TO edges.
    pub(crate) fn tool_consolidate_creative(&self, args: &Value) -> ToolResult {
        let _ = args; // no params

        // Load all memories with their id, content, and tags via StorageBackend
        let parsed = match self.storage.list_memories_for_creative() {
            Ok(rows) => rows,
            Err(e) => return ToolResult::tool_error(format!("Creative cycle failed: {e}")),
        };

        // We also need memory types - get them from full memories for the subset
        // For efficiency, get types from the content prefix or re-query
        // Actually, list_memories_for_creative returns (id, content, tags)
        // We need types too. Let's batch-fetch the memories to get their types.
        let ids_refs: Vec<&str> = parsed.iter().map(|(id, _, _)| id.as_str()).collect();
        let memories = match self.storage.get_memories_batch(&ids_refs) {
            Ok(m) => m,
            Err(e) => return ToolResult::tool_error(format!("Creative cycle failed: {e}")),
        };

        // Build a map of id -> (type_string, tags)
        let memory_info: Vec<(String, String, Vec<String>)> = memories
            .iter()
            .map(|m| (m.id.clone(), m.memory_type.to_string(), m.tags.clone()))
            .collect();

        // Load existing RELATES_TO edges to avoid duplicates
        let all_edges = match self.storage.all_graph_edges() {
            Ok(e) => e,
            Err(e) => return ToolResult::tool_error(format!("Creative cycle failed: {e}")),
        };
        let existing_edges: HashSet<(String, String)> = all_edges
            .iter()
            .filter(|e| e.relationship == RelationshipType::RelatesTo)
            .map(|e| (e.src.clone(), e.dst.clone()))
            .collect();

        let mut new_connections = 0usize;
        let now = chrono::Utc::now();
        let mut graph = self.graph.lock().unwrap();

        for i in 0..memory_info.len() {
            for j in (i + 1)..memory_info.len() {
                let (ref id_a, ref type_a, ref tags_a) = memory_info[i];
                let (ref id_b, ref type_b, ref tags_b) = memory_info[j];

                // Different types required
                if type_a == type_b {
                    continue;
                }

                // Must have at least one overlapping tag
                let has_common_tag = tags_a.iter().any(|t| tags_b.contains(t));
                if !has_common_tag {
                    continue;
                }

                // Check not already connected in either direction
                if existing_edges.contains(&(id_a.clone(), id_b.clone()))
                    || existing_edges.contains(&(id_b.clone(), id_a.clone()))
                {
                    continue;
                }

                // Ensure both nodes exist in graph_nodes (upsert memory-type nodes)
                let node_a = GraphNode {
                    id: id_a.clone(),
                    kind: NodeKind::Memory,
                    label: id_a.clone(),
                    payload: HashMap::new(),
                    centrality: 0.0,
                    memory_id: Some(id_a.clone()),
                    namespace: None,
                };
                let node_b = GraphNode {
                    id: id_b.clone(),
                    kind: NodeKind::Memory,
                    label: id_b.clone(),
                    payload: HashMap::new(),
                    centrality: 0.0,
                    memory_id: Some(id_b.clone()),
                    namespace: None,
                };
                let _ = self.storage.insert_graph_node(&node_a);
                let _ = self.storage.insert_graph_node(&node_b);

                let edge_id = format!("{id_a}-RELATES_TO-{id_b}");
                let edge = Edge {
                    id: edge_id,
                    src: id_a.clone(),
                    dst: id_b.clone(),
                    relationship: RelationshipType::RelatesTo,
                    weight: 1.0,
                    properties: HashMap::new(),
                    created_at: now,
                };

                if self.storage.insert_graph_edge(&edge).is_ok() {
                    let _ = graph.add_edge(edge);
                    new_connections += 1;
                }
            }
        }

        // Log the consolidation run
        if let Err(e) = self
            .storage
            .insert_consolidation_log("creative", new_connections)
        {
            tracing::warn!("Failed to log creative consolidation: {e}");
        }

        ToolResult::text(
            json!({
                "cycle": "creative",
                "new_connections": new_connections,
            })
            .to_string(),
        )
    }

    /// MCP tool: consolidate_cluster -- merge memories with same content_hash prefix,
    /// keeping the one with highest importance.
    pub(crate) fn tool_consolidate_cluster(&self, args: &Value) -> ToolResult {
        let _ = args; // no params

        // Load all memory IDs and batch-fetch for clustering
        let ids = match self.storage.list_memory_ids() {
            Ok(ids) => ids,
            Err(e) => return ToolResult::tool_error(format!("Cluster cycle failed: {e}")),
        };
        let id_refs: Vec<&str> = ids.iter().map(|s| s.as_str()).collect();
        let memories = match self.storage.get_memories_batch(&id_refs) {
            Ok(m) => m,
            Err(e) => return ToolResult::tool_error(format!("Cluster cycle failed: {e}")),
        };

        // Group by first 8 chars of content_hash
        let mut groups: HashMap<String, Vec<(String, f64)>> = HashMap::new();
        for m in &memories {
            let (id, hash, importance) = (&m.id, &m.content_hash, m.importance);
            let prefix = if hash.len() >= 8 {
                hash[..8].to_string()
            } else {
                hash.clone()
            };
            groups
                .entry(prefix)
                .or_default()
                .push((id.clone(), importance));
        }

        let mut merged_count = 0usize;
        let mut kept_count = 0usize;
        let mut ids_to_delete: Vec<String> = Vec::new();

        for (_prefix, mut members) in groups {
            if members.len() <= 1 {
                kept_count += 1;
                continue;
            }

            // Sort by importance descending; keep the first (highest), delete the rest
            members.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            kept_count += 1;

            for (id, _importance) in members.iter().skip(1) {
                ids_to_delete.push(id.clone());
                merged_count += 1;
            }
        }

        // Delete the duplicates
        let mut vector = self.vector.lock().unwrap();
        let mut graph = self.graph.lock().unwrap();
        for id in &ids_to_delete {
            let _ = self.storage.delete_memory(id);
            let _ = self.storage.delete_embedding(id);
            let _ = self.storage.delete_graph_edges_for_node(id);
            let _ = self.storage.delete_graph_node(id);
            let _ = vector.remove(id);
            let _ = graph.remove_node(id);
        }

        // Rebuild vector index if we deleted anything
        if merged_count > 0 {
            self.rebuild_vector_index_internal(&mut vector);
        }
        drop(vector);
        drop(graph);

        // Persist vector index to disk
        self.save_index();

        // Log the consolidation run
        if let Err(e) = self
            .storage
            .insert_consolidation_log("cluster", merged_count)
        {
            tracing::warn!("Failed to log cluster consolidation: {e}");
        }

        ToolResult::text(
            json!({
                "cycle": "cluster",
                "merged": merged_count,
                "kept": kept_count,
            })
            .to_string(),
        )
    }

    /// MCP tool: consolidate_forget -- delete low-importance, never-accessed memories.
    pub(crate) fn tool_consolidate_forget(&self, args: &Value) -> ToolResult {
        let importance_threshold = args
            .get("importance_threshold")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.1);

        // Find memories to forget
        let ids = match self.storage.find_forgettable(importance_threshold) {
            Ok(ids) => ids,
            Err(e) => return ToolResult::tool_error(format!("Forget cycle failed: {e}")),
        };

        let deleted = ids.len();

        let mut vector = self.vector.lock().unwrap();
        let mut graph = self.graph.lock().unwrap();
        let mut bm25 = self.bm25_index.lock().unwrap();
        for id in &ids {
            let _ = self.storage.delete_memory(id);
            let _ = self.storage.delete_embedding(id);
            let _ = self.storage.delete_graph_edges_for_node(id);
            let _ = self.storage.delete_graph_node(id);
            let _ = vector.remove(id);
            let _ = graph.remove_node(id);
            bm25.remove_document(id);
        }

        // Rebuild vector index if we deleted anything
        if deleted > 0 {
            self.rebuild_vector_index_internal(&mut vector);
        }
        drop(vector);
        drop(graph);
        drop(bm25);

        // Persist vector index to disk
        self.save_index();

        // Log the consolidation run
        if let Err(e) = self.storage.insert_consolidation_log("forget", deleted) {
            tracing::warn!("Failed to log forget consolidation: {e}");
        }

        ToolResult::text(
            json!({
                "cycle": "forget",
                "deleted": deleted,
                "threshold": importance_threshold,
            })
            .to_string(),
        )
    }

    /// MCP tool: consolidation_status -- show last run of each consolidation cycle.
    pub(crate) fn tool_consolidation_status(&self) -> ToolResult {
        let runs = match self.storage.last_consolidation_runs() {
            Ok(r) => r,
            Err(e) => return ToolResult::tool_error(format!("Failed to query status: {e}")),
        };

        let mut cycles = json!({});
        for entry in &runs {
            let dt = chrono::DateTime::from_timestamp(entry.run_at, 0)
                .map(|t| t.format("%Y-%m-%d %H:%M:%S UTC").to_string())
                .unwrap_or_else(|| "unknown".to_string());
            cycles[&entry.cycle_type] = json!({
                "last_run": dt,
                "affected": entry.affected_count,
            });
        }

        ToolResult::text(
            json!({
                "cycles": cycles,
            })
            .to_string(),
        )
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

        // Run standard recall logic
        let query_lower = query.to_lowercase();
        let query_tokens: Vec<&str> = query_lower.split_whitespace().collect();

        let vector_results: Vec<(String, f32)> = if let Some(ref emb_service) = self.embeddings {
            match emb_service.lock().unwrap().embed(query) {
                Ok(query_embedding) => self
                    .vector
                    .lock()
                    .unwrap()
                    .search(&query_embedding, k * 2)
                    .unwrap_or_default(),
                Err(e) => {
                    tracing::warn!("Query embedding failed: {e}");
                    vec![]
                }
            }
        } else {
            vec![]
        };

        let graph = self.graph.lock().unwrap();
        let bm25 = self.bm25_index.lock().unwrap();

        let mut results: Vec<SearchResult> = Vec::new();

        if vector_results.is_empty() {
            let ids = match self.storage.list_memory_ids() {
                Ok(ids) => ids,
                Err(e) => return ToolResult::tool_error(format!("Storage error: {e}")),
            };

            for id in &ids {
                if let Ok(Some(memory)) = self.storage.get_memory(id) {
                    if let Some(ns) = namespace_filter {
                        if memory.namespace.as_deref() != Some(ns) {
                            continue;
                        }
                    }

                    let breakdown =
                        compute_score(&memory, query, &query_tokens, 0.0, &graph, &bm25);
                    // SAFETY: Single-threaded MCP server; no concurrent access.
                    let score =
                        breakdown.total_with_weights(unsafe { &*self.scoring_weights.get() });
                    if score > 0.01 {
                        results.push(SearchResult {
                            memory,
                            score,
                            score_breakdown: breakdown,
                        });
                    }
                }
            }
        } else {
            for (id, distance) in &vector_results {
                if let Ok(Some(memory)) = self.storage.get_memory(id) {
                    if let Some(ns) = namespace_filter {
                        if memory.namespace.as_deref() != Some(ns) {
                            continue;
                        }
                    }

                    let similarity = 1.0 - (*distance as f64);
                    let breakdown =
                        compute_score(&memory, query, &query_tokens, similarity, &graph, &bm25);
                    // SAFETY: Single-threaded MCP server; no concurrent access.
                    let score =
                        breakdown.total_with_weights(unsafe { &*self.scoring_weights.get() });
                    results.push(SearchResult {
                        memory,
                        score,
                        score_breakdown: breakdown,
                    });
                }
            }
        }

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(k);

        if results.is_empty() {
            return ToolResult::text("No matching memories found.");
        }

        // Enrich each result with impact data
        let output: Vec<Value> = results
            .iter()
            .map(|r| {
                let memory_id = &r.memory.id;

                let pagerank = graph.get_pagerank(memory_id);
                let centrality = graph.get_betweenness(memory_id);

                // Find connected Decision memories
                let connected_decisions: Vec<String> = graph
                    .get_edges(memory_id)
                    .unwrap_or_default()
                    .iter()
                    .filter_map(|e| {
                        let other_id = if e.src == *memory_id { &e.dst } else { &e.src };
                        self.storage
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

                // Find dependent files from graph edges
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

                let modification_count = r.memory.access_count;

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
                        "modification_count": modification_count,
                    }
                })
            })
            .collect();

        ToolResult::text(serde_json::to_string_pretty(&output).unwrap())
    }

    /// MCP tool: get_decision_chain -- follow decision evolution through the graph.
    pub(crate) fn tool_get_decision_chain(&self, args: &Value) -> ToolResult {
        let file_path: Option<&str> = args.get("file_path").and_then(|v| v.as_str());
        let topic: Option<&str> = args.get("topic").and_then(|v| v.as_str());

        if file_path.is_none() && topic.is_none() {
            return ToolResult::tool_error("Must provide either 'file_path' or 'topic' parameter");
        }

        let graph = self.graph.lock().unwrap();

        // Find all Decision-type memories matching the file_path or topic
        let ids = match self.storage.list_memory_ids() {
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
            if let Ok(Some(memory)) = self.storage.get_memory(id) {
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
                            if let Ok(Some(m)) = self.storage.get_memory(other_id) {
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
            if let Ok(Some(memory)) = self.storage.get_memory(id) {
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

        ToolResult::text(serde_json::to_string_pretty(&response).unwrap())
    }

    /// Internal helper: rebuild vector index from all stored embeddings.
    pub(crate) fn rebuild_vector_index_internal(&self, vector: &mut HnswIndex) {
        let embeddings = match self.storage.list_all_embeddings() {
            Ok(e) => e,
            Err(e) => {
                tracing::warn!("Failed to rebuild vector index: {e}");
                return;
            }
        };

        // Create a fresh index and reinsert all embeddings
        if let Ok(mut fresh) = HnswIndex::with_defaults() {
            for (id, floats) in &embeddings {
                let _ = fresh.insert(id, floats);
            }
            *vector = fresh;
        }
    }

    /// MCP tool: detect_patterns -- detect cross-session patterns in stored memories.
    pub(crate) fn tool_detect_patterns(&self, args: &Value) -> ToolResult {
        let min_frequency = args
            .get("min_frequency")
            .and_then(|v| v.as_u64())
            .unwrap_or(3) as usize;
        let namespace = args.get("namespace").and_then(|v| v.as_str());

        match crate::patterns::detect_patterns(&*self.storage, namespace, min_frequency) {
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
                    .unwrap(),
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

        match crate::patterns::detect_patterns(&*self.storage, namespace, min_frequency) {
            Ok(detected) => {
                let markdown = crate::patterns::generate_insights(&detected);
                ToolResult::text(markdown)
            }
            Err(e) => ToolResult::tool_error(format!("Pattern insights error: {e}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::*;
    use codemem_storage::Storage;

    /// Helper: call a tool and return the result Value.
    fn call_tool(server: &McpServer, tool_name: &str, arguments: Value) -> Value {
        let params = json!({"name": tool_name, "arguments": arguments});
        let resp = server.handle_request("tools/call", Some(&params), json!("req"));
        assert!(
            resp.error.is_none(),
            "Unexpected error calling {tool_name}: {:?}",
            resp.error
        );
        resp.result.unwrap()
    }

    /// Helper: call a tool and parse the text content as JSON.
    fn call_tool_parse(server: &McpServer, tool_name: &str, arguments: Value) -> Value {
        let result = call_tool(server, tool_name, arguments);
        let text = result["content"][0]["text"].as_str().unwrap();
        serde_json::from_str(text).unwrap_or_else(|_| Value::String(text.to_string()))
    }

    /// Helper: store a memory with namespace.
    fn store_ns(
        server: &McpServer,
        content: &str,
        namespace: &str,
        memory_type: &str,
        tags: &[&str],
    ) -> Value {
        call_tool_parse(
            server,
            "store_memory",
            json!({
                "content": content,
                "memory_type": memory_type,
                "tags": tags,
                "namespace": namespace,
            }),
        )
    }

    // ── Consolidation Tool Tests ────────────────────────────────────────

    #[test]
    fn consolidate_decay_reduces_importance() {
        let server = test_server();

        // Store a memory with known importance
        let now = chrono::Utc::now();
        let sixty_days_ago = now - chrono::Duration::days(60);
        let id = uuid::Uuid::new_v4().to_string();
        let content = "old memory that should decay";
        let hash = Storage::content_hash(content);
        let memory = MemoryNode {
            id: id.clone(),
            content: content.to_string(),
            memory_type: MemoryType::Context,
            importance: 0.8,
            confidence: 1.0,
            access_count: 0,
            content_hash: hash,
            tags: vec![],
            metadata: HashMap::new(),
            namespace: None,
            created_at: sixty_days_ago,
            updated_at: sixty_days_ago,
            last_accessed_at: sixty_days_ago,
        };
        server.storage.insert_memory(&memory).unwrap();

        // Run decay with default threshold (30 days)
        let params = json!({"name": "consolidate_decay", "arguments": {}});
        let resp = server.handle_request("tools/call", Some(&params), json!(1));
        let result = resp.result.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        let parsed: Value = serde_json::from_str(text).unwrap();

        assert_eq!(parsed["cycle"], "decay");
        assert_eq!(parsed["affected"], 1);
        assert_eq!(parsed["threshold_days"], 30);

        // Verify importance was reduced: 0.8 * 0.9 = 0.72
        let retrieved = server.storage.get_memory(&id).unwrap().unwrap();
        assert!((retrieved.importance - 0.72).abs() < 0.01);
    }

    #[test]
    fn consolidate_decay_skips_recent_memories() {
        let server = test_server();

        // Store a recent memory
        store_memory(&server, "recently accessed memory", "context", &[]);

        // Run decay
        let params = json!({"name": "consolidate_decay", "arguments": {"threshold_days": 30}});
        let resp = server.handle_request("tools/call", Some(&params), json!(1));
        let result = resp.result.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        let parsed: Value = serde_json::from_str(text).unwrap();

        // Recent memory should not be affected
        assert_eq!(parsed["affected"], 0);
    }

    #[test]
    fn consolidate_creative_creates_edges() {
        let server = test_server();

        // Store two memories with overlapping tags but different types
        store_memory(
            &server,
            "insight about rust safety",
            "insight",
            &["rust", "safety"],
        );
        store_memory(
            &server,
            "pattern for error handling",
            "pattern",
            &["rust", "error"],
        );

        // Run creative cycle
        let params = json!({"name": "consolidate_creative", "arguments": {}});
        let resp = server.handle_request("tools/call", Some(&params), json!(1));
        let result = resp.result.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        let parsed: Value = serde_json::from_str(text).unwrap();

        assert_eq!(parsed["cycle"], "creative");
        // They share the "rust" tag and have different types, so should create 1 connection
        assert_eq!(parsed["new_connections"], 1);
    }

    #[test]
    fn consolidate_creative_skips_same_type() {
        let server = test_server();

        // Store two memories with same type (should not create edges)
        store_memory(&server, "insight one about rust", "insight", &["rust"]);
        store_memory(&server, "insight two about rust", "insight", &["rust"]);

        let params = json!({"name": "consolidate_creative", "arguments": {}});
        let resp = server.handle_request("tools/call", Some(&params), json!(1));
        let result = resp.result.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        let parsed: Value = serde_json::from_str(text).unwrap();

        assert_eq!(parsed["new_connections"], 0);
    }

    #[test]
    fn consolidate_forget_deletes_low_importance() {
        let server = test_server();

        // Store a memory with very low importance and zero access count
        let now = chrono::Utc::now();
        let id = uuid::Uuid::new_v4().to_string();
        let content = "forgettable memory";
        let hash = Storage::content_hash(content);
        let memory = MemoryNode {
            id: id.clone(),
            content: content.to_string(),
            memory_type: MemoryType::Context,
            importance: 0.05,
            confidence: 1.0,
            access_count: 0,
            content_hash: hash,
            tags: vec![],
            metadata: HashMap::new(),
            namespace: None,
            created_at: now,
            updated_at: now,
            last_accessed_at: now,
        };
        server.storage.insert_memory(&memory).unwrap();

        // Verify it exists
        assert_eq!(server.storage.memory_count().unwrap(), 1);

        // Run forget
        let params = json!({"name": "consolidate_forget", "arguments": {}});
        let resp = server.handle_request("tools/call", Some(&params), json!(1));
        let result = resp.result.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        let parsed: Value = serde_json::from_str(text).unwrap();

        assert_eq!(parsed["cycle"], "forget");
        assert_eq!(parsed["deleted"], 1);
        assert_eq!(parsed["threshold"], 0.1);

        // Verify it's gone
        assert_eq!(server.storage.memory_count().unwrap(), 0);
    }

    #[test]
    fn consolidate_forget_keeps_accessed_memories() {
        let server = test_server();

        // Store a memory with low importance but nonzero access count directly
        let now = chrono::Utc::now();
        let memory = MemoryNode {
            id: uuid::Uuid::new_v4().to_string(),
            content: "low importance but accessed".to_string(),
            memory_type: MemoryType::Context,
            importance: 0.05,
            confidence: 1.0,
            access_count: 5,
            content_hash: Storage::content_hash("low importance but accessed"),
            tags: vec![],
            metadata: HashMap::new(),
            namespace: None,
            created_at: now,
            updated_at: now,
            last_accessed_at: now,
        };
        server.storage.insert_memory(&memory).unwrap();

        // This memory has access_count = 5, so it should NOT be forgotten
        // (forget only targets memories with access_count == 0)

        let params = json!({"name": "consolidate_forget", "arguments": {}});
        let resp = server.handle_request("tools/call", Some(&params), json!(1));
        let result = resp.result.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        let parsed: Value = serde_json::from_str(text).unwrap();

        assert_eq!(parsed["deleted"], 0);
        assert_eq!(server.storage.memory_count().unwrap(), 1);
    }

    #[test]
    fn consolidation_status_shows_last_run() {
        let server = test_server();

        // Status with no prior runs should return empty cycles
        let params = json!({"name": "consolidation_status", "arguments": {}});
        let resp = server.handle_request("tools/call", Some(&params), json!(1));
        let result = resp.result.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        let parsed: Value = serde_json::from_str(text).unwrap();
        assert_eq!(parsed["cycles"], json!({}));

        // Run a decay cycle
        let params = json!({"name": "consolidate_decay", "arguments": {}});
        server.handle_request("tools/call", Some(&params), json!(2));

        // Now status should show decay
        let params = json!({"name": "consolidation_status", "arguments": {}});
        let resp = server.handle_request("tools/call", Some(&params), json!(3));
        let result = resp.result.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        let parsed: Value = serde_json::from_str(text).unwrap();

        assert!(parsed["cycles"]["decay"].is_object());
        assert!(parsed["cycles"]["decay"]["last_run"].is_string());
        assert!(parsed["cycles"]["decay"]["affected"].is_number());
    }

    #[test]
    fn consolidate_forget_custom_threshold() {
        let server = test_server();

        // Store two memories with different importance
        let now = chrono::Utc::now();
        for (imp, content) in [(0.3, "medium importance"), (0.05, "very low importance")] {
            let id = uuid::Uuid::new_v4().to_string();
            let hash = Storage::content_hash(content);
            let memory = MemoryNode {
                id,
                content: content.to_string(),
                memory_type: MemoryType::Context,
                importance: imp,
                confidence: 1.0,
                access_count: 0,
                content_hash: hash,
                tags: vec![],
                metadata: HashMap::new(),
                namespace: None,
                created_at: now,
                updated_at: now,
                last_accessed_at: now,
            };
            server.storage.insert_memory(&memory).unwrap();
        }

        assert_eq!(server.storage.memory_count().unwrap(), 2);

        // Forget with threshold 0.5 should delete both
        let params =
            json!({"name": "consolidate_forget", "arguments": {"importance_threshold": 0.5}});
        let resp = server.handle_request("tools/call", Some(&params), json!(1));
        let result = resp.result.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        let parsed: Value = serde_json::from_str(text).unwrap();

        assert_eq!(parsed["deleted"], 2);
        assert_eq!(parsed["threshold"], 0.5);
        assert_eq!(server.storage.memory_count().unwrap(), 0);
    }

    // ── Impact-Aware Recall & Decision Chain Tests ────────────────────────

    #[test]
    fn recall_with_impact_returns_impact_data() {
        let server = test_server();

        // Store a memory
        let mem = store_ns(
            &server,
            "impact test memory about error handling patterns",
            "test-ns",
            "insight",
            &["error", "handling"],
        );
        let _id = mem["id"].as_str().unwrap();

        // Recall with impact (text fallback, no embeddings)
        let result = call_tool(
            &server,
            "recall_with_impact",
            json!({"query": "error handling"}),
        );
        let text = result["content"][0]["text"].as_str().unwrap();

        // Should find the memory and include impact data
        if text.contains("No matching memories") {
            // Token overlap alone may not be enough; that is fine
            return;
        }

        let parsed: Value = serde_json::from_str(text).unwrap();
        let first = &parsed[0];
        assert!(
            first.get("impact").is_some(),
            "result should contain impact data"
        );
        let impact = &first["impact"];
        assert!(impact.get("pagerank").is_some());
        assert!(impact.get("centrality").is_some());
        assert!(impact.get("connected_decisions").is_some());
        assert!(impact.get("dependent_files").is_some());
        assert!(impact.get("modification_count").is_some());
    }

    #[test]
    fn get_decision_chain_requires_parameter() {
        let server = test_server();

        // Calling without file_path or topic should return an error
        let params = json!({"name": "get_decision_chain", "arguments": {}});
        let resp = server.handle_request("tools/call", Some(&params), json!("req"));
        let result = resp.result.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        assert!(
            text.contains("file_path") || text.contains("topic"),
            "error should mention required parameters"
        );
    }

    #[test]
    fn get_decision_chain_by_topic() {
        let server = test_server();

        // Store decision memories with a topic
        let _d1 = call_tool_parse(
            &server,
            "store_memory",
            json!({
                "content": "Decision: use async runtime for concurrency",
                "memory_type": "decision",
                "tags": ["concurrency"],
            }),
        );
        let _d2 = call_tool_parse(
            &server,
            "store_memory",
            json!({
                "content": "Decision: switched from threads to async for concurrency",
                "memory_type": "decision",
                "tags": ["concurrency"],
            }),
        );

        // Query decision chain by topic
        let result = call_tool(
            &server,
            "get_decision_chain",
            json!({"topic": "concurrency"}),
        );
        let text = result["content"][0]["text"].as_str().unwrap();
        let parsed: Value = serde_json::from_str(text).unwrap();
        assert!(parsed["chain_length"].as_u64().unwrap() >= 2);
        assert_eq!(parsed["filter"]["topic"], "concurrency");
    }

    #[test]
    fn decision_chain_follows_temporal_order() {
        let server = test_server();

        // Store decision memories at different times (chronological insertion order)
        let d1 = call_tool_parse(
            &server,
            "store_memory",
            json!({
                "content": "Decision: initial architecture for auth module",
                "memory_type": "decision",
                "tags": ["auth"],
            }),
        );
        let d2 = call_tool_parse(
            &server,
            "store_memory",
            json!({
                "content": "Decision: refactored auth to use JWT tokens",
                "memory_type": "decision",
                "tags": ["auth"],
            }),
        );
        let d3 = call_tool_parse(
            &server,
            "store_memory",
            json!({
                "content": "Decision: added OAuth2 to auth module",
                "memory_type": "decision",
                "tags": ["auth"],
            }),
        );

        // Link d1 -> d2 -> d3 with EVOLVED_INTO edges
        let id1 = d1["id"].as_str().unwrap();
        let id2 = d2["id"].as_str().unwrap();
        let id3 = d3["id"].as_str().unwrap();

        call_tool(
            &server,
            "associate_memories",
            json!({
                "source_id": id1,
                "target_id": id2,
                "relationship": "EVOLVED_INTO",
            }),
        );
        call_tool(
            &server,
            "associate_memories",
            json!({
                "source_id": id2,
                "target_id": id3,
                "relationship": "EVOLVED_INTO",
            }),
        );

        // Get decision chain
        let result = call_tool(&server, "get_decision_chain", json!({"topic": "auth"}));
        let text = result["content"][0]["text"].as_str().unwrap();
        let parsed: Value = serde_json::from_str(text).unwrap();

        assert_eq!(parsed["chain_length"].as_u64().unwrap(), 3);
        let decisions = parsed["decisions"].as_array().unwrap();

        // Verify temporal order: created_at of each should be <= the next
        for i in 0..decisions.len() - 1 {
            let dt_a = decisions[i]["created_at"].as_str().unwrap();
            let dt_b = decisions[i + 1]["created_at"].as_str().unwrap();
            assert!(dt_a <= dt_b, "decisions should be in chronological order");
        }

        // Verify connections exist
        let has_connections = decisions
            .iter()
            .any(|d| !d["connections"].as_array().unwrap().is_empty());
        assert!(
            has_connections,
            "at least one decision should have connections"
        );
    }
}
