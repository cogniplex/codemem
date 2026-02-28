//! Advanced recall & namespace tools: recall_with_expansion, list_namespaces,
//! namespace_stats, delete_namespace, export_memories, import_memories.

use crate::scoring::{compute_score, truncate_str};
use crate::types::ToolResult;
use crate::McpServer;
use codemem_core::{
    CodememError, GraphBackend, GraphNode, MemoryNode, MemoryType, NodeKind, SearchResult,
    VectorBackend,
};
use codemem_storage::Storage;
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};

impl McpServer {
    /// MCP tool: recall_with_expansion -- vector search + graph expansion.
    pub(crate) fn tool_recall_with_expansion(&self, args: &Value) -> ToolResult {
        let query = match args.get("query").and_then(|v| v.as_str()) {
            Some(q) if !q.is_empty() => q,
            _ => return ToolResult::tool_error("Missing or empty 'query' parameter"),
        };

        let k = args.get("k").and_then(|v| v.as_u64()).unwrap_or(5) as usize;
        let expansion_depth = args
            .get("expansion_depth")
            .and_then(|v| v.as_u64())
            .unwrap_or(1) as usize;
        let namespace_filter: Option<&str> = args.get("namespace").and_then(|v| v.as_str());

        let query_lower = query.to_lowercase();
        let query_tokens: Vec<&str> = query_lower.split_whitespace().collect();

        // Step 1: Run normal vector search (or text fallback)
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

        // Collect initial seed memories with their vector similarity
        struct ScoredMemory {
            memory: MemoryNode,
            vector_sim: f64,
            expansion_path: String,
        }

        let mut all_memories: Vec<ScoredMemory> = Vec::new();
        let mut seen_ids: HashSet<String> = HashSet::new();

        if vector_results.is_empty() {
            // Fallback: text search over all memories
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
                    let score =
                        breakdown.total_with_weights(unsafe { &*self.scoring_weights.get() });
                    if score > 0.01 {
                        seen_ids.insert(memory.id.clone());
                        all_memories.push(ScoredMemory {
                            memory,
                            vector_sim: 0.0,
                            expansion_path: "direct".to_string(),
                        });
                    }
                }
            }
        } else {
            // Vector search path
            for (id, distance) in &vector_results {
                if let Ok(Some(memory)) = self.storage.get_memory(id) {
                    if let Some(ns) = namespace_filter {
                        if memory.namespace.as_deref() != Some(ns) {
                            continue;
                        }
                    }
                    seen_ids.insert(memory.id.clone());
                    let similarity = 1.0 - (*distance as f64);
                    all_memories.push(ScoredMemory {
                        memory,
                        vector_sim: similarity,
                        expansion_path: "direct".to_string(),
                    });
                }
            }
        }

        // Step 2-4: Graph expansion from each direct result
        // Collect the IDs of direct results for expansion
        let direct_ids: Vec<String> = all_memories.iter().map(|m| m.memory.id.clone()).collect();

        for direct_id in &direct_ids {
            // Use BFS expansion from this memory's graph node
            if let Ok(expanded_nodes) = graph.bfs(direct_id, expansion_depth) {
                for expanded_node in &expanded_nodes {
                    // Skip the start node itself (already in results)
                    if expanded_node.id == *direct_id {
                        continue;
                    }

                    // Only consider memory nodes
                    if expanded_node.kind != NodeKind::Memory {
                        continue;
                    }

                    // Get the memory_id from the graph node
                    let memory_id = expanded_node
                        .memory_id
                        .as_deref()
                        .unwrap_or(&expanded_node.id);

                    // Skip if already seen
                    if seen_ids.contains(memory_id) {
                        continue;
                    }

                    // Fetch the memory
                    if let Ok(Some(memory)) = self.storage.get_memory(memory_id) {
                        if let Some(ns) = namespace_filter {
                            if memory.namespace.as_deref() != Some(ns) {
                                continue;
                            }
                        }

                        // Build expansion path description
                        let expansion_path = if let Ok(edges) = graph.get_edges(direct_id) {
                            edges
                                .iter()
                                .find(|e| e.dst == expanded_node.id || e.src == expanded_node.id)
                                .map(|e| format!("via {} from {}", e.relationship, direct_id))
                                .unwrap_or_else(|| format!("via graph from {direct_id}"))
                        } else {
                            format!("via graph from {direct_id}")
                        };

                        seen_ids.insert(memory_id.to_string());
                        all_memories.push(ScoredMemory {
                            memory,
                            vector_sim: 0.0,
                            expansion_path,
                        });
                    }
                }
            }
        }

        // Step 5-6: Score all memories and sort
        let mut scored_results: Vec<(SearchResult, String)> = all_memories
            .into_iter()
            .map(|sm| {
                let breakdown = compute_score(
                    &sm.memory,
                    query,
                    &query_tokens,
                    sm.vector_sim,
                    &graph,
                    &bm25,
                );
                let score = breakdown.total_with_weights(unsafe { &*self.scoring_weights.get() });
                (
                    SearchResult {
                        memory: sm.memory,
                        score,
                        score_breakdown: breakdown,
                    },
                    sm.expansion_path,
                )
            })
            .collect();

        scored_results.sort_by(|a, b| {
            b.0.score
                .partial_cmp(&a.0.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored_results.truncate(k);

        // Step 7: Format results with expansion_path
        if scored_results.is_empty() {
            return ToolResult::text("No matching memories found.");
        }

        let output: Vec<Value> = scored_results
            .iter()
            .map(|(r, path)| {
                json!({
                    "id": r.memory.id,
                    "content": r.memory.content,
                    "memory_type": r.memory.memory_type.to_string(),
                    "score": format!("{:.4}", r.score),
                    "importance": r.memory.importance,
                    "tags": r.memory.tags,
                    "access_count": r.memory.access_count,
                    "expansion_path": path,
                })
            })
            .collect();

        ToolResult::text(serde_json::to_string_pretty(&output).unwrap())
    }

    /// MCP tool: list_namespaces -- list all namespaces with memory counts.
    pub(crate) fn tool_list_namespaces(&self) -> ToolResult {
        let namespaces = match self.storage.list_namespaces() {
            Ok(ns) => ns,
            Err(e) => return ToolResult::tool_error(format!("Storage error: {e}")),
        };

        let mut ns_list: Vec<Value> = Vec::new();
        for ns in &namespaces {
            let count = match self.storage.list_memory_ids_for_namespace(ns) {
                Ok(ids) => ids.len(),
                Err(_) => 0,
            };
            ns_list.push(json!({
                "name": ns,
                "memory_count": count,
            }));
        }

        let response = json!({ "namespaces": ns_list });
        ToolResult::text(serde_json::to_string_pretty(&response).unwrap())
    }

    /// MCP tool: namespace_stats -- detailed stats for a single namespace.
    pub(crate) fn tool_namespace_stats(&self, args: &Value) -> ToolResult {
        let namespace = match args.get("namespace").and_then(|v| v.as_str()) {
            Some(ns) if !ns.is_empty() => ns,
            _ => return ToolResult::tool_error("Missing or empty 'namespace' parameter"),
        };

        let ids = match self.storage.list_memory_ids_for_namespace(namespace) {
            Ok(ids) => ids,
            Err(e) => return ToolResult::tool_error(format!("Storage error: {e}")),
        };

        if ids.is_empty() {
            return ToolResult::text(
                serde_json::to_string_pretty(&json!({
                    "namespace": namespace,
                    "count": 0,
                    "message": "No memories found in this namespace"
                }))
                .unwrap(),
            );
        }

        let mut total_importance = 0.0;
        let mut total_confidence = 0.0;
        let mut type_distribution: HashMap<String, usize> = HashMap::new();
        let mut tag_frequency: HashMap<String, usize> = HashMap::new();
        let mut oldest: Option<chrono::DateTime<chrono::Utc>> = None;
        let mut newest: Option<chrono::DateTime<chrono::Utc>> = None;
        let mut count = 0usize;

        for id in &ids {
            if let Ok(Some(memory)) = self.storage.get_memory(id) {
                count += 1;
                total_importance += memory.importance;
                total_confidence += memory.confidence;

                *type_distribution
                    .entry(memory.memory_type.to_string())
                    .or_insert(0) += 1;

                for tag in &memory.tags {
                    *tag_frequency.entry(tag.clone()).or_insert(0) += 1;
                }

                match oldest {
                    None => oldest = Some(memory.created_at),
                    Some(ref o) if memory.created_at < *o => oldest = Some(memory.created_at),
                    _ => {}
                }
                match newest {
                    None => newest = Some(memory.created_at),
                    Some(ref n) if memory.created_at > *n => newest = Some(memory.created_at),
                    _ => {}
                }
            }
        }

        let avg_importance = if count > 0 {
            total_importance / count as f64
        } else {
            0.0
        };
        let avg_confidence = if count > 0 {
            total_confidence / count as f64
        } else {
            0.0
        };

        let response = json!({
            "namespace": namespace,
            "count": count,
            "avg_importance": format!("{:.4}", avg_importance),
            "avg_confidence": format!("{:.4}", avg_confidence),
            "type_distribution": type_distribution,
            "tag_frequency": tag_frequency,
            "oldest": oldest.map(|d| d.to_rfc3339()),
            "newest": newest.map(|d| d.to_rfc3339()),
        });

        ToolResult::text(serde_json::to_string_pretty(&response).unwrap())
    }

    /// MCP tool: delete_namespace -- delete all memories in a namespace.
    pub(crate) fn tool_delete_namespace(&self, args: &Value) -> ToolResult {
        let namespace = match args.get("namespace").and_then(|v| v.as_str()) {
            Some(ns) if !ns.is_empty() => ns,
            _ => return ToolResult::tool_error("Missing or empty 'namespace' parameter"),
        };

        let confirm = args
            .get("confirm")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        if !confirm {
            return ToolResult::tool_error(
                "Destructive operation requires 'confirm': true parameter",
            );
        }

        let ids = match self.storage.list_memory_ids_for_namespace(namespace) {
            Ok(ids) => ids,
            Err(e) => return ToolResult::tool_error(format!("Storage error: {e}")),
        };

        let mut deleted = 0usize;
        let mut graph = self.graph.lock().unwrap();
        let mut vector = self.vector.lock().unwrap();
        let mut bm25 = self.bm25_index.lock().unwrap();

        for id in &ids {
            // Delete memory from storage
            if let Ok(true) = self.storage.delete_memory(id) {
                deleted += 1;

                // Remove from vector index
                let _ = vector.remove(id);

                // Remove from in-memory graph
                let _ = graph.remove_node(id);

                // Remove graph node and edges from SQLite
                let _ = self.storage.delete_graph_edges_for_node(id);
                let _ = self.storage.delete_graph_node(id);

                // Remove embedding from SQLite
                let _ = self.storage.delete_embedding(id);

                // Remove from BM25 index
                bm25.remove_document(id);
            }
        }

        // Drop locks before calling save_index (which acquires vector lock)
        drop(graph);
        drop(vector);
        drop(bm25);

        // Persist vector index to disk
        self.save_index();

        let response = json!({
            "deleted": deleted,
            "namespace": namespace,
        });

        ToolResult::text(serde_json::to_string_pretty(&response).unwrap())
    }

    // ── Export/Import Tools ─────────────────────────────────────────────────

    /// MCP tool: export_memories -- export memories as a JSON array.
    pub(crate) fn tool_export_memories(&self, args: &Value) -> ToolResult {
        let namespace_filter: Option<&str> = args.get("namespace").and_then(|v| v.as_str());
        let memory_type_filter: Option<MemoryType> = args
            .get("memory_type")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse().ok());
        let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(100) as usize;

        let ids = match namespace_filter {
            Some(ns) => match self.storage.list_memory_ids_for_namespace(ns) {
                Ok(ids) => ids,
                Err(e) => return ToolResult::tool_error(format!("Storage error: {e}")),
            },
            None => match self.storage.list_memory_ids() {
                Ok(ids) => ids,
                Err(e) => return ToolResult::tool_error(format!("Storage error: {e}")),
            },
        };

        let mut exported: Vec<Value> = Vec::new();

        for id in &ids {
            if exported.len() >= limit {
                break;
            }
            if let Ok(Some(memory)) = self.storage.get_memory(id) {
                // Apply memory_type filter
                if let Some(ref filter_type) = memory_type_filter {
                    if memory.memory_type != *filter_type {
                        continue;
                    }
                }

                // Get edges for this memory
                let edges: Vec<Value> = self
                    .storage
                    .get_edges_for_node(id)
                    .unwrap_or_default()
                    .iter()
                    .map(|e| {
                        json!({
                            "id": e.id,
                            "src": e.src,
                            "dst": e.dst,
                            "relationship": e.relationship.to_string(),
                            "weight": e.weight,
                        })
                    })
                    .collect();

                exported.push(json!({
                    "id": memory.id,
                    "content": memory.content,
                    "memory_type": memory.memory_type.to_string(),
                    "importance": memory.importance,
                    "confidence": memory.confidence,
                    "tags": memory.tags,
                    "namespace": memory.namespace,
                    "metadata": memory.metadata,
                    "created_at": memory.created_at.to_rfc3339(),
                    "updated_at": memory.updated_at.to_rfc3339(),
                    "edges": edges,
                }));
            }
        }

        ToolResult::text(serde_json::to_string_pretty(&exported).unwrap())
    }

    /// MCP tool: import_memories -- import memories from a JSON array.
    pub(crate) fn tool_import_memories(&self, args: &Value) -> ToolResult {
        let memories_arr = match args.get("memories").and_then(|v| v.as_array()) {
            Some(arr) => arr,
            None => return ToolResult::tool_error("Missing 'memories' parameter (expected array)"),
        };

        let mut imported = 0usize;
        let mut skipped = 0usize;
        let mut ids: Vec<String> = Vec::new();

        for mem_val in memories_arr {
            let content = match mem_val.get("content").and_then(|v| v.as_str()) {
                Some(c) if !c.is_empty() => c,
                _ => {
                    skipped += 1;
                    continue;
                }
            };

            let memory_type: MemoryType = mem_val
                .get("memory_type")
                .and_then(|v| v.as_str())
                .and_then(|s| s.parse().ok())
                .unwrap_or(MemoryType::Context);

            let importance = mem_val
                .get("importance")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.5);

            let confidence = mem_val
                .get("confidence")
                .and_then(|v| v.as_f64())
                .unwrap_or(1.0);

            let tags: Vec<String> = mem_val
                .get("tags")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default();

            let namespace = mem_val
                .get("namespace")
                .and_then(|v| v.as_str())
                .map(String::from);

            let metadata: HashMap<String, serde_json::Value> = mem_val
                .get("metadata")
                .and_then(|v| serde_json::from_value(v.clone()).ok())
                .unwrap_or_default();

            let now = chrono::Utc::now();
            let id = uuid::Uuid::new_v4().to_string();
            let hash = Storage::content_hash(content);

            let memory = MemoryNode {
                id: id.clone(),
                content: content.to_string(),
                memory_type,
                importance,
                confidence,
                access_count: 0,
                content_hash: hash,
                tags,
                metadata,
                namespace,
                created_at: now,
                updated_at: now,
                last_accessed_at: now,
            };

            match self.storage.insert_memory(&memory) {
                Ok(()) => {
                    // Update BM25 index
                    self.bm25_index.lock().unwrap().add_document(&id, content);

                    // Create graph node first (so enrichment can reference it)
                    let graph_node = GraphNode {
                        id: id.clone(),
                        kind: NodeKind::Memory,
                        label: truncate_str(content, 80),
                        payload: HashMap::new(),
                        centrality: 0.0,
                        memory_id: Some(id.clone()),
                        namespace: None,
                    };
                    let _ = self.storage.insert_graph_node(&graph_node);
                    let _ = self.graph.lock().unwrap().add_node(graph_node);

                    // Generate contextual embedding and insert into vector index
                    if let Some(ref emb_service) = self.embeddings {
                        let enriched = self.enrich_memory_text(
                            content,
                            memory_type,
                            &memory.tags,
                            memory.namespace.as_deref(),
                            Some(&id),
                        );
                        if let Ok(embedding) = emb_service.lock().unwrap().embed(&enriched) {
                            let _ = self.storage.store_embedding(&id, &embedding);
                            let _ = self.vector.lock().unwrap().insert(&id, &embedding);
                        }
                    }

                    ids.push(id);
                    imported += 1;
                }
                Err(CodememError::Duplicate(_)) => {
                    skipped += 1;
                }
                Err(_) => {
                    skipped += 1;
                }
            }
        }

        // Persist vector index to disk
        self.save_index();

        ToolResult::text(
            serde_json::to_string_pretty(&json!({
                "imported": imported,
                "skipped": skipped,
                "ids": ids,
            }))
            .unwrap(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::*;

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

    #[test]
    fn recall_with_expansion_no_embeddings() {
        let server = test_server();

        // Store two memories and link them
        let mem_a = store_ns(
            &server,
            "graph expansion base memory about architecture",
            "test-ns",
            "insight",
            &["arch"],
        );
        let id_a = mem_a["id"].as_str().unwrap();

        let mem_b = store_ns(
            &server,
            "related memory about design patterns",
            "test-ns",
            "pattern",
            &["design"],
        );
        let id_b = mem_b["id"].as_str().unwrap();

        // Associate them
        call_tool(
            &server,
            "associate_memories",
            json!({
                "source_id": id_a,
                "target_id": id_b,
                "relationship": "RELATES_TO",
            }),
        );

        // Recall with expansion (no embeddings = text fallback)
        let result = call_tool(
            &server,
            "recall_with_expansion",
            json!({
                "query": "architecture",
                "k": 5,
                "expansion_depth": 1,
            }),
        );
        let text = result["content"][0]["text"].as_str().unwrap();
        // Should find at least the base memory via token overlap
        assert!(text.contains("architecture") || text.contains("design"));
    }

    #[test]
    fn list_namespaces_empty() {
        let server = test_server();

        let result = call_tool_parse(&server, "list_namespaces", json!({}));
        let namespaces = result["namespaces"].as_array().unwrap();
        assert_eq!(namespaces.len(), 0);
    }

    #[test]
    fn list_namespaces_with_data() {
        let server = test_server();

        // Store memories in two namespaces
        store_ns(
            &server,
            "memory alpha one about rust",
            "ns-alpha",
            "insight",
            &["rust"],
        );
        store_ns(
            &server,
            "memory alpha two about safety",
            "ns-alpha",
            "pattern",
            &["safety"],
        );
        store_ns(
            &server,
            "memory beta one about python",
            "ns-beta",
            "context",
            &["python"],
        );

        let result = call_tool_parse(&server, "list_namespaces", json!({}));
        let namespaces = result["namespaces"].as_array().unwrap();
        assert_eq!(namespaces.len(), 2);

        // Verify names and counts
        let ns_names: Vec<&str> = namespaces
            .iter()
            .filter_map(|n| n["name"].as_str())
            .collect();
        assert!(ns_names.contains(&"ns-alpha"));
        assert!(ns_names.contains(&"ns-beta"));

        for ns in namespaces {
            if ns["name"].as_str().unwrap() == "ns-alpha" {
                assert_eq!(ns["memory_count"], 2);
            } else if ns["name"].as_str().unwrap() == "ns-beta" {
                assert_eq!(ns["memory_count"], 1);
            }
        }
    }

    #[test]
    fn namespace_stats_basic() {
        let server = test_server();

        store_ns(
            &server,
            "insight about architecture patterns",
            "stats-ns",
            "insight",
            &["arch", "patterns"],
        );
        store_ns(
            &server,
            "pattern for error handling in rust",
            "stats-ns",
            "pattern",
            &["rust", "errors"],
        );

        let result = call_tool_parse(&server, "namespace_stats", json!({"namespace": "stats-ns"}));
        assert_eq!(result["namespace"], "stats-ns");
        assert_eq!(result["count"], 2);

        // Check type distribution
        let types = &result["type_distribution"];
        assert_eq!(types["insight"], 1);
        assert_eq!(types["pattern"], 1);

        // Check tag frequency
        let tags = &result["tag_frequency"];
        assert_eq!(tags["arch"], 1);
        assert_eq!(tags["patterns"], 1);
        assert_eq!(tags["rust"], 1);
        assert_eq!(tags["errors"], 1);

        // Dates should be present
        assert!(result["oldest"].is_string());
        assert!(result["newest"].is_string());
    }

    #[test]
    fn delete_namespace_requires_confirm() {
        let server = test_server();

        store_ns(
            &server,
            "memory to be protected",
            "protected-ns",
            "context",
            &[],
        );

        // Try to delete without confirm
        let result = call_tool(
            &server,
            "delete_namespace",
            json!({
                "namespace": "protected-ns",
                "confirm": false,
            }),
        );
        let text = result["content"][0]["text"].as_str().unwrap();
        assert_eq!(result["isError"], true);
        assert!(text.contains("confirm"));

        // Memory should still exist
        let stats = call_tool_parse(
            &server,
            "namespace_stats",
            json!({"namespace": "protected-ns"}),
        );
        assert_eq!(stats["count"], 1);
    }

    #[test]
    fn delete_namespace_with_confirm() {
        let server = test_server();

        store_ns(
            &server,
            "memory to delete alpha",
            "delete-ns",
            "insight",
            &["test"],
        );
        store_ns(
            &server,
            "memory to delete beta",
            "delete-ns",
            "pattern",
            &["test"],
        );

        // Verify they exist
        let stats = call_tool_parse(
            &server,
            "namespace_stats",
            json!({"namespace": "delete-ns"}),
        );
        assert_eq!(stats["count"], 2);

        // Delete with confirm
        let result = call_tool_parse(
            &server,
            "delete_namespace",
            json!({
                "namespace": "delete-ns",
                "confirm": true,
            }),
        );
        assert_eq!(result["deleted"], 2);
        assert_eq!(result["namespace"], "delete-ns");

        // Verify they are gone
        let stats_after = call_tool_parse(
            &server,
            "namespace_stats",
            json!({"namespace": "delete-ns"}),
        );
        assert_eq!(stats_after["count"], 0);
    }

    // ── Export/Import Tests ─────────────────────────────────────────────

    #[test]
    fn export_memories_empty() {
        let server = test_server();
        let params = json!({"name": "export_memories", "arguments": {}});
        let resp = server.handle_request("tools/call", Some(&params), json!(400));
        let result = resp.result.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        let exported: Vec<Value> = serde_json::from_str(text).unwrap();
        assert!(exported.is_empty());
    }

    #[test]
    fn import_and_export_roundtrip() {
        let server = test_server();

        // Import 2 memories
        let import_params = json!({
            "name": "import_memories",
            "arguments": {
                "memories": [
                    {
                        "content": "roundtrip memory one about rust",
                        "memory_type": "insight",
                        "importance": 0.8,
                        "tags": ["rust", "test"]
                    },
                    {
                        "content": "roundtrip memory two about python",
                        "memory_type": "pattern",
                        "tags": ["python"]
                    }
                ]
            }
        });
        let resp = server.handle_request("tools/call", Some(&import_params), json!(401));
        let result = resp.result.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        let import_result: Value = serde_json::from_str(text).unwrap();
        assert_eq!(import_result["imported"], 2);
        assert_eq!(import_result["skipped"], 0);
        assert_eq!(import_result["ids"].as_array().unwrap().len(), 2);

        // Export all memories
        let export_params = json!({"name": "export_memories", "arguments": {}});
        let resp = server.handle_request("tools/call", Some(&export_params), json!(402));
        let result = resp.result.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        let exported: Vec<Value> = serde_json::from_str(text).unwrap();
        assert_eq!(exported.len(), 2);

        // Verify content matches
        let contents: Vec<&str> = exported
            .iter()
            .filter_map(|e| e["content"].as_str())
            .collect();
        assert!(contents.contains(&"roundtrip memory one about rust"));
        assert!(contents.contains(&"roundtrip memory two about python"));

        // Verify memory types
        let types: Vec<&str> = exported
            .iter()
            .filter_map(|e| e["memory_type"].as_str())
            .collect();
        assert!(types.contains(&"insight"));
        assert!(types.contains(&"pattern"));
    }

    #[test]
    fn export_with_namespace_filter() {
        let server = test_server();

        // Import memories with different namespaces
        let import_params = json!({
            "name": "import_memories",
            "arguments": {
                "memories": [
                    {
                        "content": "project-a memory about architecture",
                        "memory_type": "decision",
                        "namespace": "/projects/a"
                    },
                    {
                        "content": "project-b memory about testing",
                        "memory_type": "insight",
                        "namespace": "/projects/b"
                    },
                    {
                        "content": "project-a memory about patterns",
                        "memory_type": "pattern",
                        "namespace": "/projects/a"
                    }
                ]
            }
        });
        let resp = server.handle_request("tools/call", Some(&import_params), json!(403));
        let result = resp.result.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        let import_result: Value = serde_json::from_str(text).unwrap();
        assert_eq!(import_result["imported"], 3);

        // Export only namespace /projects/a
        let export_params = json!({
            "name": "export_memories",
            "arguments": {"namespace": "/projects/a"}
        });
        let resp = server.handle_request("tools/call", Some(&export_params), json!(404));
        let result = resp.result.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        let exported: Vec<Value> = serde_json::from_str(text).unwrap();
        assert_eq!(exported.len(), 2);

        // All exported should be from /projects/a
        for mem in &exported {
            assert_eq!(mem["namespace"].as_str().unwrap(), "/projects/a");
        }
    }
}
