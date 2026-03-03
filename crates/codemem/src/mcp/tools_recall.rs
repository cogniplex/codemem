//! Advanced recall & namespace tools: recall_with_expansion, list_namespaces,
//! namespace_stats, delete_namespace, export_memories, import_memories.

use super::args::{parse_opt_string, parse_string_array};
use super::scoring::compute_score;
use super::types::ToolResult;
use super::McpServer;
use codemem_core::{GraphBackend, MemoryNode, MemoryType, NodeKind, SearchResult, VectorBackend};
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
        let vector_results: Vec<(String, f32)> = if let Some(emb_guard) =
            match self.lock_embeddings() {
                Ok(g) => g,
                Err(e) => return ToolResult::tool_error(format!("Lock error: {e}")),
            } {
            match emb_guard.embed(query) {
                Ok(query_embedding) => {
                    drop(emb_guard);
                    let vec = match self.lock_vector() {
                        Ok(v) => v,
                        Err(e) => return ToolResult::tool_error(format!("Lock error: {e}")),
                    };
                    vec.search(&query_embedding, k * 2).unwrap_or_default()
                }
                Err(e) => {
                    tracing::warn!("Query embedding failed: {e}");
                    vec![]
                }
            }
        } else {
            vec![]
        };

        let graph = match self.lock_graph() {
            Ok(g) => g,
            Err(e) => return ToolResult::tool_error(format!("Lock error: {e}")),
        };
        let bm25 = match self.lock_bm25() {
            Ok(b) => b,
            Err(e) => return ToolResult::tool_error(format!("Lock error: {e}")),
        };

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
            let ids = match self.engine.storage.list_memory_ids() {
                Ok(ids) => ids,
                Err(e) => return ToolResult::tool_error(format!("Storage error: {e}")),
            };

            for id in &ids {
                if let Ok(Some(memory)) = self.engine.storage.get_memory(id) {
                    if let Some(ns) = namespace_filter {
                        if memory.namespace.as_deref() != Some(ns) {
                            continue;
                        }
                    }
                    let breakdown =
                        compute_score(&memory, query, &query_tokens, 0.0, &graph, &bm25);
                    let weights = match self.scoring_weights() {
                        Ok(w) => w,
                        Err(e) => return ToolResult::tool_error(format!("Lock error: {e}")),
                    };
                    let score = breakdown.total_with_weights(&weights);
                    drop(weights);
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
                if let Ok(Some(memory)) = self.engine.storage.get_memory(id) {
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
            if let Ok(expanded_nodes) =
                graph.bfs_filtered(direct_id, expansion_depth, &[NodeKind::Chunk], None)
            {
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
                    if let Ok(Some(memory)) = self.engine.storage.get_memory(memory_id) {
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
        let weights = match self.scoring_weights() {
            Ok(w) => w,
            Err(e) => return ToolResult::tool_error(format!("Lock error: {e}")),
        };
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
                let score = breakdown.total_with_weights(&weights);
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
        drop(weights);

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

        ToolResult::text(
            serde_json::to_string_pretty(&output).expect("JSON serialization of literal"),
        )
    }

    /// MCP tool: list_namespaces -- list all namespaces with memory counts.
    pub(crate) fn tool_list_namespaces(&self) -> ToolResult {
        let namespaces = match self.engine.storage.list_namespaces() {
            Ok(ns) => ns,
            Err(e) => return ToolResult::tool_error(format!("Storage error: {e}")),
        };

        let mut ns_list: Vec<Value> = Vec::new();
        for ns in &namespaces {
            let count = match self.engine.storage.list_memory_ids_for_namespace(ns) {
                Ok(ids) => ids.len(),
                Err(_) => 0,
            };
            ns_list.push(json!({
                "name": ns,
                "memory_count": count,
            }));
        }

        let response = json!({ "namespaces": ns_list });
        ToolResult::text(
            serde_json::to_string_pretty(&response).expect("JSON serialization of literal"),
        )
    }

    /// MCP tool: namespace_stats -- detailed stats for a single namespace.
    pub(crate) fn tool_namespace_stats(&self, args: &Value) -> ToolResult {
        let namespace = match args.get("namespace").and_then(|v| v.as_str()) {
            Some(ns) if !ns.is_empty() => ns,
            _ => return ToolResult::tool_error("Missing or empty 'namespace' parameter"),
        };

        let ids = match self.engine.storage.list_memory_ids_for_namespace(namespace) {
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
                .expect("JSON serialization of literal"),
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
            if let Ok(Some(memory)) = self.engine.storage.get_memory(id) {
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

        ToolResult::text(
            serde_json::to_string_pretty(&response).expect("JSON serialization of literal"),
        )
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

        let ids = match self.engine.storage.list_memory_ids_for_namespace(namespace) {
            Ok(ids) => ids,
            Err(e) => return ToolResult::tool_error(format!("Storage error: {e}")),
        };

        let mut deleted = 0usize;
        let mut graph = match self.lock_graph() {
            Ok(g) => g,
            Err(e) => return ToolResult::tool_error(format!("Lock error: {e}")),
        };
        let mut vector = match self.lock_vector() {
            Ok(v) => v,
            Err(e) => return ToolResult::tool_error(format!("Lock error: {e}")),
        };
        let mut bm25 = match self.lock_bm25() {
            Ok(b) => b,
            Err(e) => return ToolResult::tool_error(format!("Lock error: {e}")),
        };

        for id in &ids {
            // Delete memory from storage
            if let Ok(true) = self.engine.storage.delete_memory(id) {
                deleted += 1;

                // Remove from vector index
                let _ = vector.remove(id);

                // Remove from in-memory graph
                let _ = graph.remove_node(id);

                // Remove graph node and edges from SQLite
                let _ = self.engine.storage.delete_graph_edges_for_node(id);
                let _ = self.engine.storage.delete_graph_node(id);

                // Remove embedding from SQLite
                let _ = self.engine.storage.delete_embedding(id);

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

        ToolResult::text(
            serde_json::to_string_pretty(&response).expect("JSON serialization of literal"),
        )
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
            Some(ns) => match self.engine.storage.list_memory_ids_for_namespace(ns) {
                Ok(ids) => ids,
                Err(e) => return ToolResult::tool_error(format!("Storage error: {e}")),
            },
            None => match self.engine.storage.list_memory_ids() {
                Ok(ids) => ids,
                Err(e) => return ToolResult::tool_error(format!("Storage error: {e}")),
            },
        };

        let mut exported: Vec<Value> = Vec::new();

        for id in &ids {
            if exported.len() >= limit {
                break;
            }
            if let Ok(Some(memory)) = self.engine.storage.get_memory(id) {
                // Apply memory_type filter
                if let Some(ref filter_type) = memory_type_filter {
                    if memory.memory_type != *filter_type {
                        continue;
                    }
                }

                // Get edges for this memory
                let edges: Vec<Value> = self
                    .engine
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

        ToolResult::text(
            serde_json::to_string_pretty(&exported).expect("JSON serialization of literal"),
        )
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

            let tags = parse_string_array(mem_val, "tags");
            let namespace = parse_opt_string(mem_val, "namespace");

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

            match self.persist_memory(&memory) {
                Ok(()) => {
                    ids.push(id);
                    imported += 1;
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
            .expect("JSON serialization of literal"),
        )
    }
}

#[cfg(test)]
#[path = "tests/tools_recall_tests.rs"]
mod tests;
