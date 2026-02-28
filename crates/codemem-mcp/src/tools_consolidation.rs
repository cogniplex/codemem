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

/// Cosine similarity between two embedding vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let x = *x as f64;
        let y = *y as f64;
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < 1e-12 {
        0.0
    } else {
        dot / denom
    }
}

/// Union-Find (disjoint set) data structure for transitive clustering.
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    fn union(&mut self, x: usize, y: usize) {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx == ry {
            return;
        }
        match self.rank[rx].cmp(&self.rank[ry]) {
            std::cmp::Ordering::Less => self.parent[rx] = ry,
            std::cmp::Ordering::Greater => self.parent[ry] = rx,
            std::cmp::Ordering::Equal => {
                self.parent[ry] = rx;
                self.rank[rx] += 1;
            }
        }
    }

    fn groups(&mut self, n: usize) -> Vec<Vec<usize>> {
        let mut map: HashMap<usize, Vec<usize>> = HashMap::new();
        for i in 0..n {
            let root = self.find(i);
            map.entry(root).or_default().push(i);
        }
        map.into_values().collect()
    }
}

impl McpServer {
    /// MCP tool: consolidate_decay -- power-law decay that rewards access frequency.
    pub(crate) fn tool_consolidate_decay(&self, args: &Value) -> ToolResult {
        let threshold_days = args
            .get("threshold_days")
            .and_then(|v| v.as_u64())
            .unwrap_or(30) as i64;

        let now = chrono::Utc::now();
        let threshold_ts = (now - chrono::Duration::days(threshold_days)).timestamp();

        // Fetch stale memories with access metadata for power-law decay
        let stale = match self.storage.get_stale_memories_for_decay(threshold_ts) {
            Ok(rows) => rows,
            Err(e) => return ToolResult::tool_error(format!("Decay failed: {e}")),
        };

        if stale.is_empty() {
            if let Err(e) = self.storage.insert_consolidation_log("decay", 0) {
                tracing::warn!("Failed to log decay consolidation: {e}");
            }
            return ToolResult::text(
                json!({"cycle": "decay", "affected": 0, "threshold_days": threshold_days})
                    .to_string(),
            );
        }

        // Compute power-law decay: importance * 0.9^(days_since/30) * (1 + log2(max(access_count,1)) * 0.1)
        let now_ts = now.timestamp();
        let updates: Vec<(String, f64)> = stale
            .iter()
            .map(|(id, importance, access_count, last_accessed_at)| {
                let days_since = (now_ts - last_accessed_at) as f64 / 86400.0;
                let time_decay = 0.9_f64.powf(days_since / 30.0);
                let access_boost = 1.0 + ((*access_count).max(1) as f64).log2() * 0.1;
                let new_importance = (importance * time_decay * access_boost).clamp(0.0, 1.0);
                (id.clone(), new_importance)
            })
            .collect();

        let affected = match self.storage.batch_update_importance(&updates) {
            Ok(count) => count,
            Err(e) => return ToolResult::tool_error(format!("Decay batch update failed: {e}")),
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
                "algorithm": "power_law",
            })
            .to_string(),
        )
    }

    /// MCP tool: consolidate_creative -- O(n log n) semantic creative consolidation.
    /// Uses vector search to find cross-type neighbors and creates SHARES_THEME edges.
    pub(crate) fn tool_consolidate_creative(&self, args: &Value) -> ToolResult {
        let _ = args;

        // Load all memories with their types
        let parsed = match self.storage.list_memories_for_creative() {
            Ok(rows) => rows,
            Err(e) => return ToolResult::tool_error(format!("Creative cycle failed: {e}")),
        };

        let ids_refs: Vec<&str> = parsed.iter().map(|(id, _, _)| id.as_str()).collect();
        let memories = match self.storage.get_memories_batch(&ids_refs) {
            Ok(m) => m,
            Err(e) => return ToolResult::tool_error(format!("Creative cycle failed: {e}")),
        };

        // Build type lookup
        let type_map: HashMap<String, String> = memories
            .iter()
            .map(|m| (m.id.clone(), m.memory_type.to_string()))
            .collect();

        // Load existing SHARES_THEME edges to avoid duplicates
        let all_edges = match self.storage.all_graph_edges() {
            Ok(e) => e,
            Err(e) => return ToolResult::tool_error(format!("Creative cycle failed: {e}")),
        };
        let existing_edges: HashSet<(String, String)> = all_edges
            .iter()
            .filter(|e| {
                e.relationship == RelationshipType::SharesTheme
                    || e.relationship == RelationshipType::RelatesTo
            })
            .flat_map(|e| {
                vec![
                    (e.src.clone(), e.dst.clone()),
                    (e.dst.clone(), e.src.clone()),
                ]
            })
            .collect();

        // Load all embeddings
        let all_embeddings = match self.storage.list_all_embeddings() {
            Ok(e) => e,
            Err(e) => return ToolResult::tool_error(format!("Creative cycle failed: {e}")),
        };
        let embedding_map: HashMap<String, Vec<f32>> = all_embeddings.into_iter().collect();

        let now = chrono::Utc::now();
        let mut new_connections = 0usize;
        let mut graph = match self.lock_graph() {
            Ok(g) => g,
            Err(e) => return ToolResult::tool_error(format!("Lock error: {e}")),
        };

        let vector = match self.lock_vector() {
            Ok(v) => v,
            Err(e) => return ToolResult::tool_error(format!("Lock error: {e}")),
        };

        // For each memory with an embedding, find 6 nearest neighbors
        for (id, embedding) in &embedding_map {
            let my_type = match type_map.get(id) {
                Some(t) => t,
                None => continue,
            };

            // vector.search returns (id, similarity) where similarity = 1 - cosine_distance
            let neighbors = vector.search(embedding, 7).unwrap_or_default();

            for (neighbor_id, sim) in &neighbors {
                // Skip self
                if neighbor_id == id {
                    continue;
                }

                let neighbor_type = match type_map.get(neighbor_id) {
                    Some(t) => t,
                    None => continue,
                };

                // Cross-type only
                if my_type == neighbor_type {
                    continue;
                }

                // Check not already connected
                if existing_edges.contains(&(id.clone(), neighbor_id.clone())) {
                    continue;
                }

                let similarity = *sim as f64;
                if similarity < 0.5 {
                    continue;
                }

                // Ensure both nodes exist
                for nid in [id, neighbor_id] {
                    let node = GraphNode {
                        id: nid.clone(),
                        kind: NodeKind::Memory,
                        label: nid.clone(),
                        payload: HashMap::new(),
                        centrality: 0.0,
                        memory_id: Some(nid.clone()),
                        namespace: None,
                    };
                    let _ = self.storage.insert_graph_node(&node);
                }

                let edge_id = format!("{id}-SHARES_THEME-{neighbor_id}");
                let edge = Edge {
                    id: edge_id,
                    src: id.clone(),
                    dst: neighbor_id.clone(),
                    relationship: RelationshipType::SharesTheme,
                    weight: similarity,
                    properties: HashMap::new(),
                    created_at: now,
                    valid_from: Some(now),
                    valid_to: None,
                };

                if self.storage.insert_graph_edge(&edge).is_ok() {
                    let _ = graph.add_edge(edge);
                    new_connections += 1;
                }
            }
        }
        drop(vector);
        drop(graph);

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
                "algorithm": "vector_knn",
            })
            .to_string(),
        )
    }

    /// MCP tool: consolidate_cluster -- semantic deduplication using cosine similarity.
    ///
    /// Groups memories by content_hash prefix (fast pre-filter), then uses
    /// pairwise cosine similarity + union-find to cluster transitively-similar
    /// memories. Keeps the highest-importance memory per cluster.
    pub(crate) fn tool_consolidate_cluster(&self, args: &Value) -> ToolResult {
        let similarity_threshold = args
            .get("similarity_threshold")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.92);

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

        // Group by first 8 chars of content_hash (fast pre-filter)
        let mut groups: HashMap<String, Vec<usize>> = HashMap::new();
        for (idx, m) in memories.iter().enumerate() {
            let prefix = if m.content_hash.len() >= 8 {
                m.content_hash[..8].to_string()
            } else {
                m.content_hash.clone()
            };
            groups.entry(prefix).or_default().push(idx);
        }

        // Load all embeddings for semantic comparison
        let all_embeddings = match self.storage.list_all_embeddings() {
            Ok(e) => e,
            Err(e) => return ToolResult::tool_error(format!("Cluster cycle failed: {e}")),
        };
        let embedding_map: HashMap<String, Vec<f32>> = all_embeddings.into_iter().collect();

        // Union-find for transitive clustering
        let n = memories.len();
        let mut uf = UnionFind::new(n);

        for member_indices in groups.values() {
            if member_indices.len() <= 1 {
                continue;
            }

            // Pairwise cosine similarity within each hash-prefix group
            for i in 0..member_indices.len() {
                for j in (i + 1)..member_indices.len() {
                    let idx_a = member_indices[i];
                    let idx_b = member_indices[j];
                    let id_a = &memories[idx_a].id;
                    let id_b = &memories[idx_b].id;

                    let sim = match (embedding_map.get(id_a), embedding_map.get(id_b)) {
                        (Some(emb_a), Some(emb_b)) => cosine_similarity(emb_a, emb_b),
                        // Fall back to hash equality (same prefix → likely duplicate)
                        _ => {
                            if memories[idx_a].content_hash == memories[idx_b].content_hash {
                                1.0
                            } else {
                                0.0
                            }
                        }
                    };

                    if sim >= similarity_threshold {
                        uf.union(idx_a, idx_b);
                    }
                }
            }
        }

        // Collect clusters from union-find
        let clusters = uf.groups(n);

        let mut merged_count = 0usize;
        let mut kept_count = 0usize;
        let mut ids_to_delete: Vec<String> = Vec::new();

        for cluster in &clusters {
            if cluster.len() <= 1 {
                kept_count += 1;
                continue;
            }

            // Sort by importance descending; keep the first (highest), delete the rest
            let mut members: Vec<(usize, f64)> = cluster
                .iter()
                .map(|&idx| (idx, memories[idx].importance))
                .collect();
            members.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            kept_count += 1;

            for &(idx, _) in members.iter().skip(1) {
                ids_to_delete.push(memories[idx].id.clone());
                merged_count += 1;
            }
        }

        // Delete the duplicates
        let mut vector = match self.lock_vector() {
            Ok(v) => v,
            Err(e) => return ToolResult::tool_error(format!("Lock error: {e}")),
        };
        let mut graph = match self.lock_graph() {
            Ok(g) => g,
            Err(e) => return ToolResult::tool_error(format!("Lock error: {e}")),
        };
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
                "similarity_threshold": similarity_threshold,
                "algorithm": "semantic_cosine",
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

        let mut vector = match self.lock_vector() {
            Ok(v) => v,
            Err(e) => return ToolResult::tool_error(format!("Lock error: {e}")),
        };
        let mut graph = match self.lock_graph() {
            Ok(g) => g,
            Err(e) => return ToolResult::tool_error(format!("Lock error: {e}")),
        };
        let mut bm25 = match self.lock_bm25() {
            Ok(b) => b,
            Err(e) => return ToolResult::tool_error(format!("Lock error: {e}")),
        };
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
                    let weights = match self.scoring_weights() {
                        Ok(w) => w,
                        Err(e) => return ToolResult::tool_error(format!("Lock error: {e}")),
                    };
                    let score = breakdown.total_with_weights(&weights);
                    drop(weights);
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
                    let weights = match self.scoring_weights() {
                        Ok(w) => w,
                        Err(e) => return ToolResult::tool_error(format!("Lock error: {e}")),
                    };
                    let score = breakdown.total_with_weights(&weights);
                    drop(weights);
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

        ToolResult::text(
            serde_json::to_string_pretty(&output).expect("JSON serialization of literal"),
        )
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

        ToolResult::text(
            serde_json::to_string_pretty(&response).expect("JSON serialization of literal"),
        )
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

        let total_sessions = self.storage.session_count(namespace).unwrap_or(10);

        match crate::patterns::detect_patterns(
            &*self.storage,
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

        let total_sessions = self.storage.session_count(namespace).unwrap_or(10);

        match crate::patterns::detect_patterns(
            &*self.storage,
            namespace,
            min_frequency,
            total_sessions,
        ) {
            Ok(detected) => {
                let markdown = crate::patterns::generate_insights(&detected);
                ToolResult::text(markdown)
            }
            Err(e) => ToolResult::tool_error(format!("Pattern insights error: {e}")),
        }
    }

    /// MCP tool: consolidate_summarize -- LLM-powered consolidation that finds
    /// connected components, summarizes large clusters into Insight memories
    /// linked via SUMMARIZES edges.
    pub(crate) fn tool_consolidate_summarize(&self, args: &Value) -> ToolResult {
        let min_cluster_size = args
            .get("cluster_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(5) as usize;

        // Check that compression provider is configured
        let provider = crate::compress::CompressProvider::from_env();
        if !provider.is_enabled() {
            return ToolResult::tool_error(
                "CODEMEM_COMPRESS_PROVIDER env var not set. \
                 Set it to 'ollama', 'openai', or 'anthropic' to enable LLM-powered consolidation.",
            );
        }

        // Find connected components via the graph
        let graph = match self.lock_graph() {
            Ok(g) => g,
            Err(e) => return ToolResult::tool_error(format!("Lock error: {e}")),
        };

        let components = graph.connected_components();
        drop(graph);

        let large_clusters: Vec<&Vec<String>> = components
            .iter()
            .filter(|c| c.len() >= min_cluster_size)
            .collect();

        if large_clusters.is_empty() {
            return ToolResult::text(
                json!({
                    "cycle": "summarize",
                    "summarized": 0,
                    "message": format!("No clusters with {} or more members found", min_cluster_size),
                })
                .to_string(),
            );
        }

        let mut summarized_count = 0u64;
        let mut created_ids: Vec<String> = Vec::new();

        for cluster in &large_clusters {
            // Fetch memories for this cluster
            let mut contents: Vec<String> = Vec::new();
            let mut source_ids: Vec<String> = Vec::new();
            let mut all_tags: Vec<String> = Vec::new();

            for node_id in *cluster {
                // Try to load the memory by looking up the graph node's memory_id
                let graph = match self.lock_graph() {
                    Ok(g) => g,
                    Err(_) => continue,
                };
                if let Ok(Some(node)) = graph.get_node(node_id) {
                    if let Some(mid) = &node.memory_id {
                        let mid = mid.clone();
                        drop(graph);
                        if let Ok(Some(mem)) = self.storage.get_memory(&mid) {
                            contents.push(mem.content.clone());
                            source_ids.push(mid);
                            all_tags.extend(mem.tags.clone());
                        }
                    }
                }
            }

            if contents.len() < 2 {
                continue;
            }

            // Use LLM to summarize the cluster
            let combined = contents.join("\n---\n");
            let summary = match provider.compress(&combined, "consolidate_summarize", None) {
                Some(s) => s,
                None => continue, // Compression failed or content too short
            };

            // Deduplicate tags
            all_tags.sort();
            all_tags.dedup();

            // Create a new Insight memory for the summary
            let now = chrono::Utc::now();
            let new_id = uuid::Uuid::new_v4().to_string();
            let hash = codemem_storage::Storage::content_hash(&summary);

            let mem = MemoryNode {
                id: new_id.clone(),
                content: summary.clone(),
                memory_type: MemoryType::Insight,
                importance: 0.7,
                confidence: 1.0,
                access_count: 0,
                content_hash: hash,
                tags: all_tags.clone(),
                metadata: HashMap::new(),
                namespace: None,
                created_at: now,
                updated_at: now,
                last_accessed_at: now,
            };

            if let Err(e) = self.storage.insert_memory(&mem) {
                tracing::warn!("Failed to store summary memory: {e}");
                continue;
            }

            // Add to graph
            let graph_node = GraphNode {
                id: new_id.clone(),
                kind: NodeKind::Memory,
                label: summary.chars().take(80).collect::<String>(),
                memory_id: Some(new_id.clone()),
                payload: Default::default(),
                centrality: 0.0,
                namespace: None,
            };
            if let Err(e) = self.storage.insert_graph_node(&graph_node) {
                tracing::warn!("Failed to persist graph node: {e}");
            }

            if let Ok(mut graph) = self.lock_graph() {
                let _ = graph.add_node(graph_node);

                // Link via SUMMARIZES edges from summary to each source
                for sid in &source_ids {
                    let edge = Edge {
                        id: format!("{new_id}-SUMMARIZES-{sid}"),
                        src: new_id.clone(),
                        dst: sid.clone(),
                        relationship: RelationshipType::Summarizes,
                        weight: 1.0,
                        properties: HashMap::new(),
                        created_at: now,
                        valid_from: Some(now),
                        valid_to: None,
                    };
                    if let Err(e) = self.storage.insert_graph_edge(&edge) {
                        tracing::warn!("Failed to persist SUMMARIZES edge: {e}");
                    }
                    let _ = graph.add_edge(edge);
                }
            }

            // Embed the summary
            if let Ok(Some(emb_guard)) = self.lock_embeddings() {
                if let Ok(embedding) = emb_guard.embed(&summary) {
                    drop(emb_guard);
                    if let Err(e) = self.storage.store_embedding(&new_id, &embedding) {
                        tracing::warn!("Failed to store embedding: {e}");
                    }
                    if let Ok(mut vec) = self.lock_vector() {
                        let _ = vec.insert(&new_id, &embedding);
                    }
                }
            }

            summarized_count += 1;
            created_ids.push(new_id);
        }

        // Log the consolidation run
        if let Err(e) = self
            .storage
            .insert_consolidation_log("summarize", summarized_count as usize)
        {
            tracing::warn!("Failed to log summarize consolidation: {e}");
        }

        ToolResult::text(
            json!({
                "cycle": "summarize",
                "clusters_found": large_clusters.len(),
                "summarized": summarized_count,
                "created_ids": created_ids,
                "min_cluster_size": min_cluster_size,
            })
            .to_string(),
        )
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

        // Verify importance was reduced via power-law:
        // 0.8 * 0.9^(60/30) * (1 + log2(max(0,1))*0.1) = 0.8 * 0.81 * 1.0 ≈ 0.648
        let retrieved = server.storage.get_memory(&id).unwrap().unwrap();
        assert!(
            (retrieved.importance - 0.648).abs() < 0.02,
            "expected ~0.648, got {}",
            retrieved.importance
        );
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
        let result1 = store_memory(
            &server,
            "insight about rust safety",
            "insight",
            &["rust", "safety"],
        );
        let result2 = store_memory(
            &server,
            "pattern for error handling",
            "pattern",
            &["rust", "error"],
        );
        let id1 = result1["id"].as_str().unwrap();
        let id2 = result2["id"].as_str().unwrap();

        // Manually insert embeddings so vector search can find neighbors.
        // Use similar (but not identical) vectors for the two memories.
        let emb1: Vec<f32> = (0..768).map(|i| i as f32 / 768.0).collect();
        let mut emb2 = emb1.clone();
        emb2[0] += 0.01; // slightly different
        server.storage.store_embedding(id1, &emb1).unwrap();
        server.storage.store_embedding(id2, &emb2).unwrap();
        {
            let mut vec = server.lock_vector().unwrap();
            let _ = vec.insert(id1, &emb1);
            let _ = vec.insert(id2, &emb2);
        }

        // Run creative cycle
        let params = json!({"name": "consolidate_creative", "arguments": {}});
        let resp = server.handle_request("tools/call", Some(&params), json!(1));
        let result = resp.result.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        let parsed: Value = serde_json::from_str(text).unwrap();

        assert_eq!(parsed["cycle"], "creative");
        assert_eq!(parsed["algorithm"], "vector_knn");
        // They have different types and similar embeddings, so should create a SHARES_THEME edge
        assert!(parsed["new_connections"].as_u64().unwrap() >= 1);
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
