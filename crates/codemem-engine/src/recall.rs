//! Memory recall with hybrid scoring.

use crate::scoring::compute_score;
use crate::CodememEngine;
use chrono::Utc;
use codemem_core::{
    CodememError, GraphBackend, MemoryNode, MemoryType, NodeKind, SearchResult, VectorBackend,
};
use std::collections::{HashMap, HashSet};

/// A recall result that includes the expansion path taken to reach the memory.
#[derive(Debug, Clone)]
pub struct ExpandedResult {
    pub result: SearchResult,
    pub expansion_path: String,
}

/// Aggregated stats for a single namespace.
#[derive(Debug, Clone)]
pub struct NamespaceStats {
    pub namespace: String,
    pub count: usize,
    pub avg_importance: f64,
    pub avg_confidence: f64,
    pub type_distribution: HashMap<String, usize>,
    pub tag_frequency: HashMap<String, usize>,
    pub oldest: Option<chrono::DateTime<chrono::Utc>>,
    pub newest: Option<chrono::DateTime<chrono::Utc>>,
}

/// Parameters for the recall query.
#[derive(Debug, Clone)]
pub struct RecallQuery<'a> {
    pub query: &'a str,
    pub k: usize,
    pub memory_type_filter: Option<MemoryType>,
    pub namespace_filter: Option<&'a str>,
    pub exclude_tags: &'a [String],
    pub min_importance: Option<f64>,
    pub min_confidence: Option<f64>,
}

impl<'a> RecallQuery<'a> {
    /// Create a minimal recall query with just the search text and result limit.
    pub fn new(query: &'a str, k: usize) -> Self {
        Self {
            query,
            k,
            memory_type_filter: None,
            namespace_filter: None,
            exclude_tags: &[],
            min_importance: None,
            min_confidence: None,
        }
    }
}

impl CodememEngine {
    /// Core recall logic: search storage with hybrid scoring and return ranked results.
    ///
    /// Combines vector search (if embeddings available), BM25, graph strength,
    /// temporal, tag matching, importance, confidence, and recency into a
    /// 9-component hybrid score. Supports filtering by memory type, namespace,
    /// tag exclusion, and minimum importance/confidence thresholds.
    pub fn recall(&self, q: &RecallQuery<'_>) -> Result<Vec<SearchResult>, CodememError> {
        // Opportunistic cleanup of expired memories (rate-limited to once per 60s)
        self.sweep_expired_memories();

        // Try vector search first (if embeddings available)
        let vector_results: Vec<(String, f32)> = if let Some(emb_guard) = self.lock_embeddings()? {
            match emb_guard.embed(q.query) {
                Ok(query_embedding) => {
                    drop(emb_guard);
                    let vec = self.lock_vector()?;
                    vec.search(&query_embedding, q.k * 2) // over-fetch for re-ranking
                        .unwrap_or_default()
                }
                Err(e) => {
                    tracing::warn!("Query embedding failed: {e}");
                    vec![]
                }
            }
        } else {
            vec![]
        };

        // H1: Use code-aware tokenizer for query tokens so that compound identifiers
        // like "parseFunction" are split into ["parse", "function"] — matching the
        // tokenization used when documents were added to the BM25 index.
        let query_tokens: Vec<String> = crate::bm25::tokenize(q.query);
        let query_token_refs: Vec<&str> = query_tokens.iter().map(|s| s.as_str()).collect();

        // Graph and BM25 intentionally load different data: graph stores structural relationships
        // (nodes/edges), while BM25 indexes memory content for text search. This is by design,
        // not duplication.
        let mut graph = self.lock_graph()?;
        // C1: Lazily compute betweenness centrality before scoring so the
        // betweenness component (30% of graph_strength) is not permanently zero.
        graph.ensure_betweenness_computed();
        let bm25 = self.lock_bm25()?;
        let now = Utc::now();

        // Entity expansion: find memories connected to code entities mentioned in the query.
        // This ensures that structurally related memories are candidates even when they are
        // semantically distant from the query text.
        let entity_memory_ids = self.resolve_entity_memories(q.query, &graph, now);

        let mut results: Vec<SearchResult> = Vec::new();
        let weights = self.scoring_weights()?;

        if vector_results.is_empty() {
            // Fallback: batch-load all memories matching filters in one query
            let type_str = q.memory_type_filter.as_ref().map(|t| t.to_string());
            let all_memories = self
                .storage
                .list_memories_filtered(q.namespace_filter, type_str.as_deref())?;

            for memory in all_memories {
                if !Self::passes_quality_filters(&memory, q) {
                    continue;
                }

                let breakdown = compute_score(&memory, &query_token_refs, 0.0, &graph, &bm25, now);
                let score = breakdown.total_with_weights(&weights);
                if score > 0.01 {
                    results.push(SearchResult {
                        memory,
                        score,
                        score_breakdown: breakdown,
                    });
                }
            }
        } else {
            // Vector search path: batch-fetch all candidate memories + entity-connected memories
            let mut all_candidate_ids: HashSet<&str> =
                vector_results.iter().map(|(id, _)| id.as_str()).collect();

            // Merge entity-connected memory IDs into the candidate pool
            for eid in &entity_memory_ids {
                all_candidate_ids.insert(eid.as_str());
            }

            let candidate_id_vec: Vec<&str> = all_candidate_ids.into_iter().collect();
            let candidate_memories = self.storage.get_memories_batch(&candidate_id_vec)?;

            // Build similarity lookup (entity memories will get 0.0 similarity)
            let sim_map: HashMap<&str, f64> = vector_results
                .iter()
                .map(|(id, sim)| (id.as_str(), *sim as f64))
                .collect();

            for memory in candidate_memories {
                // Apply memory_type filter
                if let Some(ref filter_type) = q.memory_type_filter {
                    if memory.memory_type != *filter_type {
                        continue;
                    }
                }
                // Apply namespace filter
                if let Some(ns) = q.namespace_filter {
                    if memory.namespace.as_deref() != Some(ns) {
                        continue;
                    }
                }
                if !Self::passes_quality_filters(&memory, q) {
                    continue;
                }

                let similarity = sim_map.get(memory.id.as_str()).copied().unwrap_or(0.0);
                let breakdown =
                    compute_score(&memory, &query_token_refs, similarity, &graph, &bm25, now);
                let score = breakdown.total_with_weights(&weights);
                if score > 0.01 {
                    results.push(SearchResult {
                        memory,
                        score,
                        score_breakdown: breakdown,
                    });
                }
            }
        }

        // Sort by score descending, take top k
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(q.k);

        Ok(results)
    }

    /// Check expiry, exclude_tags, min_importance, and min_confidence filters.
    fn passes_quality_filters(memory: &MemoryNode, q: &RecallQuery<'_>) -> bool {
        // Skip expired memories (their embeddings may linger in HNSW until next sweep)
        if memory.expires_at.is_some_and(|dt| dt <= Utc::now()) {
            return false;
        }
        if !q.exclude_tags.is_empty() && memory.tags.iter().any(|t| q.exclude_tags.contains(t)) {
            return false;
        }
        if let Some(min) = q.min_importance {
            if memory.importance < min {
                return false;
            }
        }
        if let Some(min) = q.min_confidence {
            if memory.confidence < min {
                return false;
            }
        }
        true
    }

    /// Recall with graph expansion: vector search (or BM25 fallback) for seed
    /// memories, then BFS expansion from each seed through the graph, scoring
    /// all candidates with the 9-component hybrid scorer.
    pub fn recall_with_expansion(
        &self,
        query: &str,
        k: usize,
        expansion_depth: usize,
        namespace_filter: Option<&str>,
    ) -> Result<Vec<ExpandedResult>, CodememError> {
        // Opportunistic cleanup of expired memories (rate-limited to once per 60s)
        self.sweep_expired_memories();

        // H1: Code-aware tokenization for consistent BM25 scoring
        let query_tokens: Vec<String> = crate::bm25::tokenize(query);
        let query_token_refs: Vec<&str> = query_tokens.iter().map(|s| s.as_str()).collect();

        // Step 1: Run normal vector search (or text fallback)
        let vector_results: Vec<(String, f32)> = if let Some(emb_guard) = self.lock_embeddings()? {
            match emb_guard.embed(query) {
                Ok(query_embedding) => {
                    drop(emb_guard);
                    let vec = self.lock_vector()?;
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

        let mut graph = self.lock_graph()?;
        // C1: Lazily compute betweenness centrality before scoring
        graph.ensure_betweenness_computed();
        let bm25 = self.lock_bm25()?;
        let now = Utc::now();

        // Collect initial seed memories with their vector similarity
        struct ScoredMemory {
            memory: MemoryNode,
            vector_sim: f64,
            expansion_path: String,
        }

        let mut all_memories: Vec<ScoredMemory> = Vec::new();
        let mut seen_ids: HashSet<String> = HashSet::new();

        if vector_results.is_empty() {
            // Fallback: batch-load all memories matching namespace in one query
            let all = self
                .storage
                .list_memories_filtered(namespace_filter, None)?;
            let weights = self.scoring_weights()?;

            for memory in all {
                if memory.expires_at.is_some_and(|dt| dt <= now) {
                    continue;
                }
                let breakdown = compute_score(&memory, &query_token_refs, 0.0, &graph, &bm25, now);
                let score = breakdown.total_with_weights(&weights);
                if score > 0.01 {
                    seen_ids.insert(memory.id.clone());
                    all_memories.push(ScoredMemory {
                        memory,
                        vector_sim: 0.0,
                        expansion_path: "direct".to_string(),
                    });
                }
            }
        } else {
            // Vector search path: batch-fetch all candidate memories
            let candidate_ids: Vec<&str> =
                vector_results.iter().map(|(id, _)| id.as_str()).collect();
            let candidate_memories = self.storage.get_memories_batch(&candidate_ids)?;

            let sim_map: HashMap<&str, f64> = vector_results
                .iter()
                .map(|(id, sim)| (id.as_str(), *sim as f64))
                .collect();

            for memory in candidate_memories {
                if memory.expires_at.is_some_and(|dt| dt <= now) {
                    continue;
                }
                if let Some(ns) = namespace_filter {
                    if memory.namespace.as_deref() != Some(ns) {
                        continue;
                    }
                }
                let similarity = sim_map.get(memory.id.as_str()).copied().unwrap_or(0.0);
                seen_ids.insert(memory.id.clone());
                all_memories.push(ScoredMemory {
                    memory,
                    vector_sim: similarity,
                    expansion_path: "direct".to_string(),
                });
            }
        }

        // Step 2-4: Graph expansion from each direct result
        // A7: BFS traverses through ALL node kinds (including code nodes like
        // File, Function, etc.) as intermediaries, but only COLLECTS Memory nodes.
        // A6: Apply temporal edge filtering — skip edges whose valid_to < now.
        let direct_ids: Vec<String> = all_memories.iter().map(|m| m.memory.id.clone()).collect();

        for direct_id in &direct_ids {
            // Cache edges for this direct node outside the inner loop,
            // filtering out expired temporal edges (A6)
            let direct_edges: Vec<_> = graph
                .get_edges(direct_id)
                .unwrap_or_default()
                .into_iter()
                .filter(|e| is_edge_active(e, now))
                .collect();

            // A7: Only exclude Chunk from BFS traversal (noisy), but allow
            // File, Function, Class, etc. as intermediaries to reach more Memory nodes
            if let Ok(expanded_nodes) =
                graph.bfs_filtered(direct_id, expansion_depth, &[NodeKind::Chunk], None)
            {
                for expanded_node in &expanded_nodes {
                    // Skip the start node itself (already in results)
                    if expanded_node.id == *direct_id {
                        continue;
                    }

                    // A7: Only COLLECT Memory nodes in results, but we
                    // traversed through all other node kinds to reach them
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

                    // Fetch the memory (no-touch to avoid inflating access_count)
                    if let Ok(Some(memory)) = self.storage.get_memory_no_touch(memory_id) {
                        if memory.expires_at.is_some_and(|dt| dt <= now) {
                            continue;
                        }
                        if let Some(ns) = namespace_filter {
                            if memory.namespace.as_deref() != Some(ns) {
                                continue;
                            }
                        }

                        // Build expansion path description using cached edges
                        let expansion_path = direct_edges
                            .iter()
                            .find(|e| e.dst == expanded_node.id || e.src == expanded_node.id)
                            .map(|e| format!("via {} from {}", e.relationship, direct_id))
                            .unwrap_or_else(|| format!("via graph from {direct_id}"));

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
        let weights = self.scoring_weights()?;
        let mut scored_results: Vec<ExpandedResult> = all_memories
            .into_iter()
            .map(|sm| {
                let breakdown = compute_score(
                    &sm.memory,
                    &query_token_refs,
                    sm.vector_sim,
                    &graph,
                    &bm25,
                    now,
                );
                let score = breakdown.total_with_weights(&weights);
                ExpandedResult {
                    result: SearchResult {
                        memory: sm.memory,
                        score,
                        score_breakdown: breakdown,
                    },
                    expansion_path: sm.expansion_path,
                }
            })
            .collect();

        scored_results.sort_by(|a, b| {
            b.result
                .score
                .partial_cmp(&a.result.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored_results.truncate(k);

        Ok(scored_results)
    }

    /// Resolve entity references from a query to memory IDs connected to those entities.
    ///
    /// Extracts code references (CamelCase identifiers, qualified paths, file paths,
    /// backtick-wrapped code) from the query, matches them to graph nodes, and returns
    /// the IDs of Memory nodes within one hop of each matched entity. This ensures
    /// structurally related memories are recall candidates even when semantically distant.
    pub(crate) fn resolve_entity_memories(
        &self,
        query: &str,
        graph: &codemem_storage::graph::GraphEngine,
        now: chrono::DateTime<chrono::Utc>,
    ) -> HashSet<String> {
        let entity_refs = crate::search::extract_code_references(query);
        let mut memory_ids: HashSet<String> = HashSet::new();

        for entity_ref in &entity_refs {
            // Try common ID patterns: sym:Name, file:path, or direct ID match
            let candidate_ids = [
                format!("sym:{entity_ref}"),
                format!("file:{entity_ref}"),
                entity_ref.clone(),
            ];

            for candidate_id in &candidate_ids {
                if graph.get_node_ref(candidate_id).is_none() {
                    continue;
                }
                // Found a matching node — collect one-hop Memory neighbors
                for edge in graph.get_edges_ref(candidate_id) {
                    if !is_edge_active(edge, now) {
                        continue;
                    }
                    let neighbor_id = if edge.src == *candidate_id {
                        &edge.dst
                    } else {
                        &edge.src
                    };
                    if let Some(node) = graph.get_node_ref(neighbor_id) {
                        if node.kind == NodeKind::Memory {
                            let mem_id = node.memory_id.as_deref().unwrap_or(&node.id);
                            memory_ids.insert(mem_id.to_string());
                        }
                    }
                }
                break; // Found the node, no need to try other ID patterns
            }
        }

        memory_ids
    }

    /// Compute detailed stats for a single namespace: count, averages,
    /// type distribution, tag frequency, and date range.
    pub fn namespace_stats(&self, namespace: &str) -> Result<NamespaceStats, CodememError> {
        let ids = self.storage.list_memory_ids_for_namespace(namespace)?;

        if ids.is_empty() {
            return Ok(NamespaceStats {
                namespace: namespace.to_string(),
                count: 0,
                avg_importance: 0.0,
                avg_confidence: 0.0,
                type_distribution: HashMap::new(),
                tag_frequency: HashMap::new(),
                oldest: None,
                newest: None,
            });
        }

        let mut total_importance = 0.0;
        let mut total_confidence = 0.0;
        let mut type_distribution: HashMap<String, usize> = HashMap::new();
        let mut tag_frequency: HashMap<String, usize> = HashMap::new();
        let mut oldest: Option<chrono::DateTime<chrono::Utc>> = None;
        let mut newest: Option<chrono::DateTime<chrono::Utc>> = None;
        let mut count = 0usize;

        // M2: Batch-fetch all memories in one query instead of per-ID get_memory_no_touch.
        // get_memories_batch does not increment access_count (pure SELECT), so it is
        // equivalent to get_memory_no_touch for stats purposes.
        let id_refs: Vec<&str> = ids.iter().map(|s| s.as_str()).collect();
        let memories = self.storage.get_memories_batch(&id_refs)?;

        for memory in &memories {
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

        Ok(NamespaceStats {
            namespace: namespace.to_string(),
            count,
            avg_importance,
            avg_confidence,
            type_distribution,
            tag_frequency,
            oldest,
            newest,
        })
    }

    /// Delete all memories in a namespace from all subsystems (storage, vector,
    /// graph, BM25). Returns the number of memories deleted.
    pub fn delete_namespace(&self, namespace: &str) -> Result<usize, CodememError> {
        let ids = self.storage.list_memory_ids_for_namespace(namespace)?;

        let mut deleted = 0usize;
        let mut graph = self.lock_graph()?;
        let mut vector = self.lock_vector()?;
        let mut bm25 = self.lock_bm25()?;

        for id in &ids {
            // Use cascade delete: atomic transaction deleting memory + graph + embedding from SQLite
            if let Ok(true) = self.storage.delete_memory_cascade(id) {
                deleted += 1;

                // Remove from in-memory indexes
                let _ = vector.remove(id);
                let _ = graph.remove_node(id);
                bm25.remove_document(id);
            }
        }

        // Drop locks before calling save_index (which acquires vector lock)
        drop(graph);
        drop(vector);
        drop(bm25);

        // Persist vector index to disk
        self.save_index();

        Ok(deleted)
    }
}

/// Check if an edge is currently active based on its temporal bounds.
/// An edge is active if:
/// - `valid_from` is None or <= `now`
/// - `valid_to` is None or > `now`
pub(crate) fn is_edge_active(
    edge: &codemem_core::Edge,
    now: chrono::DateTime<chrono::Utc>,
) -> bool {
    if let Some(valid_to) = edge.valid_to {
        if valid_to < now {
            return false;
        }
    }
    if let Some(valid_from) = edge.valid_from {
        if valid_from > now {
            return false;
        }
    }
    true
}
