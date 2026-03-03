//! Memory recall with hybrid scoring.

use crate::scoring::compute_score;
use crate::CodememEngine;
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

impl CodememEngine {
    /// Core recall logic: search storage with hybrid scoring and return ranked results.
    ///
    /// Combines vector search (if embeddings available), BM25, graph strength,
    /// temporal, tag matching, importance, confidence, and recency into a
    /// 9-component hybrid score. Supports filtering by memory type, namespace,
    /// tag exclusion, and minimum importance/confidence thresholds.
    #[allow(clippy::too_many_arguments)]
    pub fn recall(
        &self,
        query: &str,
        k: usize,
        memory_type_filter: Option<MemoryType>,
        namespace_filter: Option<&str>,
        exclude_tags: &[String],
        min_importance: Option<f64>,
        min_confidence: Option<f64>,
    ) -> Result<Vec<SearchResult>, CodememError> {
        // Try vector search first (if embeddings available)
        let vector_results: Vec<(String, f32)> = if let Some(emb_guard) = self.lock_embeddings()? {
            match emb_guard.embed(query) {
                Ok(query_embedding) => {
                    drop(emb_guard);
                    let vec = self.lock_vector()?;
                    vec.search(&query_embedding, k * 2) // over-fetch for re-ranking
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

        let query_lower = query.to_lowercase();
        let query_tokens: Vec<&str> = query_lower.split_whitespace().collect();

        let graph = self.lock_graph()?;
        let bm25 = self.lock_bm25()?;

        let mut results: Vec<SearchResult> = Vec::new();

        let candidate_ids: Vec<(String, f64)> = if vector_results.is_empty() {
            // Fallback: text search over all memories
            let ids = self.storage.list_memory_ids()?;
            ids.into_iter().map(|id| (id, 0.0)).collect()
        } else {
            vector_results
                .iter()
                .map(|(id, similarity)| (id.clone(), *similarity as f64))
                .collect()
        };

        let weights = self.scoring_weights()?;

        for (id, similarity) in &candidate_ids {
            if let Ok(Some(memory)) = self.storage.get_memory(id) {
                // Apply memory_type filter
                if let Some(ref filter_type) = memory_type_filter {
                    if memory.memory_type != *filter_type {
                        continue;
                    }
                }
                // Apply namespace filter
                if let Some(ns) = namespace_filter {
                    if memory.namespace.as_deref() != Some(ns) {
                        continue;
                    }
                }
                // Apply quality filters
                if !exclude_tags.is_empty() && memory.tags.iter().any(|t| exclude_tags.contains(t))
                {
                    continue;
                }
                if let Some(min) = min_importance {
                    if memory.importance < min {
                        continue;
                    }
                }
                if let Some(min) = min_confidence {
                    if memory.confidence < min {
                        continue;
                    }
                }

                let breakdown =
                    compute_score(&memory, query, &query_tokens, *similarity, &graph, &bm25);
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
        results.truncate(k);

        Ok(results)
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
        let query_lower = query.to_lowercase();
        let query_tokens: Vec<&str> = query_lower.split_whitespace().collect();

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

        let graph = self.lock_graph()?;
        let bm25 = self.lock_bm25()?;

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
            let ids = self.storage.list_memory_ids()?;
            let weights = self.scoring_weights()?;

            for id in &ids {
                if let Ok(Some(memory)) = self.storage.get_memory(id) {
                    if let Some(ns) = namespace_filter {
                        if memory.namespace.as_deref() != Some(ns) {
                            continue;
                        }
                    }
                    let breakdown =
                        compute_score(&memory, query, &query_tokens, 0.0, &graph, &bm25);
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
            }
        } else {
            // Vector search path
            for (id, similarity) in &vector_results {
                if let Ok(Some(memory)) = self.storage.get_memory(id) {
                    if let Some(ns) = namespace_filter {
                        if memory.namespace.as_deref() != Some(ns) {
                            continue;
                        }
                    }
                    seen_ids.insert(memory.id.clone());
                    all_memories.push(ScoredMemory {
                        memory,
                        vector_sim: *similarity as f64,
                        expansion_path: "direct".to_string(),
                    });
                }
            }
        }

        // Step 2-4: Graph expansion from each direct result
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
        let weights = self.scoring_weights()?;
        let mut scored_results: Vec<ExpandedResult> = all_memories
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

        Ok(deleted)
    }
}
