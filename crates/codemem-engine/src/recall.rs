//! Memory recall with hybrid scoring.

use crate::scoring::compute_score;
use crate::CodememEngine;
use codemem_core::{CodememError, MemoryType, SearchResult, VectorBackend};

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
}
