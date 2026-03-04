//! Hybrid scoring for memory recall.

use crate::bm25;
use chrono::{DateTime, Utc};
use codemem_core::{MemoryNode, ScoreBreakdown};
use codemem_storage::graph::GraphEngine;

/// Compute graph strength for a memory node by combining raw graph metrics.
///
/// Uses PageRank, betweenness centrality, connectivity, and edge weights
/// from the memory's code-graph neighbors to produce a 0.0-1.0 score.
/// Weights: PageRank 40%, betweenness 30%, connectivity 20%, edge weight 10%.
pub fn graph_strength_for_memory(graph: &GraphEngine, memory_id: &str) -> f64 {
    let metrics = match graph.raw_graph_metrics_for_memory(memory_id) {
        Some(m) => m,
        None => return 0.0,
    };

    if metrics.code_neighbor_count == 0 {
        return 0.0;
    }

    let connectivity_bonus = (metrics.code_neighbor_count as f64 / 5.0).min(1.0);
    let edge_weight_bonus =
        (metrics.total_edge_weight / metrics.code_neighbor_count as f64).min(1.0);

    (0.4 * metrics.max_pagerank
        + 0.3 * metrics.max_betweenness
        + 0.2 * connectivity_bonus
        + 0.1 * edge_weight_bonus)
        .min(1.0)
}

/// Truncate a string to `max` bytes, appending "..." if truncated.
/// Handles multi-byte UTF-8 safely by finding the nearest char boundary.
pub fn truncate_content(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        let mut end = max;
        while end > 0 && !s.is_char_boundary(end) {
            end -= 1;
        }
        format!("{}...", &s[..end])
    }
}

/// Compute 9-component hybrid score for a memory against a query.
/// The `graph` parameter is used to look up edge counts for graph strength scoring.
/// The `bm25` parameter provides BM25-based token overlap scoring; if the memory
/// is in the index it uses the indexed score, otherwise falls back to `score_text`.
/// The `now` parameter makes scoring deterministic and testable by avoiding internal clock reads.
pub fn compute_score(
    memory: &MemoryNode,
    query_tokens: &[&str],
    vector_similarity: f64,
    graph: &GraphEngine,
    bm25: &bm25::Bm25Index,
    now: DateTime<Utc>,
) -> ScoreBreakdown {
    // BM25 token overlap (replaces naive split+intersect)
    // Use pre-tokenized query tokens to avoid re-tokenizing per document.
    let token_overlap = if query_tokens.is_empty() {
        0.0
    } else {
        // Try indexed score first (memory already in the BM25 index),
        // fall back to scoring against raw text for unindexed documents.
        let indexed_score = bm25.score_with_tokens_str(query_tokens, &memory.id);
        if indexed_score > 0.0 {
            indexed_score
        } else {
            bm25.score_text_with_tokens_str(query_tokens, &memory.content)
        }
    };

    // Temporal: how recently updated (exponential decay over 30 days)
    let age_hours = (now - memory.updated_at).num_hours().max(0) as f64;
    let temporal = (-age_hours / (30.0 * 24.0)).exp();

    // Tag matching: fraction of query tokens found in tags.
    // Per-memory `tags.join().to_lowercase()` is O(tags) which is typically <10 strings,
    // so allocation is negligible.
    let tag_matching = if !query_tokens.is_empty() {
        let tag_str: String = memory.tags.join(" ").to_lowercase();
        let matches = query_tokens
            .iter()
            .filter(|qt| tag_str.contains(**qt))
            .count();
        matches as f64 / query_tokens.len() as f64
    } else {
        0.0
    };

    // Recency: based on last access time (decay over 7 days)
    let access_hours = (now - memory.last_accessed_at).num_hours().max(0) as f64;
    let recency = (-access_hours / (7.0 * 24.0)).exp();

    // Enhanced graph scoring: bridge memory UUIDs to code-graph centrality.
    // Memory nodes live in a separate ID space from code nodes (sym:, file:),
    // so we collect raw metrics from code-graph neighbors and apply the
    // scoring formula here in the engine.
    let graph_strength = graph_strength_for_memory(graph, &memory.id);

    ScoreBreakdown {
        vector_similarity,
        graph_strength,
        token_overlap,
        temporal,
        tag_matching,
        importance: memory.importance,
        confidence: memory.confidence,
        recency,
    }
}

#[cfg(test)]
#[path = "tests/scoring_tests.rs"]
mod tests;
