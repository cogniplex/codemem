//! Hybrid scoring for memory recall.

use crate::bm25;
use codemem_core::{MemoryNode, ScoreBreakdown};
use codemem_storage::graph::GraphEngine;

/// Truncate a string to `max` characters, appending "..." if truncated.
pub fn truncate_content(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}...", &s[..max])
    }
}

/// Compute 9-component hybrid score for a memory against a query.
/// The `graph` parameter is used to look up edge counts for graph strength scoring.
/// The `bm25` parameter provides BM25-based token overlap scoring; if the memory
/// is in the index it uses the indexed score, otherwise falls back to `score_text`.
pub fn compute_score(
    memory: &MemoryNode,
    query: &str,
    query_tokens: &[&str],
    vector_similarity: f64,
    graph: &GraphEngine,
    bm25: &bm25::Bm25Index,
) -> ScoreBreakdown {
    // BM25 token overlap (replaces naive split+intersect)
    let token_overlap = if query.is_empty() {
        0.0
    } else {
        // Try indexed score first (memory already in the BM25 index),
        // fall back to scoring against raw text for unindexed documents.
        let indexed_score = bm25.score(query, &memory.id);
        if indexed_score > 0.0 {
            indexed_score
        } else {
            bm25.score_text(query, &memory.content)
        }
    };

    // Temporal: how recently updated (exponential decay over 30 days)
    let age_hours = (chrono::Utc::now() - memory.updated_at).num_hours().max(0) as f64;
    let temporal = (-age_hours / (30.0 * 24.0)).exp();

    // Tag matching: fraction of query tokens found in tags
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
    let access_hours = (chrono::Utc::now() - memory.last_accessed_at)
        .num_hours()
        .max(0) as f64;
    let recency = (-access_hours / (7.0 * 24.0)).exp();

    // Enhanced graph scoring: bridge memory UUIDs to code-graph centrality.
    // Memory nodes live in a separate ID space from code nodes (sym:, file:),
    // so we use graph_strength_for_memory() which traverses neighbors to find
    // connected code nodes and aggregates their PageRank/betweenness.
    let graph_strength = graph.graph_strength_for_memory(&memory.id);

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
