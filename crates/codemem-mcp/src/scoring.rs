//! Scoring & utility functions for the MCP server.

use crate::bm25;
use crate::types::{JsonRpcResponse, ToolResult};
use codemem_core::{MemoryNode, ScoreBreakdown, SearchResult};
use codemem_graph::GraphEngine;
use serde_json::{json, Value};
use std::io::Write;

/// Write a JSON-RPC response as a single line to stdout.
pub(crate) fn write_response(
    writer: &mut impl Write,
    response: &JsonRpcResponse,
) -> std::io::Result<()> {
    let json = serde_json::to_string(response)?;
    writeln!(writer, "{json}")?;
    writer.flush()
}

pub(crate) fn truncate_str(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}...", &s[..max])
    }
}

/// Format search results into a ToolResult. If `_repo_label` is provided,
/// a "repo" field is added to each result.
pub(crate) fn format_recall_results(
    results: &[SearchResult],
    _repo_label: Option<&str>,
) -> ToolResult {
    if results.is_empty() {
        return ToolResult::text("No matching memories found.");
    }

    let output: Vec<Value> = results
        .iter()
        .map(|r| {
            json!({
                "id": r.memory.id,
                "content": r.memory.content,
                "memory_type": r.memory.memory_type.to_string(),
                "score": format!("{:.4}", r.score),
                "importance": r.memory.importance,
                "tags": r.memory.tags,
                "access_count": r.memory.access_count,
            })
        })
        .collect();

    ToolResult::text(serde_json::to_string_pretty(&output).unwrap())
}

/// Compute 9-component hybrid score for a memory against a query.
/// The `graph` parameter is used to look up edge counts for graph strength scoring.
/// The `bm25` parameter provides BM25-based token overlap scoring; if the memory
/// is in the index it uses the indexed score, otherwise falls back to `score_text`.
pub(crate) fn compute_score(
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

    // Enhanced graph scoring using cached centrality metrics.
    // Combines PageRank, betweenness centrality, normalized degree,
    // and a cluster bonus for richer graph-awareness.
    let pagerank = graph.get_pagerank(&memory.id);
    let betweenness = graph.get_betweenness(&memory.id);
    let degree = graph.neighbors(&memory.id).map(|n| n.len()).unwrap_or(0) as f64;
    let max_degree = graph.max_degree();
    let normalized_degree = degree / max_degree.max(1.0);

    // Cluster bonus: if the memory has many neighbors, it gets a small bonus (capped at 1.0).
    let cluster_bonus = graph
        .neighbors(&memory.id)
        .map(|n| (n.len() as f64 / 10.0).min(1.0))
        .unwrap_or(0.0);

    let graph_strength =
        (0.4 * pagerank + 0.3 * betweenness + 0.2 * normalized_degree + 0.1 * cluster_bonus)
            .min(1.0);

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
mod tests {
    use super::*;

    #[test]
    fn write_response_newline_delimited() {
        let resp = JsonRpcResponse::success(json!(1), json!({"ok": true}));
        let mut buf = Vec::new();
        write_response(&mut buf, &resp).unwrap();
        let output = String::from_utf8(buf).unwrap();
        assert!(output.ends_with('\n'));
        assert!(!output.contains("Content-Length"));
    }
}
