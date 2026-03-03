//! Scoring & utility functions for the MCP server.

use super::types::{JsonRpcResponse, ToolResult};
use codemem_core::SearchResult;
use serde_json::{json, Value};
use std::io::Write;

// Re-export compute_score from engine so existing `crate::scoring::compute_score` paths work.
pub(crate) use codemem_engine::scoring::compute_score;

/// Write a JSON-RPC response as a single line to stdout.
pub(crate) fn write_response(
    writer: &mut impl Write,
    response: &JsonRpcResponse,
) -> std::io::Result<()> {
    let json = serde_json::to_string(response)?;
    writeln!(writer, "{json}")?;
    writer.flush()
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
                "namespace": r.memory.namespace,
                "breakdown": {
                    "vector_similarity": r.score_breakdown.vector_similarity,
                    "graph_strength": r.score_breakdown.graph_strength,
                    "token_overlap": r.score_breakdown.token_overlap,
                    "temporal": r.score_breakdown.temporal,
                    "tag_matching": r.score_breakdown.tag_matching,
                    "importance": r.score_breakdown.importance,
                    "confidence": r.score_breakdown.confidence,
                    "recency": r.score_breakdown.recency,
                },
            })
        })
        .collect();

    ToolResult::text(
        serde_json::to_string_pretty(&output).expect("JSON serialization of search results"),
    )
}

#[cfg(test)]
#[path = "tests/scoring_tests.rs"]
mod tests;
