//! Engine facade methods for graph algorithms.
//!
//! These wrap lock acquisition + graph algorithm calls so the MCP/API
//! transport layers don't interact with the graph mutex directly.

use crate::CodememEngine;
use codemem_core::{CodememError, Edge, GraphNode, NodeKind, RelationshipType};
use std::collections::HashMap;

// ── Result Types ─────────────────────────────────────────────────────────────

/// A node with its PageRank score.
#[derive(Debug, Clone)]
pub struct RankedNode {
    pub id: String,
    pub score: f64,
    pub kind: Option<String>,
    pub label: Option<String>,
}

/// In-memory graph statistics snapshot.
#[derive(Debug, Clone)]
pub struct GraphStats {
    pub node_count: usize,
    pub edge_count: usize,
    pub node_kind_counts: HashMap<String, usize>,
    pub relationship_type_counts: HashMap<String, usize>,
}

// ── Engine Methods ───────────────────────────────────────────────────────────

impl CodememEngine {
    /// BFS or DFS traversal from a start node, with optional kind/relationship filters.
    pub fn graph_traverse(
        &self,
        start_id: &str,
        depth: usize,
        algorithm: &str,
        exclude_kinds: &[NodeKind],
        include_relationships: Option<&[RelationshipType]>,
    ) -> Result<Vec<GraphNode>, CodememError> {
        let graph = self.lock_graph()?;
        let has_filters = !exclude_kinds.is_empty() || include_relationships.is_some();

        if has_filters {
            match algorithm {
                "bfs" => graph.bfs_filtered(start_id, depth, exclude_kinds, include_relationships),
                "dfs" => graph.dfs_filtered(start_id, depth, exclude_kinds, include_relationships),
                _ => Err(CodememError::InvalidInput(format!(
                    "Unknown algorithm: {algorithm}"
                ))),
            }
        } else {
            match algorithm {
                "bfs" => graph.bfs(start_id, depth),
                "dfs" => graph.dfs(start_id, depth),
                _ => Err(CodememError::InvalidInput(format!(
                    "Unknown algorithm: {algorithm}"
                ))),
            }
        }
    }

    /// Get in-memory graph statistics.
    pub fn graph_stats(&self) -> Result<GraphStats, CodememError> {
        let graph = self.lock_graph()?;
        let stats = graph.stats();
        Ok(GraphStats {
            node_count: stats.node_count,
            edge_count: stats.edge_count,
            node_kind_counts: stats.node_kind_counts,
            relationship_type_counts: stats.relationship_type_counts,
        })
    }

    /// Get all edges for a node.
    pub fn get_node_edges(&self, node_id: &str) -> Result<Vec<Edge>, CodememError> {
        let graph = self.lock_graph()?;
        graph.get_edges(node_id)
    }

    /// Run Louvain community detection at the given resolution.
    pub fn louvain_communities(&self, resolution: f64) -> Result<Vec<Vec<String>>, CodememError> {
        let graph = self.lock_graph()?;
        Ok(graph.louvain_communities(resolution))
    }

    /// Compute PageRank and return the top-k nodes with their scores,
    /// kinds, and labels.
    pub fn find_important_nodes(
        &self,
        top_k: usize,
        damping: f64,
    ) -> Result<Vec<RankedNode>, CodememError> {
        let graph = self.lock_graph()?;
        let scores = graph.pagerank(damping, 100, 1e-6);

        let mut sorted: Vec<(String, f64)> = scores.into_iter().collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted.truncate(top_k);

        let results = sorted
            .into_iter()
            .map(|(id, score)| {
                let node = graph.get_node(&id).ok().flatten();
                RankedNode {
                    id,
                    score,
                    kind: node.as_ref().map(|n| n.kind.to_string()),
                    label: node.as_ref().map(|n| n.label.clone()),
                }
            })
            .collect();

        Ok(results)
    }
}
