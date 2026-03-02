//! codemem-graph: Graph engine with petgraph algorithms and SQLite persistence.
//!
//! Provides BFS, DFS, shortest path, and connected components over
//! a knowledge graph with 6 node types and 15 relationship types.

mod algorithms;
mod traversal;

#[cfg(test)]
use codemem_core::NodeKind;
use codemem_core::{CodememError, Edge, GraphBackend, GraphNode};
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::Direction;
use std::collections::{HashMap, HashSet, VecDeque};

/// In-memory graph backed by petgraph, synced to SQLite via codemem-storage.
pub struct GraphEngine {
    pub(crate) graph: DiGraph<String, f64>,
    /// Map from string node IDs to petgraph NodeIndex.
    pub(crate) id_to_index: HashMap<String, NodeIndex>,
    /// Node data by ID.
    pub(crate) nodes: HashMap<String, GraphNode>,
    /// Edge data by ID.
    pub(crate) edges: HashMap<String, Edge>,
    /// Cached PageRank scores (populated by `recompute_centrality()`).
    pub(crate) cached_pagerank: HashMap<String, f64>,
    /// Cached betweenness centrality scores (populated by `recompute_centrality()`).
    pub(crate) cached_betweenness: HashMap<String, f64>,
}

impl GraphEngine {
    /// Create a new empty graph.
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            id_to_index: HashMap::new(),
            nodes: HashMap::new(),
            edges: HashMap::new(),
            cached_pagerank: HashMap::new(),
            cached_betweenness: HashMap::new(),
        }
    }

    /// Load graph from storage.
    pub fn from_storage(storage: &dyn codemem_core::StorageBackend) -> Result<Self, CodememError> {
        let mut engine = Self::new();

        // Load all nodes
        let nodes = storage.all_graph_nodes()?;
        for node in nodes {
            engine.add_node(node)?;
        }

        // Load all edges
        let edges = storage.all_graph_edges()?;
        for edge in edges {
            engine.add_edge(edge)?;
        }

        // Compute degree centrality so subgraph queries can rank nodes
        engine.compute_centrality();

        Ok(engine)
    }

    /// Get the number of nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get the number of edges.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Multi-hop expansion: given a set of node IDs, expand N hops to find related nodes.
    pub fn expand(
        &self,
        start_ids: &[String],
        max_hops: usize,
    ) -> Result<Vec<GraphNode>, CodememError> {
        let mut visited = std::collections::HashSet::new();
        let mut result = Vec::new();

        for start_id in start_ids {
            let nodes = self.bfs(start_id, max_hops)?;
            for node in nodes {
                if visited.insert(node.id.clone()) {
                    result.push(node);
                }
            }
        }

        Ok(result)
    }

    /// Get neighbors of a node (1-hop).
    pub fn neighbors(&self, node_id: &str) -> Result<Vec<GraphNode>, CodememError> {
        let idx = self
            .id_to_index
            .get(node_id)
            .ok_or_else(|| CodememError::NotFound(format!("Node {node_id}")))?;

        let mut result = Vec::new();
        for neighbor_idx in self.graph.neighbors(*idx) {
            if let Some(neighbor_id) = self.graph.node_weight(neighbor_idx) {
                if let Some(node) = self.nodes.get(neighbor_id) {
                    result.push(node.clone());
                }
            }
        }

        Ok(result)
    }

    /// Return groups of connected node IDs.
    ///
    /// Treats the directed graph as undirected: two nodes are in the same
    /// component if there is a path between them in either direction.
    /// Each inner `Vec<String>` is one connected component.
    pub fn connected_components(&self) -> Vec<Vec<String>> {
        let mut visited: HashSet<NodeIndex> = HashSet::new();
        let mut components: Vec<Vec<String>> = Vec::new();

        for &start_idx in self.id_to_index.values() {
            if visited.contains(&start_idx) {
                continue;
            }

            // BFS treating edges as undirected
            let mut component: Vec<String> = Vec::new();
            let mut queue: VecDeque<NodeIndex> = VecDeque::new();
            queue.push_back(start_idx);
            visited.insert(start_idx);

            while let Some(current) = queue.pop_front() {
                if let Some(node_id) = self.graph.node_weight(current) {
                    component.push(node_id.clone());
                }

                // Follow outgoing edges
                for neighbor in self.graph.neighbors_directed(current, Direction::Outgoing) {
                    if visited.insert(neighbor) {
                        queue.push_back(neighbor);
                    }
                }

                // Follow incoming edges (treat as undirected)
                for neighbor in self.graph.neighbors_directed(current, Direction::Incoming) {
                    if visited.insert(neighbor) {
                        queue.push_back(neighbor);
                    }
                }
            }

            component.sort();
            components.push(component);
        }

        components.sort();
        components
    }

    /// Compute degree centrality for every node and update their `centrality` field.
    ///
    /// Degree centrality for node *v* is defined as:
    ///   `(in_degree(v) + out_degree(v)) / (N - 1)`
    /// where *N* is the total number of nodes.  When N <= 1, centrality is 0.
    pub fn compute_centrality(&mut self) {
        let n = self.nodes.len();
        if n <= 1 {
            for node in self.nodes.values_mut() {
                node.centrality = 0.0;
            }
            return;
        }

        let denominator = (n - 1) as f64;

        // Pre-compute centrality values by node ID.
        let centrality_map: HashMap<String, f64> = self
            .id_to_index
            .iter()
            .map(|(id, &idx)| {
                let in_deg = self
                    .graph
                    .neighbors_directed(idx, Direction::Incoming)
                    .count();
                let out_deg = self
                    .graph
                    .neighbors_directed(idx, Direction::Outgoing)
                    .count();
                let centrality = (in_deg + out_deg) as f64 / denominator;
                (id.clone(), centrality)
            })
            .collect();

        // Apply centrality values to the stored nodes.
        for (id, centrality) in &centrality_map {
            if let Some(node) = self.nodes.get_mut(id) {
                node.centrality = *centrality;
            }
        }
    }

    /// Return all nodes currently in the graph.
    pub fn get_all_nodes(&self) -> Vec<GraphNode> {
        self.nodes.values().cloned().collect()
    }

    /// Recompute and cache PageRank and betweenness centrality scores.
    ///
    /// This should be called after loading the graph (e.g., on server start)
    /// and periodically when the graph changes significantly.
    pub fn recompute_centrality(&mut self) {
        self.cached_pagerank = self.pagerank(0.85, 100, 1e-6);
        self.cached_betweenness = self.betweenness_centrality();
    }

    /// Get the cached PageRank score for a node. Returns 0.0 if not found.
    pub fn get_pagerank(&self, node_id: &str) -> f64 {
        self.cached_pagerank.get(node_id).copied().unwrap_or(0.0)
    }

    /// Get the cached betweenness centrality score for a node. Returns 0.0 if not found.
    pub fn get_betweenness(&self, node_id: &str) -> f64 {
        self.cached_betweenness.get(node_id).copied().unwrap_or(0.0)
    }

    /// Get the maximum degree (in + out) across all nodes in the graph.
    /// Returns 1.0 if the graph has fewer than 2 nodes to avoid division by zero.
    pub fn max_degree(&self) -> f64 {
        if self.nodes.len() <= 1 {
            return 1.0;
        }
        self.id_to_index
            .values()
            .map(|&idx| {
                let in_deg = self
                    .graph
                    .neighbors_directed(idx, Direction::Incoming)
                    .count();
                let out_deg = self
                    .graph
                    .neighbors_directed(idx, Direction::Outgoing)
                    .count();
                (in_deg + out_deg) as f64
            })
            .fold(1.0f64, f64::max)
    }
}

#[cfg(test)]
#[path = "tests/lib_tests.rs"]
mod tests;
