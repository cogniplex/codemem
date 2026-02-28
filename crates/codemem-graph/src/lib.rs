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
mod tests {
    use super::*;
    use codemem_core::RelationshipType;

    fn file_node(id: &str, label: &str) -> GraphNode {
        GraphNode {
            id: id.to_string(),
            kind: NodeKind::File,
            label: label.to_string(),
            payload: HashMap::new(),
            centrality: 0.0,
            memory_id: None,
            namespace: None,
        }
    }

    fn test_edge(src: &str, dst: &str) -> Edge {
        Edge {
            id: format!("{src}->{dst}"),
            src: src.to_string(),
            dst: dst.to_string(),
            relationship: RelationshipType::Contains,
            weight: 1.0,
            properties: HashMap::new(),
            created_at: chrono::Utc::now(),
            valid_from: None,
            valid_to: None,
        }
    }

    #[test]
    fn connected_components_single_component() {
        let mut graph = GraphEngine::new();
        graph.add_node(file_node("a", "a.rs")).unwrap();
        graph.add_node(file_node("b", "b.rs")).unwrap();
        graph.add_node(file_node("c", "c.rs")).unwrap();
        graph.add_edge(test_edge("a", "b")).unwrap();
        graph.add_edge(test_edge("b", "c")).unwrap();

        let components = graph.connected_components();
        assert_eq!(components.len(), 1);
        assert_eq!(components[0], vec!["a", "b", "c"]);
    }

    #[test]
    fn connected_components_multiple() {
        let mut graph = GraphEngine::new();
        graph.add_node(file_node("a", "a.rs")).unwrap();
        graph.add_node(file_node("b", "b.rs")).unwrap();
        graph.add_node(file_node("c", "c.rs")).unwrap();
        graph.add_node(file_node("d", "d.rs")).unwrap();
        graph.add_edge(test_edge("a", "b")).unwrap();
        graph.add_edge(test_edge("c", "d")).unwrap();

        let components = graph.connected_components();
        assert_eq!(components.len(), 2);
        assert_eq!(components[0], vec!["a", "b"]);
        assert_eq!(components[1], vec!["c", "d"]);
    }

    #[test]
    fn connected_components_isolated_node() {
        let mut graph = GraphEngine::new();
        graph.add_node(file_node("a", "a.rs")).unwrap();
        graph.add_node(file_node("b", "b.rs")).unwrap();
        graph.add_node(file_node("c", "c.rs")).unwrap();
        graph.add_edge(test_edge("a", "b")).unwrap();
        // "c" is isolated

        let components = graph.connected_components();
        assert_eq!(components.len(), 2);
        // Sorted: ["a","b"] comes before ["c"]
        assert_eq!(components[0], vec!["a", "b"]);
        assert_eq!(components[1], vec!["c"]);
    }

    #[test]
    fn connected_components_reverse_edge_connects() {
        // Directed edge c->a should still put a and c in the same component
        // when treated as undirected.
        let mut graph = GraphEngine::new();
        graph.add_node(file_node("a", "a.rs")).unwrap();
        graph.add_node(file_node("b", "b.rs")).unwrap();
        graph.add_node(file_node("c", "c.rs")).unwrap();
        graph.add_edge(test_edge("a", "b")).unwrap();
        graph.add_edge(test_edge("c", "a")).unwrap();

        let components = graph.connected_components();
        assert_eq!(components.len(), 1);
        assert_eq!(components[0], vec!["a", "b", "c"]);
    }

    #[test]
    fn connected_components_empty_graph() {
        let graph = GraphEngine::new();
        let components = graph.connected_components();
        assert!(components.is_empty());
    }

    #[test]
    fn compute_centrality_simple() {
        // Graph: a -> b -> c
        // Node a: out=1, in=0 => centrality = 1/2 = 0.5
        // Node b: out=1, in=1 => centrality = 2/2 = 1.0
        // Node c: out=0, in=1 => centrality = 1/2 = 0.5
        let mut graph = GraphEngine::new();
        graph.add_node(file_node("a", "a.rs")).unwrap();
        graph.add_node(file_node("b", "b.rs")).unwrap();
        graph.add_node(file_node("c", "c.rs")).unwrap();
        graph.add_edge(test_edge("a", "b")).unwrap();
        graph.add_edge(test_edge("b", "c")).unwrap();

        graph.compute_centrality();

        let a = graph.get_node("a").unwrap().unwrap();
        let b = graph.get_node("b").unwrap().unwrap();
        let c = graph.get_node("c").unwrap().unwrap();

        assert!((a.centrality - 0.5).abs() < f64::EPSILON);
        assert!((b.centrality - 1.0).abs() < f64::EPSILON);
        assert!((c.centrality - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn compute_centrality_star() {
        // Graph: a -> b, a -> c, a -> d (star topology)
        // Node a: out=3, in=0 => centrality = 3/3 = 1.0
        // Node b: out=0, in=1 => centrality = 1/3
        // Node c: out=0, in=1 => centrality = 1/3
        // Node d: out=0, in=1 => centrality = 1/3
        let mut graph = GraphEngine::new();
        graph.add_node(file_node("a", "a.rs")).unwrap();
        graph.add_node(file_node("b", "b.rs")).unwrap();
        graph.add_node(file_node("c", "c.rs")).unwrap();
        graph.add_node(file_node("d", "d.rs")).unwrap();
        graph.add_edge(test_edge("a", "b")).unwrap();
        graph.add_edge(test_edge("a", "c")).unwrap();
        graph.add_edge(test_edge("a", "d")).unwrap();

        graph.compute_centrality();

        let a = graph.get_node("a").unwrap().unwrap();
        let b = graph.get_node("b").unwrap().unwrap();

        assert!((a.centrality - 1.0).abs() < f64::EPSILON);
        assert!((b.centrality - 1.0 / 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn compute_centrality_single_node() {
        let mut graph = GraphEngine::new();
        graph.add_node(file_node("a", "a.rs")).unwrap();

        graph.compute_centrality();

        let a = graph.get_node("a").unwrap().unwrap();
        assert!((a.centrality - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn compute_centrality_no_edges() {
        let mut graph = GraphEngine::new();
        graph.add_node(file_node("a", "a.rs")).unwrap();
        graph.add_node(file_node("b", "b.rs")).unwrap();

        graph.compute_centrality();

        let a = graph.get_node("a").unwrap().unwrap();
        let b = graph.get_node("b").unwrap().unwrap();
        assert!((a.centrality - 0.0).abs() < f64::EPSILON);
        assert!((b.centrality - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn get_all_nodes_returns_all() {
        let mut graph = GraphEngine::new();
        graph.add_node(file_node("a", "a.rs")).unwrap();
        graph.add_node(file_node("b", "b.rs")).unwrap();
        graph.add_node(file_node("c", "c.rs")).unwrap();

        let mut all = graph.get_all_nodes();
        all.sort_by(|x, y| x.id.cmp(&y.id));
        assert_eq!(all.len(), 3);
        assert_eq!(all[0].id, "a");
        assert_eq!(all[1].id, "b");
        assert_eq!(all[2].id, "c");
    }

    // ── Centrality Caching Tests ────────────────────────────────────────────

    #[test]
    fn recompute_centrality_caches_pagerank() {
        let mut graph = GraphEngine::new();
        graph.add_node(file_node("a", "a.rs")).unwrap();
        graph.add_node(file_node("b", "b.rs")).unwrap();
        graph.add_node(file_node("c", "c.rs")).unwrap();
        graph.add_edge(test_edge("a", "b")).unwrap();
        graph.add_edge(test_edge("b", "c")).unwrap();

        // Before recompute, cached values should be 0.0
        assert_eq!(graph.get_pagerank("a"), 0.0);
        assert_eq!(graph.get_betweenness("a"), 0.0);

        graph.recompute_centrality();

        // After recompute, cached PageRank values should be non-zero
        assert!(graph.get_pagerank("a") > 0.0);
        assert!(graph.get_pagerank("b") > 0.0);
        assert!(graph.get_pagerank("c") > 0.0);

        // c should have highest PageRank (sink node in a -> b -> c)
        assert!(
            graph.get_pagerank("c") > graph.get_pagerank("a"),
            "c ({}) should have higher PageRank than a ({})",
            graph.get_pagerank("c"),
            graph.get_pagerank("a")
        );

        // b should have highest betweenness (middle of chain)
        assert!(
            graph.get_betweenness("b") > graph.get_betweenness("a"),
            "b ({}) should have higher betweenness than a ({})",
            graph.get_betweenness("b"),
            graph.get_betweenness("a")
        );
    }

    #[test]
    fn get_pagerank_returns_zero_for_unknown_node() {
        let graph = GraphEngine::new();
        assert_eq!(graph.get_pagerank("nonexistent"), 0.0);
        assert_eq!(graph.get_betweenness("nonexistent"), 0.0);
    }

    #[test]
    fn max_degree_returns_correct_value() {
        let mut graph = GraphEngine::new();
        graph.add_node(file_node("a", "a.rs")).unwrap();
        graph.add_node(file_node("b", "b.rs")).unwrap();
        graph.add_node(file_node("c", "c.rs")).unwrap();
        graph.add_node(file_node("d", "d.rs")).unwrap();
        // a -> b, a -> c, a -> d (star: a has degree 3)
        graph.add_edge(test_edge("a", "b")).unwrap();
        graph.add_edge(test_edge("a", "c")).unwrap();
        graph.add_edge(test_edge("a", "d")).unwrap();

        assert!((graph.max_degree() - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn enhanced_graph_strength_differs_from_simple_edge_count() {
        // Build a graph where PageRank/betweenness differ from simple edge count.
        // a -> b -> c -> d (chain)
        // b is in the middle with betweenness, c gets more PageRank flow
        let mut graph = GraphEngine::new();
        for id in &["a", "b", "c", "d"] {
            graph.add_node(file_node(id, &format!("{id}.rs"))).unwrap();
        }
        graph.add_edge(test_edge("a", "b")).unwrap();
        graph.add_edge(test_edge("b", "c")).unwrap();
        graph.add_edge(test_edge("c", "d")).unwrap();
        graph.recompute_centrality();

        // Nodes b and c both have 2 edges (in+out), but different centrality profiles.
        // b has higher betweenness (on path a->c and a->d).
        // Simple edge count would give them equal scores.
        let edges_b = graph.get_edges("b").unwrap().len();
        let edges_c = graph.get_edges("c").unwrap().len();
        assert_eq!(edges_b, edges_c, "b and c should have same edge count");

        // But their centrality profiles should differ
        let pr_b = graph.get_pagerank("b");
        let pr_c = graph.get_pagerank("c");
        let bt_b = graph.get_betweenness("b");
        let bt_c = graph.get_betweenness("c");

        // At least one centrality metric should differ between b and c
        let centrality_differs = (pr_b - pr_c).abs() > 1e-6 || (bt_b - bt_c).abs() > 1e-6;
        assert!(
            centrality_differs,
            "Centrality should differ: b(pr={pr_b}, bt={bt_b}) vs c(pr={pr_c}, bt={bt_c})"
        );
    }
}
