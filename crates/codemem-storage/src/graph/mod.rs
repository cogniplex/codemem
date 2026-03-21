//! codemem-graph: Graph engine with petgraph algorithms and SQLite persistence.
//!
//! Provides BFS, DFS, shortest path, and connected components over
//! a knowledge graph with 13 node kinds and 24 relationship types.

mod algorithms;
mod traversal;

use codemem_core::{CodememError, Edge, GraphBackend, GraphNode, NodeKind, RawGraphMetrics};
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::Direction;
use std::collections::{HashMap, HashSet, VecDeque};

/// In-memory graph engine backed by petgraph, synced to SQLite via codemem-storage.
///
/// # Design: intentional in-memory architecture
///
/// All graph data (nodes, edges, adjacency) is held entirely in memory using
/// `HashMap`-based structures. This is deliberate: graph traversals, centrality
/// algorithms, and multi-hop expansions benefit enormously from avoiding disk
/// I/O on every edge follow. The trade-off is higher memory usage, which is
/// acceptable for the typical code-graph sizes this engine targets.
///
/// # Memory characteristics
///
/// - **`nodes`**: `HashMap<String, GraphNode>` — ~200 bytes per node (ID, kind,
///   label, namespace, metadata, centrality).
/// - **`edges`**: `HashMap<String, Edge>` — ~150 bytes per edge (ID, src, dst,
///   relationship, weight, properties, timestamps).
/// - **`edge_adj`**: `HashMap<String, Vec<String>>` — adjacency index mapping
///   node IDs to incident edge IDs for O(degree) lookups.
/// - **`id_to_index`**: maps string IDs to petgraph `NodeIndex` values.
/// - **`cached_pagerank` / `cached_betweenness`**: centrality caches populated
///   by [`recompute_centrality()`](Self::recompute_centrality).
///
/// Use [`CodememEngine::graph_memory_estimate()`](../../codemem_engine) for a
/// byte-level sizing estimate based on current node and edge counts.
///
/// # Thread safety
///
/// `GraphEngine` is **not** `Sync` — it stores mutable graph state without
/// internal locking. Callers in codemem-engine wrap it in `Mutex<GraphEngine>`
/// (via `lock_graph()`) to ensure exclusive access. All public `&mut self`
/// methods (e.g., `add_node`, `recompute_centrality`) require the caller to
/// hold the lock.
pub struct GraphEngine {
    pub(crate) graph: DiGraph<String, f64>,
    /// Map from string node IDs to petgraph `NodeIndex`.
    pub(crate) id_to_index: HashMap<String, NodeIndex>,
    /// Node data by ID.
    pub(crate) nodes: HashMap<String, GraphNode>,
    /// Edge data by ID.
    pub(crate) edges: HashMap<String, Edge>,
    /// Edge adjacency index: maps node IDs to the IDs of edges incident on that node.
    ///
    /// Maintained alongside `edges` to allow O(degree) edge lookups instead of O(E).
    /// The string duplication (~40 bytes/edge for source+target node ID copies) is
    /// intentional: using `Arc<str>` for shared ownership would be too invasive for
    /// the marginal memory savings, and the adjacency index is critical for
    /// performance in `get_edges()`, `bfs_filtered()`, and `raw_graph_metrics_for_memory()`.
    pub(crate) edge_adj: HashMap<String, Vec<String>>,
    /// Cached PageRank scores (populated by [`recompute_centrality()`](Self::recompute_centrality)).
    pub(crate) cached_pagerank: HashMap<String, f64>,
    /// Cached betweenness centrality scores (populated by [`recompute_centrality()`](Self::recompute_centrality)).
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
            edge_adj: HashMap::new(),
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

    /// Return a reference to a node without cloning. Returns `None` if not found.
    pub fn get_node_ref(&self, id: &str) -> Option<&GraphNode> {
        self.nodes.get(id)
    }

    /// Return references to edges incident on a node without cloning.
    ///
    /// This is the zero-copy variant of [`GraphBackend::get_edges()`] — same
    /// lookup logic via `edge_adj`, but returns `&Edge` instead of owned `Edge`.
    pub fn get_edges_ref(&self, node_id: &str) -> Vec<&Edge> {
        self.edge_adj
            .get(node_id)
            .map(|edge_ids| {
                edge_ids
                    .iter()
                    .filter_map(|eid| self.edges.get(eid))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Recompute and cache PageRank and betweenness centrality scores.
    ///
    /// This should be called after loading the graph (e.g., on server start)
    /// and periodically when the graph changes significantly.
    pub fn recompute_centrality(&mut self) {
        self.recompute_centrality_with_options(true);
    }

    /// Recompute PageRank for a single namespace and update the cache.
    ///
    /// Only scores for nodes in `namespace` are written; scores for nodes in
    /// other namespaces are left unchanged. This prevents cross-project
    /// pollution when the shared database holds multiple indexed projects.
    ///
    /// Stale scores for deleted nodes in this namespace are evicted.
    pub fn recompute_centrality_for_namespace(&mut self, namespace: &str) {
        // Get set of node IDs currently in this namespace
        let current_node_ids: HashSet<String> = self.nodes
            .iter()
            .filter(|(_, n)| n.namespace.as_deref() == Some(namespace))
            .map(|(id, _)| id.clone())
            .collect();

        // Evict stale scores: remove cached entries for deleted nodes in this namespace
        self.cached_pagerank.retain(|id, _| {
            // Keep score if node doesn't exist OR node belongs to different namespace
            self.nodes.get(id)
                .map(|n| n.namespace.as_deref() != Some(namespace))
                .unwrap_or(true) // Keep if node doesn't exist (safe fallback)
        });

        // Compute and insert new scores
        let scores = self.pagerank_for_namespace(
            namespace,
            codemem_core::PAGERANK_DAMPING_DEFAULT,
            codemem_core::PAGERANK_ITERATIONS_DEFAULT,
            codemem_core::PAGERANK_TOLERANCE_DEFAULT,
        );
        for (id, score) in scores {
            self.cached_pagerank.insert(id, score);
        }

        tracing::debug!(
            namespace = %namespace,
            scores_updated = self.cached_pagerank.iter().filter(|(id, _)| current_node_ids.contains(*id)).count(),
            "PageRank recomputed for namespace"
        );
    }

    /// Recompute centrality caches with control over which algorithms run.
    ///
    /// PageRank is always computed. Betweenness centrality is only computed
    /// when `include_betweenness` is true, since it is O(V * E) and can be
    /// expensive on large graphs.
    pub fn recompute_centrality_with_options(&mut self, include_betweenness: bool) {
        self.cached_pagerank = self.pagerank(
            codemem_core::PAGERANK_DAMPING_DEFAULT,
            codemem_core::PAGERANK_ITERATIONS_DEFAULT,
            codemem_core::PAGERANK_TOLERANCE_DEFAULT,
        );
        if include_betweenness {
            self.cached_betweenness = self.betweenness_centrality();
        } else {
            // L1: Clear stale betweenness cache so ensure_betweenness_computed()
            // knows it needs to recompute when lazily invoked.
            self.cached_betweenness.clear();
        }
    }

    /// Lazily ensure betweenness centrality has been computed.
    ///
    /// If `cached_betweenness` is empty (e.g., after `recompute_centrality_with_options(false)`),
    /// this method computes and caches betweenness centrality on demand. If the
    /// cache is already populated, this is a no-op.
    pub fn ensure_betweenness_computed(&mut self) {
        if self.cached_betweenness.is_empty() && self.graph.node_count() > 0 {
            self.cached_betweenness = self.betweenness_centrality();
        }
    }

    /// Get the cached PageRank score for a node. Returns 0.0 if not found.
    pub fn get_pagerank(&self, node_id: &str) -> f64 {
        self.cached_pagerank.get(node_id).copied().unwrap_or(0.0)
    }

    /// Get the cached betweenness centrality score for a node. Returns 0.0 if not found.
    pub fn get_betweenness(&self, node_id: &str) -> f64 {
        self.cached_betweenness.get(node_id).copied().unwrap_or(0.0)
    }

    /// Collect raw graph metrics for a memory node from both code-graph and
    /// memory-graph neighbors.
    ///
    /// Code neighbors (`sym:`, `file:`, `chunk:`, `pkg:`) contribute centrality
    /// data (PageRank, betweenness). Memory neighbors (UUID-based) contribute
    /// connectivity data (count + edge weights). A function with many linked
    /// memories is more important; a memory with many linked memories has richer
    /// context.
    ///
    /// Returns `None` if the memory node is not in the graph or has no neighbors.
    pub fn raw_graph_metrics_for_memory(&self, memory_id: &str) -> Option<RawGraphMetrics> {
        let idx = *self.id_to_index.get(memory_id)?;

        let mut max_pagerank = 0.0_f64;
        let mut max_betweenness = 0.0_f64;
        let mut code_neighbor_count = 0_usize;
        let mut total_edge_weight = 0.0_f64;
        let mut memory_neighbor_count = 0_usize;
        let mut memory_edge_weight = 0.0_f64;
        let now = chrono::Utc::now();

        // Iterate both outgoing and incoming neighbors
        for direction in &[Direction::Outgoing, Direction::Incoming] {
            for neighbor_idx in self.graph.neighbors_directed(idx, *direction) {
                if let Some(neighbor_id) = self.graph.node_weight(neighbor_idx) {
                    // Skip expired neighbors (valid_to in the past)
                    let neighbor_node = self.nodes.get(neighbor_id.as_str());
                    if let Some(n) = neighbor_node {
                        if n.valid_to.is_some_and(|vt| vt <= now) {
                            continue;
                        }
                    }

                    let is_code_node = neighbor_node
                        .map(|n| n.kind != NodeKind::Memory)
                        .unwrap_or(false);

                    // Collect edge weight from our edge adjacency index
                    let mut edge_w = 0.0_f64;
                    if let Some(edge_ids) = self.edge_adj.get(memory_id) {
                        for eid in edge_ids {
                            if let Some(edge) = self.edges.get(eid) {
                                if (edge.src == memory_id && edge.dst == *neighbor_id)
                                    || (edge.dst == memory_id && edge.src == *neighbor_id)
                                {
                                    edge_w = edge.weight;
                                    break;
                                }
                            }
                        }
                    }

                    if is_code_node {
                        code_neighbor_count += 1;
                        total_edge_weight += edge_w;

                        let pr = self
                            .cached_pagerank
                            .get(neighbor_id)
                            .copied()
                            .unwrap_or(0.0);
                        let bt = self
                            .cached_betweenness
                            .get(neighbor_id)
                            .copied()
                            .unwrap_or(0.0);
                        max_pagerank = max_pagerank.max(pr);
                        max_betweenness = max_betweenness.max(bt);
                    } else {
                        memory_neighbor_count += 1;
                        memory_edge_weight += edge_w;
                    }
                }
            }
        }

        if code_neighbor_count == 0 && memory_neighbor_count == 0 {
            return None;
        }

        Some(RawGraphMetrics {
            max_pagerank,
            max_betweenness,
            code_neighbor_count,
            total_edge_weight,
            memory_neighbor_count,
            memory_edge_weight,
        })
    }

    /// Get the maximum degree (in + out) across all nodes in the graph.
    /// Returns 1.0 if the graph has fewer than 2 nodes to avoid division by zero.
    #[cfg(test)]
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
#[path = "../tests/graph_tests.rs"]
mod tests;
