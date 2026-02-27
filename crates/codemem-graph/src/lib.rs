//! codemem-graph: Graph engine with petgraph algorithms and SQLite persistence.
//!
//! Provides BFS, DFS, shortest path, and connected components over
//! a knowledge graph with 6 node types and 15 relationship types.

#[cfg(test)]
use codemem_core::NodeKind;
use codemem_core::{CodememError, Edge, GraphBackend, GraphNode, GraphStats};
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::Bfs;
use petgraph::Direction;
use std::collections::{HashMap, HashSet, VecDeque};

/// In-memory graph backed by petgraph, synced to SQLite via codemem-storage.
pub struct GraphEngine {
    graph: DiGraph<String, f64>,
    /// Map from string node IDs to petgraph NodeIndex.
    id_to_index: HashMap<String, NodeIndex>,
    /// Node data by ID.
    nodes: HashMap<String, GraphNode>,
    /// Edge data by ID.
    edges: HashMap<String, Edge>,
    /// Cached PageRank scores (populated by `recompute_centrality()`).
    cached_pagerank: HashMap<String, f64>,
    /// Cached betweenness centrality scores (populated by `recompute_centrality()`).
    cached_betweenness: HashMap<String, f64>,
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
    pub fn from_storage(storage: &codemem_storage::Storage) -> Result<Self, CodememError> {
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

impl Default for GraphEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphBackend for GraphEngine {
    fn add_node(&mut self, node: GraphNode) -> Result<(), CodememError> {
        let id = node.id.clone();

        if !self.id_to_index.contains_key(&id) {
            let idx = self.graph.add_node(id.clone());
            self.id_to_index.insert(id.clone(), idx);
        }

        self.nodes.insert(id, node);
        Ok(())
    }

    fn get_node(&self, id: &str) -> Result<Option<GraphNode>, CodememError> {
        Ok(self.nodes.get(id).cloned())
    }

    fn remove_node(&mut self, id: &str) -> Result<bool, CodememError> {
        if let Some(idx) = self.id_to_index.remove(id) {
            self.graph.remove_node(idx);
            self.nodes.remove(id);

            // Remove associated edges
            let edge_ids: Vec<String> = self
                .edges
                .iter()
                .filter(|(_, e)| e.src == id || e.dst == id)
                .map(|(eid, _)| eid.clone())
                .collect();
            for eid in edge_ids {
                self.edges.remove(&eid);
            }

            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn add_edge(&mut self, edge: Edge) -> Result<(), CodememError> {
        let src_idx = self
            .id_to_index
            .get(&edge.src)
            .ok_or_else(|| CodememError::NotFound(format!("Source node {}", edge.src)))?;
        let dst_idx = self
            .id_to_index
            .get(&edge.dst)
            .ok_or_else(|| CodememError::NotFound(format!("Destination node {}", edge.dst)))?;

        self.graph.add_edge(*src_idx, *dst_idx, edge.weight);
        self.edges.insert(edge.id.clone(), edge);
        Ok(())
    }

    fn get_edges(&self, node_id: &str) -> Result<Vec<Edge>, CodememError> {
        let edges: Vec<Edge> = self
            .edges
            .values()
            .filter(|e| e.src == node_id || e.dst == node_id)
            .cloned()
            .collect();
        Ok(edges)
    }

    fn remove_edge(&mut self, id: &str) -> Result<bool, CodememError> {
        if let Some(edge) = self.edges.remove(id) {
            // Also remove from petgraph
            if let (Some(&src_idx), Some(&dst_idx)) = (
                self.id_to_index.get(&edge.src),
                self.id_to_index.get(&edge.dst),
            ) {
                if let Some(edge_idx) = self.graph.find_edge(src_idx, dst_idx) {
                    self.graph.remove_edge(edge_idx);
                }
            }
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn bfs(&self, start_id: &str, max_depth: usize) -> Result<Vec<GraphNode>, CodememError> {
        let start_idx = self
            .id_to_index
            .get(start_id)
            .ok_or_else(|| CodememError::NotFound(format!("Node {start_id}")))?;

        let mut visited = HashSet::new();
        let mut result = Vec::new();
        let mut bfs = Bfs::new(&self.graph, *start_idx);
        let mut depth_map: HashMap<NodeIndex, usize> = HashMap::new();
        depth_map.insert(*start_idx, 0);

        while let Some(node_idx) = bfs.next(&self.graph) {
            let depth = depth_map.get(&node_idx).copied().unwrap_or(0);
            if depth > max_depth {
                continue;
            }

            if visited.insert(node_idx) {
                if let Some(node_id) = self.graph.node_weight(node_idx) {
                    if let Some(node) = self.nodes.get(node_id) {
                        result.push(node.clone());
                    }
                }
            }

            // Set depth for neighbors
            for neighbor in self.graph.neighbors(node_idx) {
                depth_map.entry(neighbor).or_insert(depth + 1);
            }
        }

        Ok(result)
    }

    fn dfs(&self, start_id: &str, max_depth: usize) -> Result<Vec<GraphNode>, CodememError> {
        let start_idx = self
            .id_to_index
            .get(start_id)
            .ok_or_else(|| CodememError::NotFound(format!("Node {start_id}")))?;

        let mut visited = HashSet::new();
        let mut result = Vec::new();
        let mut stack: Vec<(NodeIndex, usize)> = vec![(*start_idx, 0)];

        while let Some((node_idx, depth)) = stack.pop() {
            if depth > max_depth || !visited.insert(node_idx) {
                continue;
            }

            if let Some(node_id) = self.graph.node_weight(node_idx) {
                if let Some(node) = self.nodes.get(node_id) {
                    result.push(node.clone());
                }
            }

            for neighbor in self.graph.neighbors(node_idx) {
                if !visited.contains(&neighbor) {
                    stack.push((neighbor, depth + 1));
                }
            }
        }

        Ok(result)
    }

    fn shortest_path(&self, from: &str, to: &str) -> Result<Vec<String>, CodememError> {
        let from_idx = self
            .id_to_index
            .get(from)
            .ok_or_else(|| CodememError::NotFound(format!("Node {from}")))?;
        let to_idx = self
            .id_to_index
            .get(to)
            .ok_or_else(|| CodememError::NotFound(format!("Node {to}")))?;

        // BFS shortest path (unweighted)
        use petgraph::algo::astar;
        let path = astar(
            &self.graph,
            *from_idx,
            |finish| finish == *to_idx,
            |_| 1.0f64,
            |_| 0.0f64,
        );

        match path {
            Some((_cost, nodes)) => {
                let ids: Vec<String> = nodes
                    .iter()
                    .filter_map(|idx| self.graph.node_weight(*idx).cloned())
                    .collect();
                Ok(ids)
            }
            None => Ok(vec![]),
        }
    }

    fn stats(&self) -> GraphStats {
        let mut node_kind_counts = HashMap::new();
        for node in self.nodes.values() {
            *node_kind_counts.entry(node.kind.to_string()).or_insert(0) += 1;
        }

        let mut relationship_type_counts = HashMap::new();
        for edge in self.edges.values() {
            *relationship_type_counts
                .entry(edge.relationship.to_string())
                .or_insert(0) += 1;
        }

        GraphStats {
            node_count: self.nodes.len(),
            edge_count: self.edges.len(),
            node_kind_counts,
            relationship_type_counts,
        }
    }
}

// ── Advanced Graph Algorithms ───────────────────────────────────────────────

impl GraphEngine {
    /// Compute PageRank scores for all nodes using power iteration.
    ///
    /// - `damping`: probability of following an edge (default 0.85)
    /// - `iterations`: max number of power iterations (default 100)
    /// - `tolerance`: convergence threshold (default 1e-6)
    ///
    /// Returns a map from node ID to PageRank score.
    pub fn pagerank(
        &self,
        damping: f64,
        iterations: usize,
        tolerance: f64,
    ) -> HashMap<String, f64> {
        let n = self.graph.node_count();
        if n == 0 {
            return HashMap::new();
        }

        let nf = n as f64;
        let initial = 1.0 / nf;

        // Collect all node indices in a stable order
        let indices: Vec<NodeIndex> = self.graph.node_indices().collect();
        let idx_pos: HashMap<NodeIndex, usize> = indices
            .iter()
            .enumerate()
            .map(|(i, &idx)| (idx, i))
            .collect();

        let mut scores = vec![initial; n];

        // Precompute out-degrees
        let out_degree: Vec<usize> = indices
            .iter()
            .map(|&idx| {
                self.graph
                    .neighbors_directed(idx, Direction::Outgoing)
                    .count()
            })
            .collect();

        for _ in 0..iterations {
            let mut new_scores = vec![(1.0 - damping) / nf; n];

            // Distribute rank from each node to its out-neighbors
            for (i, &idx) in indices.iter().enumerate() {
                let deg = out_degree[i];
                if deg == 0 {
                    // Dangling node: distribute evenly to all nodes
                    let share = damping * scores[i] / nf;
                    for ns in new_scores.iter_mut() {
                        *ns += share;
                    }
                } else {
                    let share = damping * scores[i] / deg as f64;
                    for neighbor in self.graph.neighbors_directed(idx, Direction::Outgoing) {
                        if let Some(&pos) = idx_pos.get(&neighbor) {
                            new_scores[pos] += share;
                        }
                    }
                }
            }

            // Check convergence
            let diff: f64 = scores
                .iter()
                .zip(new_scores.iter())
                .map(|(a, b)| (a - b).abs())
                .sum();

            scores = new_scores;

            if diff < tolerance {
                break;
            }
        }

        // Map back to node IDs
        indices
            .iter()
            .enumerate()
            .filter_map(|(i, &idx)| {
                self.graph
                    .node_weight(idx)
                    .map(|id| (id.clone(), scores[i]))
            })
            .collect()
    }

    /// Compute Personalized PageRank with custom teleport weights.
    ///
    /// `seed_weights` maps node IDs to teleport probabilities (will be normalized).
    /// Nodes not in seed_weights get zero teleport probability.
    ///
    /// Used for blast-radius analysis and HippoRAG-2-style retrieval.
    pub fn personalized_pagerank(
        &self,
        seed_weights: &HashMap<String, f64>,
        damping: f64,
        iterations: usize,
        tolerance: f64,
    ) -> HashMap<String, f64> {
        let n = self.graph.node_count();
        if n == 0 {
            return HashMap::new();
        }

        let nf = n as f64;

        let indices: Vec<NodeIndex> = self.graph.node_indices().collect();
        let idx_pos: HashMap<NodeIndex, usize> = indices
            .iter()
            .enumerate()
            .map(|(i, &idx)| (idx, i))
            .collect();

        // Build and normalize the teleport vector
        let mut teleport = vec![0.0f64; n];
        let mut teleport_sum = 0.0;
        for (i, &idx) in indices.iter().enumerate() {
            if let Some(node_id) = self.graph.node_weight(idx) {
                if let Some(&w) = seed_weights.get(node_id) {
                    teleport[i] = w;
                    teleport_sum += w;
                }
            }
        }
        // Normalize; if no seeds provided, fall back to uniform
        if teleport_sum > 0.0 {
            for t in teleport.iter_mut() {
                *t /= teleport_sum;
            }
        } else {
            for t in teleport.iter_mut() {
                *t = 1.0 / nf;
            }
        }

        let initial = 1.0 / nf;
        let mut scores = vec![initial; n];

        let out_degree: Vec<usize> = indices
            .iter()
            .map(|&idx| {
                self.graph
                    .neighbors_directed(idx, Direction::Outgoing)
                    .count()
            })
            .collect();

        for _ in 0..iterations {
            let mut new_scores: Vec<f64> = teleport.iter().map(|&t| (1.0 - damping) * t).collect();

            for (i, &idx) in indices.iter().enumerate() {
                let deg = out_degree[i];
                if deg == 0 {
                    // Dangling node: distribute to teleport targets
                    let share = damping * scores[i];
                    for (j, t) in teleport.iter().enumerate() {
                        new_scores[j] += share * t;
                    }
                } else {
                    let share = damping * scores[i] / deg as f64;
                    for neighbor in self.graph.neighbors_directed(idx, Direction::Outgoing) {
                        if let Some(&pos) = idx_pos.get(&neighbor) {
                            new_scores[pos] += share;
                        }
                    }
                }
            }

            let diff: f64 = scores
                .iter()
                .zip(new_scores.iter())
                .map(|(a, b)| (a - b).abs())
                .sum();

            scores = new_scores;

            if diff < tolerance {
                break;
            }
        }

        indices
            .iter()
            .enumerate()
            .filter_map(|(i, &idx)| {
                self.graph
                    .node_weight(idx)
                    .map(|id| (id.clone(), scores[i]))
            })
            .collect()
    }

    /// Detect communities using the Louvain algorithm.
    ///
    /// Treats the directed graph as undirected for modularity computation.
    /// `resolution` controls community granularity (1.0 = standard modularity).
    /// Returns groups of node IDs, one group per community.
    pub fn louvain_communities(&self, resolution: f64) -> Vec<Vec<String>> {
        let n = self.graph.node_count();
        if n == 0 {
            return Vec::new();
        }

        let indices: Vec<NodeIndex> = self.graph.node_indices().collect();
        let idx_pos: HashMap<NodeIndex, usize> = indices
            .iter()
            .enumerate()
            .map(|(i, &idx)| (idx, i))
            .collect();

        // Build undirected adjacency with weights.
        // adj[i] contains (j, weight) for each undirected neighbor.
        let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        let mut total_weight = 0.0;

        for edge_ref in self.graph.edge_indices() {
            if let Some((src_idx, dst_idx)) = self.graph.edge_endpoints(edge_ref) {
                let w = self.graph[edge_ref];
                if let (Some(&si), Some(&di)) = (idx_pos.get(&src_idx), idx_pos.get(&dst_idx)) {
                    adj[si].push((di, w));
                    adj[di].push((si, w));
                    total_weight += w; // Each undirected edge contributes w (counted once)
                }
            }
        }

        if total_weight == 0.0 {
            // No edges: each node is its own community
            return indices
                .iter()
                .filter_map(|&idx| self.graph.node_weight(idx).map(|id| vec![id.clone()]))
                .collect();
        }

        // m = total edge weight (for undirected: sum of all edge weights)
        let m = total_weight;
        let m2 = 2.0 * m;

        // Weighted degree of each node (sum of incident edge weights, undirected)
        let k: Vec<f64> = (0..n)
            .map(|i| adj[i].iter().map(|&(_, w)| w).sum())
            .collect();

        // Initial assignment: each node in its own community
        let mut community: Vec<usize> = (0..n).collect();

        // Iteratively move nodes to improve modularity
        let mut improved = true;
        let max_passes = 100;
        let mut pass = 0;

        while improved && pass < max_passes {
            improved = false;
            pass += 1;

            for i in 0..n {
                let current_comm = community[i];

                // Compute weights to each neighboring community
                let mut comm_weights: HashMap<usize, f64> = HashMap::new();
                for &(j, w) in &adj[i] {
                    *comm_weights.entry(community[j]).or_insert(0.0) += w;
                }

                // Sum of degrees in each community (excluding node i for its own community)
                let mut comm_degree_sum: HashMap<usize, f64> = HashMap::new();
                for j in 0..n {
                    *comm_degree_sum.entry(community[j]).or_insert(0.0) += k[j];
                }

                let ki = k[i];

                // Modularity gain for removing i from its current community
                let w_in_current = comm_weights.get(&current_comm).copied().unwrap_or(0.0);
                let sigma_current = comm_degree_sum.get(&current_comm).copied().unwrap_or(0.0);
                let remove_cost = w_in_current - resolution * ki * (sigma_current - ki) / m2;

                // Find best community to move to
                let mut best_comm = current_comm;
                let mut best_gain = 0.0;

                for (&comm, &w_in_comm) in &comm_weights {
                    if comm == current_comm {
                        continue;
                    }
                    let sigma_comm = comm_degree_sum.get(&comm).copied().unwrap_or(0.0);
                    let gain = w_in_comm - resolution * ki * sigma_comm / m2 - remove_cost;
                    if gain > best_gain {
                        best_gain = gain;
                        best_comm = comm;
                    }
                }

                if best_comm != current_comm {
                    community[i] = best_comm;
                    improved = true;
                }
            }
        }

        // Group nodes by community
        let mut groups: HashMap<usize, Vec<String>> = HashMap::new();
        for (i, &idx) in indices.iter().enumerate() {
            if let Some(node_id) = self.graph.node_weight(idx) {
                groups
                    .entry(community[i])
                    .or_default()
                    .push(node_id.clone());
            }
        }

        let mut result: Vec<Vec<String>> = groups.into_values().collect();
        for group in result.iter_mut() {
            group.sort();
        }
        result.sort();
        result
    }

    /// Compute betweenness centrality for all nodes using Brandes' algorithm.
    ///
    /// For graphs with more than 1000 nodes, samples sqrt(n) source nodes
    /// for approximate computation.
    ///
    /// Returns a map from node ID to betweenness centrality score (normalized by
    /// 1/((n-1)(n-2)) for directed graphs).
    pub fn betweenness_centrality(&self) -> HashMap<String, f64> {
        let n = self.graph.node_count();
        if n <= 2 {
            return self
                .graph
                .node_indices()
                .filter_map(|idx| self.graph.node_weight(idx).map(|id| (id.clone(), 0.0)))
                .collect();
        }

        let indices: Vec<NodeIndex> = self.graph.node_indices().collect();
        let idx_pos: HashMap<NodeIndex, usize> = indices
            .iter()
            .enumerate()
            .map(|(i, &idx)| (idx, i))
            .collect();

        let mut centrality = vec![0.0f64; n];

        // Determine source nodes (sample for large graphs)
        let sources: Vec<usize> = if n > 1000 {
            let sample_size = (n as f64).sqrt() as usize;
            // Deterministic sampling: evenly spaced
            let step = n / sample_size;
            (0..sample_size).map(|i| i * step).collect()
        } else {
            (0..n).collect()
        };

        let scale = if n > 1000 {
            n as f64 / sources.len() as f64
        } else {
            1.0
        };

        for &s in &sources {
            // Brandes' algorithm from source s
            let mut stack: Vec<usize> = Vec::new();
            let mut predecessors: Vec<Vec<usize>> = vec![Vec::new(); n];
            let mut sigma = vec![0.0f64; n]; // number of shortest paths
            sigma[s] = 1.0;
            let mut dist: Vec<i64> = vec![-1; n];
            dist[s] = 0;

            let mut queue: VecDeque<usize> = VecDeque::new();
            queue.push_back(s);

            while let Some(v) = queue.pop_front() {
                stack.push(v);
                let v_idx = indices[v];
                for neighbor in self.graph.neighbors_directed(v_idx, Direction::Outgoing) {
                    if let Some(&w) = idx_pos.get(&neighbor) {
                        if dist[w] < 0 {
                            dist[w] = dist[v] + 1;
                            queue.push_back(w);
                        }
                        if dist[w] == dist[v] + 1 {
                            sigma[w] += sigma[v];
                            predecessors[w].push(v);
                        }
                    }
                }
            }

            let mut delta = vec![0.0f64; n];
            while let Some(w) = stack.pop() {
                for &v in &predecessors[w] {
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w]);
                }
                if w != s {
                    centrality[w] += delta[w];
                }
            }
        }

        // Apply sampling scale and normalize
        let norm = ((n - 1) * (n - 2)) as f64;
        indices
            .iter()
            .enumerate()
            .filter_map(|(i, &idx)| {
                self.graph
                    .node_weight(idx)
                    .map(|id| (id.clone(), centrality[i] * scale / norm))
            })
            .collect()
    }

    /// Find all strongly connected components using Tarjan's algorithm.
    ///
    /// Returns groups of node IDs. Each group is a strongly connected component
    /// where every node can reach every other node via directed edges.
    pub fn strongly_connected_components(&self) -> Vec<Vec<String>> {
        let sccs = petgraph::algo::tarjan_scc(&self.graph);

        let mut result: Vec<Vec<String>> = sccs
            .into_iter()
            .map(|component| {
                let mut ids: Vec<String> = component
                    .into_iter()
                    .filter_map(|idx| self.graph.node_weight(idx).cloned())
                    .collect();
                ids.sort();
                ids
            })
            .collect();

        result.sort();
        result
    }

    /// Compute topological layers using Kahn's algorithm.
    ///
    /// Returns layers where all nodes in layer i have no dependencies on nodes
    /// in layer i or later. For cyclic graphs, SCCs are condensed into single
    /// super-nodes first, then the resulting DAG is topologically sorted.
    ///
    /// Each inner Vec contains the node IDs at that layer.
    pub fn topological_layers(&self) -> Vec<Vec<String>> {
        let n = self.graph.node_count();
        if n == 0 {
            return Vec::new();
        }

        let indices: Vec<NodeIndex> = self.graph.node_indices().collect();
        let idx_pos: HashMap<NodeIndex, usize> = indices
            .iter()
            .enumerate()
            .map(|(i, &idx)| (idx, i))
            .collect();

        // Step 1: Find SCCs
        let sccs = petgraph::algo::tarjan_scc(&self.graph);

        // Map each node position to its SCC index
        let mut node_to_scc = vec![0usize; n];
        for (scc_idx, scc) in sccs.iter().enumerate() {
            for &node_idx in scc {
                if let Some(&pos) = idx_pos.get(&node_idx) {
                    node_to_scc[pos] = scc_idx;
                }
            }
        }

        let num_sccs = sccs.len();

        // Step 2: Build condensed DAG (SCC graph)
        let mut condensed_adj: Vec<HashSet<usize>> = vec![HashSet::new(); num_sccs];
        let mut condensed_in_degree = vec![0usize; num_sccs];

        for &idx in &indices {
            if let Some(&src_pos) = idx_pos.get(&idx) {
                let src_scc = node_to_scc[src_pos];
                for neighbor in self.graph.neighbors_directed(idx, Direction::Outgoing) {
                    if let Some(&dst_pos) = idx_pos.get(&neighbor) {
                        let dst_scc = node_to_scc[dst_pos];
                        if src_scc != dst_scc && condensed_adj[src_scc].insert(dst_scc) {
                            condensed_in_degree[dst_scc] += 1;
                        }
                    }
                }
            }
        }

        // Step 3: Kahn's algorithm on the condensed DAG
        let mut queue: VecDeque<usize> = VecDeque::new();
        for (i, &deg) in condensed_in_degree.iter().enumerate().take(num_sccs) {
            if deg == 0 {
                queue.push_back(i);
            }
        }

        let mut scc_layers: Vec<Vec<usize>> = Vec::new();
        while !queue.is_empty() {
            let mut layer = Vec::new();
            let mut next_queue = VecDeque::new();

            while let Some(scc_idx) = queue.pop_front() {
                layer.push(scc_idx);
                for &neighbor_scc in &condensed_adj[scc_idx] {
                    condensed_in_degree[neighbor_scc] -= 1;
                    if condensed_in_degree[neighbor_scc] == 0 {
                        next_queue.push_back(neighbor_scc);
                    }
                }
            }

            scc_layers.push(layer);
            queue = next_queue;
        }

        // Step 4: Expand SCC layers back to node IDs
        let mut result: Vec<Vec<String>> = Vec::new();
        for scc_layer in scc_layers {
            let mut layer_nodes: Vec<String> = Vec::new();
            for scc_idx in scc_layer {
                for &node_idx in &sccs[scc_idx] {
                    if let Some(id) = self.graph.node_weight(node_idx) {
                        layer_nodes.push(id.clone());
                    }
                }
            }
            layer_nodes.sort();
            result.push(layer_nodes);
        }

        result
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
        }
    }

    #[test]
    fn add_nodes_and_edges() {
        let mut graph = GraphEngine::new();
        graph.add_node(file_node("a", "a.rs")).unwrap();
        graph.add_node(file_node("b", "b.rs")).unwrap();
        graph.add_edge(test_edge("a", "b")).unwrap();

        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.edge_count(), 1);
    }

    #[test]
    fn bfs_traversal() {
        let mut graph = GraphEngine::new();
        graph.add_node(file_node("a", "a.rs")).unwrap();
        graph.add_node(file_node("b", "b.rs")).unwrap();
        graph.add_node(file_node("c", "c.rs")).unwrap();
        graph.add_edge(test_edge("a", "b")).unwrap();
        graph.add_edge(test_edge("b", "c")).unwrap();

        let nodes = graph.bfs("a", 1).unwrap();
        assert_eq!(nodes.len(), 2); // a and b (c is at depth 2)
    }

    #[test]
    fn shortest_path() {
        let mut graph = GraphEngine::new();
        graph.add_node(file_node("a", "a.rs")).unwrap();
        graph.add_node(file_node("b", "b.rs")).unwrap();
        graph.add_node(file_node("c", "c.rs")).unwrap();
        graph.add_edge(test_edge("a", "b")).unwrap();
        graph.add_edge(test_edge("b", "c")).unwrap();

        let path = graph.shortest_path("a", "c").unwrap();
        assert_eq!(path, vec!["a", "b", "c"]);
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
    // ── PageRank Tests ──────────────────────────────────────────────────────

    #[test]
    fn pagerank_chain() {
        // a -> b -> c
        // c is a sink (dangling node) that redistributes rank uniformly.
        // Rank flows a -> b -> c, with c accumulating the most. Order: c > b > a.
        let mut graph = GraphEngine::new();
        graph.add_node(file_node("a", "a.rs")).unwrap();
        graph.add_node(file_node("b", "b.rs")).unwrap();
        graph.add_node(file_node("c", "c.rs")).unwrap();
        graph.add_edge(test_edge("a", "b")).unwrap();
        graph.add_edge(test_edge("b", "c")).unwrap();

        let ranks = graph.pagerank(0.85, 100, 1e-6);
        assert_eq!(ranks.len(), 3);
        assert!(
            ranks["c"] > ranks["b"],
            "c ({}) should rank higher than b ({})",
            ranks["c"],
            ranks["b"]
        );
        assert!(
            ranks["b"] > ranks["a"],
            "b ({}) should rank higher than a ({})",
            ranks["b"],
            ranks["a"]
        );
    }

    #[test]
    fn pagerank_star() {
        // a -> b, a -> c, a -> d
        // b, c, d are dangling nodes that redistribute rank uniformly.
        // They each receive direct rank from a, plus redistribution.
        // a only receives redistributed rank from the dangling nodes.
        // So each leaf should rank higher than the hub.
        let mut graph = GraphEngine::new();
        graph.add_node(file_node("a", "a.rs")).unwrap();
        graph.add_node(file_node("b", "b.rs")).unwrap();
        graph.add_node(file_node("c", "c.rs")).unwrap();
        graph.add_node(file_node("d", "d.rs")).unwrap();
        graph.add_edge(test_edge("a", "b")).unwrap();
        graph.add_edge(test_edge("a", "c")).unwrap();
        graph.add_edge(test_edge("a", "d")).unwrap();

        let ranks = graph.pagerank(0.85, 100, 1e-6);
        assert_eq!(ranks.len(), 4);
        // Leaves get direct rank from a AND redistribute back uniformly.
        // b, c, d should be approximately equal and each higher than a.
        assert!(
            ranks["b"] > ranks["a"],
            "b ({}) should rank higher than a ({})",
            ranks["b"],
            ranks["a"]
        );
        // b, c, d should be approximately equal
        assert!(
            (ranks["b"] - ranks["c"]).abs() < 0.01,
            "b ({}) and c ({}) should be approximately equal",
            ranks["b"],
            ranks["c"]
        );
    }

    #[test]
    fn pagerank_empty_graph() {
        let graph = GraphEngine::new();
        let ranks = graph.pagerank(0.85, 100, 1e-6);
        assert!(ranks.is_empty());
    }

    #[test]
    fn pagerank_single_node() {
        let mut graph = GraphEngine::new();
        graph.add_node(file_node("a", "a.rs")).unwrap();

        let ranks = graph.pagerank(0.85, 100, 1e-6);
        assert_eq!(ranks.len(), 1);
        assert!((ranks["a"] - 1.0).abs() < 0.01);
    }

    // ── Personalized PageRank Tests ─────────────────────────────────────────

    #[test]
    fn personalized_pagerank_cycle_seed_c() {
        // a -> b -> c -> a (cycle)
        // Seed on c: c and its neighbors should rank highest
        let mut graph = GraphEngine::new();
        graph.add_node(file_node("a", "a.rs")).unwrap();
        graph.add_node(file_node("b", "b.rs")).unwrap();
        graph.add_node(file_node("c", "c.rs")).unwrap();
        graph.add_edge(test_edge("a", "b")).unwrap();
        graph.add_edge(test_edge("b", "c")).unwrap();
        graph.add_edge(test_edge("c", "a")).unwrap();

        let mut seeds = HashMap::new();
        seeds.insert("c".to_string(), 1.0);

        let ranks = graph.personalized_pagerank(&seeds, 0.85, 100, 1e-6);
        assert_eq!(ranks.len(), 3);
        // c should have highest rank (it's the seed and receives teleport)
        // a is c's out-neighbor so it should be next
        assert!(
            ranks["c"] > ranks["b"],
            "c ({}) should rank higher than b ({})",
            ranks["c"],
            ranks["b"]
        );
        assert!(
            ranks["a"] > ranks["b"],
            "a ({}) should rank higher than b ({}) since c->a",
            ranks["a"],
            ranks["b"]
        );
    }

    #[test]
    fn personalized_pagerank_empty_seeds() {
        // With no seeds, should fall back to uniform (same as regular pagerank)
        let mut graph = GraphEngine::new();
        graph.add_node(file_node("a", "a.rs")).unwrap();
        graph.add_node(file_node("b", "b.rs")).unwrap();
        graph.add_edge(test_edge("a", "b")).unwrap();

        let seeds = HashMap::new();
        let ppr = graph.personalized_pagerank(&seeds, 0.85, 100, 1e-6);
        let pr = graph.pagerank(0.85, 100, 1e-6);

        // Should be approximately equal
        assert!((ppr["a"] - pr["a"]).abs() < 0.01);
        assert!((ppr["b"] - pr["b"]).abs() < 0.01);
    }

    // ── Louvain Community Detection Tests ───────────────────────────────────

    #[test]
    fn louvain_two_disconnected_cliques() {
        // Clique 1: a <-> b <-> c <-> a
        // Clique 2: d <-> e <-> f <-> d
        let mut graph = GraphEngine::new();
        for id in &["a", "b", "c", "d", "e", "f"] {
            graph.add_node(file_node(id, &format!("{id}.rs"))).unwrap();
        }
        // Clique 1
        graph.add_edge(test_edge("a", "b")).unwrap();
        graph.add_edge(test_edge("b", "a")).unwrap();
        graph.add_edge(test_edge("b", "c")).unwrap();
        graph.add_edge(test_edge("c", "b")).unwrap();
        graph.add_edge(test_edge("a", "c")).unwrap();
        graph.add_edge(test_edge("c", "a")).unwrap();
        // Clique 2
        graph.add_edge(test_edge("d", "e")).unwrap();
        graph.add_edge(test_edge("e", "d")).unwrap();
        graph.add_edge(test_edge("e", "f")).unwrap();
        graph.add_edge(test_edge("f", "e")).unwrap();
        graph.add_edge(test_edge("d", "f")).unwrap();
        graph.add_edge(test_edge("f", "d")).unwrap();

        let communities = graph.louvain_communities(1.0);
        assert_eq!(
            communities.len(),
            2,
            "Expected 2 communities, got {}: {:?}",
            communities.len(),
            communities
        );
        // Each community should have 3 nodes
        assert_eq!(communities[0].len(), 3);
        assert_eq!(communities[1].len(), 3);
        // Check that each clique is in a separate community
        let comm0_set: HashSet<&str> = communities[0].iter().map(|s| s.as_str()).collect();
        let has_abc = comm0_set.contains("a") && comm0_set.contains("b") && comm0_set.contains("c");
        let has_def = comm0_set.contains("d") && comm0_set.contains("e") && comm0_set.contains("f");
        assert!(
            has_abc || has_def,
            "First community should be one of the cliques: {:?}",
            communities[0]
        );
    }

    #[test]
    fn louvain_empty_graph() {
        let graph = GraphEngine::new();
        let communities = graph.louvain_communities(1.0);
        assert!(communities.is_empty());
    }

    #[test]
    fn louvain_single_node() {
        let mut graph = GraphEngine::new();
        graph.add_node(file_node("a", "a.rs")).unwrap();
        let communities = graph.louvain_communities(1.0);
        assert_eq!(communities.len(), 1);
        assert_eq!(communities[0], vec!["a"]);
    }

    // ── Betweenness Centrality Tests ────────────────────────────────────────

    #[test]
    fn betweenness_chain_middle_highest() {
        // a -> b -> c
        // b is on the shortest path from a to c, so it should have highest betweenness
        let mut graph = GraphEngine::new();
        graph.add_node(file_node("a", "a.rs")).unwrap();
        graph.add_node(file_node("b", "b.rs")).unwrap();
        graph.add_node(file_node("c", "c.rs")).unwrap();
        graph.add_edge(test_edge("a", "b")).unwrap();
        graph.add_edge(test_edge("b", "c")).unwrap();

        let bc = graph.betweenness_centrality();
        assert_eq!(bc.len(), 3);
        assert!(
            bc["b"] > bc["a"],
            "b ({}) should have higher betweenness than a ({})",
            bc["b"],
            bc["a"]
        );
        assert!(
            bc["b"] > bc["c"],
            "b ({}) should have higher betweenness than c ({})",
            bc["b"],
            bc["c"]
        );
        // a and c should have 0 betweenness (they are endpoints)
        assert!(
            bc["a"].abs() < f64::EPSILON,
            "a should have 0 betweenness, got {}",
            bc["a"]
        );
        assert!(
            bc["c"].abs() < f64::EPSILON,
            "c should have 0 betweenness, got {}",
            bc["c"]
        );
    }

    #[test]
    fn betweenness_empty_graph() {
        let graph = GraphEngine::new();
        let bc = graph.betweenness_centrality();
        assert!(bc.is_empty());
    }

    #[test]
    fn betweenness_two_nodes() {
        let mut graph = GraphEngine::new();
        graph.add_node(file_node("a", "a.rs")).unwrap();
        graph.add_node(file_node("b", "b.rs")).unwrap();
        graph.add_edge(test_edge("a", "b")).unwrap();

        let bc = graph.betweenness_centrality();
        assert_eq!(bc.len(), 2);
        assert!((bc["a"]).abs() < f64::EPSILON);
        assert!((bc["b"]).abs() < f64::EPSILON);
    }

    // ── Strongly Connected Components Tests ─────────────────────────────────

    #[test]
    fn scc_cycle_all_in_one() {
        // a -> b -> c -> a: all three should be in one SCC
        let mut graph = GraphEngine::new();
        graph.add_node(file_node("a", "a.rs")).unwrap();
        graph.add_node(file_node("b", "b.rs")).unwrap();
        graph.add_node(file_node("c", "c.rs")).unwrap();
        graph.add_edge(test_edge("a", "b")).unwrap();
        graph.add_edge(test_edge("b", "c")).unwrap();
        graph.add_edge(test_edge("c", "a")).unwrap();

        let sccs = graph.strongly_connected_components();
        assert_eq!(
            sccs.len(),
            1,
            "Expected 1 SCC, got {}: {:?}",
            sccs.len(),
            sccs
        );
        assert_eq!(sccs[0], vec!["a", "b", "c"]);
    }

    #[test]
    fn scc_chain_each_separate() {
        // a -> b -> c: no cycles, each node is its own SCC
        let mut graph = GraphEngine::new();
        graph.add_node(file_node("a", "a.rs")).unwrap();
        graph.add_node(file_node("b", "b.rs")).unwrap();
        graph.add_node(file_node("c", "c.rs")).unwrap();
        graph.add_edge(test_edge("a", "b")).unwrap();
        graph.add_edge(test_edge("b", "c")).unwrap();

        let sccs = graph.strongly_connected_components();
        assert_eq!(
            sccs.len(),
            3,
            "Expected 3 SCCs, got {}: {:?}",
            sccs.len(),
            sccs
        );
    }

    #[test]
    fn scc_empty_graph() {
        let graph = GraphEngine::new();
        let sccs = graph.strongly_connected_components();
        assert!(sccs.is_empty());
    }

    // ── Topological Sort Tests ──────────────────────────────────────────────

    #[test]
    fn topological_layers_dag() {
        // a -> b, a -> c, b -> d, c -> d
        // Layer 0: [a], Layer 1: [b, c], Layer 2: [d]
        let mut graph = GraphEngine::new();
        graph.add_node(file_node("a", "a.rs")).unwrap();
        graph.add_node(file_node("b", "b.rs")).unwrap();
        graph.add_node(file_node("c", "c.rs")).unwrap();
        graph.add_node(file_node("d", "d.rs")).unwrap();
        graph.add_edge(test_edge("a", "b")).unwrap();
        graph.add_edge(test_edge("a", "c")).unwrap();
        graph.add_edge(test_edge("b", "d")).unwrap();
        graph.add_edge(test_edge("c", "d")).unwrap();

        let layers = graph.topological_layers();
        assert_eq!(
            layers.len(),
            3,
            "Expected 3 layers, got {}: {:?}",
            layers.len(),
            layers
        );
        assert_eq!(layers[0], vec!["a"]);
        assert_eq!(layers[1], vec!["b", "c"]); // sorted within layer
        assert_eq!(layers[2], vec!["d"]);
    }

    #[test]
    fn topological_layers_with_cycle() {
        // a -> b -> c -> b (cycle between b and c), a -> d
        // SCCs: {a}, {b, c}, {d}
        // After condensation: {a} -> {b,c} and {a} -> {d}
        // Layer 0: [a], Layer 1: [b, c, d] (b and c condensed, d also depends on a)
        let mut graph = GraphEngine::new();
        graph.add_node(file_node("a", "a.rs")).unwrap();
        graph.add_node(file_node("b", "b.rs")).unwrap();
        graph.add_node(file_node("c", "c.rs")).unwrap();
        graph.add_node(file_node("d", "d.rs")).unwrap();
        graph.add_edge(test_edge("a", "b")).unwrap();
        graph.add_edge(test_edge("b", "c")).unwrap();
        graph.add_edge(test_edge("c", "b")).unwrap();
        graph.add_edge(test_edge("a", "d")).unwrap();

        let layers = graph.topological_layers();
        assert_eq!(
            layers.len(),
            2,
            "Expected 2 layers, got {}: {:?}",
            layers.len(),
            layers
        );
        assert_eq!(layers[0], vec!["a"]);
        // Layer 1 should contain b, c (from the cycle SCC) and d
        assert!(layers[1].contains(&"b".to_string()));
        assert!(layers[1].contains(&"c".to_string()));
        assert!(layers[1].contains(&"d".to_string()));
    }

    #[test]
    fn topological_layers_empty_graph() {
        let graph = GraphEngine::new();
        let layers = graph.topological_layers();
        assert!(layers.is_empty());
    }

    #[test]
    fn topological_layers_single_node() {
        let mut graph = GraphEngine::new();
        graph.add_node(file_node("a", "a.rs")).unwrap();
        let layers = graph.topological_layers();
        assert_eq!(layers.len(), 1);
        assert_eq!(layers[0], vec!["a"]);
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
