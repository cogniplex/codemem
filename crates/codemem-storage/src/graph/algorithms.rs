use super::GraphEngine;
use codemem_core::{Edge, GraphNode, NodeKind};
use petgraph::graph::NodeIndex;
use petgraph::Direction;
use std::collections::{HashMap, HashSet, VecDeque};

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
        // Deduplicate bidirectional edges: for A->B and B->A, merge into one
        // undirected edge with combined weight.
        let mut undirected_weights: HashMap<(usize, usize), f64> = HashMap::new();
        for edge_ref in self.graph.edge_indices() {
            if let Some((src_idx, dst_idx)) = self.graph.edge_endpoints(edge_ref) {
                let w = self.graph[edge_ref];
                if let (Some(&si), Some(&di)) = (idx_pos.get(&src_idx), idx_pos.get(&dst_idx)) {
                    let key = if si <= di { (si, di) } else { (di, si) };
                    *undirected_weights.entry(key).or_insert(0.0) += w;
                }
            }
        }

        let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        let mut total_weight = 0.0;

        for (&(si, di), &w) in &undirected_weights {
            adj[si].push((di, w));
            if si != di {
                adj[di].push((si, w));
            }
            total_weight += w;
        }

        if total_weight == 0.0 {
            // No edges: each node is its own community
            return indices
                .iter()
                .filter_map(|&idx| self.graph.node_weight(idx).map(|id| vec![id.clone()]))
                .collect();
        }

        // m = total undirected edge weight
        let m = total_weight;
        let m2 = 2.0 * m;

        // Weighted degree of each node (sum of incident undirected edge weights)
        let k: Vec<f64> = (0..n)
            .map(|i| adj[i].iter().map(|&(_, w)| w).sum())
            .collect();

        // Initial assignment: each node in its own community
        let mut community: Vec<usize> = (0..n).collect();

        // sigma_tot[c] = sum of degrees of nodes in community c.
        // Maintained incrementally to avoid O(n^2) per pass.
        let mut sigma_tot: Vec<f64> = k.clone();

        // Iteratively move nodes to improve modularity
        let mut improved = true;
        let max_passes = 100;
        let mut pass = 0;

        while improved && pass < max_passes {
            improved = false;
            pass += 1;

            for i in 0..n {
                let current_comm = community[i];
                let ki = k[i];

                // Compute weights to each neighboring community
                let mut comm_weights: HashMap<usize, f64> = HashMap::new();
                for &(j, w) in &adj[i] {
                    *comm_weights.entry(community[j]).or_insert(0.0) += w;
                }

                // Standard Louvain delta-Q formula:
                // delta_Q = [w_in_new/m - resolution * ki * sigma_new / m2]
                //         - [w_in_current/m - resolution * ki * (sigma_current - ki) / m2]
                let w_in_current = comm_weights.get(&current_comm).copied().unwrap_or(0.0);
                let sigma_current = sigma_tot[current_comm];
                let remove_cost =
                    w_in_current / m - resolution * ki * (sigma_current - ki) / (m2 * m);

                // Find best community to move to
                let mut best_comm = current_comm;
                let mut best_gain = 0.0;

                for (&comm, &w_in_comm) in &comm_weights {
                    if comm == current_comm {
                        continue;
                    }
                    let sigma_comm = sigma_tot[comm];
                    let gain =
                        w_in_comm / m - resolution * ki * sigma_comm / (m2 * m) - remove_cost;
                    if gain > best_gain {
                        best_gain = gain;
                        best_comm = comm;
                    }
                }

                if best_comm != current_comm {
                    // Update sigma_tot incrementally
                    sigma_tot[current_comm] -= ki;
                    sigma_tot[best_comm] += ki;
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

    /// Return top-N nodes by centrality and edges between them.
    /// Optionally filter by namespace and/or node kinds.
    pub fn subgraph_top_n(
        &self,
        n: usize,
        namespace: Option<&str>,
        kinds: Option<&[NodeKind]>,
    ) -> (Vec<GraphNode>, Vec<Edge>) {
        let mut candidates: Vec<&GraphNode> = self
            .nodes
            .values()
            .filter(|node| {
                if let Some(ns) = namespace {
                    match &node.namespace {
                        Some(node_ns) => node_ns == ns,
                        None => false,
                    }
                } else {
                    true
                }
            })
            .filter(|node| {
                if let Some(k) = kinds {
                    k.contains(&node.kind)
                } else {
                    true
                }
            })
            .collect();

        // Sort by centrality descending
        candidates.sort_by(|a, b| {
            b.centrality
                .partial_cmp(&a.centrality)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Take top N
        candidates.truncate(n);

        let top_ids: HashSet<&str> = candidates.iter().map(|node| node.id.as_str()).collect();
        let nodes_vec: Vec<GraphNode> = candidates.into_iter().cloned().collect();

        // Collect edges where both src and dst are in the top-N set
        let edges_vec: Vec<Edge> = self
            .edges
            .values()
            .filter(|edge| {
                top_ids.contains(edge.src.as_str()) && top_ids.contains(edge.dst.as_str())
            })
            .cloned()
            .collect();

        (nodes_vec, edges_vec)
    }

    /// Return node-to-community-ID mapping for Louvain.
    pub fn louvain_with_assignment(&self, resolution: f64) -> HashMap<String, usize> {
        let communities = self.louvain_communities(resolution);
        let mut assignment = HashMap::new();
        for (idx, community) in communities.into_iter().enumerate() {
            for node_id in community {
                assignment.insert(node_id, idx);
            }
        }
        assignment
    }
}

#[cfg(test)]
#[path = "../tests/graph_algorithms_tests.rs"]
mod tests;
