use super::GraphEngine;
use codemem_core::{
    CodememError, Edge, GraphBackend, GraphNode, GraphStats, NodeKind, RelationshipType,
};
use petgraph::graph::NodeIndex;
use petgraph::visit::EdgeRef;
use petgraph::Direction;
use std::collections::{HashMap, HashSet, VecDeque};

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
            // petgraph::DiGraph::remove_node swaps the last node into the removed
            // slot, invalidating the last node's NodeIndex. We must fix id_to_index.
            let last_idx = NodeIndex::new(self.graph.node_count() - 1);
            self.graph.remove_node(idx);
            // After removal, the node that was at `last_idx` is now at `idx`
            // (unless we removed the last node itself).
            if idx != last_idx {
                if let Some(swapped_id) = self.graph.node_weight(idx) {
                    self.id_to_index.insert(swapped_id.clone(), idx);
                }
            }
            self.nodes.remove(id);

            // Remove associated edges using edge adjacency index
            if let Some(edge_ids) = self.edge_adj.remove(id) {
                for eid in &edge_ids {
                    if let Some(edge) = self.edges.remove(eid) {
                        // Also remove from the other endpoint's adjacency list
                        let other = if edge.src == id { &edge.dst } else { &edge.src };
                        if let Some(other_edges) = self.edge_adj.get_mut(other) {
                            other_edges.retain(|e| e != eid);
                        }
                    }
                }
            }

            // Clean up centrality caches
            self.cached_pagerank.remove(id);
            self.cached_betweenness.remove(id);

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
        // Maintain edge adjacency index
        self.edge_adj
            .entry(edge.src.clone())
            .or_default()
            .push(edge.id.clone());
        self.edge_adj
            .entry(edge.dst.clone())
            .or_default()
            .push(edge.id.clone());
        self.edges.insert(edge.id.clone(), edge);
        Ok(())
    }

    fn get_edges(&self, node_id: &str) -> Result<Vec<Edge>, CodememError> {
        let edges: Vec<Edge> = self
            .edge_adj
            .get(node_id)
            .map(|edge_ids| {
                edge_ids
                    .iter()
                    .filter_map(|eid| self.edges.get(eid).cloned())
                    .collect()
            })
            .unwrap_or_default();
        Ok(edges)
    }

    fn remove_edge(&mut self, id: &str) -> Result<bool, CodememError> {
        if let Some(edge) = self.edges.remove(id) {
            // Remove from petgraph — match by weight to handle parallel edges
            if let (Some(&src_idx), Some(&dst_idx)) = (
                self.id_to_index.get(&edge.src),
                self.id_to_index.get(&edge.dst),
            ) {
                // Iterate edges_connecting to find the correct one by weight
                let target_weight = edge.weight;
                let petgraph_edge_idx = self
                    .graph
                    .edges_connecting(src_idx, dst_idx)
                    .find(|e| (*e.weight() - target_weight).abs() < f64::EPSILON)
                    .map(|e| e.id());
                if let Some(eidx) = petgraph_edge_idx {
                    self.graph.remove_edge(eidx);
                }
            }
            // Maintain edge adjacency index
            if let Some(src_edges) = self.edge_adj.get_mut(&edge.src) {
                src_edges.retain(|e| e != id);
            }
            if let Some(dst_edges) = self.edge_adj.get_mut(&edge.dst) {
                dst_edges.retain(|e| e != id);
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
        let mut queue: VecDeque<(NodeIndex, usize)> = VecDeque::new();
        queue.push_back((*start_idx, 0));
        visited.insert(*start_idx);

        while let Some((node_idx, depth)) = queue.pop_front() {
            if let Some(node_id) = self.graph.node_weight(node_idx) {
                if let Some(node) = self.nodes.get(node_id) {
                    result.push(node.clone());
                }
            }

            if depth >= max_depth {
                continue;
            }

            // Traverse edges in both directions so we find parents and children
            for neighbor in self.graph.neighbors_undirected(node_idx) {
                if visited.insert(neighbor) {
                    queue.push_back((neighbor, depth + 1));
                }
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

            for neighbor in self.graph.neighbors_undirected(node_idx) {
                if !visited.contains(&neighbor) {
                    stack.push((neighbor, depth + 1));
                }
            }
        }

        Ok(result)
    }

    fn bfs_filtered(
        &self,
        start_id: &str,
        max_depth: usize,
        exclude_kinds: &[NodeKind],
        include_relationships: Option<&[RelationshipType]>,
    ) -> Result<Vec<GraphNode>, CodememError> {
        let start_idx = self
            .id_to_index
            .get(start_id)
            .ok_or_else(|| CodememError::NotFound(format!("Node {start_id}")))?;

        let mut visited = HashSet::new();
        let mut result = Vec::new();
        let mut queue: VecDeque<(NodeIndex, usize)> = VecDeque::new();
        queue.push_back((*start_idx, 0));
        visited.insert(*start_idx);

        while let Some((node_idx, depth)) = queue.pop_front() {
            // Add current node to results if not excluded
            if let Some(node_id) = self.graph.node_weight(node_idx) {
                if let Some(node) = self.nodes.get(node_id) {
                    if !exclude_kinds.contains(&node.kind) {
                        result.push(node.clone());
                    }
                }
            }

            if depth >= max_depth {
                continue;
            }

            // Explore outgoing edges with filtering
            for neighbor_idx in self.graph.neighbors_directed(node_idx, Direction::Outgoing) {
                if visited.contains(&neighbor_idx) {
                    continue;
                }

                // Check relationship filter if set, using edge adjacency index
                if let Some(allowed_rels) = include_relationships {
                    let src_id = self
                        .graph
                        .node_weight(node_idx)
                        .cloned()
                        .unwrap_or_default();
                    let dst_id = self
                        .graph
                        .node_weight(neighbor_idx)
                        .cloned()
                        .unwrap_or_default();
                    let edge_matches = self
                        .edge_adj
                        .get(&src_id)
                        .map(|edge_ids| {
                            edge_ids.iter().any(|eid| {
                                self.edges.get(eid).is_some_and(|e| {
                                    e.src == src_id
                                        && e.dst == dst_id
                                        && allowed_rels.contains(&e.relationship)
                                })
                            })
                        })
                        .unwrap_or(false);
                    if !edge_matches {
                        continue;
                    }
                }

                // Always traverse through excluded-kind nodes but don't include
                // them in results (handled above when popped from queue).
                visited.insert(neighbor_idx);
                queue.push_back((neighbor_idx, depth + 1));
            }
        }

        Ok(result)
    }

    fn dfs_filtered(
        &self,
        start_id: &str,
        max_depth: usize,
        exclude_kinds: &[NodeKind],
        include_relationships: Option<&[RelationshipType]>,
    ) -> Result<Vec<GraphNode>, CodememError> {
        let start_idx = self
            .id_to_index
            .get(start_id)
            .ok_or_else(|| CodememError::NotFound(format!("Node {start_id}")))?;

        let mut visited = HashSet::new();
        let mut result = Vec::new();
        let mut stack: Vec<(NodeIndex, usize)> = vec![(*start_idx, 0)];

        while let Some((node_idx, depth)) = stack.pop() {
            if !visited.insert(node_idx) {
                continue;
            }

            // Add current node to results if not excluded
            if let Some(node_id) = self.graph.node_weight(node_idx) {
                if let Some(node) = self.nodes.get(node_id) {
                    if !exclude_kinds.contains(&node.kind) {
                        result.push(node.clone());
                    }
                }
            }

            if depth >= max_depth {
                continue;
            }

            // Explore outgoing edges with filtering
            for neighbor_idx in self.graph.neighbors_directed(node_idx, Direction::Outgoing) {
                if visited.contains(&neighbor_idx) {
                    continue;
                }

                // Check relationship filter if set, using edge adjacency index
                if let Some(allowed_rels) = include_relationships {
                    let src_id = self
                        .graph
                        .node_weight(node_idx)
                        .cloned()
                        .unwrap_or_default();
                    let dst_id = self
                        .graph
                        .node_weight(neighbor_idx)
                        .cloned()
                        .unwrap_or_default();
                    let edge_matches = self
                        .edge_adj
                        .get(&src_id)
                        .map(|edge_ids| {
                            edge_ids.iter().any(|eid| {
                                self.edges.get(eid).is_some_and(|e| {
                                    e.src == src_id
                                        && e.dst == dst_id
                                        && allowed_rels.contains(&e.relationship)
                                })
                            })
                        })
                        .unwrap_or(false);
                    if !edge_matches {
                        continue;
                    }
                }

                // Always traverse through excluded-kind nodes but don't include
                // them in results (handled above when popped from stack).
                stack.push((neighbor_idx, depth + 1));
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

    // Note: O(n+e) per call. Could be cached if this becomes a hot path.
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

    fn get_all_nodes(&self) -> Vec<GraphNode> {
        GraphEngine::get_all_nodes(self)
    }

    fn get_node_ref(&self, id: &str) -> Option<&GraphNode> {
        GraphEngine::get_node_ref(self, id)
    }

    fn get_edges_ref(&self, node_id: &str) -> Vec<&Edge> {
        GraphEngine::get_edges_ref(self, node_id)
    }

    fn node_count(&self) -> usize {
        GraphEngine::node_count(self)
    }

    fn edge_count(&self) -> usize {
        GraphEngine::edge_count(self)
    }

    fn recompute_centrality(&mut self) {
        GraphEngine::recompute_centrality(self);
    }

    fn recompute_centrality_with_options(&mut self, include_betweenness: bool) {
        GraphEngine::recompute_centrality_with_options(self, include_betweenness);
    }

    fn ensure_betweenness_computed(&mut self) {
        GraphEngine::ensure_betweenness_computed(self);
    }

    fn compute_centrality(&mut self) {
        GraphEngine::compute_centrality(self);
    }

    fn get_pagerank(&self, node_id: &str) -> f64 {
        GraphEngine::get_pagerank(self, node_id)
    }

    fn get_betweenness(&self, node_id: &str) -> f64 {
        GraphEngine::get_betweenness(self, node_id)
    }

    fn raw_graph_metrics_for_memory(
        &self,
        memory_id: &str,
    ) -> Option<codemem_core::RawGraphMetrics> {
        GraphEngine::raw_graph_metrics_for_memory(self, memory_id)
    }

    fn connected_components(&self) -> Vec<Vec<String>> {
        GraphEngine::connected_components(self)
    }

    fn strongly_connected_components(&self) -> Vec<Vec<String>> {
        GraphEngine::strongly_connected_components(self)
    }

    fn pagerank(&self, damping: f64, iterations: usize, tolerance: f64) -> HashMap<String, f64> {
        GraphEngine::pagerank(self, damping, iterations, tolerance)
    }

    fn louvain_communities(&self, resolution: f64) -> Vec<Vec<String>> {
        GraphEngine::louvain_communities(self, resolution)
    }

    fn topological_layers(&self) -> Vec<Vec<String>> {
        GraphEngine::topological_layers(self)
    }

    fn louvain_with_assignment(&self, resolution: f64) -> HashMap<String, usize> {
        GraphEngine::louvain_with_assignment(self, resolution)
    }

    fn subgraph_top_n(
        &self,
        n: usize,
        namespace: Option<&str>,
        kinds: Option<&[codemem_core::NodeKind]>,
    ) -> (Vec<GraphNode>, Vec<Edge>) {
        GraphEngine::subgraph_top_n(self, n, namespace, kinds)
    }
}

#[cfg(test)]
#[path = "../tests/graph_traversal_tests.rs"]
mod tests;
