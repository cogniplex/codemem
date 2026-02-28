use crate::GraphEngine;
use codemem_core::{CodememError, Edge, GraphBackend, GraphNode, GraphStats};
use petgraph::graph::NodeIndex;
use petgraph::visit::Bfs;
use std::collections::{HashMap, HashSet};

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

#[cfg(test)]
mod tests {
    use crate::GraphEngine;
    use codemem_core::{Edge, GraphBackend, GraphNode, NodeKind, RelationshipType};
    use std::collections::HashMap;

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
}
