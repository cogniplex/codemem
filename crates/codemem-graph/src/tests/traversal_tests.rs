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
