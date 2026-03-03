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

fn make_node(id: &str, label: &str, kind: NodeKind) -> GraphNode {
    GraphNode {
        id: id.to_string(),
        kind,
        label: label.to_string(),
        payload: HashMap::new(),
        centrality: 0.0,
        memory_id: None,
        namespace: None,
    }
}

fn make_edge(src: &str, dst: &str, rel: RelationshipType) -> Edge {
    Edge {
        id: format!("{src}->{dst}:{rel}"),
        src: src.to_string(),
        dst: dst.to_string(),
        relationship: rel,
        weight: 1.0,
        properties: HashMap::new(),
        created_at: chrono::Utc::now(),
        valid_from: None,
        valid_to: None,
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

// ── Filtered Traversal Tests ────────────────────────────────────────

#[test]
fn bfs_filtered_excludes_chunk_nodes() {
    let mut graph = GraphEngine::new();
    graph
        .add_node(make_node("file:main.rs", "main.rs", NodeKind::File))
        .unwrap();
    graph
        .add_node(make_node("sym:main::run", "run", NodeKind::Function))
        .unwrap();
    graph
        .add_node(make_node("chunk:main.rs:0", "chunk0", NodeKind::Chunk))
        .unwrap();
    graph
        .add_node(make_node("chunk:main.rs:1", "chunk1", NodeKind::Chunk))
        .unwrap();

    graph
        .add_edge(make_edge(
            "file:main.rs",
            "sym:main::run",
            RelationshipType::Contains,
        ))
        .unwrap();
    graph
        .add_edge(make_edge(
            "file:main.rs",
            "chunk:main.rs:0",
            RelationshipType::Contains,
        ))
        .unwrap();
    graph
        .add_edge(make_edge(
            "file:main.rs",
            "chunk:main.rs:1",
            RelationshipType::Contains,
        ))
        .unwrap();

    // Unfiltered BFS should return all 4 nodes
    let all = graph.bfs("file:main.rs", 2).unwrap();
    assert_eq!(all.len(), 4);

    // Filtered BFS excluding chunks should return only file + function
    let filtered = graph
        .bfs_filtered("file:main.rs", 2, &[NodeKind::Chunk], None)
        .unwrap();
    assert_eq!(filtered.len(), 2);
    let ids: Vec<&str> = filtered.iter().map(|n| n.id.as_str()).collect();
    assert!(ids.contains(&"file:main.rs"));
    assert!(ids.contains(&"sym:main::run"));
    assert!(!ids.iter().any(|id| id.starts_with("chunk:")));
}

#[test]
fn dfs_filtered_excludes_chunk_nodes() {
    let mut graph = GraphEngine::new();
    graph
        .add_node(make_node("file:lib.rs", "lib.rs", NodeKind::File))
        .unwrap();
    graph
        .add_node(make_node("sym:lib::parse", "parse", NodeKind::Function))
        .unwrap();
    graph
        .add_node(make_node("chunk:lib.rs:0", "chunk0", NodeKind::Chunk))
        .unwrap();

    graph
        .add_edge(make_edge(
            "file:lib.rs",
            "sym:lib::parse",
            RelationshipType::Contains,
        ))
        .unwrap();
    graph
        .add_edge(make_edge(
            "file:lib.rs",
            "chunk:lib.rs:0",
            RelationshipType::Contains,
        ))
        .unwrap();

    let filtered = graph
        .dfs_filtered("file:lib.rs", 2, &[NodeKind::Chunk], None)
        .unwrap();
    assert_eq!(filtered.len(), 2);
    assert!(filtered.iter().all(|n| n.kind != NodeKind::Chunk));
}

#[test]
fn bfs_filtered_includes_only_specified_relationships() {
    let mut graph = GraphEngine::new();
    graph
        .add_node(make_node("sym:a", "a", NodeKind::Function))
        .unwrap();
    graph
        .add_node(make_node("sym:b", "b", NodeKind::Function))
        .unwrap();
    graph
        .add_node(make_node("sym:c", "c", NodeKind::Function))
        .unwrap();
    graph
        .add_node(make_node("file:x.rs", "x.rs", NodeKind::File))
        .unwrap();

    // a -CALLS-> b, a -CONTAINS-> file:x.rs, b -CALLS-> c
    graph
        .add_edge(make_edge("sym:a", "sym:b", RelationshipType::Calls))
        .unwrap();
    graph
        .add_edge(make_edge("sym:a", "file:x.rs", RelationshipType::Contains))
        .unwrap();
    graph
        .add_edge(make_edge("sym:b", "sym:c", RelationshipType::Calls))
        .unwrap();

    // Only follow CALLS edges
    let filtered = graph
        .bfs_filtered("sym:a", 3, &[], Some(&[RelationshipType::Calls]))
        .unwrap();
    let ids: Vec<&str> = filtered.iter().map(|n| n.id.as_str()).collect();
    assert!(ids.contains(&"sym:a"));
    assert!(ids.contains(&"sym:b"));
    assert!(ids.contains(&"sym:c"));
    // file:x.rs should NOT be reached since it's via CONTAINS edge
    assert!(!ids.contains(&"file:x.rs"));
}

#[test]
fn dfs_filtered_includes_only_specified_relationships() {
    let mut graph = GraphEngine::new();
    graph
        .add_node(make_node("sym:x", "x", NodeKind::Function))
        .unwrap();
    graph
        .add_node(make_node("sym:y", "y", NodeKind::Function))
        .unwrap();
    graph
        .add_node(make_node("sym:z", "z", NodeKind::Function))
        .unwrap();

    // x -IMPORTS-> y, x -CALLS-> z
    graph
        .add_edge(make_edge("sym:x", "sym:y", RelationshipType::Imports))
        .unwrap();
    graph
        .add_edge(make_edge("sym:x", "sym:z", RelationshipType::Calls))
        .unwrap();

    // Only follow IMPORTS
    let filtered = graph
        .dfs_filtered("sym:x", 3, &[], Some(&[RelationshipType::Imports]))
        .unwrap();
    let ids: Vec<&str> = filtered.iter().map(|n| n.id.as_str()).collect();
    assert!(ids.contains(&"sym:x"));
    assert!(ids.contains(&"sym:y"));
    assert!(!ids.contains(&"sym:z"));
}

#[test]
fn bfs_filtered_combines_exclude_and_relationship_filters() {
    let mut graph = GraphEngine::new();
    graph
        .add_node(make_node("file:app.rs", "app.rs", NodeKind::File))
        .unwrap();
    graph
        .add_node(make_node("sym:app::main", "main", NodeKind::Function))
        .unwrap();
    graph
        .add_node(make_node("sym:app::helper", "helper", NodeKind::Function))
        .unwrap();
    graph
        .add_node(make_node("chunk:app.rs:0", "chunk0", NodeKind::Chunk))
        .unwrap();

    // file -CONTAINS-> sym:main, file -CONTAINS-> chunk, sym:main -CALLS-> sym:helper
    graph
        .add_edge(make_edge(
            "file:app.rs",
            "sym:app::main",
            RelationshipType::Contains,
        ))
        .unwrap();
    graph
        .add_edge(make_edge(
            "file:app.rs",
            "chunk:app.rs:0",
            RelationshipType::Contains,
        ))
        .unwrap();
    graph
        .add_edge(make_edge(
            "sym:app::main",
            "sym:app::helper",
            RelationshipType::Calls,
        ))
        .unwrap();

    // Filter: exclude chunks, only follow CONTAINS edges
    let filtered = graph
        .bfs_filtered(
            "file:app.rs",
            3,
            &[NodeKind::Chunk],
            Some(&[RelationshipType::Contains]),
        )
        .unwrap();
    let ids: Vec<&str> = filtered.iter().map(|n| n.id.as_str()).collect();
    // Should reach file and sym:main via CONTAINS, but NOT chunk (excluded)
    // and NOT sym:helper (reached via CALLS which is not in include list)
    assert!(ids.contains(&"file:app.rs"));
    assert!(ids.contains(&"sym:app::main"));
    assert!(!ids.contains(&"chunk:app.rs:0"));
    assert!(!ids.contains(&"sym:app::helper"));
}
