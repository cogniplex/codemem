use crate::graph::GraphEngine;
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

// ── remove_node + Subsequent Operations Tests ────────────────────────

#[test]
fn remove_node_then_bfs_still_works() {
    // Build: a -> b -> c -> d
    let mut graph = GraphEngine::new();
    for id in &["a", "b", "c", "d"] {
        graph.add_node(file_node(id, &format!("{id}.rs"))).unwrap();
    }
    graph.add_edge(test_edge("a", "b")).unwrap();
    graph.add_edge(test_edge("b", "c")).unwrap();
    graph.add_edge(test_edge("c", "d")).unwrap();

    // Remove middle node "b"
    assert!(graph.remove_node("b").unwrap());
    assert_eq!(graph.node_count(), 3);

    // BFS from "a" should only find "a" (since b is gone, no path to c or d)
    let nodes = graph.bfs("a", 10).unwrap();
    assert_eq!(nodes.len(), 1);
    assert_eq!(nodes[0].id, "a");

    // BFS from "c" should find c and d (edge c->d still exists)
    let nodes = graph.bfs("c", 10).unwrap();
    let ids: Vec<&str> = nodes.iter().map(|n| n.id.as_str()).collect();
    assert!(ids.contains(&"c"));
    assert!(ids.contains(&"d"));
    assert!(!ids.contains(&"b"));
}

#[test]
fn remove_node_then_pagerank_still_works() {
    let mut graph = GraphEngine::new();
    for id in &["a", "b", "c", "d"] {
        graph.add_node(file_node(id, &format!("{id}.rs"))).unwrap();
    }
    graph.add_edge(test_edge("a", "b")).unwrap();
    graph.add_edge(test_edge("b", "c")).unwrap();
    graph.add_edge(test_edge("c", "d")).unwrap();

    graph.remove_node("b").unwrap();

    // PageRank should still work on the remaining nodes
    let ranks = graph.pagerank(0.85, 100, 1e-6);
    assert_eq!(ranks.len(), 3);
    assert!(ranks.contains_key("a"));
    assert!(ranks.contains_key("c"));
    assert!(ranks.contains_key("d"));
    assert!(!ranks.contains_key("b"));
}

#[test]
fn remove_node_cleans_edges() {
    let mut graph = GraphEngine::new();
    graph.add_node(file_node("a", "a.rs")).unwrap();
    graph.add_node(file_node("b", "b.rs")).unwrap();
    graph.add_node(file_node("c", "c.rs")).unwrap();
    graph.add_edge(test_edge("a", "b")).unwrap();
    graph.add_edge(test_edge("b", "c")).unwrap();

    graph.remove_node("b").unwrap();

    // Edges involving "b" should be gone
    assert_eq!(graph.edge_count(), 0);
    // get_edges for remaining nodes should be empty
    assert!(graph.get_edges("a").unwrap().is_empty());
    assert!(graph.get_edges("c").unwrap().is_empty());
}

#[test]
fn remove_node_cleans_centrality_caches() {
    let mut graph = GraphEngine::new();
    graph.add_node(file_node("a", "a.rs")).unwrap();
    graph.add_node(file_node("b", "b.rs")).unwrap();
    graph.add_edge(test_edge("a", "b")).unwrap();
    graph.recompute_centrality();

    assert!(graph.get_pagerank("a") > 0.0);
    assert!(graph.get_pagerank("b") > 0.0);

    graph.remove_node("b").unwrap();

    // Cached values for "b" should be gone
    assert_eq!(graph.get_pagerank("b"), 0.0);
    assert_eq!(graph.get_betweenness("b"), 0.0);
}

#[test]
fn remove_node_fixes_swapped_index() {
    // petgraph swap-removes: removing a node swaps the last node into the removed slot.
    // After remove, operations on the swapped node must still work.
    let mut graph = GraphEngine::new();
    graph.add_node(file_node("a", "a.rs")).unwrap();
    graph.add_node(file_node("b", "b.rs")).unwrap();
    graph.add_node(file_node("c", "c.rs")).unwrap();
    graph.add_node(file_node("d", "d.rs")).unwrap();
    graph.add_edge(test_edge("a", "b")).unwrap();
    graph.add_edge(test_edge("c", "d")).unwrap();

    // Remove "a" — "d" (the last node) gets swapped to a's old slot
    graph.remove_node("a").unwrap();

    // "d" should still be accessible and traversable
    let d_node = graph.get_node("d").unwrap();
    assert!(d_node.is_some());

    // BFS from "c" should reach "d"
    let nodes = graph.bfs("c", 10).unwrap();
    let ids: Vec<&str> = nodes.iter().map(|n| n.id.as_str()).collect();
    assert!(ids.contains(&"c"));
    assert!(ids.contains(&"d"));

    // Shortest path c->d should still work
    let path = graph.shortest_path("c", "d").unwrap();
    assert_eq!(path, vec!["c", "d"]);
}

// ── Parallel Edges + remove_edge Tests ──────────────────────────────

#[test]
fn remove_edge_with_parallel_edges() {
    // Two edges between same nodes with different relationship types
    let mut graph = GraphEngine::new();
    graph.add_node(file_node("a", "a.rs")).unwrap();
    graph.add_node(file_node("b", "b.rs")).unwrap();

    let edge1 = Edge {
        id: "e1".to_string(),
        src: "a".to_string(),
        dst: "b".to_string(),
        relationship: RelationshipType::Contains,
        weight: 1.0,
        properties: HashMap::new(),
        created_at: chrono::Utc::now(),
        valid_from: None,
        valid_to: None,
    };
    let edge2 = Edge {
        id: "e2".to_string(),
        src: "a".to_string(),
        dst: "b".to_string(),
        relationship: RelationshipType::Calls,
        weight: 2.0,
        properties: HashMap::new(),
        created_at: chrono::Utc::now(),
        valid_from: None,
        valid_to: None,
    };

    graph.add_edge(edge1).unwrap();
    graph.add_edge(edge2).unwrap();
    assert_eq!(graph.edge_count(), 2);

    // Remove edge1 — edge2 should remain
    graph.remove_edge("e1").unwrap();
    assert_eq!(graph.edge_count(), 1);

    let remaining = graph.get_edges("a").unwrap();
    assert_eq!(remaining.len(), 1);
    assert_eq!(remaining[0].id, "e2");
    assert_eq!(remaining[0].relationship, RelationshipType::Calls);
    assert!((remaining[0].weight - 2.0).abs() < f64::EPSILON);
}

// ── bfs_filtered traversal through excluded nodes ───────────────────

#[test]
fn bfs_filtered_traverses_through_excluded_nodes() {
    // Graph: File -> Chunk -> Function
    // Excluding Chunk should still reach Function (traverse through Chunk)
    let mut graph = GraphEngine::new();
    graph
        .add_node(make_node("file:main.rs", "main.rs", NodeKind::File))
        .unwrap();
    graph
        .add_node(make_node("chunk:main.rs:0", "chunk0", NodeKind::Chunk))
        .unwrap();
    graph
        .add_node(make_node("sym:main::run", "run", NodeKind::Function))
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
            "chunk:main.rs:0",
            "sym:main::run",
            RelationshipType::Contains,
        ))
        .unwrap();

    // BFS excluding Chunk should still find Function behind the Chunk
    let filtered = graph
        .bfs_filtered("file:main.rs", 3, &[NodeKind::Chunk], None)
        .unwrap();
    let ids: Vec<&str> = filtered.iter().map(|n| n.id.as_str()).collect();
    assert!(ids.contains(&"file:main.rs"), "should include start node");
    assert!(
        ids.contains(&"sym:main::run"),
        "should reach Function through excluded Chunk"
    );
    assert!(
        !ids.iter().any(|id| id.starts_with("chunk:")),
        "should not include excluded Chunk nodes"
    );
}

#[test]
fn dfs_filtered_traverses_through_excluded_nodes() {
    // Same test but with DFS
    let mut graph = GraphEngine::new();
    graph
        .add_node(make_node("file:lib.rs", "lib.rs", NodeKind::File))
        .unwrap();
    graph
        .add_node(make_node("chunk:lib.rs:0", "chunk0", NodeKind::Chunk))
        .unwrap();
    graph
        .add_node(make_node("sym:lib::parse", "parse", NodeKind::Function))
        .unwrap();

    graph
        .add_edge(make_edge(
            "file:lib.rs",
            "chunk:lib.rs:0",
            RelationshipType::Contains,
        ))
        .unwrap();
    graph
        .add_edge(make_edge(
            "chunk:lib.rs:0",
            "sym:lib::parse",
            RelationshipType::Contains,
        ))
        .unwrap();

    let filtered = graph
        .dfs_filtered("file:lib.rs", 3, &[NodeKind::Chunk], None)
        .unwrap();
    let ids: Vec<&str> = filtered.iter().map(|n| n.id.as_str()).collect();
    assert!(ids.contains(&"file:lib.rs"));
    assert!(
        ids.contains(&"sym:lib::parse"),
        "should reach Function through excluded Chunk"
    );
    assert!(!ids.iter().any(|id| id.starts_with("chunk:")));
}
