use crate::GraphEngine;
use codemem_core::{Edge, GraphBackend, GraphNode, NodeKind, RelationshipType};
use std::collections::{HashMap, HashSet};

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

// ── Subgraph Top-N Tests ──────────────────────────────────────────────

fn namespaced_node(id: &str, label: &str, namespace: Option<&str>, kind: NodeKind) -> GraphNode {
    GraphNode {
        id: id.to_string(),
        kind,
        label: label.to_string(),
        payload: HashMap::new(),
        centrality: 0.0,
        memory_id: None,
        namespace: namespace.map(|s| s.to_string()),
    }
}

#[test]
fn subgraph_top_n_basic() {
    let mut graph = GraphEngine::new();
    graph.add_node(file_node("a", "a.rs")).unwrap();
    graph.add_node(file_node("b", "b.rs")).unwrap();
    graph.add_node(file_node("c", "c.rs")).unwrap();
    graph.add_edge(test_edge("a", "b")).unwrap();
    graph.add_edge(test_edge("b", "c")).unwrap();
    graph.add_edge(test_edge("a", "c")).unwrap();

    // Compute centrality so nodes have different scores
    graph.compute_centrality();

    // Take top 2 nodes by centrality
    let (nodes, edges) = graph.subgraph_top_n(2, None, None);
    assert_eq!(nodes.len(), 2);

    // All returned edges should connect top-N nodes only
    let top_ids: HashSet<&str> = nodes.iter().map(|n| n.id.as_str()).collect();
    for edge in &edges {
        assert!(top_ids.contains(edge.src.as_str()));
        assert!(top_ids.contains(edge.dst.as_str()));
    }
}

#[test]
fn subgraph_top_n_with_namespace_filter() {
    let mut graph = GraphEngine::new();
    graph
        .add_node(namespaced_node("a", "a.rs", Some("proj1"), NodeKind::File))
        .unwrap();
    graph
        .add_node(namespaced_node("b", "b.rs", Some("proj1"), NodeKind::File))
        .unwrap();
    graph
        .add_node(namespaced_node("c", "c.rs", Some("proj2"), NodeKind::File))
        .unwrap();
    graph.add_edge(test_edge("a", "b")).unwrap();
    graph.add_edge(test_edge("a", "c")).unwrap();
    graph.compute_centrality();

    let (nodes, _edges) = graph.subgraph_top_n(10, Some("proj1"), None);
    assert_eq!(nodes.len(), 2);
    for node in &nodes {
        assert_eq!(node.namespace.as_deref(), Some("proj1"));
    }
}

#[test]
fn subgraph_top_n_with_kind_filter() {
    let mut graph = GraphEngine::new();
    graph
        .add_node(namespaced_node("a", "a.rs", None, NodeKind::File))
        .unwrap();
    graph
        .add_node(namespaced_node("b", "do_stuff", None, NodeKind::Function))
        .unwrap();
    graph
        .add_node(namespaced_node("c", "MyClass", None, NodeKind::Class))
        .unwrap();
    graph.add_edge(test_edge("a", "b")).unwrap();
    graph.add_edge(test_edge("a", "c")).unwrap();
    graph.compute_centrality();

    let (nodes, _edges) =
        graph.subgraph_top_n(10, None, Some(&[NodeKind::Function, NodeKind::Class]));
    assert_eq!(nodes.len(), 2);
    for node in &nodes {
        assert!(
            node.kind == NodeKind::Function || node.kind == NodeKind::Class,
            "unexpected kind: {:?}",
            node.kind
        );
    }
}

#[test]
fn subgraph_top_n_empty_graph() {
    let graph = GraphEngine::new();
    let (nodes, edges) = graph.subgraph_top_n(5, None, None);
    assert!(nodes.is_empty());
    assert!(edges.is_empty());
}

#[test]
fn subgraph_top_n_n_larger_than_graph() {
    let mut graph = GraphEngine::new();
    graph.add_node(file_node("a", "a.rs")).unwrap();
    graph.add_node(file_node("b", "b.rs")).unwrap();
    graph.add_edge(test_edge("a", "b")).unwrap();
    graph.compute_centrality();

    let (nodes, edges) = graph.subgraph_top_n(100, None, None);
    assert_eq!(nodes.len(), 2);
    assert_eq!(edges.len(), 1);
}

#[test]
fn subgraph_top_n_edges_only_between_top() {
    // a -> b -> c -> d, take top 2 (b and c have highest centrality)
    // Only edge b->c should be returned
    let mut graph = GraphEngine::new();
    graph.add_node(file_node("a", "a.rs")).unwrap();
    graph.add_node(file_node("b", "b.rs")).unwrap();
    graph.add_node(file_node("c", "c.rs")).unwrap();
    graph.add_node(file_node("d", "d.rs")).unwrap();
    graph.add_edge(test_edge("a", "b")).unwrap();
    graph.add_edge(test_edge("b", "c")).unwrap();
    graph.add_edge(test_edge("c", "d")).unwrap();
    graph.compute_centrality();

    let (nodes, edges) = graph.subgraph_top_n(2, None, None);
    assert_eq!(nodes.len(), 2);

    let top_ids: HashSet<&str> = nodes.iter().map(|n| n.id.as_str()).collect();
    for edge in &edges {
        assert!(
            top_ids.contains(edge.src.as_str()) && top_ids.contains(edge.dst.as_str()),
            "edge {}-->{} should only connect top-N nodes",
            edge.src,
            edge.dst
        );
    }
}

// ── Louvain With Assignment Tests ─────────────────────────────────────

#[test]
fn louvain_with_assignment_two_cliques() {
    let mut graph = GraphEngine::new();
    for id in &["a", "b", "c", "d", "e", "f"] {
        graph
            .add_node(file_node(id, &format!("{id}.rs")))
            .unwrap();
    }
    // Clique 1: a <-> b <-> c <-> a
    graph.add_edge(test_edge("a", "b")).unwrap();
    graph.add_edge(test_edge("b", "a")).unwrap();
    graph.add_edge(test_edge("b", "c")).unwrap();
    graph.add_edge(test_edge("c", "b")).unwrap();
    graph.add_edge(test_edge("a", "c")).unwrap();
    graph.add_edge(test_edge("c", "a")).unwrap();
    // Clique 2: d <-> e <-> f <-> d
    graph.add_edge(test_edge("d", "e")).unwrap();
    graph.add_edge(test_edge("e", "d")).unwrap();
    graph.add_edge(test_edge("e", "f")).unwrap();
    graph.add_edge(test_edge("f", "e")).unwrap();
    graph.add_edge(test_edge("d", "f")).unwrap();
    graph.add_edge(test_edge("f", "d")).unwrap();

    let assignment = graph.louvain_with_assignment(1.0);
    assert_eq!(assignment.len(), 6);

    // All nodes in clique 1 should share the same community
    assert_eq!(assignment["a"], assignment["b"]);
    assert_eq!(assignment["b"], assignment["c"]);

    // All nodes in clique 2 should share the same community
    assert_eq!(assignment["d"], assignment["e"]);
    assert_eq!(assignment["e"], assignment["f"]);

    // The two cliques should be in different communities
    assert_ne!(assignment["a"], assignment["d"]);
}

#[test]
fn louvain_with_assignment_empty_graph() {
    let graph = GraphEngine::new();
    let assignment = graph.louvain_with_assignment(1.0);
    assert!(assignment.is_empty());
}

#[test]
fn louvain_with_assignment_single_node() {
    let mut graph = GraphEngine::new();
    graph.add_node(file_node("a", "a.rs")).unwrap();
    let assignment = graph.louvain_with_assignment(1.0);
    assert_eq!(assignment.len(), 1);
    assert!(assignment.contains_key("a"));
}

#[test]
fn louvain_with_assignment_all_nodes_present() {
    let mut graph = GraphEngine::new();
    for id in &["a", "b", "c"] {
        graph
            .add_node(file_node(id, &format!("{id}.rs")))
            .unwrap();
    }
    graph.add_edge(test_edge("a", "b")).unwrap();
    graph.add_edge(test_edge("b", "c")).unwrap();

    let assignment = graph.louvain_with_assignment(1.0);
    assert_eq!(assignment.len(), 3);
    assert!(assignment.contains_key("a"));
    assert!(assignment.contains_key("b"));
    assert!(assignment.contains_key("c"));
}
