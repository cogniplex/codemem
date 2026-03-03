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
