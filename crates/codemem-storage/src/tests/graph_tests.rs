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

    let mut components = graph.connected_components();
    components.sort_by(|a, b| a[0].cmp(&b[0]));
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

    let mut components = graph.connected_components();
    components.sort_by(|a, b| a[0].cmp(&b[0]));
    assert_eq!(components.len(), 2);
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

    assert!((a.centrality - 0.5).abs() < 1e-10);
    assert!((b.centrality - 1.0).abs() < 1e-10);
    assert!((c.centrality - 0.5).abs() < 1e-10);
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

    assert!((a.centrality - 1.0).abs() < 1e-10);
    assert!((b.centrality - 1.0 / 3.0).abs() < 1e-10);
}

#[test]
fn compute_centrality_single_node() {
    let mut graph = GraphEngine::new();
    graph.add_node(file_node("a", "a.rs")).unwrap();

    graph.compute_centrality();

    let a = graph.get_node("a").unwrap().unwrap();
    assert!((a.centrality - 0.0).abs() < 1e-10);
}

#[test]
fn compute_centrality_no_edges() {
    let mut graph = GraphEngine::new();
    graph.add_node(file_node("a", "a.rs")).unwrap();
    graph.add_node(file_node("b", "b.rs")).unwrap();

    graph.compute_centrality();

    let a = graph.get_node("a").unwrap().unwrap();
    let b = graph.get_node("b").unwrap().unwrap();
    assert!((a.centrality - 0.0).abs() < 1e-10);
    assert!((b.centrality - 0.0).abs() < 1e-10);
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

    assert!((graph.max_degree() - 3.0).abs() < 1e-10);
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

// ── raw_graph_metrics_for_memory Tests ──────────────────────────────

fn memory_node(id: &str, label: &str) -> GraphNode {
    GraphNode {
        id: id.to_string(),
        kind: NodeKind::Memory,
        label: label.to_string(),
        payload: HashMap::new(),
        centrality: 0.0,
        memory_id: None,
        namespace: None,
        valid_from: None,
        valid_to: None,
    }
}

fn code_node(id: &str, label: &str, kind: NodeKind) -> GraphNode {
    GraphNode {
        id: id.to_string(),
        kind,
        label: label.to_string(),
        payload: HashMap::new(),
        centrality: 0.0,
        memory_id: None,
        namespace: None,
        valid_from: None,
        valid_to: None,
    }
}

fn weighted_edge(src: &str, dst: &str, weight: f64) -> Edge {
    Edge {
        id: format!("{src}->{dst}"),
        src: src.to_string(),
        dst: dst.to_string(),
        relationship: RelationshipType::RelatesTo,
        weight,
        properties: HashMap::new(),
        created_at: chrono::Utc::now(),
        valid_from: None,
        valid_to: None,
    }
}

#[test]
fn raw_graph_metrics_code_neighbors_only() {
    let mut graph = GraphEngine::new();
    graph.add_node(memory_node("mem-1", "my memory")).unwrap();
    graph
        .add_node(code_node("sym:Func", "Func", NodeKind::Function))
        .unwrap();
    graph
        .add_node(code_node("file:main.rs", "main.rs", NodeKind::File))
        .unwrap();
    graph
        .add_edge(weighted_edge("mem-1", "sym:Func", 0.8))
        .unwrap();
    graph
        .add_edge(weighted_edge("mem-1", "file:main.rs", 0.6))
        .unwrap();
    graph.recompute_centrality();

    let metrics = graph.raw_graph_metrics_for_memory("mem-1").unwrap();
    assert_eq!(metrics.code_neighbor_count, 2);
    assert_eq!(metrics.memory_neighbor_count, 0);
    assert!((metrics.total_edge_weight - 1.4).abs() < 1e-6);
    assert!((metrics.memory_edge_weight - 0.0).abs() < 1e-6);
}

#[test]
fn raw_graph_metrics_memory_neighbors_only() {
    let mut graph = GraphEngine::new();
    graph
        .add_node(memory_node("mem-1", "first memory"))
        .unwrap();
    graph
        .add_node(memory_node("mem-2", "second memory"))
        .unwrap();
    graph
        .add_node(memory_node("mem-3", "third memory"))
        .unwrap();
    graph
        .add_edge(weighted_edge("mem-1", "mem-2", 0.5))
        .unwrap();
    graph
        .add_edge(weighted_edge("mem-1", "mem-3", 0.7))
        .unwrap();
    graph.recompute_centrality();

    let metrics = graph.raw_graph_metrics_for_memory("mem-1").unwrap();
    assert_eq!(metrics.code_neighbor_count, 0);
    assert_eq!(metrics.memory_neighbor_count, 2);
    assert!((metrics.memory_edge_weight - 1.2).abs() < 1e-6);
}

#[test]
fn raw_graph_metrics_both_types() {
    let mut graph = GraphEngine::new();
    graph.add_node(memory_node("mem-1", "memory")).unwrap();
    graph
        .add_node(code_node("sym:Bar", "Bar", NodeKind::Class))
        .unwrap();
    graph
        .add_node(memory_node("mem-2", "other memory"))
        .unwrap();
    graph
        .add_edge(weighted_edge("mem-1", "sym:Bar", 0.9))
        .unwrap();
    graph
        .add_edge(weighted_edge("mem-1", "mem-2", 0.4))
        .unwrap();
    graph.recompute_centrality();

    let metrics = graph.raw_graph_metrics_for_memory("mem-1").unwrap();
    assert_eq!(metrics.code_neighbor_count, 1);
    assert_eq!(metrics.memory_neighbor_count, 1);
    assert!(metrics.total_edge_weight > 0.0);
    assert!(metrics.memory_edge_weight > 0.0);
}

#[test]
fn raw_graph_metrics_no_neighbors_returns_none() {
    let mut graph = GraphEngine::new();
    graph
        .add_node(memory_node("mem-alone", "lonely memory"))
        .unwrap();
    graph.recompute_centrality();

    let metrics = graph.raw_graph_metrics_for_memory("mem-alone");
    assert!(metrics.is_none(), "Isolated node should return None");
}

#[test]
fn raw_graph_metrics_nonexistent_node_returns_none() {
    let graph = GraphEngine::new();
    assert!(graph.raw_graph_metrics_for_memory("nonexistent").is_none());
}

#[test]
fn raw_graph_metrics_endpoint_and_test_are_code_neighbors() {
    // Endpoint and Test nodes should be classified as code neighbors (not Memory)
    let mut graph = GraphEngine::new();
    graph.add_node(memory_node("mem-1", "a memory")).unwrap();
    graph
        .add_node(code_node("endpoint-1", "GET /api", NodeKind::Endpoint))
        .unwrap();
    graph
        .add_node(code_node("test-1", "test_foo", NodeKind::Test))
        .unwrap();
    graph
        .add_edge(weighted_edge("mem-1", "endpoint-1", 0.5))
        .unwrap();
    graph
        .add_edge(weighted_edge("mem-1", "test-1", 0.5))
        .unwrap();
    graph.recompute_centrality();

    let metrics = graph.raw_graph_metrics_for_memory("mem-1").unwrap();
    assert_eq!(
        metrics.code_neighbor_count, 2,
        "Endpoint and Test nodes should count as code neighbors"
    );
    assert_eq!(metrics.memory_neighbor_count, 0);
}

#[test]
fn raw_graph_metrics_incoming_edges_counted() {
    // Verify incoming edges (code -> memory) are also counted
    let mut graph = GraphEngine::new();
    graph.add_node(memory_node("mem-1", "memory")).unwrap();
    graph
        .add_node(code_node("sym:Caller", "Caller", NodeKind::Function))
        .unwrap();
    // Edge direction: code -> memory (incoming to memory)
    graph
        .add_edge(weighted_edge("sym:Caller", "mem-1", 0.6))
        .unwrap();
    graph.recompute_centrality();

    let metrics = graph.raw_graph_metrics_for_memory("mem-1").unwrap();
    assert_eq!(
        metrics.code_neighbor_count, 1,
        "Incoming edges should be counted"
    );
}

// ── Large graph stress test for PageRank/betweenness ────────────────

#[test]
fn large_graph_pagerank_betweenness_stress() {
    let mut graph = GraphEngine::new();

    // Build a graph with 500+ nodes in a "hub-and-spoke + chain" topology
    let hub_count = 5;
    let spokes_per_hub = 100;

    // Create hub nodes
    for h in 0..hub_count {
        let hub_id = format!("hub-{h}");
        graph
            .add_node(file_node(&hub_id, &format!("hub{h}.rs")))
            .unwrap();
    }

    // Create spoke nodes and connect to hubs
    for h in 0..hub_count {
        let hub_id = format!("hub-{h}");
        for s in 0..spokes_per_hub {
            let spoke_id = format!("spoke-{h}-{s}");
            graph
                .add_node(file_node(&spoke_id, &format!("spoke{h}_{s}.rs")))
                .unwrap();
            graph.add_edge(test_edge(&hub_id, &spoke_id)).unwrap();
        }
    }

    // Chain hubs together: hub-0 -> hub-1 -> ... -> hub-4
    for h in 0..(hub_count - 1) {
        let src = format!("hub-{h}");
        let dst = format!("hub-{}", h + 1);
        graph
            .add_edge(Edge {
                id: format!("{src}->chain->{dst}"),
                src: src.clone(),
                dst: dst.clone(),
                relationship: RelationshipType::Contains,
                weight: 1.0,
                properties: HashMap::new(),
                created_at: chrono::Utc::now(),
                valid_from: None,
                valid_to: None,
            })
            .unwrap();
    }

    let total_nodes = hub_count + hub_count * spokes_per_hub;
    assert!(
        graph.node_count() >= 500,
        "Should have 500+ nodes, got {}",
        graph.node_count()
    );
    assert_eq!(graph.node_count(), total_nodes);

    // Recompute centrality (PageRank + betweenness)
    graph.recompute_centrality();

    // All hubs and spokes should have non-zero PageRank after recompute
    let hub0_pr = graph.get_pagerank("hub-0");
    let spoke_pr = graph.get_pagerank("spoke-0-0");
    assert!(hub0_pr > 0.0, "Hub should have non-zero PageRank");
    assert!(spoke_pr > 0.0, "Spoke should have non-zero PageRank");

    // Middle hub in chain should have high betweenness (bridges hubs)
    let mid_hub = format!("hub-{}", hub_count / 2);
    let mid_bt = graph.get_betweenness(&mid_hub);
    let spoke_bt = graph.get_betweenness("spoke-0-0");
    assert!(
        mid_bt > spoke_bt,
        "Middle hub betweenness ({mid_bt}) should exceed spoke betweenness ({spoke_bt})"
    );

    // Verify all hubs have PageRank > 0
    for h in 0..hub_count {
        let pr = graph.get_pagerank(&format!("hub-{h}"));
        assert!(pr > 0.0, "hub-{h} should have non-zero PageRank");
    }
}
