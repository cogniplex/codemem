use codemem_core::{Edge, GraphBackend, GraphNode, NodeKind, RelationshipType};
use codemem_graph::GraphEngine;
use criterion::{criterion_group, criterion_main, Criterion};
use std::collections::HashMap;

/// Create a graph node with the given numeric ID.
fn make_node(i: usize) -> GraphNode {
    GraphNode {
        id: format!("node-{i}"),
        kind: NodeKind::File,
        label: format!("file_{i}.rs"),
        payload: HashMap::new(),
        centrality: 0.0,
        memory_id: None,
        namespace: None,
    }
}

/// Create a directed edge between two nodes.
fn make_edge(src: usize, dst: usize, edge_id: usize) -> Edge {
    Edge {
        id: format!("edge-{edge_id}"),
        src: format!("node-{src}"),
        dst: format!("node-{dst}"),
        relationship: RelationshipType::Contains,
        weight: 1.0,
        properties: HashMap::new(),
        created_at: chrono::Utc::now(),
        valid_from: None,
        valid_to: None,
    }
}

/// Build a graph with `num_nodes` nodes and `num_edges` deterministic edges.
/// Edges are distributed to create a connected, non-trivial graph topology:
///   - First (num_nodes - 1) edges form a chain: 0->1->2->...->N-1
///   - Remaining edges use a deterministic pattern to add cross-links.
fn build_graph(num_nodes: usize, num_edges: usize) -> GraphEngine {
    let mut graph = GraphEngine::new();

    // Add all nodes
    for i in 0..num_nodes {
        graph.add_node(make_node(i)).unwrap();
    }

    let mut edge_id = 0;

    // Phase 1: Chain edges to ensure connectivity (0 -> 1 -> 2 -> ... -> N-1)
    let chain_count = (num_nodes - 1).min(num_edges);
    for i in 0..chain_count {
        graph.add_edge(make_edge(i, i + 1, edge_id)).unwrap();
        edge_id += 1;
    }

    // Phase 2: Deterministic cross-links for the remaining edges
    let remaining = num_edges.saturating_sub(chain_count);
    let mut seed: usize = 42;
    for _ in 0..remaining {
        // Simple LCG to pick src/dst
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let src = seed % num_nodes;
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let dst = seed % num_nodes;
        // Skip self-loops
        if src == dst {
            // Shift dst by 1
            let dst = (dst + 1) % num_nodes;
            graph.add_edge(make_edge(src, dst, edge_id)).unwrap();
        } else {
            graph.add_edge(make_edge(src, dst, edge_id)).unwrap();
        }
        edge_id += 1;
    }

    graph
}

/// Benchmark BFS traversal on a graph with 100 nodes and 200 edges.
fn bench_graph_bfs(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_bfs");

    let graph = build_graph(100, 200);

    // BFS from node-0, depth 3 (should visit a meaningful subset)
    group.bench_function("100n_200e_depth3", |b| {
        b.iter(|| {
            let result = graph.bfs("node-0", 3).unwrap();
            assert!(!result.is_empty());
        });
    });

    // BFS from node-0, depth 10 (should reach most/all nodes via chain)
    group.bench_function("100n_200e_depth10", |b| {
        b.iter(|| {
            let result = graph.bfs("node-0", 10).unwrap();
            assert!(!result.is_empty());
        });
    });

    // BFS from a middle node
    group.bench_function("100n_200e_from_mid", |b| {
        b.iter(|| {
            let result = graph.bfs("node-50", 5).unwrap();
            assert!(!result.is_empty());
        });
    });

    group.finish();
}

/// Benchmark DFS traversal on a graph with 100 nodes and 200 edges.
fn bench_graph_dfs(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_dfs");

    let graph = build_graph(100, 200);

    // DFS from node-0, depth 3
    group.bench_function("100n_200e_depth3", |b| {
        b.iter(|| {
            let result = graph.dfs("node-0", 3).unwrap();
            assert!(!result.is_empty());
        });
    });

    // DFS from node-0, depth 10
    group.bench_function("100n_200e_depth10", |b| {
        b.iter(|| {
            let result = graph.dfs("node-0", 10).unwrap();
            assert!(!result.is_empty());
        });
    });

    // DFS from a middle node
    group.bench_function("100n_200e_from_mid", |b| {
        b.iter(|| {
            let result = graph.dfs("node-50", 5).unwrap();
            assert!(!result.is_empty());
        });
    });

    group.finish();
}

criterion_group!(benches, bench_graph_bfs, bench_graph_dfs);
criterion_main!(benches);
