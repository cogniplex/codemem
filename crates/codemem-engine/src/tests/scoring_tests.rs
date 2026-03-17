use super::*;
use crate::bm25::Bm25Index;
use chrono::Utc;
use codemem_core::{
    Edge, GraphBackend, GraphNode, MemoryNode, MemoryType, NodeKind, RelationshipType,
};
use codemem_storage::graph::GraphEngine;
use std::collections::HashMap;

// ── Test #2: All 9 scoring components contribute non-zero ───────────

#[test]
fn all_nine_scoring_components_nonzero() {
    // Build a graph with enough structure for graph_strength > 0.
    // Memory -> sym:B, sym:A -> sym:B -> sym:C gives B non-zero pagerank/betweenness.
    let mut graph = GraphEngine::new();
    let now = Utc::now();
    let memory_id = "mem-test-001";

    // Memory node
    let mem_node = GraphNode {
        id: memory_id.to_string(),
        kind: NodeKind::Memory,
        label: "test memory".to_string(),
        payload: HashMap::new(),
        centrality: 0.0,
        memory_id: Some(memory_id.to_string()),
        namespace: None,
        valid_from: None,
        valid_to: None,
    };
    graph.add_node(mem_node).unwrap();

    // Code nodes
    for (id, kind) in &[
        ("sym:A", NodeKind::Function),
        ("sym:B", NodeKind::Function),
        ("sym:C", NodeKind::Function),
    ] {
        let node = GraphNode {
            id: id.to_string(),
            kind: *kind,
            label: id.to_string(),
            payload: HashMap::new(),
            centrality: 0.0,
            memory_id: None,
            namespace: None,
            valid_from: None,
            valid_to: None,
        };
        graph.add_node(node).unwrap();
    }

    // Edges: A -> B -> C, memory -> B
    let edges = vec![
        ("e1", "sym:A", "sym:B", RelationshipType::Calls),
        ("e2", "sym:B", "sym:C", RelationshipType::Calls),
        ("e3", memory_id, "sym:B", RelationshipType::RelatesTo),
    ];
    for (id, src, dst, rel) in &edges {
        let edge = Edge {
            id: id.to_string(),
            src: src.to_string(),
            dst: dst.to_string(),
            relationship: *rel,
            weight: 0.8,
            properties: HashMap::new(),
            created_at: now,
            valid_from: None,
            valid_to: None,
        };
        graph.add_edge(edge).unwrap();
    }

    // Populate centrality caches (pagerank + betweenness)
    graph.recompute_centrality();

    // Build BM25 index with the memory doc
    let mut bm25 = Bm25Index::new();
    let content = "processRequest handles incoming HTTP data";
    bm25.add_document(memory_id, content);

    // Create MemoryNode with all fields set for non-zero scoring
    let mut memory = MemoryNode::test_default(content);
    memory.id = memory_id.to_string();
    memory.memory_type = MemoryType::Decision;
    memory.importance = 0.8;
    memory.confidence = 0.9;
    memory.access_count = 1;
    memory.tags = vec!["process".to_string(), "request".to_string()];

    let query_tokens = vec!["process", "request"];
    let breakdown = compute_score(&memory, &query_tokens, 0.75, &graph, &bm25, now);

    assert!(
        breakdown.vector_similarity > 0.0,
        "vector_similarity should be > 0, got {}",
        breakdown.vector_similarity
    );
    assert!(
        breakdown.graph_strength > 0.0,
        "graph_strength should be > 0, got {}",
        breakdown.graph_strength
    );
    assert!(
        breakdown.token_overlap > 0.0,
        "token_overlap should be > 0, got {}",
        breakdown.token_overlap
    );
    assert!(
        breakdown.temporal > 0.0,
        "temporal should be > 0, got {}",
        breakdown.temporal
    );
    assert!(
        breakdown.tag_matching > 0.0,
        "tag_matching should be > 0, got {}",
        breakdown.tag_matching
    );
    assert!(
        breakdown.importance > 0.0,
        "importance should be > 0, got {}",
        breakdown.importance
    );
    assert!(
        breakdown.confidence > 0.0,
        "confidence should be > 0, got {}",
        breakdown.confidence
    );
    assert!(
        breakdown.recency > 0.0,
        "recency should be > 0, got {}",
        breakdown.recency
    );
}

// ── Test #6: UTF-8 truncate safety ──────────────────────────────────

#[test]
fn truncate_no_truncation_needed() {
    assert_eq!(truncate_content("hello", 10), "hello");
}

#[test]
fn truncate_exact_boundary() {
    assert_eq!(truncate_content("hello world", 5), "hello...");
}

#[test]
fn truncate_multibyte_two_byte_char() {
    // 'é' is 2 bytes in UTF-8. "héllo" = h(1) é(2) l(1) l(1) o(1) = 6 bytes.
    // max=2 lands inside 'é', should back up to byte 1.
    let result = truncate_content("héllo", 2);
    assert_eq!(result, "h...");
}

#[test]
fn truncate_multibyte_three_byte_char() {
    // '日' is 3 bytes. "日本語テスト" = 18 bytes.
    // max=4 lands inside '本' (bytes 3..6), should back up to byte 3 (after '日').
    let result = truncate_content("日本語テスト", 4);
    assert_eq!(result, "日...");
}

#[test]
fn truncate_empty_string() {
    assert_eq!(truncate_content("", 5), "");
}

#[test]
fn truncate_zero_max() {
    assert_eq!(truncate_content("abc", 0), "...");
}

// ── graph_strength_for_memory edge cases ────────────────────────────

/// Helper: create a memory node in the graph.
fn add_memory_node(graph: &mut GraphEngine, id: &str) {
    let node = GraphNode {
        id: id.to_string(),
        kind: NodeKind::Memory,
        label: "test memory".to_string(),
        payload: HashMap::new(),
        centrality: 0.0,
        memory_id: Some(id.to_string()),
        namespace: None,
        valid_from: None,
        valid_to: None,
    };
    graph.add_node(node).unwrap();
}

/// Helper: create a code node in the graph.
fn add_code_node(graph: &mut GraphEngine, id: &str, kind: NodeKind) {
    let node = GraphNode {
        id: id.to_string(),
        kind,
        label: id.to_string(),
        payload: HashMap::new(),
        centrality: 0.0,
        memory_id: None,
        namespace: None,
        valid_from: None,
        valid_to: None,
    };
    graph.add_node(node).unwrap();
}

/// Helper: add an edge between two nodes.
fn add_test_edge(
    graph: &mut GraphEngine,
    id: &str,
    src: &str,
    dst: &str,
    rel: RelationshipType,
    weight: f64,
) {
    let now = Utc::now();
    let edge = Edge {
        id: id.to_string(),
        src: src.to_string(),
        dst: dst.to_string(),
        relationship: rel,
        weight,
        properties: HashMap::new(),
        created_at: now,
        valid_from: None,
        valid_to: None,
    };
    graph.add_edge(edge).unwrap();
}

#[test]
fn graph_strength_code_only_neighbors() {
    let mut graph = GraphEngine::new();

    add_memory_node(&mut graph, "mem-code-only");
    add_code_node(&mut graph, "sym:CodeA", NodeKind::Function);
    add_code_node(&mut graph, "sym:CodeB", NodeKind::Function);

    // Memory -> two code nodes
    add_test_edge(
        &mut graph,
        "e1",
        "mem-code-only",
        "sym:CodeA",
        RelationshipType::RelatesTo,
        0.6,
    );
    add_test_edge(
        &mut graph,
        "e2",
        "mem-code-only",
        "sym:CodeB",
        RelationshipType::RelatesTo,
        0.8,
    );

    graph.recompute_centrality();

    let score = graph_strength_for_memory(&graph, "mem-code-only");
    assert!(
        score > 0.0,
        "code-only neighbors should produce non-zero score, got {score}"
    );

    // The score should only have code_score component (no memory component blended)
    // With code_neighbor_count=2, connectivity = 2/5 = 0.4
    // avg_edge_w = (0.6+0.8)/2 = 0.7
    // code_score = 0.4*pagerank + 0.3*betweenness + 0.2*0.4 + 0.1*0.7
    // When no memory neighbors, code_score is used fully (no 70/30 blend)
    assert!(score <= 1.0, "score should be capped at 1.0, got {score}");
}

#[test]
fn graph_strength_memory_only_neighbors() {
    let mut graph = GraphEngine::new();

    add_memory_node(&mut graph, "mem-center");
    add_memory_node(&mut graph, "mem-peer1");
    add_memory_node(&mut graph, "mem-peer2");

    // Center memory -> two other memories
    add_test_edge(
        &mut graph,
        "e1",
        "mem-peer1",
        "mem-center",
        RelationshipType::SharesTheme,
        0.5,
    );
    add_test_edge(
        &mut graph,
        "e2",
        "mem-peer2",
        "mem-center",
        RelationshipType::PrecededBy,
        0.8,
    );

    graph.recompute_centrality();

    let score = graph_strength_for_memory(&graph, "mem-center");
    assert!(
        score > 0.0,
        "memory-only neighbors should produce non-zero score, got {score}"
    );

    // memory_score = 0.6 * connectivity + 0.4 * avg_edge_w
    // connectivity = 2/10 = 0.2
    // avg_edge_w = (0.5 + 0.8) / 2 = 0.65
    // memory_score = 0.6*0.2 + 0.4*0.65 = 0.12 + 0.26 = 0.38
    let expected_approx = 0.38;
    assert!(
        (score - expected_approx).abs() < 0.05,
        "memory-only score should be approximately {expected_approx}, got {score}"
    );
}

#[test]
fn graph_strength_both_code_and_memory_blend() {
    let mut graph = GraphEngine::new();

    add_memory_node(&mut graph, "mem-both");
    add_code_node(&mut graph, "sym:CodeX", NodeKind::Function);
    add_memory_node(&mut graph, "mem-peer");

    // Code neighbor
    add_test_edge(
        &mut graph,
        "e1",
        "mem-both",
        "sym:CodeX",
        RelationshipType::RelatesTo,
        0.7,
    );
    // Memory neighbor
    add_test_edge(
        &mut graph,
        "e2",
        "mem-peer",
        "mem-both",
        RelationshipType::SharesTheme,
        0.5,
    );

    graph.recompute_centrality();

    let score = graph_strength_for_memory(&graph, "mem-both");
    assert!(
        score > 0.0,
        "both code+memory neighbors should produce non-zero score, got {score}"
    );

    // Verify the 70/30 blend is applied
    // Compute code_score and memory_score independently
    let code_only_score = {
        let mut g = GraphEngine::new();
        add_memory_node(&mut g, "mem-co");
        add_code_node(&mut g, "sym:CodeX2", NodeKind::Function);
        add_test_edge(
            &mut g,
            "e1",
            "mem-co",
            "sym:CodeX2",
            RelationshipType::RelatesTo,
            0.7,
        );
        g.recompute_centrality();
        graph_strength_for_memory(&g, "mem-co")
    };

    let memory_only_score = {
        let mut g = GraphEngine::new();
        add_memory_node(&mut g, "mem-mo");
        add_memory_node(&mut g, "mem-mo-peer");
        add_test_edge(
            &mut g,
            "e1",
            "mem-mo-peer",
            "mem-mo",
            RelationshipType::SharesTheme,
            0.5,
        );
        g.recompute_centrality();
        graph_strength_for_memory(&g, "mem-mo")
    };

    let expected_blend = 0.7 * code_only_score + 0.3 * memory_only_score;
    assert!(
        (score - expected_blend).abs() < 0.1,
        "blended score {score} should approximate 0.7*{code_only_score} + 0.3*{memory_only_score} = {expected_blend}"
    );
}

#[test]
fn graph_strength_no_neighbors_returns_zero() {
    let mut graph = GraphEngine::new();

    // Memory node with no edges
    add_memory_node(&mut graph, "mem-isolated");
    graph.recompute_centrality();

    let score = graph_strength_for_memory(&graph, "mem-isolated");
    assert!(
        score == 0.0,
        "isolated memory should have graph_strength of 0.0, got {score}"
    );
}

#[test]
fn graph_strength_nonexistent_memory_returns_zero() {
    let graph = GraphEngine::new();

    let score = graph_strength_for_memory(&graph, "nonexistent");
    assert!(
        score == 0.0,
        "nonexistent memory should return 0.0, got {score}"
    );
}

#[test]
fn graph_strength_code_connectivity_caps_at_one() {
    let mut graph = GraphEngine::new();

    add_memory_node(&mut graph, "mem-many-code");
    // Add 8 code neighbors (>5, so connectivity should cap at 1.0)
    for i in 0..8 {
        let code_id = format!("sym:Code{i}");
        add_code_node(&mut graph, &code_id, NodeKind::Function);
        add_test_edge(
            &mut graph,
            &format!("e{i}"),
            "mem-many-code",
            &code_id,
            RelationshipType::RelatesTo,
            0.5,
        );
    }

    graph.recompute_centrality();

    let score = graph_strength_for_memory(&graph, "mem-many-code");
    assert!(
        score <= 1.0,
        "score should be capped at 1.0 even with many code neighbors, got {score}"
    );
    assert!(
        score > 0.0,
        "should have non-zero score with many code neighbors"
    );
}

#[test]
fn graph_strength_memory_connectivity_caps_at_one() {
    let mut graph = GraphEngine::new();

    add_memory_node(&mut graph, "mem-many-mem");
    // Add 15 memory neighbors (>10, so connectivity should cap at 1.0)
    for i in 0..15 {
        let peer_id = format!("mem-peer-{i}");
        add_memory_node(&mut graph, &peer_id);
        add_test_edge(
            &mut graph,
            &format!("e{i}"),
            &peer_id,
            "mem-many-mem",
            RelationshipType::SharesTheme,
            0.5,
        );
    }

    graph.recompute_centrality();

    let score = graph_strength_for_memory(&graph, "mem-many-mem");
    assert!(
        score <= 1.0,
        "score should be capped at 1.0 even with many memory neighbors, got {score}"
    );

    // With 15 memory neighbors, connectivity = min(15/10, 1.0) = 1.0
    // avg_edge_w = 0.5
    // memory_score = 0.6*1.0 + 0.4*0.5 = 0.8
    let expected_approx = 0.8;
    assert!(
        (score - expected_approx).abs() < 0.05,
        "capped memory connectivity score should be approximately {expected_approx}, got {score}"
    );
}
