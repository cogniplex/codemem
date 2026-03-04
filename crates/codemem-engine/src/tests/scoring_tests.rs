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
    let memory = MemoryNode {
        id: memory_id.to_string(),
        content: content.to_string(),
        memory_type: MemoryType::Decision,
        importance: 0.8,
        confidence: 0.9,
        access_count: 1,
        content_hash: "hash".to_string(),
        tags: vec!["process".to_string(), "request".to_string()],
        metadata: HashMap::new(),
        namespace: None,
        created_at: now,
        updated_at: now,
        last_accessed_at: now,
    };

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
