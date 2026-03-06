use super::*;
use crate::mcp::scoring::compute_score;
use crate::mcp::test_helpers::*;
use chrono::Utc;
use codemem_core::{Edge, GraphNode, NodeKind, RelationshipType};
use std::collections::HashMap;

#[test]
fn handle_unknown_tool() {
    let server = test_server();
    let params = json!({"name": "nonexistent", "arguments": {}});
    let resp = server.handle_request("tools/call", Some(&params), json!(4));
    let result = resp.result.unwrap();
    assert_eq!(result["isError"], true);
}

#[test]
fn handle_health() {
    let server = test_server();
    let params = json!({"name": "codemem_status", "arguments": {"include": ["health"]}});
    let resp = server.handle_request("tools/call", Some(&params), json!(7));
    let result = resp.result.unwrap();
    let text = result["content"][0]["text"].as_str().unwrap();
    let parsed: Value = serde_json::from_str(text).unwrap();
    assert_eq!(parsed["health"]["healthy"], true);
    assert_eq!(parsed["health"]["storage"], "ok");
}

#[test]
fn handle_stats() {
    let server = test_server();
    let params = json!({"name": "codemem_status", "arguments": {"include": ["stats"]}});
    let resp = server.handle_request("tools/call", Some(&params), json!(8));
    let result = resp.result.unwrap();
    let text = result["content"][0]["text"].as_str().unwrap();
    let parsed: Value = serde_json::from_str(text).unwrap();
    assert_eq!(parsed["stats"]["storage"]["memories"], 0);
    assert_eq!(parsed["stats"]["vector"]["dimensions"], 768);
}

// ── Graph Strength Scoring Tests ────────────────────────────────────

#[test]
fn graph_strength_zero_when_no_edges() {
    let server = test_server();
    let stored = store_memory(&server, "isolated memory", "context", &[]);
    let id = stored["id"].as_str().unwrap();

    // Verify graph strength is 0 for a memory with no edges
    let graph = server.engine.lock_graph().unwrap();
    let edges = graph.get_edges(id).unwrap();
    assert_eq!(edges.len(), 0);

    let memory = server.engine.storage().get_memory(id).unwrap().unwrap();
    let bm25 = server.engine.lock_bm25().unwrap();
    let breakdown = compute_score(&memory, &["isolated"], 0.0, &graph, &bm25, Utc::now());
    assert_eq!(breakdown.graph_strength, 0.0);
}

#[test]
fn graph_strength_increases_with_code_edges() {
    let server = test_server();
    let src = store_memory(&server, "source memory about rust", "insight", &["rust"]);
    let src_id = src["id"].as_str().unwrap();

    // Create code nodes and link the memory to them
    let sym_node = GraphNode {
        id: "sym:main::parse".to_string(),
        kind: NodeKind::Function,
        label: "parse".to_string(),
        payload: HashMap::new(),
        centrality: 0.0,
        memory_id: None,
        namespace: None,
    };
    let file_node = GraphNode {
        id: "file:src/main.rs".to_string(),
        kind: NodeKind::File,
        label: "main.rs".to_string(),
        payload: HashMap::new(),
        centrality: 0.0,
        memory_id: None,
        namespace: None,
    };

    server
        .engine
        .storage()
        .insert_graph_node(&sym_node)
        .unwrap();
    server
        .engine
        .storage()
        .insert_graph_node(&file_node)
        .unwrap();
    {
        let mut graph = server.engine.lock_graph().unwrap();
        graph.add_node(sym_node).unwrap();
        graph.add_node(file_node).unwrap();

        // Link memory -> sym node
        let edge1 = Edge {
            id: format!("{src_id}-RELATES_TO-sym:main::parse"),
            src: src_id.to_string(),
            dst: "sym:main::parse".to_string(),
            relationship: RelationshipType::RelatesTo,
            weight: 0.5,
            properties: HashMap::new(),
            created_at: chrono::Utc::now(),
            valid_from: None,
            valid_to: None,
        };
        // Link memory -> file node
        let edge2 = Edge {
            id: format!("{src_id}-RELATES_TO-file:src/main.rs"),
            src: src_id.to_string(),
            dst: "file:src/main.rs".to_string(),
            relationship: RelationshipType::RelatesTo,
            weight: 0.5,
            properties: HashMap::new(),
            created_at: chrono::Utc::now(),
            valid_from: None,
            valid_to: None,
        };
        graph.add_edge(edge1).unwrap();
        graph.add_edge(edge2).unwrap();

        // Recompute centrality so PageRank/betweenness are cached
        graph.recompute_centrality();
    }

    // Score: the memory linked to 2 code nodes should have graph_strength > 0
    let graph = server.engine.lock_graph().unwrap();
    let memory = server.engine.storage().get_memory(src_id).unwrap().unwrap();
    let bm25 = server.engine.lock_bm25().unwrap();
    let breakdown = compute_score(&memory, &["rust"], 0.0, &graph, &bm25, Utc::now());
    assert!(
        breakdown.graph_strength > 0.0,
        "graph_strength should be > 0 with code node edges, got {}",
        breakdown.graph_strength
    );
}

#[test]
fn graph_strength_caps_at_one() {
    let server = test_server();

    // Create 6 memories, connect all to the first
    let src = store_memory(&server, "hub memory with many edges", "insight", &[]);
    let src_id = src["id"].as_str().unwrap();

    for i in 0..6 {
        let dst = store_memory(&server, &format!("spoke memory number {i}"), "context", &[]);
        let dst_id = dst["id"].as_str().unwrap();
        let params = json!({
            "name": "associate_memories",
            "arguments": {
                "source_id": src_id,
                "target_id": dst_id,
                "relationship": "RELATES_TO",
            }
        });
        server.handle_request("tools/call", Some(&params), json!(20 + i));
    }

    // The graph_strength formula caps at 1.0 via .min(1.0)
    let graph = server.engine.lock_graph().unwrap();
    let memory = server.engine.storage().get_memory(src_id).unwrap().unwrap();
    let bm25 = server.engine.lock_bm25().unwrap();
    let breakdown = compute_score(&memory, &["hub"], 0.0, &graph, &bm25, Utc::now());
    assert!(
        breakdown.graph_strength <= 1.0,
        "graph_strength should be <= 1.0, got {}",
        breakdown.graph_strength
    );
}

#[test]
fn recall_uses_default_scoring_weights() {
    let server = test_server();

    // Store two memories
    store_memory(&server, "rust ownership concept", "insight", &[]);
    store_memory(
        &server,
        "rust borrowing rules",
        "pattern",
        &["rust", "borrowing", "rules"],
    );

    // Default weights: recall both (both match "rust")
    let text_default = recall_memories(&server, "rust", None);
    let results_default: Vec<Value> = serde_json::from_str(&text_default).unwrap();
    assert_eq!(results_default.len(), 2);
}

#[test]
fn tool_metrics_returns_snapshot() {
    let server = test_server();
    // Record some metrics manually
    codemem_core::Metrics::record_latency(server.engine.metrics().as_ref(), "recall_memory", 12.5);
    codemem_core::Metrics::increment_counter(
        server.engine.metrics().as_ref(),
        "tool_calls_total",
        1,
    );
    codemem_core::Metrics::record_gauge(server.engine.metrics().as_ref(), "memory_count", 7.0);

    let result = server.tool_codemem_status(&serde_json::json!({"include": ["metrics"]}));
    assert!(!result.is_error);
    let text = &result.content[0].text;
    let parsed: serde_json::Value = serde_json::from_str(text).unwrap();
    assert!(
        parsed["metrics"]["latencies"]["recall_memory"]["count"]
            .as_u64()
            .unwrap()
            >= 1
    );
    assert_eq!(parsed["metrics"]["counters"]["tool_calls_total"], 1);
    assert!(
        (parsed["metrics"]["gauges"]["memory_count"]
            .as_f64()
            .unwrap()
            - 7.0)
            .abs()
            < f64::EPSILON
    );
}
