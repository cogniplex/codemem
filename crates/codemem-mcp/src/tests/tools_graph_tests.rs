use super::*;
use crate::scoring::compute_score;
use crate::test_helpers::*;
use codemem_core::RelationshipType;

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
    let params = json!({"name": "codemem_health", "arguments": {}});
    let resp = server.handle_request("tools/call", Some(&params), json!(7));
    let result = resp.result.unwrap();
    let text = result["content"][0]["text"].as_str().unwrap();
    let health: Value = serde_json::from_str(text).unwrap();
    assert_eq!(health["healthy"], true);
    assert_eq!(health["storage"], "ok");
}

#[test]
fn handle_stats() {
    let server = test_server();
    let params = json!({"name": "codemem_stats", "arguments": {}});
    let resp = server.handle_request("tools/call", Some(&params), json!(8));
    let result = resp.result.unwrap();
    let text = result["content"][0]["text"].as_str().unwrap();
    let stats: Value = serde_json::from_str(text).unwrap();
    assert_eq!(stats["storage"]["memories"], 0);
    assert_eq!(stats["vector"]["dimensions"], 768);
}

// ── Graph Strength Scoring Tests ────────────────────────────────────

#[test]
fn graph_strength_zero_when_no_edges() {
    let server = test_server();
    let stored = store_memory(&server, "isolated memory", "context", &[]);
    let id = stored["id"].as_str().unwrap();

    // Verify graph strength is 0 for a memory with no edges
    let graph = server.graph.lock().unwrap();
    let edges = graph.get_edges(id).unwrap();
    assert_eq!(edges.len(), 0);

    let memory = server.storage.get_memory(id).unwrap().unwrap();
    let bm25 = server.bm25_index.lock().unwrap();
    let breakdown = compute_score(&memory, "isolated", &["isolated"], 0.0, &graph, &bm25);
    assert_eq!(breakdown.graph_strength, 0.0);
}

#[test]
fn graph_strength_increases_with_edges() {
    let server = test_server();
    let src = store_memory(&server, "source memory about rust", "insight", &["rust"]);
    let dst1 = store_memory(&server, "target memory one about types", "pattern", &[]);
    let dst2 = store_memory(&server, "target memory two about safety", "decision", &[]);

    let src_id = src["id"].as_str().unwrap();
    let dst1_id = dst1["id"].as_str().unwrap();
    let dst2_id = dst2["id"].as_str().unwrap();

    // Associate: src -> dst1
    let params = json!({
        "name": "associate_memories",
        "arguments": {
            "source_id": src_id,
            "target_id": dst1_id,
            "relationship": "RELATES_TO",
        }
    });
    server.handle_request("tools/call", Some(&params), json!(10));

    // Associate: src -> dst2
    let params = json!({
        "name": "associate_memories",
        "arguments": {
            "source_id": src_id,
            "target_id": dst2_id,
            "relationship": "LEADS_TO",
        }
    });
    server.handle_request("tools/call", Some(&params), json!(11));

    // Recompute centrality so PageRank/betweenness are cached
    {
        let mut graph = server.graph.lock().unwrap();
        graph.recompute_centrality();
    }

    // Score with edges: the source memory with 2 edges should have
    // a non-zero graph_strength due to enhanced scoring (PageRank + betweenness + degree)
    let graph = server.graph.lock().unwrap();
    let memory = server.storage.get_memory(src_id).unwrap().unwrap();
    let bm25 = server.bm25_index.lock().unwrap();
    let breakdown = compute_score(&memory, "rust", &["rust"], 0.0, &graph, &bm25);
    assert!(
        breakdown.graph_strength > 0.0,
        "graph_strength should be > 0 with 2 edges, got {}",
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
    let graph = server.graph.lock().unwrap();
    let memory = server.storage.get_memory(src_id).unwrap().unwrap();
    let bm25 = server.bm25_index.lock().unwrap();
    let breakdown = compute_score(&memory, "hub", &["hub"], 0.0, &graph, &bm25);
    assert!(
        breakdown.graph_strength <= 1.0,
        "graph_strength should be <= 1.0, got {}",
        breakdown.graph_strength
    );
}

// ── Structural Tool Tests ───────────────────────────────────────────

#[test]
fn search_symbols_requires_index() {
    let server = test_server();
    let params = json!({"name": "search_symbols", "arguments": {"query": "foo"}});
    let resp = server.handle_request("tools/call", Some(&params), json!(300));
    let result = resp.result.unwrap();
    assert_eq!(result["isError"], true);
    let text = result["content"][0]["text"].as_str().unwrap();
    assert!(text.contains("No codebase indexed"));
}

#[test]
fn get_symbol_info_requires_index() {
    let server = test_server();
    let params =
        json!({"name": "get_symbol_info", "arguments": {"qualified_name": "foo::bar"}});
    let resp = server.handle_request("tools/call", Some(&params), json!(301));
    let result = resp.result.unwrap();
    assert_eq!(result["isError"], true);
}

#[test]
fn get_clusters_empty_graph() {
    let server = test_server();
    let params = json!({"name": "get_clusters", "arguments": {}});
    let resp = server.handle_request("tools/call", Some(&params), json!(302));
    let result = resp.result.unwrap();
    let text = result["content"][0]["text"].as_str().unwrap();
    let parsed: Value = serde_json::from_str(text).unwrap();
    assert_eq!(parsed["cluster_count"], 0);
}

#[test]
fn get_pagerank_empty_graph() {
    let server = test_server();
    let params = json!({"name": "get_pagerank", "arguments": {}});
    let resp = server.handle_request("tools/call", Some(&params), json!(303));
    let result = resp.result.unwrap();
    let text = result["content"][0]["text"].as_str().unwrap();
    let parsed: Value = serde_json::from_str(text).unwrap();
    assert_eq!(parsed["results"].as_array().unwrap().len(), 0);
}

#[test]
fn get_cross_repo_requires_path_or_index() {
    let server = test_server();
    let params = json!({"name": "get_cross_repo", "arguments": {}});
    let resp = server.handle_request("tools/call", Some(&params), json!(304));
    let result = resp.result.unwrap();
    assert_eq!(result["isError"], true);
}

#[test]
fn index_codebase_nonexistent_path() {
    let server = test_server();
    let params =
        json!({"name": "index_codebase", "arguments": {"path": "/nonexistent/path/abc123"}});
    let resp = server.handle_request("tools/call", Some(&params), json!(306));
    let result = resp.result.unwrap();
    assert_eq!(result["isError"], true);
    let text = result["content"][0]["text"].as_str().unwrap();
    assert!(text.contains("does not exist"));
}

#[test]
fn index_codebase_and_search_symbols() {
    let server = test_server();

    // Create a temp directory with a Rust file
    let dir = tempfile::TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("lib.rs"),
        b"pub fn hello_world() { println!(\"hello\"); }\npub struct MyConfig { pub debug: bool }\n",
    )
    .unwrap();

    // Index the directory
    let params = json!({
        "name": "index_codebase",
        "arguments": {"path": dir.path().to_string_lossy()}
    });
    let resp = server.handle_request("tools/call", Some(&params), json!(307));
    let result = resp.result.unwrap();
    assert_eq!(result["isError"], false);
    let text = result["content"][0]["text"].as_str().unwrap();
    let index_result: Value = serde_json::from_str(text).unwrap();
    assert!(index_result["symbols"].as_u64().unwrap() >= 2);

    // Now search for symbols
    let params = json!({
        "name": "search_symbols",
        "arguments": {"query": "hello"}
    });
    let resp = server.handle_request("tools/call", Some(&params), json!(308));
    let result = resp.result.unwrap();
    assert_eq!(result["isError"], false);
    let text = result["content"][0]["text"].as_str().unwrap();
    assert!(text.contains("hello_world"));

    // Search by kind
    let params = json!({
        "name": "search_symbols",
        "arguments": {"query": "My", "kind": "struct"}
    });
    let resp = server.handle_request("tools/call", Some(&params), json!(309));
    let result = resp.result.unwrap();
    assert_eq!(result["isError"], false);
    let text = result["content"][0]["text"].as_str().unwrap();
    assert!(text.contains("MyConfig"));
}

#[test]
fn get_dependencies_for_symbol() {
    let server = test_server();

    // Manually add symbol nodes and an edge to the graph
    let node_a = GraphNode {
        id: "sym:module::foo".to_string(),
        kind: NodeKind::Function,
        label: "foo".to_string(),
        payload: HashMap::new(),
        centrality: 0.0,
        memory_id: None,
        namespace: None,
    };
    let node_b = GraphNode {
        id: "sym:module::bar".to_string(),
        kind: NodeKind::Function,
        label: "bar".to_string(),
        payload: HashMap::new(),
        centrality: 0.0,
        memory_id: None,
        namespace: None,
    };

    server.storage.insert_graph_node(&node_a).unwrap();
    server.storage.insert_graph_node(&node_b).unwrap();
    {
        let mut graph = server.graph.lock().unwrap();
        graph.add_node(node_a).unwrap();
        graph.add_node(node_b).unwrap();
        let edge = Edge {
            id: "ref:foo->bar:CALLS".to_string(),
            src: "sym:module::foo".to_string(),
            dst: "sym:module::bar".to_string(),
            relationship: RelationshipType::Calls,
            weight: 1.0,
            properties: HashMap::new(),
            created_at: chrono::Utc::now(),
            valid_from: None,
            valid_to: None,
        };
        graph.add_edge(edge).unwrap();
    }

    // Query outgoing deps from foo
    let params = json!({
        "name": "get_dependencies",
        "arguments": {"qualified_name": "module::foo", "direction": "outgoing"}
    });
    let resp = server.handle_request("tools/call", Some(&params), json!(310));
    let result = resp.result.unwrap();
    assert_eq!(result["isError"], false);
    let text = result["content"][0]["text"].as_str().unwrap();
    assert!(text.contains("module::bar"));
    assert!(text.contains("CALLS"));
}

#[test]
fn get_pagerank_with_nodes() {
    let server = test_server();

    // Add a small graph: A -> B -> C
    for (id, label) in [("sym:a", "a"), ("sym:b", "b"), ("sym:c", "c")] {
        let node = GraphNode {
            id: id.to_string(),
            kind: NodeKind::Function,
            label: label.to_string(),
            payload: HashMap::new(),
            centrality: 0.0,
            memory_id: None,
            namespace: None,
        };
        server.storage.insert_graph_node(&node).unwrap();
        server.graph.lock().unwrap().add_node(node).unwrap();
    }

    let edge1 = Edge {
        id: "e1".to_string(),
        src: "sym:a".to_string(),
        dst: "sym:b".to_string(),
        relationship: RelationshipType::Calls,
        weight: 1.0,
        properties: HashMap::new(),
        created_at: chrono::Utc::now(),
        valid_from: None,
        valid_to: None,
    };
    let edge2 = Edge {
        id: "e2".to_string(),
        src: "sym:b".to_string(),
        dst: "sym:c".to_string(),
        relationship: RelationshipType::Calls,
        weight: 1.0,
        properties: HashMap::new(),
        created_at: chrono::Utc::now(),
        valid_from: None,
        valid_to: None,
    };
    {
        let mut graph = server.graph.lock().unwrap();
        graph.add_edge(edge1).unwrap();
        graph.add_edge(edge2).unwrap();
    }

    let params = json!({"name": "get_pagerank", "arguments": {"top_k": 3}});
    let resp = server.handle_request("tools/call", Some(&params), json!(311));
    let result = resp.result.unwrap();
    assert_eq!(result["isError"], false);
    let text = result["content"][0]["text"].as_str().unwrap();
    let parsed: Value = serde_json::from_str(text).unwrap();
    assert_eq!(parsed["results"].as_array().unwrap().len(), 3);
}

#[test]
fn set_scoring_weights_updates_weights() {
    let server = test_server();

    // Set custom weights (all equal)
    let params = json!({
        "name": "set_scoring_weights",
        "arguments": {
            "vector_similarity": 1.0,
            "graph_strength": 1.0,
            "token_overlap": 1.0,
            "temporal": 1.0,
            "tag_matching": 1.0,
            "importance": 1.0,
            "confidence": 1.0,
            "recency": 1.0,
        }
    });
    let resp = server.handle_request("tools/call", Some(&params), json!(100));
    let result = resp.result.unwrap();
    let text = result["content"][0]["text"].as_str().unwrap();
    let parsed: Value = serde_json::from_str(text).unwrap();
    assert_eq!(parsed["updated"], true);

    // All weights should be normalized to 0.125
    let weights = &parsed["weights"];
    let expected = 0.125;
    let eps = 1e-10;
    assert!((weights["vector_similarity"].as_f64().unwrap() - expected).abs() < eps);
    assert!((weights["graph_strength"].as_f64().unwrap() - expected).abs() < eps);
    assert!((weights["token_overlap"].as_f64().unwrap() - expected).abs() < eps);
    assert!((weights["temporal"].as_f64().unwrap() - expected).abs() < eps);
    assert!((weights["tag_matching"].as_f64().unwrap() - expected).abs() < eps);
    assert!((weights["importance"].as_f64().unwrap() - expected).abs() < eps);
    assert!((weights["confidence"].as_f64().unwrap() - expected).abs() < eps);
    assert!((weights["recency"].as_f64().unwrap() - expected).abs() < eps);
}

#[test]
fn recall_uses_custom_scoring_weights() {
    let server = test_server();

    // Store two memories: one with high importance, one with many tags matching
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

    // Set weights to heavily favor tag_matching (1.0) and minimize everything else
    let params = json!({
        "name": "set_scoring_weights",
        "arguments": {
            "vector_similarity": 0.0,
            "graph_strength": 0.0,
            "token_overlap": 0.01,
            "temporal": 0.0,
            "tag_matching": 1.0,
            "importance": 0.0,
            "confidence": 0.0,
            "recency": 0.0,
        }
    });
    server.handle_request("tools/call", Some(&params), json!(200));

    // Recall again - the tagged memory should score much higher
    let text_custom = recall_memories(&server, "rust", None);
    let results_custom: Vec<Value> = serde_json::from_str(&text_custom).unwrap();
    assert!(!results_custom.is_empty());

    // The first result should be the one with more tag matches
    assert!(results_custom[0]["content"]
        .as_str()
        .unwrap()
        .contains("borrowing"));
}

#[test]
fn set_scoring_weights_with_defaults_for_omitted() {
    let server = test_server();

    // Only set vector_similarity, rest should use defaults
    let params = json!({
        "name": "set_scoring_weights",
        "arguments": {
            "vector_similarity": 0.5,
        }
    });
    let resp = server.handle_request("tools/call", Some(&params), json!(300));
    let result = resp.result.unwrap();
    let text = result["content"][0]["text"].as_str().unwrap();
    let parsed: Value = serde_json::from_str(text).unwrap();
    assert_eq!(parsed["updated"], true);

    // vector_similarity should be 0.5 normalized against the sum of all defaults
    // sum = 0.5 + 0.25 + 0.15 + 0.10 + 0.10 + 0.05 + 0.05 + 0.05 = 1.25
    // so vector_similarity = 0.5 / 1.25 = 0.4
    let vs = parsed["weights"]["vector_similarity"].as_f64().unwrap();
    assert!((vs - 0.4).abs() < 1e-10);
}

#[test]
fn tool_metrics_returns_snapshot() {
    let server = test_server();
    // Record some metrics manually
    codemem_core::Metrics::record_latency(&*server.metrics, "recall_memory", 12.5);
    codemem_core::Metrics::increment_counter(&*server.metrics, "tool_calls_total", 1);
    codemem_core::Metrics::record_gauge(&*server.metrics, "memory_count", 7.0);

    let result = server.tool_metrics();
    assert!(!result.is_error);
    let text = &result.content[0].text;
    let parsed: serde_json::Value = serde_json::from_str(text).unwrap();
    assert!(
        parsed["latencies"]["recall_memory"]["count"]
            .as_u64()
            .unwrap()
            >= 1
    );
    assert_eq!(parsed["counters"]["tool_calls_total"], 1);
    assert!((parsed["gauges"]["memory_count"].as_f64().unwrap() - 7.0).abs() < f64::EPSILON);
}
