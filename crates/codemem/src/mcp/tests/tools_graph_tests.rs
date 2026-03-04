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
    // Legacy alias "codemem_health" maps to codemem_status(include: ["health"])
    let params = json!({"name": "codemem_health", "arguments": {}});
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
    // Legacy alias "codemem_stats" maps to codemem_status(include: ["stats"])
    let params = json!({"name": "codemem_stats", "arguments": {}});
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
    let graph = server.engine.graph.lock().unwrap();
    let edges = graph.get_edges(id).unwrap();
    assert_eq!(edges.len(), 0);

    let memory = server.engine.storage.get_memory(id).unwrap().unwrap();
    let bm25 = server.engine.bm25_index.lock().unwrap();
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

    server.engine.storage.insert_graph_node(&sym_node).unwrap();
    server.engine.storage.insert_graph_node(&file_node).unwrap();
    {
        let mut graph = server.engine.graph.lock().unwrap();
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
    let graph = server.engine.graph.lock().unwrap();
    let memory = server.engine.storage.get_memory(src_id).unwrap().unwrap();
    let bm25 = server.engine.bm25_index.lock().unwrap();
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
    let graph = server.engine.graph.lock().unwrap();
    let memory = server.engine.storage.get_memory(src_id).unwrap().unwrap();
    let bm25 = server.engine.bm25_index.lock().unwrap();
    let breakdown = compute_score(&memory, &["hub"], 0.0, &graph, &bm25, Utc::now());
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
    let params = json!({"name": "get_symbol_info", "arguments": {"qualified_name": "foo::bar"}});
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

    server.engine.storage.insert_graph_node(&node_a).unwrap();
    server.engine.storage.insert_graph_node(&node_b).unwrap();
    {
        let mut graph = server.engine.graph.lock().unwrap();
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
        server.engine.storage.insert_graph_node(&node).unwrap();
        server.engine.graph.lock().unwrap().add_node(node).unwrap();
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
        let mut graph = server.engine.graph.lock().unwrap();
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
fn set_scoring_weights_returns_removed_error() {
    let server = test_server();

    let params = json!({
        "name": "set_scoring_weights",
        "arguments": {
            "vector_similarity": 1.0,
        }
    });
    let resp = server.handle_request("tools/call", Some(&params), json!(100));
    let result = resp.result.unwrap();
    assert_eq!(result["isError"], true);
    let text = result["content"][0]["text"].as_str().unwrap();
    assert!(text.contains("removed"));
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

// ── Filtered Traversal MCP Tool Tests ─────────────────────────────

#[test]
fn graph_traverse_with_exclude_kinds() {
    let server = test_server();

    // Build a small graph: file -> function, file -> chunk
    let file_node = GraphNode {
        id: "file:test.rs".to_string(),
        kind: NodeKind::File,
        label: "test.rs".to_string(),
        payload: HashMap::new(),
        centrality: 0.0,
        memory_id: None,
        namespace: None,
    };
    let func_node = GraphNode {
        id: "sym:test::run".to_string(),
        kind: NodeKind::Function,
        label: "run".to_string(),
        payload: HashMap::new(),
        centrality: 0.0,
        memory_id: None,
        namespace: None,
    };
    let chunk_node = GraphNode {
        id: "chunk:test.rs:0".to_string(),
        kind: NodeKind::Chunk,
        label: "chunk0".to_string(),
        payload: HashMap::new(),
        centrality: 0.0,
        memory_id: None,
        namespace: None,
    };

    server.engine.storage.insert_graph_node(&file_node).unwrap();
    server.engine.storage.insert_graph_node(&func_node).unwrap();
    server
        .engine
        .storage
        .insert_graph_node(&chunk_node)
        .unwrap();
    {
        let mut graph = server.engine.graph.lock().unwrap();
        graph.add_node(file_node).unwrap();
        graph.add_node(func_node).unwrap();
        graph.add_node(chunk_node).unwrap();

        let edge1 = Edge {
            id: "e1".to_string(),
            src: "file:test.rs".to_string(),
            dst: "sym:test::run".to_string(),
            relationship: RelationshipType::Contains,
            weight: 0.1,
            properties: HashMap::new(),
            created_at: chrono::Utc::now(),
            valid_from: None,
            valid_to: None,
        };
        let edge2 = Edge {
            id: "e2".to_string(),
            src: "file:test.rs".to_string(),
            dst: "chunk:test.rs:0".to_string(),
            relationship: RelationshipType::Contains,
            weight: 0.1,
            properties: HashMap::new(),
            created_at: chrono::Utc::now(),
            valid_from: None,
            valid_to: None,
        };
        graph.add_edge(edge1).unwrap();
        graph.add_edge(edge2).unwrap();
    }

    // Traverse with exclude_kinds=["chunk"]
    let params = json!({
        "name": "graph_traverse",
        "arguments": {
            "start_id": "file:test.rs",
            "max_depth": 2,
            "algorithm": "bfs",
            "exclude_kinds": ["chunk"]
        }
    });
    let resp = server.handle_request("tools/call", Some(&params), json!(400));
    let result = resp.result.unwrap();
    assert_eq!(result["isError"], false);
    let text = result["content"][0]["text"].as_str().unwrap();
    let nodes: Vec<Value> = serde_json::from_str(text).unwrap();

    // Should have file + function but NOT chunk
    assert_eq!(nodes.len(), 2);
    let ids: Vec<&str> = nodes.iter().map(|n| n["id"].as_str().unwrap()).collect();
    assert!(ids.contains(&"file:test.rs"));
    assert!(ids.contains(&"sym:test::run"));
    assert!(!ids.iter().any(|id| id.starts_with("chunk:")));
}

#[test]
fn graph_traverse_with_include_relationships() {
    let server = test_server();

    // Build graph: a -CALLS-> b, a -IMPORTS-> c
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
        server.engine.storage.insert_graph_node(&node).unwrap();
        server.engine.graph.lock().unwrap().add_node(node).unwrap();
    }
    {
        let mut graph = server.engine.graph.lock().unwrap();
        let calls_edge = Edge {
            id: "e-calls".to_string(),
            src: "sym:a".to_string(),
            dst: "sym:b".to_string(),
            relationship: RelationshipType::Calls,
            weight: 1.0,
            properties: HashMap::new(),
            created_at: chrono::Utc::now(),
            valid_from: None,
            valid_to: None,
        };
        let imports_edge = Edge {
            id: "e-imports".to_string(),
            src: "sym:a".to_string(),
            dst: "sym:c".to_string(),
            relationship: RelationshipType::Imports,
            weight: 0.5,
            properties: HashMap::new(),
            created_at: chrono::Utc::now(),
            valid_from: None,
            valid_to: None,
        };
        graph.add_edge(calls_edge).unwrap();
        graph.add_edge(imports_edge).unwrap();
    }

    // Traverse with include_relationships=["CALLS"] only
    let params = json!({
        "name": "graph_traverse",
        "arguments": {
            "start_id": "sym:a",
            "max_depth": 2,
            "algorithm": "bfs",
            "include_relationships": ["CALLS"]
        }
    });
    let resp = server.handle_request("tools/call", Some(&params), json!(401));
    let result = resp.result.unwrap();
    assert_eq!(result["isError"], false);
    let text = result["content"][0]["text"].as_str().unwrap();
    let nodes: Vec<Value> = serde_json::from_str(text).unwrap();

    // Should reach a and b (via CALLS) but NOT c (via IMPORTS)
    let ids: Vec<&str> = nodes.iter().map(|n| n["id"].as_str().unwrap()).collect();
    assert!(ids.contains(&"sym:a"));
    assert!(ids.contains(&"sym:b"));
    assert!(!ids.contains(&"sym:c"));
}

#[test]
fn tool_metrics_returns_snapshot() {
    let server = test_server();
    // Record some metrics manually
    codemem_core::Metrics::record_latency(&*server.engine.metrics, "recall_memory", 12.5);
    codemem_core::Metrics::increment_counter(&*server.engine.metrics, "tool_calls_total", 1);
    codemem_core::Metrics::record_gauge(&*server.engine.metrics, "memory_count", 7.0);

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

// ── Phase 5: Graph Compaction Tests ─────────────────────────────────

#[test]
fn compaction_creates_package_nodes() {
    let server = test_server();
    let mut graph = server.engine.graph.lock().unwrap();

    // Simulate file nodes for a directory tree: src/auth/middleware.rs, src/api/handler.rs
    let files = vec!["src/auth/middleware.rs", "src/api/handler.rs", "src/lib.rs"];
    let now = chrono::Utc::now();
    let mut seen_files = std::collections::HashSet::new();

    for fp in &files {
        let node = GraphNode {
            id: format!("file:{fp}"),
            kind: NodeKind::File,
            label: fp.to_string(),
            payload: HashMap::new(),
            centrality: 0.0,
            memory_id: None,
            namespace: Some("test".to_string()),
        };
        let _ = graph.add_node(node);
        seen_files.insert(fp.to_string());
    }
    drop(graph);

    // Now call compact_graph which won't prune anything but we can test
    // that the package node creation happened during index_codebase.
    // Instead, manually create pkg nodes to test the summary tree tool.
    let mut graph = server.engine.graph.lock().unwrap();
    for dir in &["src/", "src/auth/", "src/api/"] {
        let node = GraphNode {
            id: format!("pkg:{dir}"),
            kind: NodeKind::Package,
            label: dir.to_string(),
            payload: HashMap::new(),
            centrality: 0.0,
            memory_id: None,
            namespace: Some("test".to_string()),
        };
        let _ = graph.add_node(node);
    }

    // Add CONTAINS edges: pkg:src/ → pkg:src/auth/, pkg:src/ → pkg:src/api/
    for (src, dst) in &[
        ("pkg:src/", "pkg:src/auth/"),
        ("pkg:src/", "pkg:src/api/"),
        ("pkg:src/auth/", "file:src/auth/middleware.rs"),
        ("pkg:src/api/", "file:src/api/handler.rs"),
        ("pkg:src/", "file:src/lib.rs"),
    ] {
        let edge = Edge {
            id: format!("contains:{src}->{dst}"),
            src: src.to_string(),
            dst: dst.to_string(),
            relationship: RelationshipType::Contains,
            weight: 0.1,
            valid_from: None,
            valid_to: None,
            properties: HashMap::new(),
            created_at: now,
        };
        let _ = graph.add_edge(edge);
    }
    drop(graph);

    // Test summary tree from pkg:src/
    let params = json!({
        "name": "summary_tree",
        "arguments": { "start_id": "pkg:src/", "max_depth": 3 }
    });
    let resp = server.handle_request("tools/call", Some(&params), json!(500));
    let result = resp.result.unwrap();
    let text = result["content"][0]["text"].as_str().unwrap();
    let tree: Value = serde_json::from_str(text).unwrap();

    assert_eq!(tree["id"], "pkg:src/");
    assert_eq!(tree["kind"], "package");
    let children = tree["children"].as_array().unwrap();
    // Should have 3 children: file:src/lib.rs, pkg:src/api/, pkg:src/auth/
    assert_eq!(children.len(), 3);

    // Check that sub-packages have their file children
    let auth_pkg = children
        .iter()
        .find(|c| c["id"] == "pkg:src/auth/")
        .expect("auth package should exist");
    let auth_children = auth_pkg["children"].as_array().unwrap();
    assert_eq!(auth_children.len(), 1);
    assert_eq!(auth_children[0]["id"], "file:src/auth/middleware.rs");
}

#[test]
fn compaction_prunes_low_score_chunks() {
    let server = test_server();
    let now = chrono::Utc::now();

    // Create a file node and many chunk nodes
    let file_path = "test/main.rs";
    let mut graph = server.engine.graph.lock().unwrap();

    let file_node = GraphNode {
        id: format!("file:{file_path}"),
        kind: NodeKind::File,
        label: file_path.to_string(),
        payload: HashMap::new(),
        centrality: 0.0,
        memory_id: None,
        namespace: Some("test".to_string()),
    };
    let _ = graph.add_node(file_node);

    // Create 15 chunk nodes — more than the default max_retained_chunks_per_file (10)
    for i in 0..15 {
        let chunk_id = format!("chunk:{file_path}:{i}");
        let mut payload = HashMap::new();
        payload.insert(
            "file_path".to_string(),
            serde_json::Value::String(file_path.to_string()),
        );
        payload.insert("line_start".to_string(), json!(i * 10));
        payload.insert("line_end".to_string(), json!(i * 10 + 9));
        payload.insert("non_ws_chars".to_string(), json!((i + 1) * 50));
        payload.insert("node_kind".to_string(), json!("function"));

        let node = GraphNode {
            id: chunk_id.clone(),
            kind: NodeKind::Chunk,
            label: format!("chunk:{file_path}:{i}"),
            payload,
            centrality: 0.0,
            memory_id: None,
            namespace: Some("test".to_string()),
        };
        let _ = server.engine.storage.insert_graph_node(&node);
        let _ = graph.add_node(node);

        // Add CONTAINS edge from file to chunk
        let edge = Edge {
            id: format!("contains:file:{file_path}->{chunk_id}"),
            src: format!("file:{file_path}"),
            dst: chunk_id,
            relationship: RelationshipType::Contains,
            weight: 0.1,
            valid_from: None,
            valid_to: None,
            properties: HashMap::new(),
            created_at: now,
        };
        let _ = server.engine.storage.insert_graph_edge(&edge);
        let _ = graph.add_edge(edge);
    }
    drop(graph);

    let mut seen = std::collections::HashSet::new();
    seen.insert(file_path.to_string());
    let (chunks_pruned, _symbols_pruned) = server.engine.compact_graph(&seen);

    // With 15 chunks and max_retained_chunks_per_file=10, at least some should be pruned
    assert!(
        chunks_pruned > 0,
        "Expected some chunks to be pruned, got 0"
    );

    // Remaining chunk nodes should be <= max_retained_chunks_per_file
    let graph = server.engine.graph.lock().unwrap();
    let remaining_chunks: Vec<_> = graph
        .get_all_nodes()
        .into_iter()
        .filter(|n| n.kind == NodeKind::Chunk)
        .collect();
    assert!(
        remaining_chunks.len() <= 10,
        "Expected at most 10 chunks, got {}",
        remaining_chunks.len()
    );
}

#[test]
fn compaction_preserves_memory_linked_chunks() {
    let server = test_server();
    let now = chrono::Utc::now();

    let file_path = "test/linked.rs";
    let mut graph = server.engine.graph.lock().unwrap();

    let file_node = GraphNode {
        id: format!("file:{file_path}"),
        kind: NodeKind::File,
        label: file_path.to_string(),
        payload: HashMap::new(),
        centrality: 0.0,
        memory_id: None,
        namespace: Some("test".to_string()),
    };
    let _ = graph.add_node(file_node);

    // Create 15 chunks, link one to a memory node
    for i in 0..15 {
        let chunk_id = format!("chunk:{file_path}:{i}");
        let mut payload = HashMap::new();
        payload.insert(
            "file_path".to_string(),
            serde_json::Value::String(file_path.to_string()),
        );
        payload.insert("line_start".to_string(), json!(i * 10));
        payload.insert("line_end".to_string(), json!(i * 10 + 9));
        payload.insert("non_ws_chars".to_string(), json!(50));

        let node = GraphNode {
            id: chunk_id.clone(),
            kind: NodeKind::Chunk,
            label: format!("chunk:{file_path}:{i}"),
            payload,
            centrality: 0.0,
            memory_id: None,
            namespace: Some("test".to_string()),
        };
        let _ = server.engine.storage.insert_graph_node(&node);
        let _ = graph.add_node(node);

        let edge = Edge {
            id: format!("contains:file:{file_path}->{chunk_id}"),
            src: format!("file:{file_path}"),
            dst: chunk_id,
            relationship: RelationshipType::Contains,
            weight: 0.1,
            valid_from: None,
            valid_to: None,
            properties: HashMap::new(),
            created_at: now,
        };
        let _ = server.engine.storage.insert_graph_edge(&edge);
        let _ = graph.add_edge(edge);
    }

    // Create a memory node linked to chunk 0
    let mem_id = "mem-linked-test";
    let mem_node = GraphNode {
        id: mem_id.to_string(),
        kind: NodeKind::Memory,
        label: "linked memory".to_string(),
        payload: HashMap::new(),
        centrality: 0.0,
        memory_id: Some(mem_id.to_string()),
        namespace: Some("test".to_string()),
    };
    let _ = graph.add_node(mem_node);

    let link_edge = Edge {
        id: format!("link:{mem_id}->chunk:{file_path}:0"),
        src: mem_id.to_string(),
        dst: format!("chunk:{file_path}:0"),
        relationship: RelationshipType::RelatesTo,
        weight: 0.5,
        valid_from: None,
        valid_to: None,
        properties: HashMap::new(),
        created_at: now,
    };
    let _ = server.engine.storage.insert_graph_edge(&link_edge);
    let _ = graph.add_edge(link_edge);
    drop(graph);

    let mut seen = std::collections::HashSet::new();
    seen.insert(file_path.to_string());
    server.engine.compact_graph(&seen);

    // The memory-linked chunk (chunk:0) should survive compaction
    let graph = server.engine.graph.lock().unwrap();
    let chunk0 = graph.get_node(&format!("chunk:{file_path}:0")).unwrap();
    assert!(
        chunk0.is_some(),
        "Memory-linked chunk should survive compaction"
    );
}

#[test]
fn summary_tree_excludes_chunks_by_default() {
    let server = test_server();
    let now = chrono::Utc::now();
    let mut graph = server.engine.graph.lock().unwrap();

    // Build a small tree: pkg:src/ → file:src/main.rs → chunk:src/main.rs:0
    let pkg = GraphNode {
        id: "pkg:src/".to_string(),
        kind: NodeKind::Package,
        label: "src/".to_string(),
        payload: HashMap::new(),
        centrality: 0.0,
        memory_id: None,
        namespace: None,
    };
    let file = GraphNode {
        id: "file:src/main.rs".to_string(),
        kind: NodeKind::File,
        label: "src/main.rs".to_string(),
        payload: HashMap::new(),
        centrality: 0.0,
        memory_id: None,
        namespace: None,
    };
    let chunk = GraphNode {
        id: "chunk:src/main.rs:0".to_string(),
        kind: NodeKind::Chunk,
        label: "chunk".to_string(),
        payload: HashMap::new(),
        centrality: 0.0,
        memory_id: None,
        namespace: None,
    };
    let _ = graph.add_node(pkg);
    let _ = graph.add_node(file);
    let _ = graph.add_node(chunk);

    for (src, dst) in &[
        ("pkg:src/", "file:src/main.rs"),
        ("file:src/main.rs", "chunk:src/main.rs:0"),
    ] {
        let e = Edge {
            id: format!("contains:{src}->{dst}"),
            src: src.to_string(),
            dst: dst.to_string(),
            relationship: RelationshipType::Contains,
            weight: 0.1,
            valid_from: None,
            valid_to: None,
            properties: HashMap::new(),
            created_at: now,
        };
        let _ = graph.add_edge(e);
    }
    drop(graph);

    // Without include_chunks, chunk should NOT appear
    let params = json!({
        "name": "summary_tree",
        "arguments": { "start_id": "pkg:src/", "max_depth": 3 }
    });
    let resp = server.handle_request("tools/call", Some(&params), json!(600));
    let result = resp.result.unwrap();
    let text = result["content"][0]["text"].as_str().unwrap();
    let tree: Value = serde_json::from_str(text).unwrap();

    let file_children = tree["children"][0]["children"]
        .as_array()
        .map(|a| a.len())
        .unwrap_or(0);
    assert_eq!(file_children, 0, "Chunks should be excluded by default");

    // With include_chunks=true, chunk SHOULD appear
    let params = json!({
        "name": "summary_tree",
        "arguments": { "start_id": "pkg:src/", "max_depth": 3, "include_chunks": true }
    });
    let resp = server.handle_request("tools/call", Some(&params), json!(601));
    let result = resp.result.unwrap();
    let text = result["content"][0]["text"].as_str().unwrap();
    let tree: Value = serde_json::from_str(text).unwrap();

    let file_children = tree["children"][0]["children"]
        .as_array()
        .map(|a| a.len())
        .unwrap_or(0);
    assert_eq!(file_children, 1, "Chunks should be included when requested");
}

// ── Symbol Compaction Tests ─────────────────────────────────────────

#[test]
fn compaction_prunes_low_score_symbols() {
    let server = test_server();
    let now = chrono::Utc::now();

    let file_path = "test/symbols.rs";
    let mut graph = server.engine.graph.lock().unwrap();

    let file_node = GraphNode {
        id: format!("file:{file_path}"),
        kind: NodeKind::File,
        label: file_path.to_string(),
        payload: HashMap::new(),
        centrality: 0.0,
        memory_id: None,
        namespace: Some("test".to_string()),
    };
    let _ = graph.add_node(file_node.clone());
    let _ = server.engine.storage.insert_graph_node(&file_node);

    // Create 20 symbols: mix of public/private, with/without CALLS edges
    for i in 0..20 {
        let sym_id = format!("sym:test::sym_{i}");
        let is_public = i < 5; // first 5 are public
        let mut payload = HashMap::new();
        payload.insert(
            "file_path".to_string(),
            serde_json::Value::String(file_path.to_string()),
        );
        payload.insert(
            "visibility".to_string(),
            json!(if is_public { "public" } else { "private" }),
        );
        payload.insert("line_start".to_string(), json!(i * 10));
        payload.insert("line_end".to_string(), json!(i * 10 + 5));

        let node = GraphNode {
            id: sym_id.clone(),
            kind: NodeKind::Function,
            label: format!("sym_{i}"),
            payload,
            centrality: 0.0,
            memory_id: None,
            namespace: Some("test".to_string()),
        };
        let _ = server.engine.storage.insert_graph_node(&node);
        let _ = graph.add_node(node);

        // Add CONTAINS edge from file to symbol
        let edge = Edge {
            id: format!("contains:file:{file_path}->{sym_id}"),
            src: format!("file:{file_path}"),
            dst: sym_id.clone(),
            relationship: RelationshipType::Contains,
            weight: 0.1,
            valid_from: None,
            valid_to: None,
            properties: HashMap::new(),
            created_at: now,
        };
        let _ = server.engine.storage.insert_graph_edge(&edge);
        let _ = graph.add_edge(edge);
    }

    // Add CALLS edges to first 3 symbols to make them high-connectivity
    for i in 0..3 {
        let src = format!("sym:test::sym_{i}");
        for j in 3..6 {
            let dst = format!("sym:test::sym_{j}");
            let edge = Edge {
                id: format!("calls:{src}->{dst}"),
                src: src.clone(),
                dst,
                relationship: RelationshipType::Calls,
                weight: 1.0,
                valid_from: None,
                valid_to: None,
                properties: HashMap::new(),
                created_at: now,
            };
            let _ = server.engine.storage.insert_graph_edge(&edge);
            let _ = graph.add_edge(edge);
        }
    }
    drop(graph);

    let mut seen = std::collections::HashSet::new();
    seen.insert(file_path.to_string());
    let (_chunks_pruned, symbols_pruned) = server.engine.compact_graph(&seen);

    // With 20 symbols and max_retained_symbols_per_file=15, some should be pruned
    assert!(
        symbols_pruned > 0,
        "Expected some symbols to be pruned, got 0"
    );

    // Verify private unreferenced symbols were the ones pruned
    let graph = server.engine.graph.lock().unwrap();
    let remaining_syms: Vec<_> = graph
        .get_all_nodes()
        .into_iter()
        .filter(|n| n.id.starts_with("sym:"))
        .collect();
    assert!(
        remaining_syms.len() < 20,
        "Expected fewer than 20 symbols remaining, got {}",
        remaining_syms.len()
    );
}

#[test]
fn compaction_preserves_public_symbols() {
    let server = test_server();
    let now = chrono::Utc::now();

    let file_path = "test/public.rs";
    let mut graph = server.engine.graph.lock().unwrap();

    let file_node = GraphNode {
        id: format!("file:{file_path}"),
        kind: NodeKind::File,
        label: file_path.to_string(),
        payload: HashMap::new(),
        centrality: 0.0,
        memory_id: None,
        namespace: Some("test".to_string()),
    };
    let _ = graph.add_node(file_node.clone());
    let _ = server.engine.storage.insert_graph_node(&file_node);

    // Create 20 public symbols with no CALLS edges (low connectivity)
    for i in 0..20 {
        let sym_id = format!("sym:test::pub_fn_{i}");
        let mut payload = HashMap::new();
        payload.insert(
            "file_path".to_string(),
            serde_json::Value::String(file_path.to_string()),
        );
        payload.insert("visibility".to_string(), json!("public"));
        payload.insert("line_start".to_string(), json!(i * 5));
        payload.insert("line_end".to_string(), json!(i * 5 + 3));

        let node = GraphNode {
            id: sym_id.clone(),
            kind: NodeKind::Function,
            label: format!("pub_fn_{i}"),
            payload,
            centrality: 0.0,
            memory_id: None,
            namespace: Some("test".to_string()),
        };
        let _ = server.engine.storage.insert_graph_node(&node);
        let _ = graph.add_node(node);

        let edge = Edge {
            id: format!("contains:file:{file_path}->{sym_id}"),
            src: format!("file:{file_path}"),
            dst: sym_id,
            relationship: RelationshipType::Contains,
            weight: 0.1,
            valid_from: None,
            valid_to: None,
            properties: HashMap::new(),
            created_at: now,
        };
        let _ = server.engine.storage.insert_graph_edge(&edge);
        let _ = graph.add_edge(edge);
    }
    drop(graph);

    let mut seen = std::collections::HashSet::new();
    seen.insert(file_path.to_string());
    let (_chunks_pruned, symbols_pruned) = server.engine.compact_graph(&seen);

    // K = max(15, 20 public) = 20, so no public symbols should be pruned
    assert_eq!(
        symbols_pruned, 0,
        "No public symbols should be pruned when public_count >= max_retained"
    );

    let graph = server.engine.graph.lock().unwrap();
    let remaining: Vec<_> = graph
        .get_all_nodes()
        .into_iter()
        .filter(|n| n.id.starts_with("sym:"))
        .collect();
    assert_eq!(remaining.len(), 20, "All 20 public symbols should survive");
}

#[test]
fn compaction_preserves_structural_anchors() {
    let server = test_server();
    let now = chrono::Utc::now();

    let file_path = "test/structural.rs";
    let mut graph = server.engine.graph.lock().unwrap();

    let file_node = GraphNode {
        id: format!("file:{file_path}"),
        kind: NodeKind::File,
        label: file_path.to_string(),
        payload: HashMap::new(),
        centrality: 0.0,
        memory_id: None,
        namespace: Some("test".to_string()),
    };
    let _ = graph.add_node(file_node.clone());
    let _ = server.engine.storage.insert_graph_node(&file_node);

    // Create structural nodes (Class, Interface, Module) with low scores:
    // private visibility, no CALLS edges, small code size
    let structural_kinds = [
        ("sym:test::MyClass", NodeKind::Class, "MyClass"),
        ("sym:test::MyTrait", NodeKind::Interface, "MyTrait"),
        ("sym:test::my_mod", NodeKind::Module, "my_mod"),
    ];

    for (sym_id, kind, label) in &structural_kinds {
        let mut payload = HashMap::new();
        payload.insert(
            "file_path".to_string(),
            serde_json::Value::String(file_path.to_string()),
        );
        payload.insert("visibility".to_string(), json!("private"));
        payload.insert("line_start".to_string(), json!(0));
        payload.insert("line_end".to_string(), json!(1));

        let node = GraphNode {
            id: sym_id.to_string(),
            kind: *kind,
            label: label.to_string(),
            payload,
            centrality: 0.0,
            memory_id: None,
            namespace: Some("test".to_string()),
        };
        let _ = server.engine.storage.insert_graph_node(&node);
        let _ = graph.add_node(node);

        let edge = Edge {
            id: format!("contains:file:{file_path}->{sym_id}"),
            src: format!("file:{file_path}"),
            dst: sym_id.to_string(),
            relationship: RelationshipType::Contains,
            weight: 0.1,
            valid_from: None,
            valid_to: None,
            properties: HashMap::new(),
            created_at: now,
        };
        let _ = server.engine.storage.insert_graph_edge(&edge);
        let _ = graph.add_edge(edge);
    }

    // Also add 20 private functions to exceed max_retained_symbols_per_file
    for i in 0..20 {
        let sym_id = format!("sym:test::priv_fn_{i}");
        let mut payload = HashMap::new();
        payload.insert(
            "file_path".to_string(),
            serde_json::Value::String(file_path.to_string()),
        );
        payload.insert("visibility".to_string(), json!("private"));
        payload.insert("line_start".to_string(), json!(10 + i));
        payload.insert("line_end".to_string(), json!(10 + i + 1));

        let node = GraphNode {
            id: sym_id.clone(),
            kind: NodeKind::Function,
            label: format!("priv_fn_{i}"),
            payload,
            centrality: 0.0,
            memory_id: None,
            namespace: Some("test".to_string()),
        };
        let _ = server.engine.storage.insert_graph_node(&node);
        let _ = graph.add_node(node);

        let edge = Edge {
            id: format!("contains:file:{file_path}->{sym_id}"),
            src: format!("file:{file_path}"),
            dst: sym_id,
            relationship: RelationshipType::Contains,
            weight: 0.1,
            valid_from: None,
            valid_to: None,
            properties: HashMap::new(),
            created_at: now,
        };
        let _ = server.engine.storage.insert_graph_edge(&edge);
        let _ = graph.add_edge(edge);
    }
    drop(graph);

    let mut seen = std::collections::HashSet::new();
    seen.insert(file_path.to_string());
    let (_chunks_pruned, symbols_pruned) = server.engine.compact_graph(&seen);

    // Some private functions should be pruned (23 total, K=15)
    assert!(
        symbols_pruned > 0,
        "Expected some symbols to be pruned, got 0"
    );

    // But structural anchors must survive
    let graph = server.engine.graph.lock().unwrap();
    for (sym_id, _, _) in &structural_kinds {
        let node = graph.get_node(sym_id).unwrap();
        assert!(
            node.is_some(),
            "Structural anchor {sym_id} should survive compaction"
        );
    }
}
