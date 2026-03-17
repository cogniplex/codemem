use super::*;
use crate::mcp::test_helpers::*;
use codemem_core::{Edge, GraphNode, NodeKind, RelationshipType};
use std::collections::HashMap;

// ── Cross-Repo Tool Tests ────────────────────────────────────────────

#[test]
fn get_cross_repo_returns_enhanced_data() {
    // Create a temp directory with a Cargo.toml
    let dir = tempfile::tempdir().unwrap();
    let src_dir = dir.path().join("src");
    std::fs::create_dir_all(&src_dir).unwrap();
    std::fs::write(src_dir.join("lib.rs"), "pub fn greet() {}\n").unwrap();
    std::fs::write(
        dir.path().join("Cargo.toml"),
        "[package]\nname = \"cross-test\"\nversion = \"0.1.0\"\n\n[dependencies]\nserde = \"1.0\"\n",
    )
    .unwrap();

    let server = test_server();

    // Index the codebase via engine (index_codebase MCP tool has been removed)
    let mut indexer = codemem_engine::Indexer::new();
    let resolved = indexer.index_and_resolve(dir.path()).unwrap();
    let namespace = dir
        .path()
        .file_name()
        .and_then(|f| f.to_str())
        .unwrap_or("test");
    server
        .engine
        .persist_index_results(&resolved, Some(namespace))
        .unwrap();
    let manifests = codemem_engine::index::manifest::scan_manifests(dir.path());
    let _ = server.engine.persist_cross_repo_data(
        &manifests,
        &resolved.unresolved,
        &resolved.symbols,
        &resolved.references,
        namespace,
    );

    // Now call get_cross_repo
    let params = json!({
        "name": "get_cross_repo",
        "arguments": {"path": dir.path().to_str().unwrap()}
    });
    let resp = server.handle_request("tools/call", Some(&params), json!(201));
    let result = resp.result.unwrap();
    let text = result["content"][0]["text"].as_str().unwrap();
    let parsed: serde_json::Value = serde_json::from_str(text).unwrap();

    // Verify enhanced fields exist
    assert!(
        parsed.get("cross_namespace_edges").is_some(),
        "Response should contain 'cross_namespace_edges' field"
    );
    assert!(
        parsed.get("unresolved_ref_count").is_some(),
        "Response should contain 'unresolved_ref_count' field"
    );
    assert!(
        parsed.get("registered_packages").is_some(),
        "Response should contain 'registered_packages' field"
    );
    assert!(
        parsed.get("api_endpoints").is_some(),
        "Response should contain 'api_endpoints' field"
    );

    // Cross-namespace edges may be empty (no other namespaces indexed), but the array exists
    assert!(
        parsed["cross_namespace_edges"].is_array(),
        "cross_namespace_edges should be an array"
    );

    // Registered packages should include our package from indexing
    let reg_pkgs = parsed["registered_packages"].as_array().unwrap();
    assert!(
        reg_pkgs
            .iter()
            .any(|p| p["name"].as_str() == Some("cross-test")),
        "registered_packages should include 'cross-test'"
    );
}

#[test]
fn cross_namespace_edge_query() {
    let server = test_server();

    // Insert nodes in two different namespaces
    let node_a = GraphNode {
        id: "sym:alpha::Foo".to_string(),
        kind: NodeKind::Function,
        label: "alpha::Foo".to_string(),
        payload: HashMap::new(),
        centrality: 0.0,
        memory_id: None,
        namespace: Some("alpha".to_string()),
        valid_from: None,
        valid_to: None,
    };
    let node_b = GraphNode {
        id: "sym:beta::Bar".to_string(),
        kind: NodeKind::Function,
        label: "beta::Bar".to_string(),
        payload: HashMap::new(),
        centrality: 0.0,
        memory_id: None,
        namespace: Some("beta".to_string()),
        valid_from: None,
        valid_to: None,
    };

    server.engine.storage().insert_graph_node(&node_a).unwrap();
    server.engine.storage().insert_graph_node(&node_b).unwrap();
    {
        let mut graph = server.engine.lock_graph().unwrap();
        graph.add_node(node_a).unwrap();
        graph.add_node(node_b).unwrap();
    }

    // Create a cross-namespace edge
    let edge = Edge {
        id: "xref:sym:alpha::Foo->sym:beta::Bar:DependsOn".to_string(),
        src: "sym:alpha::Foo".to_string(),
        dst: "sym:beta::Bar".to_string(),
        relationship: RelationshipType::DependsOn,
        weight: 0.6,
        valid_from: Some(chrono::Utc::now()),
        valid_to: None,
        properties: {
            let mut props = HashMap::new();
            props.insert("cross_namespace".to_string(), serde_json::Value::Bool(true));
            props.insert(
                "src_namespace".to_string(),
                serde_json::Value::String("alpha".to_string()),
            );
            props.insert(
                "dst_namespace".to_string(),
                serde_json::Value::String("beta".to_string()),
            );
            props
        },
        created_at: chrono::Utc::now(),
    };
    server.engine.storage().insert_graph_edge(&edge).unwrap();
    {
        let mut graph = server.engine.lock_graph().unwrap();
        graph.add_edge(edge).unwrap();
    }

    // Query with cross-namespace = false: should NOT find the edge for "alpha"
    // (because both src AND dst must be in "alpha", but dst is in "beta")
    let non_cross = server
        .engine
        .storage()
        .graph_edges_for_namespace_with_cross("alpha", false);
    assert!(
        non_cross.unwrap_or_default().is_empty(),
        "Non-cross-namespace query should not find cross-namespace edges"
    );

    // Query with cross-namespace = true: should find the edge for "alpha"
    let cross = server
        .engine
        .storage()
        .graph_edges_for_namespace_with_cross("alpha", true)
        .unwrap();
    assert!(
        !cross.is_empty(),
        "Cross-namespace query should find the edge"
    );
    assert_eq!(cross[0].src, "sym:alpha::Foo");
    assert_eq!(cross[0].dst, "sym:beta::Bar");

    // Also verify from beta's perspective
    let cross_beta = server
        .engine
        .storage()
        .graph_edges_for_namespace_with_cross("beta", true)
        .unwrap();
    assert!(
        !cross_beta.is_empty(),
        "Cross-namespace query from beta should also find the edge"
    );
}
