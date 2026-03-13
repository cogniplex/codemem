use super::*;
use crate::mcp::test_helpers::*;
use codemem_core::{Edge, GraphBackend, GraphNode, NodeKind, RelationshipType};
use std::collections::HashMap;

// ── Structural Tool Tests ───────────────────────────────────────────

#[test]
fn search_symbols_no_index_returns_empty() {
    let server = test_server();
    // With no cache and no DB data, search_symbols returns empty (not an error)
    let params = json!({"name": "search_code", "arguments": {"mode": "text", "query": "foo"}});
    let resp = server.handle_request("tools/call", Some(&params), json!(300));
    let result = resp.result.unwrap();
    let text = result["content"][0]["text"].as_str().unwrap();
    assert!(
        text.contains("No matching symbols") || text == "[]",
        "Expected empty results, got: {text}"
    );
}

#[test]
fn get_symbol_info_not_found() {
    let server = test_server();
    // With no cache and no DB data, get_symbol_info returns "Symbol not found"
    let params = json!({"name": "get_symbol_info", "arguments": {"qualified_name": "foo::bar"}});
    let resp = server.handle_request("tools/call", Some(&params), json!(301));
    let result = resp.result.unwrap();
    assert_eq!(result["isError"], true);
    let text = result["content"][0]["text"].as_str().unwrap();
    assert!(
        text.contains("Symbol not found"),
        "Expected 'Symbol not found', got: {text}"
    );
}

#[test]
fn search_symbols_db_fallback() {
    let server = test_server();

    // Insert a sym:* node directly into storage (simulating persisted data)
    let mut payload = HashMap::new();
    payload.insert(
        "symbol_kind".to_string(),
        serde_json::Value::String("function".to_string()),
    );
    payload.insert(
        "signature".to_string(),
        serde_json::Value::String("fn hello_world()".to_string()),
    );
    payload.insert(
        "file_path".to_string(),
        serde_json::Value::String("src/lib.rs".to_string()),
    );
    payload.insert("line_start".to_string(), serde_json::json!(10));
    payload.insert("line_end".to_string(), serde_json::json!(20));
    payload.insert(
        "visibility".to_string(),
        serde_json::Value::String("public".to_string()),
    );

    let node = GraphNode {
        id: "sym:mymod::hello_world".to_string(),
        kind: NodeKind::Function,
        label: "mymod::hello_world".to_string(),
        payload,
        centrality: 0.0,
        memory_id: None,
        namespace: None,
    };
    server.engine.storage().insert_graph_node(&node).unwrap();

    // Cache is None — search_symbols should fall through to DB
    let results = server
        .engine
        .search_symbols("hello_world", 10, None)
        .unwrap();
    assert!(
        !results.is_empty(),
        "DB fallback should return results for hello_world"
    );
    assert_eq!(results[0].qualified_name, "mymod::hello_world");
    assert_eq!(results[0].kind, "function");

    // Verify cache was populated
    let cache = server.engine.lock_index_cache().unwrap();
    assert!(
        cache.is_some(),
        "Cache should be populated after DB fallback"
    );
    let syms = &cache.as_ref().unwrap().symbols;
    assert!(
        syms.iter()
            .any(|s| s.qualified_name == "mymod::hello_world"),
        "Cache should contain hello_world"
    );
}

#[test]
fn get_symbol_info_db_fallback() {
    let server = test_server();

    // Insert a sym:* node directly into storage
    let mut payload = HashMap::new();
    payload.insert(
        "symbol_kind".to_string(),
        serde_json::Value::String("struct".to_string()),
    );
    payload.insert(
        "signature".to_string(),
        serde_json::Value::String("pub struct Config".to_string()),
    );
    payload.insert(
        "file_path".to_string(),
        serde_json::Value::String("src/config.rs".to_string()),
    );
    payload.insert("line_start".to_string(), serde_json::json!(5));
    payload.insert("line_end".to_string(), serde_json::json!(25));
    payload.insert(
        "visibility".to_string(),
        serde_json::Value::String("public".to_string()),
    );
    payload.insert(
        "doc_comment".to_string(),
        serde_json::Value::String("Configuration struct".to_string()),
    );

    let node = GraphNode {
        id: "sym:mymod::Config".to_string(),
        kind: NodeKind::Class,
        label: "mymod::Config".to_string(),
        payload,
        centrality: 0.0,
        memory_id: None,
        namespace: None,
    };
    server.engine.storage().insert_graph_node(&node).unwrap();

    // Cache is None — get_symbol_info should fall through to DB
    let params =
        json!({"name": "get_symbol_info", "arguments": {"qualified_name": "mymod::Config"}});
    let resp = server.handle_request("tools/call", Some(&params), json!(311));
    let result = resp.result.unwrap();
    assert_ne!(result["isError"], true, "Should not be an error");
    let text = result["content"][0]["text"].as_str().unwrap();
    let parsed: Value = serde_json::from_str(text).unwrap();
    assert_eq!(parsed["qualified_name"], "mymod::Config");
    assert_eq!(parsed["kind"], "struct");
    assert_eq!(parsed["file_path"], "src/config.rs");
    assert_eq!(parsed["line_start"], 5);
    assert_eq!(parsed["doc_comment"], "Configuration struct");

    // Verify cache was populated
    let cache = server.engine.lock_index_cache().unwrap();
    assert!(cache.is_some());
    assert!(cache
        .as_ref()
        .unwrap()
        .symbols
        .iter()
        .any(|s| s.qualified_name == "mymod::Config"),);
}

#[test]
fn symbol_from_graph_node_roundtrip() {
    use codemem_engine::index::symbol::symbol_from_graph_node;

    let mut payload = HashMap::new();
    payload.insert(
        "symbol_kind".to_string(),
        serde_json::Value::String("field".to_string()),
    );
    payload.insert(
        "signature".to_string(),
        serde_json::Value::String("count: usize".to_string()),
    );
    payload.insert(
        "file_path".to_string(),
        serde_json::Value::String("src/counter.rs".to_string()),
    );
    payload.insert("line_start".to_string(), serde_json::json!(3));
    payload.insert("line_end".to_string(), serde_json::json!(3));
    payload.insert(
        "visibility".to_string(),
        serde_json::Value::String("public".to_string()),
    );
    payload.insert("is_async".to_string(), serde_json::json!(false));

    let node = GraphNode {
        id: "sym:Counter::count".to_string(),
        kind: NodeKind::Constant, // lossy mapping — Field -> Constant
        label: "Counter::count".to_string(),
        payload,
        centrality: 0.5,
        memory_id: None,
        namespace: Some("test".to_string()),
    };

    let sym = symbol_from_graph_node(&node).expect("Should reconstruct symbol");
    assert_eq!(sym.qualified_name, "Counter::count");
    assert_eq!(sym.name, "count");
    // Lossless: symbol_kind payload should give us Field, not Constant
    assert_eq!(sym.kind.to_string(), "field");
    assert_eq!(sym.file_path, "src/counter.rs");
    assert_eq!(sym.line_start, 3);
    assert_eq!(sym.visibility.to_string(), "public");
    assert!(!sym.is_async);
}

#[test]
fn get_clusters_empty_graph() {
    let server = test_server();
    let params = json!({"name": "find_related_groups", "arguments": {}});
    let resp = server.handle_request("tools/call", Some(&params), json!(302));
    let result = resp.result.unwrap();
    let text = result["content"][0]["text"].as_str().unwrap();
    let parsed: Value = serde_json::from_str(text).unwrap();
    assert_eq!(parsed["cluster_count"], 0);
}

#[test]
fn get_pagerank_empty_graph() {
    let server = test_server();
    let params = json!({"name": "find_important_nodes", "arguments": {}});
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

    server.engine.storage().insert_graph_node(&node_a).unwrap();
    server.engine.storage().insert_graph_node(&node_b).unwrap();
    {
        let mut graph = server.engine.lock_graph().unwrap();
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
        "name": "get_symbol_graph",
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
        server.engine.storage().insert_graph_node(&node).unwrap();
        server.engine.lock_graph().unwrap().add_node(node).unwrap();
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
        let mut graph = server.engine.lock_graph().unwrap();
        graph.add_edge(edge1).unwrap();
        graph.add_edge(edge2).unwrap();
    }

    let params = json!({"name": "find_important_nodes", "arguments": {"top_k": 3}});
    let resp = server.handle_request("tools/call", Some(&params), json!(311));
    let result = resp.result.unwrap();
    assert_eq!(result["isError"], false);
    let text = result["content"][0]["text"].as_str().unwrap();
    let parsed: Value = serde_json::from_str(text).unwrap();
    assert_eq!(parsed["results"].as_array().unwrap().len(), 3);
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

    server
        .engine
        .storage()
        .insert_graph_node(&file_node)
        .unwrap();
    server
        .engine
        .storage()
        .insert_graph_node(&func_node)
        .unwrap();
    server
        .engine
        .storage()
        .insert_graph_node(&chunk_node)
        .unwrap();
    {
        let mut graph = server.engine.lock_graph().unwrap();
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
        server.engine.storage().insert_graph_node(&node).unwrap();
        server.engine.lock_graph().unwrap().add_node(node).unwrap();
    }
    {
        let mut graph = server.engine.lock_graph().unwrap();
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
