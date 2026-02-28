//! Integration tests for codemem-mcp: full JSON-RPC lifecycle.
//!
//! All tests use the text-search fallback path (no ONNX embedding model).

use codemem_mcp::McpServer;
use serde_json::{json, Value};

// ── Helpers ────────────────────────────────────────────────────────────────

fn test_server() -> McpServer {
    McpServer::for_testing()
}

/// Call tools/call and return the response result value.
fn call_tool(server: &McpServer, tool_name: &str, arguments: Value) -> Value {
    let params = json!({"name": tool_name, "arguments": arguments});
    let resp = server.handle_request("tools/call", Some(&params), json!("req"));
    assert!(
        resp.error.is_none(),
        "Unexpected error calling {tool_name}: {:?}",
        resp.error
    );
    resp.result.unwrap()
}

/// Call tools/call and parse the text content from the result as JSON.
fn call_tool_parse(server: &McpServer, tool_name: &str, arguments: Value) -> Value {
    let result = call_tool(server, tool_name, arguments);
    let text = result["content"][0]["text"].as_str().unwrap();
    serde_json::from_str(text).unwrap_or_else(|_| Value::String(text.to_string()))
}

/// Store a memory and return the parsed JSON response.
fn store(server: &McpServer, content: &str, memory_type: &str, tags: &[&str]) -> Value {
    call_tool_parse(
        server,
        "store_memory",
        json!({
            "content": content,
            "memory_type": memory_type,
            "tags": tags,
        }),
    )
}

/// Recall memories and return the raw text from the response.
fn recall_text(server: &McpServer, query: &str) -> String {
    let result = call_tool(server, "recall_memory", json!({"query": query}));
    result["content"][0]["text"].as_str().unwrap().to_string()
}

// ── Full Lifecycle Tests ───────────────────────────────────────────────────

#[test]
fn full_lifecycle_initialize_store_associate_traverse_recall() {
    let server = test_server();

    // 1. Initialize
    let resp = server.handle_request("initialize", None, json!(1));
    assert!(resp.error.is_none());
    let result = resp.result.unwrap();
    assert_eq!(result["protocolVersion"], "2024-11-05");
    assert_eq!(result["serverInfo"]["name"], "codemem");

    // 2. Store two memories
    let mem_a = store(
        &server,
        "Rust ownership system prevents data races at compile time",
        "insight",
        &["rust", "safety"],
    );
    let id_a = mem_a["id"].as_str().unwrap();
    assert_eq!(mem_a["memory_type"], "insight");

    let mem_b = store(
        &server,
        "The borrow checker enforces exclusive mutable references",
        "pattern",
        &["rust", "borrowing"],
    );
    let id_b = mem_b["id"].as_str().unwrap();

    // 3. Associate: A -> B with LEADS_TO
    let assoc = call_tool_parse(
        &server,
        "associate_memories",
        json!({
            "source_id": id_a,
            "target_id": id_b,
            "relationship": "LEADS_TO",
            "weight": 0.9,
        }),
    );
    assert_eq!(assoc["relationship"], "LEADS_TO");
    assert_eq!(assoc["source"], id_a);
    assert_eq!(assoc["target"], id_b);

    // 4. Graph traverse from A (BFS, depth 2)
    let traverse = call_tool(
        &server,
        "graph_traverse",
        json!({"start_id": id_a, "max_depth": 2, "algorithm": "bfs"}),
    );
    let traverse_text = traverse["content"][0]["text"].as_str().unwrap();
    let nodes: Vec<Value> = serde_json::from_str(traverse_text).unwrap();
    // Should find at least the start node (A) and possibly B (connected via LEADS_TO)
    assert!(!nodes.is_empty());
    let node_ids: Vec<&str> = nodes.iter().filter_map(|n| n["id"].as_str()).collect();
    assert!(node_ids.contains(&id_a));
    assert!(node_ids.contains(&id_b));

    // 5. Recall memory via text search
    let recall_result = recall_text(&server, "rust ownership data races");
    assert!(recall_result.contains("ownership"));

    // 6. Verify stats reflect all stored data
    let stats = call_tool_parse(&server, "codemem_stats", json!({}));
    assert_eq!(stats["storage"]["memories"], 2);
    assert_eq!(stats["storage"]["graph_nodes"], 2);
    assert_eq!(stats["storage"]["graph_edges"], 1);
}

// ── Store and Delete Round-Trip ────────────────────────────────────────────

#[test]
fn store_delete_verify_gone() {
    let server = test_server();

    // Store a memory
    let stored = store(&server, "temporary memory to be deleted", "context", &[]);
    let id = stored["id"].as_str().unwrap();

    // Verify it exists via stats
    let stats = call_tool_parse(&server, "codemem_stats", json!({}));
    assert_eq!(stats["storage"]["memories"], 1);

    // Delete it
    let delete_result = call_tool(&server, "delete_memory", json!({"id": id}));
    let delete_text = delete_result["content"][0]["text"].as_str().unwrap();
    let parsed: Value = serde_json::from_str(delete_text).unwrap();
    assert_eq!(parsed["deleted"], true);

    // Verify it is gone via stats
    let stats_after = call_tool_parse(&server, "codemem_stats", json!({}));
    assert_eq!(stats_after["storage"]["memories"], 0);

    // Trying to recall it should find nothing
    let recall_result = recall_text(&server, "temporary memory deleted");
    assert_eq!(recall_result, "No matching memories found.");
}

// ── Error Path Tests ───────────────────────────────────────────────────────

#[test]
fn error_unknown_tool() {
    let server = test_server();
    let params = json!({"name": "nonexistent_tool", "arguments": {}});
    let resp = server.handle_request("tools/call", Some(&params), json!(1));
    let result = resp.result.unwrap();
    assert_eq!(result["isError"], true);
    let text = result["content"][0]["text"].as_str().unwrap();
    assert!(text.contains("Unknown tool"));
}

#[test]
fn error_unknown_method() {
    let server = test_server();
    let resp = server.handle_request("nonexistent/method", None, json!(1));
    assert!(resp.error.is_some());
    let err = resp.error.unwrap();
    assert_eq!(err.code, -32601);
    assert!(err.message.contains("Method not found"));
}

#[test]
fn error_missing_params_on_tools_call() {
    let server = test_server();
    let resp = server.handle_request("tools/call", None, json!(1));
    assert!(resp.error.is_some());
    let err = resp.error.unwrap();
    assert_eq!(err.code, -32602);
}

#[test]
fn error_store_memory_missing_content() {
    let server = test_server();
    let result = call_tool(&server, "store_memory", json!({}));
    assert_eq!(result["isError"], true);
    let text = result["content"][0]["text"].as_str().unwrap();
    assert!(text.contains("content"));
}

#[test]
fn error_store_memory_empty_content() {
    let server = test_server();
    let result = call_tool(&server, "store_memory", json!({"content": ""}));
    assert_eq!(result["isError"], true);
}

#[test]
fn error_recall_memory_missing_query() {
    let server = test_server();
    let result = call_tool(&server, "recall_memory", json!({}));
    assert_eq!(result["isError"], true);
}

#[test]
fn error_delete_nonexistent_memory() {
    let server = test_server();
    let result = call_tool(&server, "delete_memory", json!({"id": "does-not-exist"}));
    assert_eq!(result["isError"], true);
    let text = result["content"][0]["text"].as_str().unwrap();
    assert!(text.contains("not found"));
}

#[test]
fn error_associate_missing_params() {
    let server = test_server();

    // Missing target_id
    let result = call_tool(&server, "associate_memories", json!({"source_id": "a"}));
    assert_eq!(result["isError"], true);

    // Missing source_id
    let result = call_tool(&server, "associate_memories", json!({"target_id": "b"}));
    assert_eq!(result["isError"], true);
}

#[test]
fn error_graph_traverse_missing_start() {
    let server = test_server();
    let result = call_tool(&server, "graph_traverse", json!({}));
    assert_eq!(result["isError"], true);
}

#[test]
fn error_graph_traverse_unknown_algorithm() {
    let server = test_server();
    let mem = store(&server, "some memory content", "context", &[]);
    let id = mem["id"].as_str().unwrap();
    let result = call_tool(
        &server,
        "graph_traverse",
        json!({"start_id": id, "algorithm": "dijkstra"}),
    );
    assert_eq!(result["isError"], true);
    let text = result["content"][0]["text"].as_str().unwrap();
    assert!(text.contains("Unknown algorithm"));
}

// ── Health and Stats ───────────────────────────────────────────────────────

#[test]
fn health_check_reports_healthy() {
    let server = test_server();
    let health = call_tool_parse(&server, "codemem_health", json!({}));
    assert_eq!(health["healthy"], true);
    assert_eq!(health["storage"], "ok");
    assert_eq!(health["vector"], "ok");
    assert_eq!(health["graph"], "ok");
    // No embeddings in test mode
    assert_eq!(health["embeddings"], "not_configured");
}

#[test]
fn stats_empty_server() {
    let server = test_server();
    let stats = call_tool_parse(&server, "codemem_stats", json!({}));
    assert_eq!(stats["storage"]["memories"], 0);
    assert_eq!(stats["storage"]["embeddings"], 0);
    assert_eq!(stats["storage"]["graph_nodes"], 0);
    assert_eq!(stats["storage"]["graph_edges"], 0);
    assert_eq!(stats["vector"]["indexed"], 0);
    assert_eq!(stats["vector"]["dimensions"], 768);
    assert_eq!(stats["embeddings"]["available"], false);
}

// ── Tools List ─────────────────────────────────────────────────────────────

#[test]
fn tools_list_returns_all_38_tools() {
    let server = test_server();
    let resp = server.handle_request("tools/list", None, json!(1));
    let result = resp.result.unwrap();
    let tools = result["tools"].as_array().unwrap();
    assert_eq!(tools.len(), 38);

    let expected = [
        "store_memory",
        "recall_memory",
        "update_memory",
        "delete_memory",
        "associate_memories",
        "graph_traverse",
        "codemem_stats",
        "codemem_health",
        "index_codebase",
        "search_symbols",
        "get_symbol_info",
        "get_dependencies",
        "get_impact",
        "get_clusters",
        "get_cross_repo",
        "get_pagerank",
        "search_code",
        "set_scoring_weights",
        "export_memories",
        "import_memories",
        "recall_with_expansion",
        "list_namespaces",
        "namespace_stats",
        "delete_namespace",
        "consolidate_decay",
        "consolidate_creative",
        "consolidate_cluster",
        "consolidate_forget",
        "consolidation_status",
        "detect_patterns",
        "pattern_insights",
        "recall_with_impact",
        "get_decision_chain",
        "refine_memory",
        "split_memory",
        "merge_memories",
        "consolidate_summarize",
        "codemem_metrics",
    ];
    let names: Vec<&str> = tools.iter().filter_map(|t| t["name"].as_str()).collect();
    for expected_name in &expected {
        assert!(
            names.contains(expected_name),
            "Missing tool: {expected_name}"
        );
    }
}

// ── Ping ───────────────────────────────────────────────────────────────────

#[test]
fn ping_returns_success() {
    let server = test_server();
    let resp = server.handle_request("ping", None, json!(1));
    assert!(resp.error.is_none());
    assert!(resp.result.is_some());
}

// ── Update Memory ──────────────────────────────────────────────────────────

#[test]
fn update_memory_changes_content() {
    let server = test_server();

    let stored = store(&server, "original content here", "insight", &["test"]);
    let id = stored["id"].as_str().unwrap();

    // Update it
    let update_result = call_tool(
        &server,
        "update_memory",
        json!({"id": id, "content": "updated content here", "importance": 0.9}),
    );
    let text = update_result["content"][0]["text"].as_str().unwrap();
    let parsed: Value = serde_json::from_str(text).unwrap();
    assert_eq!(parsed["updated"], true);

    // Recall should find updated content
    let recall = recall_text(&server, "updated content");
    assert!(recall.contains("updated content here"));
}

// ── Duplicate Detection ────────────────────────────────────────────────────

#[test]
fn duplicate_content_detected() {
    let server = test_server();

    store(&server, "unique content for dedup test", "context", &[]);

    // Storing exact same content again should detect duplicate
    let result = call_tool(
        &server,
        "store_memory",
        json!({"content": "unique content for dedup test"}),
    );
    let text = result["content"][0]["text"].as_str().unwrap();
    assert!(text.contains("already exists"));
}

// ── Memory Type Filter in Recall ───────────────────────────────────────────

#[test]
fn recall_filters_by_memory_type() {
    let server = test_server();

    store(
        &server,
        "rust insight about ownership",
        "insight",
        &["rust"],
    );
    store(&server, "rust pattern about matching", "pattern", &["rust"]);
    store(
        &server,
        "rust decision to use result type",
        "decision",
        &["rust"],
    );

    // Filter for insight only
    let params =
        json!({"name": "recall_memory", "arguments": {"query": "rust", "memory_type": "insight"}});
    let resp = server.handle_request("tools/call", Some(&params), json!(1));
    let result = resp.result.unwrap();
    let text = result["content"][0]["text"].as_str().unwrap();
    let results: Vec<Value> = serde_json::from_str(text).unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0]["memory_type"], "insight");
}

// ── DFS Traversal ──────────────────────────────────────────────────────────

#[test]
fn graph_traverse_dfs() {
    let server = test_server();

    let mem_a = store(&server, "node A for DFS test", "context", &[]);
    let mem_b = store(&server, "node B for DFS test", "context", &[]);
    let id_a = mem_a["id"].as_str().unwrap();
    let id_b = mem_b["id"].as_str().unwrap();

    call_tool(
        &server,
        "associate_memories",
        json!({
            "source_id": id_a,
            "target_id": id_b,
            "relationship": "CONTAINS",
        }),
    );

    let result = call_tool(
        &server,
        "graph_traverse",
        json!({"start_id": id_a, "max_depth": 2, "algorithm": "dfs"}),
    );
    let text = result["content"][0]["text"].as_str().unwrap();
    let nodes: Vec<Value> = serde_json::from_str(text).unwrap();
    let node_ids: Vec<&str> = nodes.iter().filter_map(|n| n["id"].as_str()).collect();
    assert!(node_ids.contains(&id_a));
    assert!(node_ids.contains(&id_b));
}

// ── Multiple Associations ──────────────────────────────────────────────────

#[test]
fn multiple_associations_and_stats() {
    let server = test_server();

    let m1 = store(&server, "memory one about architecture", "decision", &[]);
    let m2 = store(&server, "memory two about patterns", "pattern", &[]);
    let m3 = store(&server, "memory three about style", "style", &[]);
    let id1 = m1["id"].as_str().unwrap();
    let id2 = m2["id"].as_str().unwrap();
    let id3 = m3["id"].as_str().unwrap();

    // Create two associations: 1->2 and 1->3
    call_tool(
        &server,
        "associate_memories",
        json!({
            "source_id": id1,
            "target_id": id2,
            "relationship": "RELATES_TO",
        }),
    );
    call_tool(
        &server,
        "associate_memories",
        json!({
            "source_id": id1,
            "target_id": id3,
            "relationship": "LEADS_TO",
        }),
    );

    // Verify stats
    let stats = call_tool_parse(&server, "codemem_stats", json!({}));
    assert_eq!(stats["storage"]["memories"], 3);
    assert_eq!(stats["storage"]["graph_nodes"], 3);
    assert_eq!(stats["storage"]["graph_edges"], 2);
    assert_eq!(stats["graph"]["nodes"], 3);
    assert_eq!(stats["graph"]["edges"], 2);
}

// ── from_db_path with tempfile ─────────────────────────────────────────────

#[test]
fn from_db_path_creates_persistent_server() {
    let dir = tempfile::TempDir::new().unwrap();
    let db_path = dir.path().join("test.db");

    let server = McpServer::from_db_path(&db_path).unwrap();

    // Store something
    store(
        &server,
        "persistent test memory",
        "context",
        &["persistent"],
    );

    // Verify
    let stats = call_tool_parse(&server, "codemem_stats", json!({}));
    assert_eq!(stats["storage"]["memories"], 1);
}
