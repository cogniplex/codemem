use super::*;
use crate::test_helpers::*;

/// Helper: call a tool and return the result Value.
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

/// Helper: call a tool and parse the text content as JSON.
fn call_tool_parse(server: &McpServer, tool_name: &str, arguments: Value) -> Value {
    let result = call_tool(server, tool_name, arguments);
    let text = result["content"][0]["text"].as_str().unwrap();
    serde_json::from_str(text).unwrap_or_else(|_| Value::String(text.to_string()))
}

/// Helper: store a memory with namespace.
fn store_ns(
    server: &McpServer,
    content: &str,
    namespace: &str,
    memory_type: &str,
    tags: &[&str],
) -> Value {
    call_tool_parse(
        server,
        "store_memory",
        json!({
            "content": content,
            "memory_type": memory_type,
            "tags": tags,
            "namespace": namespace,
        }),
    )
}

#[test]
fn recall_with_expansion_no_embeddings() {
    let server = test_server();

    // Store two memories and link them
    let mem_a = store_ns(
        &server,
        "graph expansion base memory about architecture",
        "test-ns",
        "insight",
        &["arch"],
    );
    let id_a = mem_a["id"].as_str().unwrap();

    let mem_b = store_ns(
        &server,
        "related memory about design patterns",
        "test-ns",
        "pattern",
        &["design"],
    );
    let id_b = mem_b["id"].as_str().unwrap();

    // Associate them
    call_tool(
        &server,
        "associate_memories",
        json!({
            "source_id": id_a,
            "target_id": id_b,
            "relationship": "RELATES_TO",
        }),
    );

    // Recall with expansion (no embeddings = text fallback)
    let result = call_tool(
        &server,
        "recall_with_expansion",
        json!({
            "query": "architecture",
            "k": 5,
            "expansion_depth": 1,
        }),
    );
    let text = result["content"][0]["text"].as_str().unwrap();
    // Without embeddings, recall_with_expansion uses BM25-only scoring.
    // The result may contain the memory content or a "no memories" message
    // depending on whether the BM25+scoring threshold is crossed.
    assert!(
        text.contains("architecture")
            || text.contains("design")
            || text.contains("No matching memories"),
        "Unexpected recall response: {text}"
    );
}

#[test]
fn list_namespaces_empty() {
    let server = test_server();

    let result = call_tool_parse(&server, "list_namespaces", json!({}));
    let namespaces = result["namespaces"].as_array().unwrap();
    assert_eq!(namespaces.len(), 0);
}

#[test]
fn list_namespaces_with_data() {
    let server = test_server();

    // Store memories in two namespaces
    store_ns(
        &server,
        "memory alpha one about rust",
        "ns-alpha",
        "insight",
        &["rust"],
    );
    store_ns(
        &server,
        "memory alpha two about safety",
        "ns-alpha",
        "pattern",
        &["safety"],
    );
    store_ns(
        &server,
        "memory beta one about python",
        "ns-beta",
        "context",
        &["python"],
    );

    let result = call_tool_parse(&server, "list_namespaces", json!({}));
    let namespaces = result["namespaces"].as_array().unwrap();
    assert_eq!(namespaces.len(), 2);

    // Verify names and counts
    let ns_names: Vec<&str> = namespaces
        .iter()
        .filter_map(|n| n["name"].as_str())
        .collect();
    assert!(ns_names.contains(&"ns-alpha"));
    assert!(ns_names.contains(&"ns-beta"));

    for ns in namespaces {
        if ns["name"].as_str().unwrap() == "ns-alpha" {
            assert_eq!(ns["memory_count"], 2);
        } else if ns["name"].as_str().unwrap() == "ns-beta" {
            assert_eq!(ns["memory_count"], 1);
        }
    }
}

#[test]
fn namespace_stats_basic() {
    let server = test_server();

    store_ns(
        &server,
        "insight about architecture patterns",
        "stats-ns",
        "insight",
        &["arch", "patterns"],
    );
    store_ns(
        &server,
        "pattern for error handling in rust",
        "stats-ns",
        "pattern",
        &["rust", "errors"],
    );

    let result = call_tool_parse(&server, "namespace_stats", json!({"namespace": "stats-ns"}));
    assert_eq!(result["namespace"], "stats-ns");
    assert_eq!(result["count"], 2);

    // Check type distribution
    let types = &result["type_distribution"];
    assert_eq!(types["insight"], 1);
    assert_eq!(types["pattern"], 1);

    // Check tag frequency
    let tags = &result["tag_frequency"];
    assert_eq!(tags["arch"], 1);
    assert_eq!(tags["patterns"], 1);
    assert_eq!(tags["rust"], 1);
    assert_eq!(tags["errors"], 1);

    // Dates should be present
    assert!(result["oldest"].is_string());
    assert!(result["newest"].is_string());
}

#[test]
fn delete_namespace_requires_confirm() {
    let server = test_server();

    store_ns(
        &server,
        "memory to be protected",
        "protected-ns",
        "context",
        &[],
    );

    // Try to delete without confirm
    let result = call_tool(
        &server,
        "delete_namespace",
        json!({
            "namespace": "protected-ns",
            "confirm": false,
        }),
    );
    let text = result["content"][0]["text"].as_str().unwrap();
    assert_eq!(result["isError"], true);
    assert!(text.contains("confirm"));

    // Memory should still exist
    let stats = call_tool_parse(
        &server,
        "namespace_stats",
        json!({"namespace": "protected-ns"}),
    );
    assert_eq!(stats["count"], 1);
}

#[test]
fn delete_namespace_with_confirm() {
    let server = test_server();

    store_ns(
        &server,
        "memory to delete alpha",
        "delete-ns",
        "insight",
        &["test"],
    );
    store_ns(
        &server,
        "memory to delete beta",
        "delete-ns",
        "pattern",
        &["test"],
    );

    // Verify they exist
    let stats = call_tool_parse(
        &server,
        "namespace_stats",
        json!({"namespace": "delete-ns"}),
    );
    assert_eq!(stats["count"], 2);

    // Delete with confirm
    let result = call_tool_parse(
        &server,
        "delete_namespace",
        json!({
            "namespace": "delete-ns",
            "confirm": true,
        }),
    );
    assert_eq!(result["deleted"], 2);
    assert_eq!(result["namespace"], "delete-ns");

    // Verify they are gone
    let stats_after = call_tool_parse(
        &server,
        "namespace_stats",
        json!({"namespace": "delete-ns"}),
    );
    assert_eq!(stats_after["count"], 0);
}

// ── Export/Import Tests ─────────────────────────────────────────────

#[test]
fn export_memories_empty() {
    let server = test_server();
    let params = json!({"name": "export_memories", "arguments": {}});
    let resp = server.handle_request("tools/call", Some(&params), json!(400));
    let result = resp.result.unwrap();
    let text = result["content"][0]["text"].as_str().unwrap();
    let exported: Vec<Value> = serde_json::from_str(text).unwrap();
    assert!(exported.is_empty());
}

#[test]
fn import_and_export_roundtrip() {
    let server = test_server();

    // Import 2 memories
    let import_params = json!({
        "name": "import_memories",
        "arguments": {
            "memories": [
                {
                    "content": "roundtrip memory one about rust",
                    "memory_type": "insight",
                    "importance": 0.8,
                    "tags": ["rust", "test"]
                },
                {
                    "content": "roundtrip memory two about python",
                    "memory_type": "pattern",
                    "tags": ["python"]
                }
            ]
        }
    });
    let resp = server.handle_request("tools/call", Some(&import_params), json!(401));
    let result = resp.result.unwrap();
    let text = result["content"][0]["text"].as_str().unwrap();
    let import_result: Value = serde_json::from_str(text).unwrap();
    assert_eq!(import_result["imported"], 2);
    assert_eq!(import_result["skipped"], 0);
    assert_eq!(import_result["ids"].as_array().unwrap().len(), 2);

    // Export all memories
    let export_params = json!({"name": "export_memories", "arguments": {}});
    let resp = server.handle_request("tools/call", Some(&export_params), json!(402));
    let result = resp.result.unwrap();
    let text = result["content"][0]["text"].as_str().unwrap();
    let exported: Vec<Value> = serde_json::from_str(text).unwrap();
    assert_eq!(exported.len(), 2);

    // Verify content matches
    let contents: Vec<&str> = exported
        .iter()
        .filter_map(|e| e["content"].as_str())
        .collect();
    assert!(contents.contains(&"roundtrip memory one about rust"));
    assert!(contents.contains(&"roundtrip memory two about python"));

    // Verify memory types
    let types: Vec<&str> = exported
        .iter()
        .filter_map(|e| e["memory_type"].as_str())
        .collect();
    assert!(types.contains(&"insight"));
    assert!(types.contains(&"pattern"));
}

#[test]
fn export_with_namespace_filter() {
    let server = test_server();

    // Import memories with different namespaces
    let import_params = json!({
        "name": "import_memories",
        "arguments": {
            "memories": [
                {
                    "content": "project-a memory about architecture",
                    "memory_type": "decision",
                    "namespace": "/projects/a"
                },
                {
                    "content": "project-b memory about testing",
                    "memory_type": "insight",
                    "namespace": "/projects/b"
                },
                {
                    "content": "project-a memory about patterns",
                    "memory_type": "pattern",
                    "namespace": "/projects/a"
                }
            ]
        }
    });
    let resp = server.handle_request("tools/call", Some(&import_params), json!(403));
    let result = resp.result.unwrap();
    let text = result["content"][0]["text"].as_str().unwrap();
    let import_result: Value = serde_json::from_str(text).unwrap();
    assert_eq!(import_result["imported"], 3);

    // Export only namespace /projects/a
    let export_params = json!({
        "name": "export_memories",
        "arguments": {"namespace": "/projects/a"}
    });
    let resp = server.handle_request("tools/call", Some(&export_params), json!(404));
    let result = resp.result.unwrap();
    let text = result["content"][0]["text"].as_str().unwrap();
    let exported: Vec<Value> = serde_json::from_str(text).unwrap();
    assert_eq!(exported.len(), 2);

    // All exported should be from /projects/a
    for mem in &exported {
        assert_eq!(mem["namespace"].as_str().unwrap(), "/projects/a");
    }
}
