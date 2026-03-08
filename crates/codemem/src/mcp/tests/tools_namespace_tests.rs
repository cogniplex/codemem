use super::*;
use crate::mcp::test_helpers::*;

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

// ── delete_namespace with confirm=true succeeds ─────────────────────────────

#[test]
fn delete_namespace_confirm_true_succeeds() {
    let server = test_server();

    store_ns(
        &server,
        "memory to delete one",
        "del-ns",
        "insight",
        &["test"],
    );
    store_ns(
        &server,
        "memory to delete two",
        "del-ns",
        "pattern",
        &["test"],
    );

    // Verify they exist
    let stats = call_tool_parse(&server, "namespace_stats", json!({"namespace": "del-ns"}));
    assert_eq!(stats["count"], 2);

    // Delete with confirm=true
    let result = call_tool_parse(
        &server,
        "delete_namespace",
        json!({
            "namespace": "del-ns",
            "confirm": true,
        }),
    );
    assert_eq!(result["deleted"], 2);
    assert_eq!(result["namespace"], "del-ns");

    // Verify they are gone
    let stats_after = call_tool_parse(&server, "namespace_stats", json!({"namespace": "del-ns"}));
    assert_eq!(stats_after["count"], 0);
}

// ── delete_namespace with confirm=false is rejected ─────────────────────────

#[test]
fn delete_namespace_confirm_false_is_rejected() {
    let server = test_server();

    store_ns(
        &server,
        "memory that should stay",
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
    assert!(text.contains("confirm"), "Error should mention confirm");

    // Memory should still exist
    let stats = call_tool_parse(
        &server,
        "namespace_stats",
        json!({"namespace": "protected-ns"}),
    );
    assert_eq!(stats["count"], 1);
}

// ── delete_namespace for non-existent namespace ─────────────────────────────

#[test]
fn delete_namespace_nonexistent_returns_zero_deleted() {
    let server = test_server();

    // Deleting a namespace that has no memories should succeed with deleted=0
    let result = call_tool_parse(
        &server,
        "delete_namespace",
        json!({
            "namespace": "does-not-exist",
            "confirm": true,
        }),
    );
    assert_eq!(result["deleted"], 0);
    assert_eq!(result["namespace"], "does-not-exist");
}
