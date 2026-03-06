use super::*;
use crate::mcp::test_helpers::*;
use serde_json::{json, Value};

/// Helper: call a tool and return the result Value.
fn call_tool(server: &McpServer, tool_name: &str, arguments: Value) -> Value {
    let params = json!({"name": tool_name, "arguments": arguments});
    let resp = server.handle_request("tools/call", Some(&params), json!("req"));
    assert!(
        resp.error.is_none(),
        "Unexpected JSON-RPC error calling {tool_name}: {:?}",
        resp.error
    );
    resp.result.unwrap()
}

/// Helper: call a tool and return the text content string.
fn call_tool_text(server: &McpServer, tool_name: &str, arguments: Value) -> String {
    let result = call_tool(server, tool_name, arguments);
    result["content"][0]["text"]
        .as_str()
        .unwrap()
        .to_string()
}

/// Helper: check if a tool result is flagged as an error.
fn is_tool_error(server: &McpServer, tool_name: &str, arguments: Value) -> bool {
    let result = call_tool(server, tool_name, arguments);
    result["isError"].as_bool().unwrap_or(false)
}

// ── enrich_security ─────────────────────────────────────────────────

#[test]
fn enrich_security_empty_graph_succeeds() {
    let server = test_server();
    let text = call_tool_text(&server, "enrich_security", json!({}));
    // Should succeed and return valid JSON (possibly with empty results)
    let parsed: Value = serde_json::from_str(&text).unwrap();
    assert!(parsed.is_object() || parsed.is_array());
}

#[test]
fn enrich_security_with_namespace() {
    let server = test_server();
    let text = call_tool_text(&server, "enrich_security", json!({"namespace": "test-ns"}));
    let parsed: Value = serde_json::from_str(&text).unwrap();
    assert!(parsed.is_object() || parsed.is_array());
}

// ── enrich_performance ──────────────────────────────────────────────

#[test]
fn enrich_performance_empty_graph_succeeds() {
    let server = test_server();
    let text = call_tool_text(&server, "enrich_performance", json!({}));
    let parsed: Value = serde_json::from_str(&text).unwrap();
    assert!(parsed.is_object() || parsed.is_array());
}

#[test]
fn enrich_performance_with_top_param() {
    let server = test_server();
    let text = call_tool_text(&server, "enrich_performance", json!({"top": 5}));
    let parsed: Value = serde_json::from_str(&text).unwrap();
    assert!(parsed.is_object() || parsed.is_array());
}

// ── enrich_codebase ─────────────────────────────────────────────────

#[test]
fn enrich_codebase_missing_path_returns_error() {
    let server = test_server();
    assert!(is_tool_error(&server, "enrich_codebase", json!({})));
}

#[test]
fn enrich_codebase_empty_path_returns_error() {
    let server = test_server();
    assert!(is_tool_error(&server, "enrich_codebase", json!({"path": ""})));
}

#[test]
fn enrich_codebase_nonexistent_path_returns_result() {
    let server = test_server();
    // enrich_codebase does not check path existence itself (unlike analyze_codebase);
    // individual enrichments may fail but the composite tool still returns JSON.
    let text = call_tool_text(
        &server,
        "enrich_codebase",
        json!({"path": "/nonexistent/path/abc123"}),
    );
    let parsed: Value = serde_json::from_str(&text).unwrap();
    assert!(parsed.is_object());
}

#[test]
fn enrich_codebase_selective_analyses() {
    let server = test_server();
    // Run only security analysis
    let text = call_tool_text(
        &server,
        "enrich_codebase",
        json!({"path": "/tmp", "analyses": ["security"]}),
    );
    let parsed: Value = serde_json::from_str(&text).unwrap();
    // Should have a "security" key but not "git" or "performance"
    assert!(parsed.get("security").is_some());
    assert!(parsed.get("git").is_none());
    assert!(parsed.get("performance").is_none());
}

// ── enrich_git_history ──────────────────────────────────────────────

#[test]
fn enrich_git_history_missing_path_returns_error() {
    let server = test_server();
    assert!(is_tool_error(&server, "enrich_git_history", json!({})));
}

#[test]
fn enrich_git_history_empty_path_returns_error() {
    let server = test_server();
    assert!(is_tool_error(
        &server,
        "enrich_git_history",
        json!({"path": ""})
    ));
}

#[test]
fn enrich_git_history_missing_path_message() {
    let server = test_server();
    let text = call_tool_text(&server, "enrich_git_history", json!({}));
    assert!(
        text.contains("path"),
        "error message should mention 'path', got: {text}"
    );
}

// ── analyze_codebase ────────────────────────────────────────────────

#[test]
fn analyze_codebase_missing_path_returns_error() {
    let server = test_server();
    assert!(is_tool_error(&server, "analyze_codebase", json!({})));
}

#[test]
fn analyze_codebase_empty_path_returns_error() {
    let server = test_server();
    assert!(is_tool_error(
        &server,
        "analyze_codebase",
        json!({"path": ""})
    ));
}

#[test]
fn analyze_codebase_nonexistent_path_returns_error() {
    let server = test_server();
    let text = call_tool_text(
        &server,
        "analyze_codebase",
        json!({"path": "/nonexistent/path/xyz987"}),
    );
    let result = call_tool(&server, "analyze_codebase", json!({"path": "/nonexistent/path/xyz987"}));
    let is_err = result["isError"].as_bool().unwrap_or(false);
    assert!(is_err, "should error for non-existent path");
    assert!(
        text.contains("does not exist"),
        "error should mention path does not exist, got: {text}"
    );
}

#[test]
fn analyze_codebase_missing_path_message() {
    let server = test_server();
    let text = call_tool_text(&server, "analyze_codebase", json!({}));
    assert!(
        text.contains("path"),
        "error message should mention 'path', got: {text}"
    );
}
