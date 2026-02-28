//! Shared test helpers for the MCP server tests.

use crate::McpServer;
use serde_json::{json, Value};

/// Create a test server (in-memory, no embeddings).
pub(crate) fn test_server() -> McpServer {
    McpServer::for_testing()
}

/// Helper: store a memory and return the parsed JSON result.
pub(crate) fn store_memory(
    server: &McpServer,
    content: &str,
    memory_type: &str,
    tags: &[&str],
) -> Value {
    let store_params = json!({
        "name": "store_memory",
        "arguments": {
            "content": content,
            "memory_type": memory_type,
            "tags": tags,
        }
    });
    let resp = server.handle_request("tools/call", Some(&store_params), json!(1));
    let result = resp.result.unwrap();
    let text = result["content"][0]["text"].as_str().unwrap();
    serde_json::from_str(text).unwrap()
}

/// Helper: recall memories and return the parsed JSON (array or string).
pub(crate) fn recall_memories(
    server: &McpServer,
    query: &str,
    memory_type: Option<&str>,
) -> String {
    let mut arguments = json!({"query": query});
    if let Some(mt) = memory_type {
        arguments["memory_type"] = json!(mt);
    }
    let params = json!({"name": "recall_memory", "arguments": arguments});
    let resp = server.handle_request("tools/call", Some(&params), json!(2));
    let result = resp.result.unwrap();
    result["content"][0]["text"].as_str().unwrap().to_string()
}
