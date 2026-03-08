use super::*;
use crate::mcp::test_helpers::*;
use codemem_core::{GraphNode, MemoryNode, NodeKind};
use std::collections::HashMap;

#[test]
fn handle_tools_call_store() {
    let server = test_server();
    let params = json!({"name": "store_memory", "arguments": {"content": "test content"}});
    let resp = server.handle_request("tools/call", Some(&params), json!(3));
    assert!(resp.result.is_some());
    assert!(resp.error.is_none());

    // Verify it actually stored
    let stats_resp = server.handle_request(
        "tools/call",
        Some(&json!({"name": "codemem_status", "arguments": {"include": ["stats"]}})),
        json!(4),
    );
    let stats = stats_resp.result.unwrap();
    let text = stats["content"][0]["text"].as_str().unwrap();
    let parsed: Value = serde_json::from_str(text).unwrap();
    assert_eq!(parsed["stats"]["storage"]["memories"], 1);
}

#[test]
fn handle_store_and_recall() {
    let server = test_server();

    // Store a memory
    let store_params = json!({
        "name": "store_memory",
        "arguments": {
            "content": "Rust uses ownership and borrowing for memory safety",
            "memory_type": "insight",
            "tags": ["rust", "memory"]
        }
    });
    server.handle_request("tools/call", Some(&store_params), json!(1));

    // Recall it (text search fallback, no embeddings in test)
    let recall_params = json!({
        "name": "recall",
        "arguments": {"query": "rust memory safety"}
    });
    let resp = server.handle_request("tools/call", Some(&recall_params), json!(2));
    let result = resp.result.unwrap();
    let text = result["content"][0]["text"].as_str().unwrap();
    // Should find the memory via token overlap
    assert!(text.contains("ownership") || text.contains("rust"));
}

#[test]
fn handle_store_and_delete() {
    let server = test_server();

    // Store
    let store_params = json!({
        "name": "store_memory",
        "arguments": {"content": "delete me"}
    });
    let resp = server.handle_request("tools/call", Some(&store_params), json!(1));
    let result = resp.result.unwrap();
    let text = result["content"][0]["text"].as_str().unwrap();
    let stored: Value = serde_json::from_str(text).unwrap();
    let id = stored["id"].as_str().unwrap();

    // Delete
    let delete_params = json!({
        "name": "delete_memory",
        "arguments": {"id": id}
    });
    let resp = server.handle_request("tools/call", Some(&delete_params), json!(2));
    assert!(resp.error.is_none());
}

// ── Memory Type Filter Tests ────────────────────────────────────────

#[test]
fn recall_filters_by_memory_type() {
    let server = test_server();

    // Store memories of different types, all containing "rust"
    store_memory(&server, "rust ownership insight", "insight", &["rust"]);
    store_memory(&server, "rust pattern matching", "pattern", &["rust"]);
    store_memory(&server, "rust decision to use enums", "decision", &["rust"]);

    // Recall with type filter "insight"
    let text = recall_memories(&server, "rust", Some("insight"));
    let results: Vec<Value> = serde_json::from_str(&text).unwrap();

    // Should only contain the insight memory
    assert_eq!(results.len(), 1);
    assert_eq!(results[0]["memory_type"], "insight");
    assert!(results[0]["content"]
        .as_str()
        .unwrap()
        .contains("ownership"));
}

#[test]
fn recall_without_type_filter_returns_all() {
    let server = test_server();

    store_memory(&server, "rust ownership insight", "insight", &["rust"]);
    store_memory(&server, "rust pattern matching", "pattern", &["rust"]);

    // Recall without type filter
    let text = recall_memories(&server, "rust", None);
    let results: Vec<Value> = serde_json::from_str(&text).unwrap();

    // Should return both
    assert_eq!(results.len(), 2);
}

#[test]
fn recall_with_invalid_type_filter_returns_all() {
    let server = test_server();

    store_memory(&server, "rust ownership insight", "insight", &["rust"]);

    // An invalid memory_type string should be ignored (parsed as None)
    let text = recall_memories(&server, "rust", Some("nonexistent_type"));
    let results: Vec<Value> = serde_json::from_str(&text).unwrap();

    // Should return everything (no filter applied)
    assert_eq!(results.len(), 1);
}

#[test]
fn recall_with_type_filter_no_matches() {
    let server = test_server();

    store_memory(&server, "rust ownership insight", "insight", &["rust"]);

    // Filter for a type that has no matches in the content query
    let text = recall_memories(&server, "rust", Some("habit"));
    assert_eq!(text, "No matching memories found.");
}

// ── Namespace Filter Tests ────────────────────────────────────────

#[test]
fn recall_filters_by_namespace() {
    let server = test_server();

    // Store memories with different namespaces via direct storage
    let now = chrono::Utc::now();
    for (content, ns) in [
        ("rust ownership in project-a", Some("/projects/a")),
        ("rust borrowing in project-b", Some("/projects/b")),
        ("rust global memory no namespace", None),
    ] {
        let id = uuid::Uuid::new_v4().to_string();
        let hash = Storage::content_hash(content);
        let memory = MemoryNode {
            id: id.clone(),
            content: content.to_string(),
            memory_type: MemoryType::Insight,
            importance: 0.5,
            confidence: 1.0,
            access_count: 0,
            content_hash: hash,
            tags: vec!["rust".to_string()],
            metadata: HashMap::new(),
            namespace: ns.map(String::from),
            session_id: None,
            created_at: now,
            updated_at: now,
            last_accessed_at: now,
        };
        server.engine.storage().insert_memory(&memory).unwrap();

        // Add graph node so graph scoring works
        let graph_node = GraphNode {
            id: id.clone(),
            kind: NodeKind::Memory,
            label: content.to_string(),
            payload: HashMap::new(),
            centrality: 0.0,
            memory_id: Some(id),
            namespace: None,
        };
        server
            .engine
            .storage()
            .insert_graph_node(&graph_node)
            .unwrap();
        let _ = server.engine.lock_graph().unwrap().add_node(graph_node);
    }

    // Recall with namespace filter "/projects/a"
    let params = json!({
        "name": "recall",
        "arguments": {"query": "rust", "namespace": "/projects/a"}
    });
    let resp = server.handle_request("tools/call", Some(&params), json!(100));
    let result = resp.result.unwrap();
    let text = result["content"][0]["text"].as_str().unwrap();
    let results: Vec<Value> = serde_json::from_str(text).unwrap();

    assert_eq!(results.len(), 1);
    assert!(results[0]["content"]
        .as_str()
        .unwrap()
        .contains("project-a"));
}

#[test]
fn recall_without_namespace_returns_all() {
    let server = test_server();

    // Store memories in different namespaces
    store_memory(&server, "rust memory one", "context", &["rust"]);
    store_memory(&server, "rust memory two", "context", &["rust"]);

    // Recall without namespace filter returns all
    let text = recall_memories(&server, "rust", None);
    let results: Vec<Value> = serde_json::from_str(&text).unwrap();
    assert_eq!(results.len(), 2);
}

#[test]
fn store_memory_with_namespace() {
    let server = test_server();

    let params = json!({
        "name": "store_memory",
        "arguments": {
            "content": "namespaced memory content",
            "namespace": "/my/project"
        }
    });
    let resp = server.handle_request("tools/call", Some(&params), json!(200));
    let result = resp.result.unwrap();
    let text = result["content"][0]["text"].as_str().unwrap();
    let stored: Value = serde_json::from_str(text).unwrap();
    let id = stored["id"].as_str().unwrap();

    // Retrieve and verify namespace is set
    let memory = server.engine.storage().get_memory(id).unwrap().unwrap();
    assert_eq!(memory.namespace.as_deref(), Some("/my/project"));
}

#[test]
fn store_memory_with_links() {
    let server = test_server();

    // First store two memories to get node IDs
    let m1 = store_memory(&server, "target node one", "context", &[]);
    let m2 = store_memory(&server, "target node two", "context", &[]);
    let m1_id = m1["id"].as_str().unwrap();
    let m2_id = m2["id"].as_str().unwrap();

    // Store a new memory with links to the previous two
    let params = json!({
        "name": "store_memory",
        "arguments": {
            "content": "linked memory content",
            "links": [m1_id, m2_id]
        }
    });
    let resp = server.handle_request("tools/call", Some(&params), json!(305));
    let result = resp.result.unwrap();
    assert_eq!(result["isError"], false);
    let text = result["content"][0]["text"].as_str().unwrap();
    let stored: Value = serde_json::from_str(text).unwrap();
    let linked_id = stored["id"].as_str().unwrap();

    // Verify edges were created
    let graph = server.engine.lock_graph().unwrap();
    let edges = graph.get_edges(linked_id).unwrap();
    assert_eq!(edges.len(), 2);
    for edge in &edges {
        assert_eq!(edge.src, linked_id);
        assert_eq!(edge.relationship, RelationshipType::RelatesTo);
    }
}

// ── Refine Memory Tests ────────────────────────────────────────────

#[test]
fn refine_creates_evolved_into_edge() {
    let server = test_server();
    let stored = store_memory(&server, "original content", "insight", &["rust"]);
    let old_id = stored["id"].as_str().unwrap();

    let params = json!({
        "name": "refine_memory",
        "arguments": {
            "id": old_id,
            "content": "refined content with more detail",
            "tags": ["rust", "refined"]
        }
    });
    let resp = server.handle_request("tools/call", Some(&params), json!(100));
    let result = resp.result.unwrap();
    let text = result["content"][0]["text"].as_str().unwrap();
    let parsed: Value = serde_json::from_str(text).unwrap();

    assert_eq!(parsed["old_id"].as_str().unwrap(), old_id);
    assert_eq!(parsed["relationship"], "EVOLVED_INTO");
    let new_id = parsed["new_id"].as_str().unwrap();
    assert_ne!(old_id, new_id);

    // Verify EVOLVED_INTO edge exists in storage
    let edges = server.engine.storage().get_edges_for_node(old_id).unwrap();
    let evolved_edges: Vec<_> = edges
        .iter()
        .filter(|e| e.relationship == RelationshipType::EvolvedInto)
        .collect();
    assert_eq!(evolved_edges.len(), 1);
    assert_eq!(evolved_edges[0].src, old_id);
    assert_eq!(evolved_edges[0].dst, new_id);
    assert!(evolved_edges[0].valid_from.is_some());
}

#[test]
fn refine_preserves_old_memory() {
    let server = test_server();
    let stored = store_memory(&server, "keep this content", "decision", &["arch"]);
    let old_id = stored["id"].as_str().unwrap();

    let params = json!({
        "name": "refine_memory",
        "arguments": {
            "id": old_id,
            "content": "new version of the content"
        }
    });
    server.handle_request("tools/call", Some(&params), json!(101));

    // Old memory should still exist and be unchanged
    let old_memory = server.engine.storage().get_memory(old_id).unwrap().unwrap();
    assert_eq!(old_memory.content, "keep this content");
    assert_eq!(old_memory.memory_type, MemoryType::Decision);
}

#[test]
fn refine_nonexistent_errors() {
    let server = test_server();
    let params = json!({
        "name": "refine_memory",
        "arguments": {
            "id": "nonexistent-id",
            "content": "will fail"
        }
    });
    let resp = server.handle_request("tools/call", Some(&params), json!(102));
    let result = resp.result.unwrap();
    assert_eq!(result["isError"], true);
    let text = result["content"][0]["text"].as_str().unwrap();
    assert!(text.contains("not found"));
}

// ── MCP Parameter Validation Tests ──────────────────────────────────

#[test]
fn store_memory_with_importance_above_one() {
    let server = test_server();

    // importance > 1.0 is accepted (no clamping in MCP layer)
    let params = json!({
        "name": "store_memory",
        "arguments": {
            "content": "memory with high importance",
            "importance": 1.5
        }
    });
    let resp = server.handle_request("tools/call", Some(&params), json!(200));
    let result = resp.result.unwrap();
    // Should store successfully (no server-side validation clamp)
    let text = result["content"][0]["text"].as_str().unwrap();
    let stored: Value = serde_json::from_str(text).unwrap();
    assert!(stored["id"].is_string());
}

#[test]
fn store_memory_with_importance_below_zero() {
    let server = test_server();

    // importance < 0.0 is accepted (no clamping in MCP layer)
    let params = json!({
        "name": "store_memory",
        "arguments": {
            "content": "memory with negative importance",
            "importance": -0.5
        }
    });
    let resp = server.handle_request("tools/call", Some(&params), json!(201));
    let result = resp.result.unwrap();
    let text = result["content"][0]["text"].as_str().unwrap();
    let stored: Value = serde_json::from_str(text).unwrap();
    assert!(stored["id"].is_string());
}

#[test]
fn store_memory_with_empty_content_returns_error() {
    let server = test_server();

    let params = json!({
        "name": "store_memory",
        "arguments": {
            "content": ""
        }
    });
    let resp = server.handle_request("tools/call", Some(&params), json!(202));
    let result = resp.result.unwrap();
    assert_eq!(result["isError"], true);
    let text = result["content"][0]["text"].as_str().unwrap();
    assert!(
        text.contains("content"),
        "Error should mention content parameter"
    );
}

#[test]
fn recall_with_k_zero() {
    let server = test_server();

    store_memory(&server, "some rust memory", "insight", &["rust"]);

    // k=0 should return no results (zero requested)
    let params = json!({
        "name": "recall",
        "arguments": {"query": "rust", "k": 0}
    });
    let resp = server.handle_request("tools/call", Some(&params), json!(203));
    let result = resp.result.unwrap();
    let text = result["content"][0]["text"].as_str().unwrap();
    // With k=0, either no results or the "No matching memories" message
    let is_empty = text.contains("No matching memories")
        || serde_json::from_str::<Vec<Value>>(text)
            .map(|v| v.is_empty())
            .unwrap_or(false);
    assert!(is_empty, "k=0 should yield no results, got: {text}");
}

#[test]
fn recall_with_negative_k_uses_default() {
    let server = test_server();

    store_memory(&server, "recall negative k test", "context", &["test"]);

    // Negative k via JSON: as_u64() returns None for negative, so default (10) is used
    let params = json!({
        "name": "recall",
        "arguments": {"query": "recall negative", "k": -1}
    });
    let resp = server.handle_request("tools/call", Some(&params), json!(204));
    let result = resp.result.unwrap();
    // Should not error — negative k falls back to default
    assert!(
        result["isError"].is_null() || result["isError"] == false,
        "Negative k should not cause an error"
    );
}

#[test]
fn recall_with_empty_query_returns_error() {
    let server = test_server();

    let params = json!({
        "name": "recall",
        "arguments": {"query": ""}
    });
    let resp = server.handle_request("tools/call", Some(&params), json!(205));
    let result = resp.result.unwrap();
    assert_eq!(result["isError"], true);
    let text = result["content"][0]["text"].as_str().unwrap();
    assert!(
        text.contains("query"),
        "Error should mention query parameter"
    );
}
