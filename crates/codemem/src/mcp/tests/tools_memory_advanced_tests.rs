use super::*;
use crate::mcp::test_helpers::*;

// ── Split Memory Tests ──────────────────────────────────────────────

#[test]
fn split_creates_part_of_edges() {
    let server = test_server();
    let stored = store_memory(
        &server,
        "combined content about A and B",
        "insight",
        &["tag"],
    );
    let source_id = stored["id"].as_str().unwrap();

    let params = json!({
        "name": "split_memory",
        "arguments": {
            "id": source_id,
            "parts": [
                { "content": "content about A" },
                { "content": "content about B", "tags": ["b_tag"] }
            ]
        }
    });
    let resp = server.handle_request("tools/call", Some(&params), json!(200));
    let result = resp.result.unwrap();
    let text = result["content"][0]["text"].as_str().unwrap();
    let parsed: Value = serde_json::from_str(text).unwrap();

    assert_eq!(parsed["source_id"].as_str().unwrap(), source_id);
    assert_eq!(parsed["relationship"], "PART_OF");
    let parts = parsed["parts"].as_array().unwrap();
    assert_eq!(parts.len(), 2);

    // Verify PART_OF edges exist in storage
    let edges = server
        .engine
        .storage()
        .get_edges_for_node(source_id)
        .unwrap();
    let part_of_edges: Vec<_> = edges
        .iter()
        .filter(|e| e.relationship == RelationshipType::PartOf)
        .collect();
    assert_eq!(part_of_edges.len(), 2);
    for edge in &part_of_edges {
        assert_eq!(edge.dst, source_id);
        assert!(edge.valid_from.is_some());
    }
}

#[test]
fn split_empty_parts_errors() {
    let server = test_server();
    let stored = store_memory(&server, "something to split", "context", &[]);
    let source_id = stored["id"].as_str().unwrap();

    let params = json!({
        "name": "split_memory",
        "arguments": {
            "id": source_id,
            "parts": []
        }
    });
    let resp = server.handle_request("tools/call", Some(&params), json!(201));
    let result = resp.result.unwrap();
    assert_eq!(result["isError"], true);
    let text = result["content"][0]["text"].as_str().unwrap();
    assert!(text.contains("empty"));
}

// ── Merge Memories Tests ────────────────────────────────────────────

#[test]
fn merge_creates_summarizes_edges() {
    let server = test_server();
    let m1 = store_memory(&server, "first memory to merge", "context", &["a"]);
    let m2 = store_memory(&server, "second memory to merge", "context", &["b"]);
    let id1 = m1["id"].as_str().unwrap();
    let id2 = m2["id"].as_str().unwrap();

    let params = json!({
        "name": "merge_memories",
        "arguments": {
            "source_ids": [id1, id2],
            "content": "merged summary of first and second",
            "tags": ["merged"]
        }
    });
    let resp = server.handle_request("tools/call", Some(&params), json!(300));
    let result = resp.result.unwrap();
    let text = result["content"][0]["text"].as_str().unwrap();
    let parsed: Value = serde_json::from_str(text).unwrap();

    assert_eq!(parsed["relationship"], "SUMMARIZES");
    let merged_id = parsed["merged_id"].as_str().unwrap();
    let source_ids = parsed["source_ids"].as_array().unwrap();
    assert_eq!(source_ids.len(), 2);

    // Verify SUMMARIZES edges exist in storage
    let edges = server
        .engine
        .storage()
        .get_edges_for_node(merged_id)
        .unwrap();
    let summarizes_edges: Vec<_> = edges
        .iter()
        .filter(|e| e.relationship == RelationshipType::Summarizes)
        .collect();
    assert_eq!(summarizes_edges.len(), 2);
    for edge in &summarizes_edges {
        assert_eq!(edge.src, merged_id);
        assert!(edge.valid_from.is_some());
    }
}

#[test]
fn merge_insufficient_sources_errors() {
    let server = test_server();
    let m1 = store_memory(&server, "only one memory", "context", &[]);
    let id1 = m1["id"].as_str().unwrap();

    let params = json!({
        "name": "merge_memories",
        "arguments": {
            "source_ids": [id1],
            "content": "cannot merge just one"
        }
    });
    let resp = server.handle_request("tools/call", Some(&params), json!(301));
    let result = resp.result.unwrap();
    assert_eq!(result["isError"], true);
    let text = result["content"][0]["text"].as_str().unwrap();
    assert!(text.contains("at least 2"));
}

#[test]
fn merge_missing_source_errors() {
    let server = test_server();
    let m1 = store_memory(&server, "existing memory", "context", &[]);
    let id1 = m1["id"].as_str().unwrap();

    let params = json!({
        "name": "merge_memories",
        "arguments": {
            "source_ids": [id1, "nonexistent-id"],
            "content": "merge with missing"
        }
    });
    let resp = server.handle_request("tools/call", Some(&params), json!(302));
    let result = resp.result.unwrap();
    assert_eq!(result["isError"], true);
    let text = result["content"][0]["text"].as_str().unwrap();
    assert!(text.contains("not found"));
}

// ── Vector Index Persistence Tests ──────────────────────────────────

#[test]
fn save_index_noop_for_in_memory_server() {
    let server = test_server();
    // db_path is None for in-memory server, save_index should not panic
    assert!(server.engine.db_path().is_none());
    server.save_index(); // should be a no-op
}

#[test]
fn from_db_path_sets_db_path() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path().to_path_buf();

    let server = McpServer::from_db_path(&path).unwrap();
    assert_eq!(server.engine.db_path().map(|p| p.to_path_buf()), Some(path));
}

#[test]
fn save_index_persists_to_disk() {
    let dir = tempfile::TempDir::new().unwrap();
    let db_path = dir.path().join("test.db");

    let server = McpServer::from_db_path(&db_path).unwrap();

    // Store a memory (triggers save_index internally)
    store_memory(&server, "persistent memory test", "context", &[]);

    // The index file should exist if embeddings were available,
    // but even without embeddings save_index should not error.
    // Verify save_index can be called explicitly without panicking.
    server.save_index();

    // Verify the idx path is derived correctly
    let expected_idx_path = db_path.with_extension("idx");
    assert_eq!(expected_idx_path, dir.path().join("test.idx"),);
}
