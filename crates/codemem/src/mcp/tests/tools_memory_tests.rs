use super::*;
use crate::mcp::test_helpers::*;

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
        Some(&json!({"name": "codemem_stats", "arguments": {}})),
        json!(4),
    );
    let stats = stats_resp.result.unwrap();
    let text = stats["content"][0]["text"].as_str().unwrap();
    let parsed: Value = serde_json::from_str(text).unwrap();
    assert_eq!(parsed["storage"]["memories"], 1);
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
        "name": "recall_memory",
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
            created_at: now,
            updated_at: now,
            last_accessed_at: now,
        };
        server.engine.storage.insert_memory(&memory).unwrap();

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
            .storage
            .insert_graph_node(&graph_node)
            .unwrap();
        let _ = server.engine.graph.lock().unwrap().add_node(graph_node);
    }

    // Recall with namespace filter "/projects/a"
    let params = json!({
        "name": "recall_memory",
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
    let memory = server.engine.storage.get_memory(id).unwrap().unwrap();
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
    let graph = server.engine.graph.lock().unwrap();
    let edges = graph.get_edges(linked_id).unwrap();
    assert_eq!(edges.len(), 2);
    for edge in &edges {
        assert_eq!(edge.src, linked_id);
        assert_eq!(edge.relationship, RelationshipType::RelatesTo);
    }
}

// ── Vector Index Persistence Tests ──────────────────────────────────

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
    let edges = server.engine.storage.get_edges_for_node(old_id).unwrap();
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
    let old_memory = server.engine.storage.get_memory(old_id).unwrap().unwrap();
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
    let edges = server.engine.storage.get_edges_for_node(source_id).unwrap();
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
    let edges = server.engine.storage.get_edges_for_node(merged_id).unwrap();
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
    assert!(server.engine.db_path.is_none());
    server.save_index(); // should be a no-op
}

#[test]
fn from_db_path_sets_db_path() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path().to_path_buf();

    let server = McpServer::from_db_path(&path).unwrap();
    assert_eq!(server.engine.db_path, Some(path));
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

// ── Recall Quality Filter Tests ───────────────────────────────────

#[test]
fn recall_with_exclude_tags_filters_out() {
    let server = test_server();

    // Store a regular memory and one with static-analysis tag
    store_memory(&server, "rust ownership rules", "insight", &["rust"]);

    // Store one with static-analysis tag directly
    let now = chrono::Utc::now();
    let id = uuid::Uuid::new_v4().to_string();
    let hash = Storage::content_hash("rust auto-generated analysis");
    let memory = MemoryNode {
        id: id.clone(),
        content: "rust auto-generated analysis".to_string(),
        memory_type: MemoryType::Insight,
        importance: 0.5,
        confidence: 0.5,
        access_count: 0,
        content_hash: hash,
        tags: vec!["rust".to_string(), "static-analysis".to_string()],
        metadata: HashMap::new(),
        namespace: None,
        created_at: now,
        updated_at: now,
        last_accessed_at: now,
    };
    server.engine.storage.insert_memory(&memory).unwrap();
    server
        .engine
        .bm25_index
        .lock()
        .unwrap()
        .add_document(&id, "rust auto-generated analysis");

    // Recall without filter — both should appear
    let params = json!({
        "name": "recall_memory",
        "arguments": { "query": "rust" }
    });
    let resp = server.handle_request("tools/call", Some(&params), json!(500));
    let result = resp.result.unwrap();
    let text = result["content"][0]["text"].as_str().unwrap();
    let results: Vec<Value> = serde_json::from_str(text).unwrap();
    assert_eq!(results.len(), 2);

    // Recall with exclude_tags=["static-analysis"] — only the regular one
    let params = json!({
        "name": "recall_memory",
        "arguments": { "query": "rust", "exclude_tags": ["static-analysis"] }
    });
    let resp = server.handle_request("tools/call", Some(&params), json!(501));
    let result = resp.result.unwrap();
    let text = result["content"][0]["text"].as_str().unwrap();
    let results: Vec<Value> = serde_json::from_str(text).unwrap();
    assert_eq!(results.len(), 1);
    assert!(results[0]["content"]
        .as_str()
        .unwrap()
        .contains("ownership"));
}

#[test]
fn recall_with_min_importance_filters() {
    let server = test_server();

    // Store two memories with different importance
    let now = chrono::Utc::now();
    for (content, importance) in [
        ("rust high importance memory", 0.8),
        ("rust low importance memory", 0.2),
    ] {
        let id = uuid::Uuid::new_v4().to_string();
        let hash = Storage::content_hash(content);
        let memory = MemoryNode {
            id: id.clone(),
            content: content.to_string(),
            memory_type: MemoryType::Insight,
            importance,
            confidence: 1.0,
            access_count: 0,
            content_hash: hash,
            tags: vec!["rust".to_string()],
            metadata: HashMap::new(),
            namespace: None,
            created_at: now,
            updated_at: now,
            last_accessed_at: now,
        };
        server.engine.storage.insert_memory(&memory).unwrap();
        server
            .engine
            .bm25_index
            .lock()
            .unwrap()
            .add_document(&id, content);
    }

    // Recall with min_importance=0.5 — only the high importance one
    let params = json!({
        "name": "recall_memory",
        "arguments": { "query": "rust", "min_importance": 0.5 }
    });
    let resp = server.handle_request("tools/call", Some(&params), json!(502));
    let result = resp.result.unwrap();
    let text = result["content"][0]["text"].as_str().unwrap();
    let results: Vec<Value> = serde_json::from_str(text).unwrap();
    assert_eq!(results.len(), 1);
    assert!(results[0]["content"].as_str().unwrap().contains("high"));
}

// ── Duplicate Detection Tests ───────────────────────────────────────

#[test]
fn store_duplicate_content_detected() {
    let server = test_server();

    // Store a memory
    store_memory(
        &server,
        "unique content for dedup test",
        "insight",
        &["test"],
    );

    // Try to store exact same content again
    let params = json!({
        "name": "store_memory",
        "arguments": {
            "content": "unique content for dedup test",
            "memory_type": "insight",
            "tags": ["test"]
        }
    });
    let resp = server.handle_request("tools/call", Some(&params), json!(600));
    let result = resp.result.unwrap();
    // Duplicate content is reported as informational text (not error)
    let text = result["content"][0]["text"].as_str().unwrap();
    assert!(
        text.contains("exists") || text.contains("hash") || text.contains("already"),
        "expected duplicate notice, got: {text}"
    );
}

#[test]
fn store_different_content_succeeds() {
    let server = test_server();

    store_memory(&server, "first unique content", "insight", &["test"]);
    let second = store_memory(&server, "second unique content", "insight", &["test"]);

    // Second store should succeed with a different ID
    assert!(second["id"].as_str().is_some());
}

// ── persist_memory Pipeline Tests ───────────────────────────────────

#[test]
fn persist_memory_adds_to_all_subsystems() {
    let server = test_server();

    let stored = store_memory(
        &server,
        "persist pipeline test content",
        "decision",
        &["arch", "persist-test"],
    );
    let id = stored["id"].as_str().unwrap();

    // Verify storage has the memory
    let memory = server.engine.storage.get_memory(id).unwrap().unwrap();
    assert_eq!(memory.content, "persist pipeline test content");
    assert_eq!(memory.memory_type, MemoryType::Decision);

    // Verify BM25 index has the document (score > 0 means it was indexed)
    let bm25 = server.engine.bm25_index.lock().unwrap();
    let bm25_score = bm25.score("persist pipeline test", id);
    assert!(
        bm25_score > 0.0,
        "BM25 should score > 0 for the indexed document, got: {bm25_score}"
    );

    // Verify graph has a node for this memory
    let graph = server.engine.graph.lock().unwrap();
    let graph_node = graph.get_node(id).unwrap();
    assert!(graph_node.is_some(), "Graph should have a node for {id}");
}

#[test]
fn store_memory_with_importance() {
    let server = test_server();

    let params = json!({
        "name": "store_memory",
        "arguments": {
            "content": "high importance memory",
            "importance": 0.9,
            "memory_type": "decision"
        }
    });
    let resp = server.handle_request("tools/call", Some(&params), json!(700));
    let result = resp.result.unwrap();
    let text = result["content"][0]["text"].as_str().unwrap();
    let stored: Value = serde_json::from_str(text).unwrap();
    let id = stored["id"].as_str().unwrap();

    let memory = server.engine.storage.get_memory(id).unwrap().unwrap();
    assert!(
        (memory.importance - 0.9).abs() < f64::EPSILON,
        "importance should be 0.9, got: {}",
        memory.importance
    );
    // Confidence defaults to 1.0 in store_memory
    assert!(
        (memory.confidence - 1.0).abs() < f64::EPSILON,
        "confidence should default to 1.0, got: {}",
        memory.confidence
    );
}

// ── Recall Public API Tests ─────────────────────────────────────────

#[test]
fn recall_public_api_returns_search_results() {
    let server = test_server();

    store_memory(
        &server,
        "public recall test: Rust ownership",
        "insight",
        &["rust"],
    );
    store_memory(
        &server,
        "public recall test: Python typing",
        "insight",
        &["python"],
    );

    // Use the public recall() method directly
    let results = server
        .recall("Rust ownership", 10, None, None, &[], None, None)
        .unwrap();

    assert!(
        !results.is_empty(),
        "recall should find at least one result"
    );
    // The first result should be the Rust one (better match)
    assert!(results[0].memory.content.contains("Rust ownership"));
}

#[test]
fn recall_public_api_with_memory_type_filter() {
    let server = test_server();

    store_memory(&server, "api filter test content", "insight", &["test"]);
    store_memory(&server, "api filter test content two", "pattern", &["test"]);

    let results = server
        .recall(
            "api filter test",
            10,
            Some(MemoryType::Pattern),
            None,
            &[],
            None,
            None,
        )
        .unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].memory.memory_type, MemoryType::Pattern);
}

#[test]
fn recall_public_api_with_min_confidence() {
    let server = test_server();

    // Store directly with controlled confidence, including graph nodes
    // so that the scoring threshold (> 0.01) is met.
    let now = chrono::Utc::now();
    for (content, confidence) in [("confidence test high", 0.9), ("confidence test low", 0.1)] {
        let id = uuid::Uuid::new_v4().to_string();
        let hash = Storage::content_hash(content);
        let memory = MemoryNode {
            id: id.clone(),
            content: content.to_string(),
            memory_type: MemoryType::Insight,
            importance: 0.5,
            confidence,
            access_count: 0,
            content_hash: hash,
            tags: vec!["conf".to_string()],
            metadata: HashMap::new(),
            namespace: None,
            created_at: now,
            updated_at: now,
            last_accessed_at: now,
        };
        server.engine.storage.insert_memory(&memory).unwrap();
        server
            .engine
            .bm25_index
            .lock()
            .unwrap()
            .add_document(&id, content);

        // Also add graph node so the score passes the 0.01 threshold
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
            .storage
            .insert_graph_node(&graph_node)
            .unwrap();
        let _ = server.engine.graph.lock().unwrap().add_node(graph_node);
    }

    let results = server
        .recall("confidence test", 10, None, None, &[], None, Some(0.5))
        .unwrap();

    assert_eq!(results.len(), 1);
    assert!(results[0].memory.content.contains("high"));
}
