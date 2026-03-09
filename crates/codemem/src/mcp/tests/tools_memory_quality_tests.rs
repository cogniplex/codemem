use super::*;
use crate::mcp::test_helpers::*;
use codemem_core::{GraphBackend, GraphNode, MemoryNode, NodeKind};
use std::collections::HashMap;

// ── Recall Quality Filter Tests ───────────────────────────────────

#[test]
fn recall_with_exclude_tags_filters_out() {
    let server = test_server();

    // Store a regular memory and one with static-analysis tag
    store_memory(&server, "rust ownership rules", "insight", &["rust"]);

    // Store one with static-analysis tag directly
    let mut memory = MemoryNode::test_default("rust auto-generated analysis");
    memory.memory_type = MemoryType::Insight;
    memory.confidence = 0.5;
    memory.tags = vec!["rust".to_string(), "static-analysis".to_string()];
    let id = memory.id.clone();
    server.engine.storage().insert_memory(&memory).unwrap();
    server
        .engine
        .lock_bm25()
        .unwrap()
        .add_document(&id, "rust auto-generated analysis");

    // Recall without filter — both should appear
    let params = json!({
        "name": "recall",
        "arguments": { "query": "rust" }
    });
    let resp = server.handle_request("tools/call", Some(&params), json!(500));
    let result = resp.result.unwrap();
    let text = result["content"][0]["text"].as_str().unwrap();
    let results: Vec<Value> = serde_json::from_str(text).unwrap();
    assert_eq!(results.len(), 2);

    // Recall with exclude_tags=["static-analysis"] — only the regular one
    let params = json!({
        "name": "recall",
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
    for (content, importance) in [
        ("rust high importance memory", 0.8),
        ("rust low importance memory", 0.2),
    ] {
        let mut memory = MemoryNode::test_default(content);
        memory.memory_type = MemoryType::Insight;
        memory.importance = importance;
        memory.tags = vec!["rust".to_string()];
        let id = memory.id.clone();
        server.engine.storage().insert_memory(&memory).unwrap();
        server
            .engine
            .lock_bm25()
            .unwrap()
            .add_document(&id, content);
    }

    // Recall with min_importance=0.5 — only the high importance one
    let params = json!({
        "name": "recall",
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
    let memory = server.engine.storage().get_memory(id).unwrap().unwrap();
    assert_eq!(memory.content, "persist pipeline test content");
    assert_eq!(memory.memory_type, MemoryType::Decision);

    // Verify BM25 index has the document (score > 0 means it was indexed)
    let bm25 = server.engine.lock_bm25().unwrap();
    let bm25_score = bm25.score("persist pipeline test", id);
    assert!(
        bm25_score > 0.0,
        "BM25 should score > 0 for the indexed document, got: {bm25_score}"
    );

    // Verify graph has a node for this memory
    let graph = server.engine.lock_graph().unwrap();
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

    let memory = server.engine.storage().get_memory(id).unwrap().unwrap();
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
    for (content, confidence) in [("confidence test high", 0.9), ("confidence test low", 0.1)] {
        let mut memory = MemoryNode::test_default(content);
        memory.memory_type = MemoryType::Insight;
        memory.confidence = confidence;
        memory.tags = vec!["conf".to_string()];
        let id = memory.id.clone();
        server.engine.storage().insert_memory(&memory).unwrap();
        server
            .engine
            .lock_bm25()
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
            .storage()
            .insert_graph_node(&graph_node)
            .unwrap();
        let _ = server.engine.lock_graph().unwrap().add_node(graph_node);
    }

    let results = server
        .recall("confidence test", 10, None, None, &[], None, Some(0.5))
        .unwrap();

    assert_eq!(results.len(), 1);
    assert!(results[0].memory.content.contains("high"));
}
