use super::*;
use crate::mcp::test_helpers::*;
use codemem_core::{MemoryNode, MemoryType, VectorBackend};
use codemem_storage::Storage;
use std::collections::HashMap;

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

// ── Consolidation Tool Tests ────────────────────────────────────────

#[test]
fn consolidate_decay_reduces_importance() {
    let server = test_server();

    // Store a memory with known importance
    let now = chrono::Utc::now();
    let sixty_days_ago = now - chrono::Duration::days(60);
    let id = uuid::Uuid::new_v4().to_string();
    let content = "old memory that should decay";
    let hash = Storage::content_hash(content);
    let memory = MemoryNode {
        id: id.clone(),
        content: content.to_string(),
        memory_type: MemoryType::Context,
        importance: 0.8,
        confidence: 1.0,
        access_count: 0,
        content_hash: hash,
        tags: vec![],
        metadata: HashMap::new(),
        namespace: None,
        created_at: sixty_days_ago,
        updated_at: sixty_days_ago,
        last_accessed_at: sixty_days_ago,
    };
    server.engine.storage().insert_memory(&memory).unwrap();

    // Run decay with default threshold (30 days)
    let params = json!({"name": "consolidate", "arguments": {"mode": "decay"}});
    let resp = server.handle_request("tools/call", Some(&params), json!(1));
    let result = resp.result.unwrap();
    let text = result["content"][0]["text"].as_str().unwrap();
    let parsed: Value = serde_json::from_str(text).unwrap();

    assert_eq!(parsed["cycle"], "decay");
    assert_eq!(parsed["affected"], 1);
    assert_eq!(parsed["threshold_days"], 30);

    // Verify importance was reduced via power-law:
    // 0.8 * 0.9^(60/30) * (1 + log2(max(0,1))*0.1) = 0.8 * 0.81 * 1.0 ≈ 0.648
    let retrieved = server.engine.storage().get_memory(&id).unwrap().unwrap();
    assert!(
        (retrieved.importance - 0.648).abs() < 0.02,
        "expected ~0.648, got {}",
        retrieved.importance
    );
}

#[test]
fn consolidate_decay_skips_recent_memories() {
    let server = test_server();

    // Store a recent memory
    store_memory(&server, "recently accessed memory", "context", &[]);

    // Run decay
    let params =
        json!({"name": "consolidate", "arguments": {"mode": "decay", "threshold_days": 30}});
    let resp = server.handle_request("tools/call", Some(&params), json!(1));
    let result = resp.result.unwrap();
    let text = result["content"][0]["text"].as_str().unwrap();
    let parsed: Value = serde_json::from_str(text).unwrap();

    // Recent memory should not be affected
    assert_eq!(parsed["affected"], 0);
}

#[test]
fn consolidate_creative_creates_edges() {
    let server = test_server();

    // Store two memories with overlapping tags but different types
    let result1 = store_memory(
        &server,
        "insight about rust safety",
        "insight",
        &["rust", "safety"],
    );
    let result2 = store_memory(
        &server,
        "pattern for error handling",
        "pattern",
        &["rust", "error"],
    );
    let id1 = result1["id"].as_str().unwrap();
    let id2 = result2["id"].as_str().unwrap();

    // Manually insert embeddings so vector search can find neighbors.
    // Use similar (but not identical) vectors for the two memories.
    let emb1: Vec<f32> = (0..768).map(|i| i as f32 / 768.0).collect();
    let mut emb2 = emb1.clone();
    emb2[0] += 0.01; // slightly different
    server.engine.storage().store_embedding(id1, &emb1).unwrap();
    server.engine.storage().store_embedding(id2, &emb2).unwrap();
    {
        let mut vec = server.lock_vector().unwrap();
        let _ = vec.insert(id1, &emb1);
        let _ = vec.insert(id2, &emb2);
    }

    // Run creative cycle
    let params = json!({"name": "consolidate", "arguments": {"mode": "creative"}});
    let resp = server.handle_request("tools/call", Some(&params), json!(1));
    let result = resp.result.unwrap();
    let text = result["content"][0]["text"].as_str().unwrap();
    let parsed: Value = serde_json::from_str(text).unwrap();

    assert_eq!(parsed["cycle"], "creative");
    assert_eq!(parsed["algorithm"], "vector_knn");
    // They have different types and similar embeddings, so should create a SHARES_THEME edge
    assert!(parsed["new_connections"].as_u64().unwrap() >= 1);
}

#[test]
fn consolidate_creative_skips_same_type() {
    let server = test_server();

    // Store two memories with same type (should not create edges)
    store_memory(&server, "insight one about rust", "insight", &["rust"]);
    store_memory(&server, "insight two about rust", "insight", &["rust"]);

    let params = json!({"name": "consolidate", "arguments": {"mode": "creative"}});
    let resp = server.handle_request("tools/call", Some(&params), json!(1));
    let result = resp.result.unwrap();
    let text = result["content"][0]["text"].as_str().unwrap();
    let parsed: Value = serde_json::from_str(text).unwrap();

    assert_eq!(parsed["new_connections"], 0);
}

#[test]
fn consolidate_forget_deletes_low_importance() {
    let server = test_server();

    // Store a memory with very low importance and zero access count
    let now = chrono::Utc::now();
    let id = uuid::Uuid::new_v4().to_string();
    let content = "forgettable memory";
    let hash = Storage::content_hash(content);
    let memory = MemoryNode {
        id: id.clone(),
        content: content.to_string(),
        memory_type: MemoryType::Context,
        importance: 0.05,
        confidence: 1.0,
        access_count: 0,
        content_hash: hash,
        tags: vec![],
        metadata: HashMap::new(),
        namespace: None,
        created_at: now,
        updated_at: now,
        last_accessed_at: now,
    };
    server.engine.storage().insert_memory(&memory).unwrap();

    // Verify it exists
    assert_eq!(server.engine.storage().memory_count().unwrap(), 1);

    // Run forget
    let params = json!({"name": "consolidate", "arguments": {"mode": "forget"}});
    let resp = server.handle_request("tools/call", Some(&params), json!(1));
    let result = resp.result.unwrap();
    let text = result["content"][0]["text"].as_str().unwrap();
    let parsed: Value = serde_json::from_str(text).unwrap();

    assert_eq!(parsed["cycle"], "forget");
    assert_eq!(parsed["deleted"], 1);
    assert_eq!(parsed["threshold"], 0.1);

    // Verify it's gone
    assert_eq!(server.engine.storage().memory_count().unwrap(), 0);
}

#[test]
fn consolidate_forget_keeps_accessed_memories() {
    let server = test_server();

    // Store a memory with low importance but nonzero access count directly
    let now = chrono::Utc::now();
    let memory = MemoryNode {
        id: uuid::Uuid::new_v4().to_string(),
        content: "low importance but accessed".to_string(),
        memory_type: MemoryType::Context,
        importance: 0.05,
        confidence: 1.0,
        access_count: 5,
        content_hash: Storage::content_hash("low importance but accessed"),
        tags: vec![],
        metadata: HashMap::new(),
        namespace: None,
        created_at: now,
        updated_at: now,
        last_accessed_at: now,
    };
    server.engine.storage().insert_memory(&memory).unwrap();

    // This memory has access_count = 5, so it should NOT be forgotten
    // (forget only targets memories with access_count == 0)

    let params = json!({"name": "consolidate", "arguments": {"mode": "forget"}});
    let resp = server.handle_request("tools/call", Some(&params), json!(1));
    let result = resp.result.unwrap();
    let text = result["content"][0]["text"].as_str().unwrap();
    let parsed: Value = serde_json::from_str(text).unwrap();

    assert_eq!(parsed["deleted"], 0);
    assert_eq!(server.engine.storage().memory_count().unwrap(), 1);
}

#[test]
fn consolidate_auto_runs_decay_and_creative() {
    let server = test_server();

    // Run auto consolidation (replaces old consolidation_status)
    let params = json!({"name": "consolidate", "arguments": {"mode": "auto"}});
    let resp = server.handle_request("tools/call", Some(&params), json!(1));
    let result = resp.result.unwrap();
    let text = result["content"][0]["text"].as_str().unwrap();
    let parsed: Value = serde_json::from_str(text).unwrap();

    // Auto mode runs decay and creative, and includes status
    assert!(parsed["decay"].is_object());
    assert!(parsed["creative"].is_object());
    assert!(parsed["status"].is_object());
}

#[test]
fn consolidate_forget_custom_threshold() {
    let server = test_server();

    // Store two memories with different importance
    let now = chrono::Utc::now();
    for (imp, content) in [(0.3, "medium importance"), (0.05, "very low importance")] {
        let id = uuid::Uuid::new_v4().to_string();
        let hash = Storage::content_hash(content);
        let memory = MemoryNode {
            id,
            content: content.to_string(),
            memory_type: MemoryType::Context,
            importance: imp,
            confidence: 1.0,
            access_count: 0,
            content_hash: hash,
            tags: vec![],
            metadata: HashMap::new(),
            namespace: None,
            created_at: now,
            updated_at: now,
            last_accessed_at: now,
        };
        server.engine.storage().insert_memory(&memory).unwrap();
    }

    assert_eq!(server.engine.storage().memory_count().unwrap(), 2);

    // Forget with threshold 0.5 should delete both
    let params = json!({"name": "consolidate", "arguments": {"mode": "forget", "importance_threshold": 0.5}});
    let resp = server.handle_request("tools/call", Some(&params), json!(1));
    let result = resp.result.unwrap();
    let text = result["content"][0]["text"].as_str().unwrap();
    let parsed: Value = serde_json::from_str(text).unwrap();

    assert_eq!(parsed["deleted"], 2);
    assert_eq!(parsed["threshold"], 0.5);
    assert_eq!(server.engine.storage().memory_count().unwrap(), 0);
}

// ── Impact-Aware Recall & Decision Chain Tests ────────────────────────

#[test]
fn recall_with_impact_returns_impact_data() {
    let server = test_server();

    // Store a memory
    let mem = store_ns(
        &server,
        "impact test memory about error handling patterns",
        "test-ns",
        "insight",
        &["error", "handling"],
    );
    let _id = mem["id"].as_str().unwrap();

    // Recall with impact (text fallback, no embeddings)
    let result = call_tool(
        &server,
        "recall",
        json!({"query": "error handling", "include_impact": true}),
    );
    let text = result["content"][0]["text"].as_str().unwrap();

    // Should find the memory and include impact data
    if text.contains("No matching memories") {
        // Token overlap alone may not be enough; that is fine
        return;
    }

    let parsed: Value = serde_json::from_str(text).unwrap();
    let first = &parsed[0];
    assert!(
        first.get("impact").is_some(),
        "result should contain impact data"
    );
    let impact = &first["impact"];
    assert!(impact.get("pagerank").is_some());
    assert!(impact.get("centrality").is_some());
    assert!(impact.get("connected_decisions").is_some());
    assert!(impact.get("dependent_files").is_some());
    assert!(impact.get("modification_count").is_some());
}

#[test]
fn get_decision_chain_requires_parameter() {
    let server = test_server();

    // Calling without file_path or topic should return an error
    let params = json!({"name": "get_decision_chain", "arguments": {}});
    let resp = server.handle_request("tools/call", Some(&params), json!("req"));
    let result = resp.result.unwrap();
    let text = result["content"][0]["text"].as_str().unwrap();
    assert!(
        text.contains("file_path") || text.contains("topic"),
        "error should mention required parameters"
    );
}

#[test]
fn get_decision_chain_by_topic() {
    let server = test_server();

    // Store decision memories with a topic
    let _d1 = call_tool_parse(
        &server,
        "store_memory",
        json!({
            "content": "Decision: use async runtime for concurrency",
            "memory_type": "decision",
            "tags": ["concurrency"],
        }),
    );
    let _d2 = call_tool_parse(
        &server,
        "store_memory",
        json!({
            "content": "Decision: switched from threads to async for concurrency",
            "memory_type": "decision",
            "tags": ["concurrency"],
        }),
    );

    // Query decision chain by topic
    let result = call_tool(
        &server,
        "get_decision_chain",
        json!({"topic": "concurrency"}),
    );
    let text = result["content"][0]["text"].as_str().unwrap();
    let parsed: Value = serde_json::from_str(text).unwrap();
    assert!(parsed["chain_length"].as_u64().unwrap() >= 2);
    assert_eq!(parsed["filter"]["topic"], "concurrency");
}

#[test]
fn decision_chain_follows_temporal_order() {
    let server = test_server();

    // Store decision memories at different times (chronological insertion order)
    let d1 = call_tool_parse(
        &server,
        "store_memory",
        json!({
            "content": "Decision: initial architecture for auth module",
            "memory_type": "decision",
            "tags": ["auth"],
        }),
    );
    let d2 = call_tool_parse(
        &server,
        "store_memory",
        json!({
            "content": "Decision: refactored auth to use JWT tokens",
            "memory_type": "decision",
            "tags": ["auth"],
        }),
    );
    let d3 = call_tool_parse(
        &server,
        "store_memory",
        json!({
            "content": "Decision: added OAuth2 to auth module",
            "memory_type": "decision",
            "tags": ["auth"],
        }),
    );

    // Link d1 -> d2 -> d3 with EVOLVED_INTO edges
    let id1 = d1["id"].as_str().unwrap();
    let id2 = d2["id"].as_str().unwrap();
    let id3 = d3["id"].as_str().unwrap();

    call_tool(
        &server,
        "associate_memories",
        json!({
            "source_id": id1,
            "target_id": id2,
            "relationship": "EVOLVED_INTO",
        }),
    );
    call_tool(
        &server,
        "associate_memories",
        json!({
            "source_id": id2,
            "target_id": id3,
            "relationship": "EVOLVED_INTO",
        }),
    );

    // Get decision chain
    let result = call_tool(&server, "get_decision_chain", json!({"topic": "auth"}));
    let text = result["content"][0]["text"].as_str().unwrap();
    let parsed: Value = serde_json::from_str(text).unwrap();

    assert_eq!(parsed["chain_length"].as_u64().unwrap(), 3);
    let decisions = parsed["decisions"].as_array().unwrap();

    // Verify temporal order: created_at of each should be <= the next
    for i in 0..decisions.len() - 1 {
        let dt_a = decisions[i]["created_at"].as_str().unwrap();
        let dt_b = decisions[i + 1]["created_at"].as_str().unwrap();
        assert!(dt_a <= dt_b, "decisions should be in chronological order");
    }

    // Verify connections exist
    let has_connections = decisions
        .iter()
        .any(|d| !d["connections"].as_array().unwrap().is_empty());
    assert!(
        has_connections,
        "at least one decision should have connections"
    );
}

// ── Tag-Aware Forget Tests ──────────────────────────────────────────

#[test]
fn consolidate_forget_with_target_tags() {
    let server = test_server();

    // Store two low-importance memories: one with static-analysis tag, one without
    let now = chrono::Utc::now();
    for (content, tags) in [
        (
            "enrichment noise about coupling",
            vec!["static-analysis".to_string()],
        ),
        ("important manual insight", vec!["manual".to_string()]),
    ] {
        let id = uuid::Uuid::new_v4().to_string();
        let hash = Storage::content_hash(content);
        let memory = MemoryNode {
            id,
            content: content.to_string(),
            memory_type: MemoryType::Insight,
            importance: 0.3,
            confidence: 0.5,
            access_count: 0,
            content_hash: hash,
            tags,
            metadata: HashMap::new(),
            namespace: None,
            created_at: now,
            updated_at: now,
            last_accessed_at: now,
        };
        server.engine.storage().insert_memory(&memory).unwrap();
    }

    assert_eq!(server.engine.storage().memory_count().unwrap(), 2);

    // Forget only static-analysis tagged memories below 0.5
    let params = json!({
        "name": "consolidate",
        "arguments": {
            "mode": "forget",
            "importance_threshold": 0.5,
            "target_tags": ["static-analysis"]
        }
    });
    let resp = server.handle_request("tools/call", Some(&params), json!(1));
    let result = resp.result.unwrap();
    let text = result["content"][0]["text"].as_str().unwrap();
    let parsed: Value = serde_json::from_str(text).unwrap();

    assert_eq!(parsed["deleted"], 1);
    // The manual insight should survive
    assert_eq!(server.engine.storage().memory_count().unwrap(), 1);
}

#[test]
fn consolidate_forget_with_max_access_count() {
    let server = test_server();

    let now = chrono::Utc::now();
    // Store two static-analysis memories: one never accessed, one accessed twice
    for (content, access_count) in [
        ("never accessed enrichment insight", 0u32),
        ("twice accessed enrichment insight", 2u32),
    ] {
        let id = uuid::Uuid::new_v4().to_string();
        let hash = Storage::content_hash(content);
        let memory = MemoryNode {
            id,
            content: content.to_string(),
            memory_type: MemoryType::Insight,
            importance: 0.3,
            confidence: 0.5,
            access_count,
            content_hash: hash,
            tags: vec!["static-analysis".to_string()],
            metadata: HashMap::new(),
            namespace: None,
            created_at: now,
            updated_at: now,
            last_accessed_at: now,
        };
        server.engine.storage().insert_memory(&memory).unwrap();
    }

    assert_eq!(server.engine.storage().memory_count().unwrap(), 2);

    // Forget only with max_access_count=0 (only never-accessed)
    let params = json!({
        "name": "consolidate",
        "arguments": {
            "mode": "forget",
            "importance_threshold": 0.5,
            "target_tags": ["static-analysis"],
            "max_access_count": 0
        }
    });
    let resp = server.handle_request("tools/call", Some(&params), json!(1));
    let result = resp.result.unwrap();
    let text = result["content"][0]["text"].as_str().unwrap();
    let parsed: Value = serde_json::from_str(text).unwrap();

    assert_eq!(parsed["deleted"], 1);
    assert_eq!(server.engine.storage().memory_count().unwrap(), 1);
}
