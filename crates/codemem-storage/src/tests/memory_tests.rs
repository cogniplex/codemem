use crate::Storage;
use codemem_core::{
    CodememError, Edge, GraphNode, MemoryNode, MemoryType, NodeKind, RelationshipType,
};
use std::collections::HashMap;

fn test_memory() -> MemoryNode {
    let mut m = MemoryNode::new("Test memory content", MemoryType::Context);
    m.importance = 0.7;
    m.tags = vec!["test".to_string()];
    m
}

#[test]
fn insert_and_get_memory() {
    let storage = Storage::open_in_memory().unwrap();
    let memory = test_memory();
    storage.insert_memory(&memory).unwrap();

    let retrieved = storage.get_memory(&memory.id).unwrap().unwrap();
    assert_eq!(retrieved.id, memory.id);
    assert_eq!(retrieved.content, memory.content);
    assert_eq!(retrieved.access_count, 1); // bumped on get
}

#[test]
fn dedup_by_content_hash() {
    let storage = Storage::open_in_memory().unwrap();
    let m1 = test_memory();
    storage.insert_memory(&m1).unwrap();

    let mut m2 = test_memory();
    m2.id = uuid::Uuid::new_v4().to_string();
    m2.content_hash = m1.content_hash.clone(); // same hash

    assert!(matches!(
        storage.insert_memory(&m2),
        Err(CodememError::Duplicate(_))
    ));
}

#[test]
fn delete_memory() {
    let storage = Storage::open_in_memory().unwrap();
    let memory = test_memory();
    storage.insert_memory(&memory).unwrap();
    assert!(storage.delete_memory(&memory.id).unwrap());
    assert!(storage.get_memory(&memory.id).unwrap().is_none());
}

// ── Test #4: Cascade delete atomicity ───────────────────────────────

#[test]
fn cascade_delete_removes_all_related_data() {
    let storage = Storage::open_in_memory().unwrap();
    let memory = test_memory();
    let memory_id = memory.id.clone();
    storage.insert_memory(&memory).unwrap();

    // Insert graph node linked to this memory
    let node = GraphNode {
        id: format!("node-for-{memory_id}"),
        kind: NodeKind::Memory,
        label: "test node".to_string(),
        payload: HashMap::new(),
        centrality: 0.0,
        memory_id: Some(memory_id.clone()),
        namespace: None,
        valid_from: None,
        valid_to: None,
    };
    storage.insert_graph_node(&node).unwrap();

    // Insert a second graph node as edge target (FK constraint requires it)
    let target_node = GraphNode {
        id: "sym:SomeFunc".to_string(),
        kind: NodeKind::Function,
        label: "SomeFunc".to_string(),
        payload: HashMap::new(),
        centrality: 0.0,
        memory_id: None,
        namespace: None,
        valid_from: None,
        valid_to: None,
    };
    storage.insert_graph_node(&target_node).unwrap();

    // Insert edge referencing that graph node
    let now = chrono::Utc::now();
    let edge = Edge {
        id: format!("edge-for-{memory_id}"),
        src: node.id.clone(),
        dst: "sym:SomeFunc".to_string(),
        relationship: RelationshipType::RelatesTo,
        weight: 0.5,
        properties: HashMap::new(),
        created_at: now,
        valid_from: None,
        valid_to: None,
    };
    storage.insert_graph_edge(&edge).unwrap();

    // Insert embedding
    let fake_embedding = vec![0.1_f32; 768];
    storage
        .store_embedding(&memory_id, &fake_embedding)
        .unwrap();

    // Verify everything exists before cascade delete
    assert!(storage.get_memory_no_touch(&memory_id).unwrap().is_some());
    assert!(storage.get_graph_node(&node.id).unwrap().is_some());
    assert!(storage.get_embedding(&memory_id).unwrap().is_some());

    // Cascade delete
    let deleted = storage.delete_memory_cascade(&memory_id).unwrap();
    assert!(
        deleted,
        "cascade delete should return true for existing memory"
    );

    // Verify everything is gone
    assert!(
        storage.get_memory_no_touch(&memory_id).unwrap().is_none(),
        "memory should be deleted"
    );
    assert!(
        storage.get_graph_node(&node.id).unwrap().is_none(),
        "graph node should be deleted"
    );
    assert!(
        storage.get_embedding(&memory_id).unwrap().is_none(),
        "embedding should be deleted"
    );

    // Second call returns false (already deleted)
    let deleted_again = storage.delete_memory_cascade(&memory_id).unwrap();
    assert!(
        !deleted_again,
        "cascade delete should return false for already-deleted memory"
    );
}

// ── Namespace dedup tests ───────────────────────────────────────────

#[test]
fn dedup_null_namespace_coalesce() {
    // Two inserts with same content_hash but NULL namespace should dedup
    let storage = Storage::open_in_memory().unwrap();
    let m1 = test_memory(); // namespace = None
    storage.insert_memory(&m1).unwrap();

    let mut m2 = test_memory(); // same content, same hash, namespace = None
    m2.id = uuid::Uuid::new_v4().to_string();
    // m2 has the same content_hash as m1 because test_memory() uses the same content

    assert!(
        matches!(storage.insert_memory(&m2), Err(CodememError::Duplicate(_))),
        "Same content_hash with NULL namespace should be treated as duplicate"
    );
}

#[test]
fn same_hash_different_namespaces_both_succeed() {
    let storage = Storage::open_in_memory().unwrap();
    let content = "identical content for ns test";

    let mut m1 = MemoryNode::new(content, MemoryType::Context);
    m1.namespace = Some("project-a".to_string());

    let mut m2 = MemoryNode::new(content, MemoryType::Context);
    m2.namespace = Some("project-b".to_string());

    storage.insert_memory(&m1).unwrap();
    storage.insert_memory(&m2).unwrap();

    // Both should exist
    assert!(storage.get_memory(&m1.id).unwrap().is_some());
    assert!(storage.get_memory(&m2.id).unwrap().is_some());
}

// ── update_memory tests ─────────────────────────────────────────────

#[test]
fn update_memory_content_and_importance() {
    let storage = Storage::open_in_memory().unwrap();
    let memory = test_memory();
    let id = memory.id.clone();
    storage.insert_memory(&memory).unwrap();

    storage
        .update_memory(&id, "Updated content", Some(0.9))
        .unwrap();

    let updated = storage.get_memory(&id).unwrap().unwrap();
    assert_eq!(updated.content, "Updated content");
    assert!((updated.importance - 0.9).abs() < f64::EPSILON);
    assert_eq!(
        updated.content_hash,
        Storage::content_hash("Updated content")
    );
}

#[test]
fn update_memory_content_only() {
    let storage = Storage::open_in_memory().unwrap();
    let memory = test_memory();
    let id = memory.id.clone();
    let original_importance = memory.importance;
    storage.insert_memory(&memory).unwrap();

    storage.update_memory(&id, "New content", None).unwrap();

    let updated = storage.get_memory(&id).unwrap().unwrap();
    assert_eq!(updated.content, "New content");
    assert!(
        (updated.importance - original_importance).abs() < f64::EPSILON,
        "Importance should remain unchanged when None is passed"
    );
}

#[test]
fn update_memory_nonexistent_returns_not_found() {
    let storage = Storage::open_in_memory().unwrap();
    let result = storage.update_memory("nonexistent-id", "content", None);
    assert!(
        matches!(result, Err(CodememError::NotFound(_))),
        "Updating a non-existent memory should return NotFound"
    );
}

// ── delete non-existent memory ──────────────────────────────────────

#[test]
fn delete_nonexistent_memory_returns_false() {
    let storage = Storage::open_in_memory().unwrap();
    let result = storage.delete_memory("does-not-exist").unwrap();
    assert!(
        !result,
        "Deleting a non-existent memory should return false"
    );
}

// ── Memory Expiration Tests ─────────────────────────────────────────

#[test]
fn insert_memory_with_expires_at() {
    let storage = Storage::open_in_memory().unwrap();
    let mut m = test_memory();
    let future = chrono::Utc::now() + chrono::Duration::hours(24);
    m.expires_at = Some(future);
    storage.insert_memory(&m).unwrap();

    let retrieved = storage.get_memory(&m.id).unwrap().unwrap();
    assert!(retrieved.expires_at.is_some());
    // Timestamps lose sub-second precision in SQLite (stored as epoch seconds)
    assert_eq!(
        retrieved.expires_at.unwrap().timestamp(),
        future.timestamp()
    );
}

#[test]
fn insert_memory_without_expires_at() {
    let storage = Storage::open_in_memory().unwrap();
    let m = test_memory();
    storage.insert_memory(&m).unwrap();

    let retrieved = storage.get_memory(&m.id).unwrap().unwrap();
    assert!(retrieved.expires_at.is_none());
}

#[test]
fn delete_expired_memories_removes_past() {
    let storage = Storage::open_in_memory().unwrap();

    // Memory that expired 1 hour ago
    let mut expired = test_memory();
    expired.id = "expired-1".to_string();
    expired.content = "expired content unique".to_string();
    expired.content_hash = codemem_core::content_hash(&expired.content);
    expired.expires_at = Some(chrono::Utc::now() - chrono::Duration::hours(1));
    storage.insert_memory(&expired).unwrap();

    // Memory that expires in 24 hours (should survive)
    let mut future = test_memory();
    future.id = "future-1".to_string();
    future.content = "future content unique".to_string();
    future.content_hash = codemem_core::content_hash(&future.content);
    future.expires_at = Some(chrono::Utc::now() + chrono::Duration::hours(24));
    storage.insert_memory(&future).unwrap();

    // Memory with no expiry (should survive)
    let mut permanent = test_memory();
    permanent.id = "permanent-1".to_string();
    permanent.content = "permanent content unique".to_string();
    permanent.content_hash = codemem_core::content_hash(&permanent.content);
    storage.insert_memory(&permanent).unwrap();

    let deleted = storage.delete_expired_memories().unwrap();
    assert_eq!(deleted, 1);

    // Expired memory is gone
    assert!(storage.get_memory_no_touch("expired-1").unwrap().is_none());
    // Future and permanent memories survive
    assert!(storage.get_memory_no_touch("future-1").unwrap().is_some());
    assert!(storage
        .get_memory_no_touch("permanent-1")
        .unwrap()
        .is_some());
}

#[test]
fn list_memories_filtered_excludes_expired() {
    use codemem_core::StorageBackend;

    let storage = Storage::open_in_memory().unwrap();

    let mut active = test_memory();
    active.namespace = Some("ns1".to_string());
    storage.insert_memory(&active).unwrap();

    let mut expired = test_memory();
    expired.id = uuid::Uuid::new_v4().to_string();
    expired.content = "expired filtered content".to_string();
    expired.content_hash = codemem_core::content_hash(&expired.content);
    expired.namespace = Some("ns1".to_string());
    expired.expires_at = Some(chrono::Utc::now() - chrono::Duration::hours(1));
    storage.insert_memory(&expired).unwrap();

    let results = storage.list_memories_filtered(Some("ns1"), None).unwrap();
    assert_eq!(
        results.len(),
        1,
        "Expired memory should be excluded from list_memories_filtered"
    );
    assert_eq!(results[0].id, active.id);
}

#[test]
fn expire_memories_for_file_via_memory_id_and_relates_to() {
    let storage = Storage::open_in_memory().unwrap();

    // 1. Memory linked via graph_nodes.memory_id (primary link)
    let mut mem_primary = test_memory();
    mem_primary.id = "mem-primary".to_string();
    mem_primary.content = "primary linked enrichment aaa".to_string();
    mem_primary.content_hash = codemem_core::content_hash(&mem_primary.content);
    mem_primary.tags = vec!["static-analysis".to_string()];
    storage.insert_memory(&mem_primary).unwrap();

    let primary_node = GraphNode {
        id: "sym:foo::bar".to_string(),
        kind: NodeKind::Function,
        label: "bar".to_string(),
        payload: {
            let mut m = HashMap::new();
            m.insert("file_path".to_string(), serde_json::json!("src/lib.rs"));
            m
        },
        centrality: 0.0,
        memory_id: Some("mem-primary".to_string()),
        namespace: None,
        valid_from: None,
        valid_to: None,
    };
    storage.insert_graph_node(&primary_node).unwrap();

    // 2. SCIP doc memory linked via RELATES_TO edge (secondary link)
    let mut mem_edge = test_memory();
    mem_edge.id = "mem-edge-linked".to_string();
    mem_edge.content = "edge linked scip doc bbb".to_string();
    mem_edge.content_hash = codemem_core::content_hash(&mem_edge.content);
    mem_edge.tags = vec!["static-analysis".to_string()];
    storage.insert_memory(&mem_edge).unwrap();

    let edge_node = GraphNode {
        id: "sym:foo::baz".to_string(),
        kind: NodeKind::Function,
        label: "baz".to_string(),
        payload: {
            let mut m = HashMap::new();
            m.insert("file_path".to_string(), serde_json::json!("src/lib.rs"));
            m
        },
        centrality: 0.0,
        memory_id: None,
        namespace: None,
        valid_from: None,
        valid_to: None,
    };
    storage.insert_graph_node(&edge_node).unwrap();

    // Need a graph node for the mem: target (FK constraint on edges)
    let mem_node = GraphNode {
        id: "mem:mem-edge-linked".to_string(),
        kind: NodeKind::Memory,
        label: "edge linked scip doc".to_string(),
        payload: HashMap::new(),
        centrality: 0.0,
        memory_id: Some("mem-edge-linked".to_string()),
        namespace: None,
        valid_from: None,
        valid_to: None,
    };
    storage.insert_graph_node(&mem_node).unwrap();

    // RELATES_TO edge from symbol to mem:
    let edge = Edge {
        id: "edge-relates".to_string(),
        src: "sym:foo::baz".to_string(),
        dst: "mem:mem-edge-linked".to_string(),
        relationship: RelationshipType::RelatesTo,
        weight: 1.0,
        properties: HashMap::new(),
        created_at: chrono::Utc::now(),
        valid_from: None,
        valid_to: None,
    };
    storage.insert_graph_edge(&edge).unwrap();

    // 3. Memory for a DIFFERENT file (should NOT be expired)
    let mut mem_other = test_memory();
    mem_other.id = "mem-other-file".to_string();
    mem_other.content = "other file enrichment ccc".to_string();
    mem_other.content_hash = codemem_core::content_hash(&mem_other.content);
    mem_other.tags = vec!["static-analysis".to_string()];
    storage.insert_memory(&mem_other).unwrap();

    let other_node = GraphNode {
        id: "sym:other::func".to_string(),
        kind: NodeKind::Function,
        label: "func".to_string(),
        payload: {
            let mut m = HashMap::new();
            m.insert("file_path".to_string(), serde_json::json!("src/other.rs"));
            m
        },
        centrality: 0.0,
        memory_id: Some("mem-other-file".to_string()),
        namespace: None,
        valid_from: None,
        valid_to: None,
    };
    storage.insert_graph_node(&other_node).unwrap();

    // 4. Memory WITHOUT static-analysis tag (should NOT be expired)
    let mut mem_no_tag = test_memory();
    mem_no_tag.id = "mem-no-tag".to_string();
    mem_no_tag.content = "no tag enrichment ddd".to_string();
    mem_no_tag.content_hash = codemem_core::content_hash(&mem_no_tag.content);
    mem_no_tag.tags = vec!["user-created".to_string()];
    storage.insert_memory(&mem_no_tag).unwrap();

    let no_tag_node = GraphNode {
        id: "sym:foo::no_tag".to_string(),
        kind: NodeKind::Function,
        label: "no_tag".to_string(),
        payload: {
            let mut m = HashMap::new();
            m.insert("file_path".to_string(), serde_json::json!("src/lib.rs"));
            m
        },
        centrality: 0.0,
        memory_id: Some("mem-no-tag".to_string()),
        namespace: None,
        valid_from: None,
        valid_to: None,
    };
    storage.insert_graph_node(&no_tag_node).unwrap();

    // Act
    let expired_count = storage.expire_memories_for_file("src/lib.rs").unwrap();
    assert_eq!(
        expired_count, 2,
        "Should expire both primary-linked and edge-linked memories"
    );

    // Verify: both src/lib.rs memories are expired
    let m1 = storage.get_memory_no_touch("mem-primary").unwrap().unwrap();
    assert!(
        m1.expires_at.is_some(),
        "Primary-linked memory should be expired"
    );

    let m2 = storage
        .get_memory_no_touch("mem-edge-linked")
        .unwrap()
        .unwrap();
    assert!(
        m2.expires_at.is_some(),
        "Edge-linked memory should be expired"
    );

    // Verify: other file's memory is untouched
    let m3 = storage
        .get_memory_no_touch("mem-other-file")
        .unwrap()
        .unwrap();
    assert!(
        m3.expires_at.is_none(),
        "Other file's memory should NOT be expired"
    );

    // Verify: non-static-analysis memory is untouched
    let m4 = storage.get_memory_no_touch("mem-no-tag").unwrap().unwrap();
    assert!(
        m4.expires_at.is_none(),
        "Non-static-analysis memory should NOT be expired"
    );
}
