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
