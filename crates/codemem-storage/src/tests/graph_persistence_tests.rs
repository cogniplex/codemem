use crate::Storage;
use codemem_core::{GraphNode, MemoryNode, MemoryType, NodeKind};
use std::collections::HashMap;

fn test_memory() -> MemoryNode {
    let now = chrono::Utc::now();
    let content = "Test memory content";
    MemoryNode {
        id: uuid::Uuid::new_v4().to_string(),
        content: content.to_string(),
        memory_type: MemoryType::Context,
        importance: 0.7,
        confidence: 1.0,
        access_count: 0,
        content_hash: Storage::content_hash(content),
        tags: vec!["test".to_string()],
        metadata: HashMap::new(),
        namespace: None,
        created_at: now,
        updated_at: now,
        last_accessed_at: now,
    }
}

#[test]
fn store_and_get_embedding() {
    let storage = Storage::open_in_memory().unwrap();
    let memory = test_memory();
    storage.insert_memory(&memory).unwrap();

    let embedding: Vec<f32> = (0..768).map(|i| i as f32 / 768.0).collect();
    storage.store_embedding(&memory.id, &embedding).unwrap();

    let retrieved = storage.get_embedding(&memory.id).unwrap().unwrap();
    assert_eq!(retrieved.len(), 768);
    assert!((retrieved[0] - embedding[0]).abs() < f32::EPSILON);
}

#[test]
fn graph_node_crud() {
    let storage = Storage::open_in_memory().unwrap();
    let node = GraphNode {
        id: "file:src/main.rs".to_string(),
        kind: NodeKind::File,
        label: "src/main.rs".to_string(),
        payload: HashMap::new(),
        centrality: 0.0,
        memory_id: None,
        namespace: None,
    };

    storage.insert_graph_node(&node).unwrap();
    let retrieved = storage.get_graph_node(&node.id).unwrap().unwrap();
    assert_eq!(retrieved.kind, NodeKind::File);
    assert!(storage.delete_graph_node(&node.id).unwrap());
}
