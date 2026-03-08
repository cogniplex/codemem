use crate::Storage;
use codemem_core::{MemoryNode, MemoryType, StorageBackend};
use std::collections::HashMap;

fn test_memory() -> MemoryNode {
    let mut m = MemoryNode::new("Test memory content", MemoryType::Context);
    m.importance = 0.7;
    m.tags = vec!["test".to_string()];
    m
}

#[test]
fn get_memories_batch_returns_multiple() {
    let storage = Storage::open_in_memory().unwrap();
    let m1 = test_memory();
    let mut m2 = test_memory();
    m2.id = uuid::Uuid::new_v4().to_string();
    m2.content = "Different content".to_string();
    m2.content_hash = Storage::content_hash(&m2.content);

    storage.insert_memory(&m1).unwrap();
    storage.insert_memory(&m2).unwrap();

    let backend: &dyn StorageBackend = &storage;
    let batch = backend.get_memories_batch(&[&m1.id, &m2.id]).unwrap();
    assert_eq!(batch.len(), 2);
}

#[test]
fn get_memories_batch_empty() {
    let storage = Storage::open_in_memory().unwrap();
    let backend: &dyn StorageBackend = &storage;
    let batch = backend.get_memories_batch(&[]).unwrap();
    assert!(batch.is_empty());
}

#[test]
fn storage_backend_trait_object() {
    let storage = Storage::open_in_memory().unwrap();
    let backend: Box<dyn StorageBackend> = Box::new(storage);

    let m = test_memory();
    backend.insert_memory(&m).unwrap();
    let retrieved = backend.get_memory(&m.id).unwrap().unwrap();
    assert_eq!(retrieved.id, m.id);
}

#[test]
fn file_hashes_roundtrip() {
    let storage = Storage::open_in_memory().unwrap();
    let backend: &dyn StorageBackend = &storage;

    let mut hashes = HashMap::new();
    hashes.insert("src/main.rs".to_string(), "abc123".to_string());
    hashes.insert("src/lib.rs".to_string(), "def456".to_string());

    backend.save_file_hashes(&hashes).unwrap();
    let loaded = backend.load_file_hashes().unwrap();
    assert_eq!(loaded.len(), 2);
    assert_eq!(loaded.get("src/main.rs"), Some(&"abc123".to_string()));
}

#[test]
fn decay_stale_memories_updates() {
    let storage = Storage::open_in_memory().unwrap();
    let backend: &dyn StorageBackend = &storage;

    let m = test_memory();
    backend.insert_memory(&m).unwrap();

    // Decay memories older than far future = none affected
    let count = backend.decay_stale_memories(0, 0.5).unwrap();
    assert_eq!(count, 0);

    // Decay all memories (threshold in the future)
    let count = backend.decay_stale_memories(i64::MAX, 0.5).unwrap();
    assert_eq!(count, 1);
}

#[test]
fn find_forgettable_returns_low_importance() {
    let storage = Storage::open_in_memory().unwrap();
    let backend: &dyn StorageBackend = &storage;

    let mut m = test_memory();
    m.importance = 0.1;
    backend.insert_memory(&m).unwrap();

    let forgettable = backend.find_forgettable(0.5).unwrap();
    assert_eq!(forgettable.len(), 1);
    assert_eq!(forgettable[0], m.id);

    let forgettable = backend.find_forgettable(0.05).unwrap();
    assert!(forgettable.is_empty());
}
