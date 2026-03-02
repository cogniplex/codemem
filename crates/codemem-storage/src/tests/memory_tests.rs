use crate::Storage;
use codemem_core::{CodememError, MemoryNode, MemoryType};
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
