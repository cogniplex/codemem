use codemem_core::{MemoryNode, MemoryType};
use codemem_storage::Storage;
use criterion::{criterion_group, criterion_main, Criterion};
use std::collections::HashMap;

/// Create a test memory with a unique content hash derived from the index.
fn make_memory(i: usize) -> MemoryNode {
    let now = chrono::Utc::now();
    let content = format!(
        "Benchmark memory entry number {i}: testing storage CRUD performance with varied content"
    );
    MemoryNode {
        id: uuid::Uuid::new_v4().to_string(),
        content: content.clone(),
        memory_type: MemoryType::Context,
        importance: 0.5,
        confidence: 1.0,
        access_count: 0,
        content_hash: Storage::content_hash(&content),
        tags: vec!["bench".to_string(), format!("group-{}", i % 10)],
        metadata: HashMap::new(),
        namespace: None,
        created_at: now,
        updated_at: now,
        last_accessed_at: now,
    }
}

/// Benchmark inserting memories into an in-memory SQLite database.
fn bench_storage_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("storage_insert");

    // Pre-generate memories for each iteration (content hashes must be unique)
    group.bench_function("1000_memories", |b| {
        b.iter(|| {
            let storage = Storage::open_in_memory().unwrap();
            for i in 0..1000 {
                storage.insert_memory(&make_memory(i)).unwrap();
            }
        });
    });

    group.finish();
}

/// Benchmark getting a single memory by ID from a pre-populated database.
fn bench_storage_get(c: &mut Criterion) {
    let mut group = c.benchmark_group("storage_get");

    let storage = Storage::open_in_memory().unwrap();
    let mut ids = Vec::with_capacity(1000);
    for i in 0..1000 {
        let mem = make_memory(i);
        ids.push(mem.id.clone());
        storage.insert_memory(&mem).unwrap();
    }

    // Benchmark retrieving a memory near the middle of the dataset
    let target_id = ids[500].clone();

    group.bench_function("single_by_id", |b| {
        b.iter(|| {
            let result = storage.get_memory(&target_id).unwrap();
            assert!(result.is_some());
        });
    });

    group.finish();
}

/// Benchmark listing all memory IDs from a pre-populated database.
fn bench_storage_list_ids(c: &mut Criterion) {
    let mut group = c.benchmark_group("storage_list_ids");

    let storage = Storage::open_in_memory().unwrap();
    for i in 0..1000 {
        storage.insert_memory(&make_memory(i)).unwrap();
    }

    group.bench_function("1000_entries", |b| {
        b.iter(|| {
            let ids = storage.list_memory_ids().unwrap();
            assert_eq!(ids.len(), 1000);
        });
    });

    group.finish();
}

/// Benchmark deleting a memory by ID.
fn bench_storage_delete(c: &mut Criterion) {
    let mut group = c.benchmark_group("storage_delete");

    // Each iteration needs a fresh database since deletion is destructive.
    // Pre-generate a set of memories to insert.
    group.bench_function("delete_from_1000", |b| {
        b.iter_custom(|iters| {
            let mut total = std::time::Duration::ZERO;
            for _ in 0..iters {
                // Setup: create storage and insert 1000 memories
                let storage = Storage::open_in_memory().unwrap();
                let mut ids = Vec::with_capacity(1000);
                for i in 0..1000 {
                    let mem = make_memory(i);
                    ids.push(mem.id.clone());
                    storage.insert_memory(&mem).unwrap();
                }

                // Timed: delete the middle memory
                let target_id = &ids[500];
                let start = std::time::Instant::now();
                let deleted = storage.delete_memory(target_id).unwrap();
                total += start.elapsed();

                assert!(deleted);
            }
            total
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_storage_insert,
    bench_storage_get,
    bench_storage_list_ids,
    bench_storage_delete
);
criterion_main!(benches);
