use super::*;

fn random_vector(dim: usize) -> Vec<f32> {
    (0..dim)
        .map(|i| ((i * 7 + 3) % 100) as f32 / 100.0)
        .collect()
}

#[test]
fn insert_and_search() {
    let mut index = HnswIndex::with_defaults().unwrap();
    let v1 = random_vector(768);
    index.insert("mem-1", &v1).unwrap();

    let results = index.search(&v1, 1).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, "mem-1");
    assert!(results[0].1 > 0.99); // self-similarity should be ~1.0
}

#[test]
fn remove_vector() {
    let mut index = HnswIndex::with_defaults().unwrap();
    let v1 = random_vector(768);
    index.insert("mem-1", &v1).unwrap();
    assert!(index.remove("mem-1").unwrap());
    assert!(!index.remove("mem-1").unwrap()); // already removed
}

#[test]
fn dimension_mismatch() {
    let mut index = HnswIndex::with_defaults().unwrap();
    let bad = vec![1.0f32; 128]; // wrong dimensions
    assert!(index.insert("bad", &bad).is_err());
}

#[test]
fn stats() {
    let index = HnswIndex::with_defaults().unwrap();
    let stats = index.stats();
    assert_eq!(stats.count, 0);
    assert_eq!(stats.dimensions, 768);
}

// ── rebuild_from_entries Tests ──────────────────────────────────────

fn deterministic_vector(dim: usize, seed: usize) -> Vec<f32> {
    (0..dim)
        .map(|i| ((i * 7 + seed * 13 + 3) % 100) as f32 / 100.0)
        .collect()
}

#[test]
fn rebuild_from_entries_with_existing_data() {
    let mut index = HnswIndex::with_defaults().unwrap();

    // Insert some initial data
    let v1 = deterministic_vector(768, 1);
    let v2 = deterministic_vector(768, 2);
    index.insert("old-1", &v1).unwrap();
    index.insert("old-2", &v2).unwrap();
    assert_eq!(index.len(), 2);

    // Rebuild with different entries
    let v3 = deterministic_vector(768, 3);
    let v4 = deterministic_vector(768, 4);
    let v5 = deterministic_vector(768, 5);
    let entries = vec![
        ("new-1".to_string(), v3.clone()),
        ("new-2".to_string(), v4.clone()),
        ("new-3".to_string(), v5.clone()),
    ];
    index.rebuild_from_entries(&entries).unwrap();

    // Old entries should be gone, new ones present
    assert_eq!(index.len(), 3);

    // Search for new-1 should find it
    let results = index.search(&v3, 1).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, "new-1");

    // Search for old-1's vector should not return old-1
    let results = index.search(&v1, 3).unwrap();
    let found_ids: Vec<&str> = results.iter().map(|(id, _)| id.as_str()).collect();
    assert!(
        !found_ids.contains(&"old-1"),
        "Old entry should not be found after rebuild"
    );
}

#[test]
fn rebuild_from_entries_empty() {
    let mut index = HnswIndex::with_defaults().unwrap();
    let v1 = deterministic_vector(768, 1);
    index.insert("mem-1", &v1).unwrap();
    assert_eq!(index.len(), 1);

    // Rebuild with empty entries
    index.rebuild_from_entries(&[]).unwrap();
    assert_eq!(index.len(), 0);
    assert_eq!(index.ghost_count(), 0);
}

#[test]
fn search_empty_index_returns_no_results() {
    let index = HnswIndex::with_defaults().unwrap();
    let query = deterministic_vector(768, 42);
    let results = index.search(&query, 10).unwrap();
    assert!(results.is_empty(), "Empty index should return no results");
}

#[test]
fn insert_same_id_twice_updates() {
    let mut index = HnswIndex::with_defaults().unwrap();
    let v1 = deterministic_vector(768, 1);
    let v2 = deterministic_vector(768, 2);

    index.insert("mem-1", &v1).unwrap();
    assert_eq!(index.len(), 1);

    // Insert same ID with different vector
    index.insert("mem-1", &v2).unwrap();
    // Should still be 1 entry (usearch size may show 2 due to ghost, but logical count is 1)
    // The id_to_key map should have exactly 1 entry
    let results = index.search(&v2, 1).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, "mem-1");
    assert!(
        results[0].1 > 0.99,
        "Self-similarity should be high after update"
    );

    // The ghost count should be 1 (old entry marked removed)
    assert_eq!(index.ghost_count(), 1);

    // Searching with old vector should still find mem-1 (it's the only entry),
    // but similarity should be lower than the new vector
    let results_old = index.search(&v1, 1).unwrap();
    assert_eq!(results_old[0].0, "mem-1");
}
