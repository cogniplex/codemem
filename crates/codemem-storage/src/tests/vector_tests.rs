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
