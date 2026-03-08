use super::*;

// ── print_check ─────────────────────────────────────────────────────

#[test]
fn print_check_ok_does_not_panic() {
    print_check("test", true, "");
}

#[test]
fn print_check_fail_with_detail_does_not_panic() {
    print_check("test", false, "some detail");
}

#[test]
fn print_check_ok_with_detail_does_not_panic() {
    print_check("Database", true, "42 memories");
}

// ── Doctor health check subsystems ──────────────────────────────────
//
// cmd_doctor itself hits the real filesystem, but we can test the
// individual subsystem checks it depends on.

#[test]
fn doctor_storage_integrity_check() {
    let storage = codemem_storage::Storage::open_in_memory().unwrap();
    let result = storage.integrity_check().unwrap();
    assert!(result, "in-memory DB should pass integrity check");
}

#[test]
fn doctor_storage_schema_version() {
    let storage = codemem_storage::Storage::open_in_memory().unwrap();
    let version = storage.schema_version().unwrap();
    assert!(version > 0, "schema version should be positive");
}

#[test]
fn doctor_storage_stats_on_empty_db() {
    let storage = codemem_storage::Storage::open_in_memory().unwrap();
    let stats = storage.stats().unwrap();
    assert_eq!(stats.memory_count, 0);
    assert_eq!(stats.node_count, 0);
    assert_eq!(stats.edge_count, 0);
}

#[test]
fn doctor_storage_stats_with_data() {
    let storage = codemem_storage::Storage::open_in_memory().unwrap();

    let mut memory = codemem_core::MemoryNode::test_default("test memory");
    memory.id = "doc-1".to_string();
    memory.confidence = 0.9;
    storage.insert_memory(&memory).unwrap();

    let stats = storage.stats().unwrap();
    assert_eq!(stats.memory_count, 1);
}

#[test]
fn doctor_vector_index_creation() {
    let vector = codemem_storage::HnswIndex::with_defaults().unwrap();
    assert_eq!(vector.len(), 0, "new vector index should be empty");
}

#[test]
fn doctor_vector_index_save_and_load() {
    let dir = tempfile::tempdir().unwrap();
    let idx_path = dir.path().join("test.idx");

    let mut vector = codemem_storage::HnswIndex::with_defaults().unwrap();
    // Insert a dummy vector
    let dummy = vec![0.1f32; 768];
    vector.insert("test-vec", &dummy).unwrap();
    vector.save(&idx_path).unwrap();

    // Load into a new instance
    let mut loaded = codemem_storage::HnswIndex::with_defaults().unwrap();
    loaded.load(&idx_path).unwrap();
    assert_eq!(loaded.len(), 1);
}

#[test]
fn doctor_config_load_or_default() {
    let config = codemem_core::CodememConfig::load_or_default();
    // Should always succeed, returning defaults if no config file
    assert!(config.vector.dimensions > 0);
    assert!(config.scoring.vector_similarity > 0.0);
}
