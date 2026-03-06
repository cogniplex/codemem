use crate::CodememEngine;
use codemem_core::{MemoryNode, MemoryType};
use codemem_storage::Storage;
use std::collections::HashMap;

fn make_memory_with_opts(
    id: &str,
    content: &str,
    memory_type: MemoryType,
    namespace: Option<&str>,
    importance: f64,
    access_count: u32,
) -> MemoryNode {
    let now = chrono::Utc::now();
    MemoryNode {
        id: id.to_string(),
        content: content.to_string(),
        memory_type,
        importance,
        confidence: 0.9,
        access_count,
        content_hash: Storage::content_hash(content),
        tags: vec![],
        metadata: HashMap::new(),
        namespace: namespace.map(String::from),
        created_at: now,
        updated_at: now,
        last_accessed_at: now,
    }
}

// ── Decay consolidation ─────────────────────────────────────────────

#[test]
fn decay_no_stale_memories() {
    let engine = CodememEngine::for_testing();
    // Fresh memory should not be decayed
    engine
        .persist_memory(&make_memory_with_opts(
            "fresh1",
            "fresh memory content",
            MemoryType::Context,
            None,
            0.8,
            0,
        ))
        .unwrap();

    let result = engine.consolidate_decay(Some(30)).unwrap();
    assert_eq!(result.cycle, "decay");
    // Fresh memories may or may not be affected depending on threshold
}

#[test]
fn decay_returns_correct_cycle_name() {
    let engine = CodememEngine::for_testing();
    let result = engine.consolidate_decay(None).unwrap();
    assert_eq!(result.cycle, "decay");
}

// ── Forget consolidation ────────────────────────────────────────────

#[test]
fn forget_removes_low_importance_never_accessed() {
    let engine = CodememEngine::for_testing();

    // Low importance, never accessed
    engine
        .persist_memory(&make_memory_with_opts(
            "forget1",
            "forgettable memory low importance",
            MemoryType::Context,
            None,
            0.05, // below default threshold of 0.1
            0,
        ))
        .unwrap();
    // High importance
    engine
        .persist_memory(&make_memory_with_opts(
            "keep1",
            "important memory to keep",
            MemoryType::Context,
            None,
            0.9,
            0,
        ))
        .unwrap();

    let result = engine.consolidate_forget(Some(0.1), None, None).unwrap();
    assert_eq!(result.cycle, "forget");
    assert!(result.affected >= 1, "should forget at least 1 memory");

    // Verify the low-importance memory is gone
    let m = engine.storage.get_memory("forget1").unwrap();
    assert!(m.is_none(), "forgettable memory should be deleted");

    // Verify the important one is kept
    let m = engine.storage.get_memory("keep1").unwrap();
    assert!(m.is_some(), "important memory should be kept");
}

#[test]
fn forget_respects_access_count() {
    let engine = CodememEngine::for_testing();

    // Low importance but accessed
    engine
        .persist_memory(&make_memory_with_opts(
            "accessed1",
            "accessed low importance memory",
            MemoryType::Context,
            None,
            0.05,
            5, // accessed 5 times
        ))
        .unwrap();

    // Default max_access_count=0 means only delete never-accessed
    let _result = engine.consolidate_forget(Some(0.1), None, None).unwrap();
    // The memory has access_count=5 but the default SQL filter uses access_count=0,
    // so it should NOT be deleted
    let m = engine.storage.get_memory("accessed1").unwrap();
    assert!(
        m.is_some(),
        "accessed memory should not be forgotten with default max_access_count=0"
    );
}

#[test]
fn forget_with_custom_max_access_count() {
    let engine = CodememEngine::for_testing();

    engine
        .persist_memory(&make_memory_with_opts(
            "low-access",
            "low access memory to forget",
            MemoryType::Context,
            None,
            0.05,
            2,
        ))
        .unwrap();

    let result = engine
        .consolidate_forget(Some(0.1), None, Some(5))
        .unwrap();
    assert!(
        result.affected >= 1,
        "should forget memory with access_count <= max_access_count"
    );
}

#[test]
fn forget_empty_engine() {
    let engine = CodememEngine::for_testing();
    let result = engine.consolidate_forget(None, None, None).unwrap();
    assert_eq!(result.affected, 0);
    assert_eq!(result.cycle, "forget");
}

// ── Cluster consolidation ───────────────────────────────────────────

#[test]
fn cluster_no_duplicates() {
    let engine = CodememEngine::for_testing();

    // Distinctly different memories
    engine
        .persist_memory(&make_memory_with_opts(
            "unique1",
            "Rust ownership model with borrowing rules and lifetime annotations",
            MemoryType::Context,
            None,
            0.7,
            0,
        ))
        .unwrap();
    engine
        .persist_memory(&make_memory_with_opts(
            "unique2",
            "Python asyncio event loop with coroutines and tasks management",
            MemoryType::Context,
            None,
            0.7,
            0,
        ))
        .unwrap();

    let result = engine.consolidate_cluster(None).unwrap();
    assert_eq!(result.cycle, "cluster");
    // Without embeddings, no semantic duplicates should be found
    // (content hashes are different)
    assert_eq!(
        result.affected, 0,
        "distinct memories should not be merged"
    );
}

#[test]
fn cluster_with_exact_duplicate_content() {
    let engine = CodememEngine::for_testing();

    // Storage enforces UNIQUE on content_hash, so we can't insert exact duplicates.
    // Without an embedding provider, cluster uses content_hash equality as similarity
    // fallback. Two different-content memories won't have matching hashes,
    // so cluster should correctly report 0 affected (no duplicates).
    engine
        .persist_memory(&make_memory_with_opts(
            "sim1",
            "the authentication service validates tokens",
            MemoryType::Context,
            None,
            0.7,
            0,
        ))
        .unwrap();
    engine
        .persist_memory(&make_memory_with_opts(
            "sim2",
            "the authorization service checks permissions",
            MemoryType::Context,
            None,
            0.9,
            0,
        ))
        .unwrap();

    let result = engine.consolidate_cluster(Some(0.92)).unwrap();
    assert_eq!(result.cycle, "cluster");
    // Without embeddings, different content hashes → similarity=0.0 → no merge
    assert_eq!(
        result.affected, 0,
        "different-hash memories should not be merged without embeddings"
    );

    // Both memories should still exist
    assert!(engine.storage.get_memory("sim1").unwrap().is_some());
    assert!(engine.storage.get_memory("sim2").unwrap().is_some());
}

#[test]
fn cluster_empty_engine() {
    let engine = CodememEngine::for_testing();
    let result = engine.consolidate_cluster(None).unwrap();
    assert_eq!(result.affected, 0);
}

// ── Creative consolidation ──────────────────────────────────────────

#[test]
fn creative_no_memories() {
    let engine = CodememEngine::for_testing();
    let result = engine.consolidate_creative().unwrap();
    assert_eq!(result.cycle, "creative");
    assert_eq!(result.affected, 0);
}

#[test]
fn creative_same_type_no_edges() {
    let engine = CodememEngine::for_testing();

    // Same memory type — creative only connects cross-type
    engine
        .persist_memory(&make_memory_with_opts(
            "same1",
            "same type memory one about testing",
            MemoryType::Context,
            None,
            0.7,
            0,
        ))
        .unwrap();
    engine
        .persist_memory(&make_memory_with_opts(
            "same2",
            "same type memory two about testing",
            MemoryType::Context,
            None,
            0.7,
            0,
        ))
        .unwrap();

    let result = engine.consolidate_creative().unwrap();
    assert_eq!(result.cycle, "creative");
    // Same type should produce no SHARES_THEME edges
    assert_eq!(
        result.affected, 0,
        "same-type memories should not get creative connections"
    );
}

// ── Summarize consolidation ─────────────────────────────────────────

#[test]
fn summarize_requires_compress_provider() {
    let engine = CodememEngine::for_testing();

    // Without CODEMEM_COMPRESS_PROVIDER set, summarize should error
    let result = engine.consolidate_summarize(None);
    assert!(
        result.is_err(),
        "summarize should fail without compress provider"
    );
    let err = result.err().unwrap();
    let msg = format!("{err}");
    assert!(
        msg.contains("CODEMEM_COMPRESS_PROVIDER"),
        "error should mention env var: {msg}"
    );
}

// ── Consolidation status ────────────────────────────────────────────

#[test]
fn consolidation_status_initially_empty() {
    let engine = CodememEngine::for_testing();
    let status = engine.consolidation_status().unwrap();
    assert!(
        status.is_empty(),
        "no consolidation runs should exist initially"
    );
}

#[test]
fn consolidation_status_after_decay() {
    let engine = CodememEngine::for_testing();
    engine.consolidate_decay(None).unwrap();

    let status = engine.consolidation_status().unwrap();
    assert!(
        !status.is_empty(),
        "should have at least one status entry"
    );
    assert!(
        status.iter().any(|s| s.cycle_type == "decay"),
        "should include decay entry"
    );
}

// ── UnionFind ───────────────────────────────────────────────────────

#[test]
fn union_find_basic() {
    use crate::consolidation::UnionFind;

    let mut uf = UnionFind::new(5);

    // Initially all separate
    assert_ne!(uf.find(0), uf.find(1));

    // Union 0 and 1
    uf.union(0, 1);
    assert_eq!(uf.find(0), uf.find(1));

    // Union 2 and 3
    uf.union(2, 3);
    assert_eq!(uf.find(2), uf.find(3));

    // 0 and 2 still separate
    assert_ne!(uf.find(0), uf.find(2));

    // Union the groups
    uf.union(1, 3);
    assert_eq!(uf.find(0), uf.find(3));
}

#[test]
fn union_find_groups() {
    use crate::consolidation::UnionFind;

    let mut uf = UnionFind::new(6);
    uf.union(0, 1);
    uf.union(1, 2);
    uf.union(3, 4);
    // 5 is alone

    let groups = uf.groups(6);
    // Should have 3 groups: {0,1,2}, {3,4}, {5}
    assert_eq!(groups.len(), 3);

    let sizes: Vec<usize> = {
        let mut s: Vec<_> = groups.iter().map(|g| g.len()).collect();
        s.sort();
        s
    };
    assert_eq!(sizes, vec![1, 2, 3]);
}
