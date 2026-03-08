use crate::CodememEngine;
use codemem_core::{MemoryNode, MemoryType};

fn make_memory_with_opts(
    id: &str,
    content: &str,
    memory_type: MemoryType,
    namespace: Option<&str>,
    importance: f64,
    access_count: u32,
) -> MemoryNode {
    let mut m = MemoryNode::test_default(content);
    m.id = id.to_string();
    m.memory_type = memory_type;
    m.importance = importance;
    m.confidence = 0.9;
    m.access_count = access_count;
    m.namespace = namespace.map(String::from);
    m
}

/// Create a memory with a timestamp in the past (days_ago days before now).
fn make_old_memory(
    id: &str,
    content: &str,
    memory_type: MemoryType,
    importance: f64,
    access_count: u32,
    days_ago: i64,
) -> MemoryNode {
    let past = chrono::Utc::now() - chrono::Duration::days(days_ago);
    let mut m = MemoryNode::test_default(content);
    m.id = id.to_string();
    m.memory_type = memory_type;
    m.importance = importance;
    m.confidence = 0.9;
    m.access_count = access_count;
    m.created_at = past;
    m.updated_at = past;
    m.last_accessed_at = past;
    m
}

/// Create a memory with specific tags.
fn make_tagged_memory(
    id: &str,
    content: &str,
    importance: f64,
    access_count: u32,
    tags: Vec<String>,
) -> MemoryNode {
    let mut m = MemoryNode::test_default(content);
    m.id = id.to_string();
    m.importance = importance;
    m.confidence = 0.9;
    m.access_count = access_count;
    m.tags = tags;
    m
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

    let result = engine.consolidate_forget(Some(0.1), None, Some(5)).unwrap();
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
    assert_eq!(result.affected, 0, "distinct memories should not be merged");
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
    assert!(!status.is_empty(), "should have at least one status entry");
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

// ── Decay: old memories lose importance ─────────────────────────────

#[test]
fn decay_reduces_importance_of_old_memories() {
    let engine = CodememEngine::for_testing();

    // Store a memory that was last accessed 90 days ago with high importance
    let mem = make_old_memory(
        "old1",
        "old memory that should decay significantly",
        MemoryType::Context,
        0.8,
        0,
        90,
    );
    engine.persist_memory(&mem).unwrap();

    let result = engine.consolidate_decay(Some(1)).unwrap();
    assert_eq!(result.cycle, "decay");
    assert!(result.affected >= 1, "should decay at least 1 stale memory");

    // Verify importance decreased
    let updated = engine.storage.get_memory("old1").unwrap().unwrap();
    assert!(
        updated.importance < 0.8,
        "importance should decrease from 0.8 after 90 days of decay, got {}",
        updated.importance
    );
}

#[test]
fn decay_power_law_formula_more_days_more_decay() {
    let engine = CodememEngine::for_testing();

    // Two memories with different ages
    let mem_60 = make_old_memory(
        "age60",
        "sixty day old memory content abc",
        MemoryType::Context,
        0.8,
        0,
        60,
    );
    let mem_180 = make_old_memory(
        "age180",
        "one hundred eighty day old memory content xyz",
        MemoryType::Context,
        0.8,
        0,
        180,
    );
    engine.persist_memory(&mem_60).unwrap();
    engine.persist_memory(&mem_180).unwrap();

    engine.consolidate_decay(Some(1)).unwrap();

    let imp_60 = engine
        .storage
        .get_memory("age60")
        .unwrap()
        .unwrap()
        .importance;
    let imp_180 = engine
        .storage
        .get_memory("age180")
        .unwrap()
        .unwrap()
        .importance;

    assert!(
        imp_60 > imp_180,
        "60-day-old memory should retain more importance ({}) than 180-day-old ({})",
        imp_60,
        imp_180
    );
}

#[test]
fn decay_access_count_provides_boost() {
    let engine = CodememEngine::for_testing();

    // Two memories same age but different access counts
    let mut mem_no_access = make_old_memory(
        "no_access",
        "never accessed old memory content aaa",
        MemoryType::Context,
        0.8,
        0,
        60,
    );
    mem_no_access.access_count = 0;

    let mut mem_accessed = make_old_memory(
        "accessed",
        "frequently accessed old memory content bbb",
        MemoryType::Context,
        0.8,
        20,
        60,
    );
    mem_accessed.access_count = 20;

    engine.persist_memory(&mem_no_access).unwrap();
    engine.persist_memory(&mem_accessed).unwrap();

    engine.consolidate_decay(Some(1)).unwrap();

    let imp_no_access = engine
        .storage
        .get_memory("no_access")
        .unwrap()
        .unwrap()
        .importance;
    let imp_accessed = engine
        .storage
        .get_memory("accessed")
        .unwrap()
        .unwrap()
        .importance;

    assert!(
        imp_accessed > imp_no_access,
        "accessed memory ({}) should retain more importance than unaccessed ({})",
        imp_accessed,
        imp_no_access
    );
}

#[test]
fn decay_logs_consolidation_status() {
    let engine = CodememEngine::for_testing();

    let mem = make_old_memory(
        "decay_log_test",
        "memory for decay log testing xyz",
        MemoryType::Context,
        0.5,
        0,
        45,
    );
    engine.persist_memory(&mem).unwrap();

    engine.consolidate_decay(Some(1)).unwrap();

    let status = engine.consolidation_status().unwrap();
    assert!(
        status.iter().any(|s| s.cycle_type == "decay"),
        "consolidation status should include decay entry after running"
    );
}

// ── Forget: boundary edge cases ─────────────────────────────────────

#[test]
fn forget_importance_exactly_at_threshold_not_deleted() {
    let engine = CodememEngine::for_testing();

    // importance == threshold means it should NOT be deleted (< not <=)
    engine
        .persist_memory(&make_memory_with_opts(
            "boundary",
            "boundary importance memory exactly at threshold",
            MemoryType::Context,
            None,
            0.1, // exactly at default threshold
            0,
        ))
        .unwrap();

    let result = engine.consolidate_forget(Some(0.1), None, None).unwrap();
    let m = engine.storage.get_memory("boundary").unwrap();
    assert!(
        m.is_some(),
        "memory at exactly the threshold should NOT be deleted (strict < comparison)"
    );
    assert_eq!(result.affected, 0);
}

#[test]
fn forget_access_count_exactly_at_max_is_deleted() {
    let engine = CodememEngine::for_testing();

    engine
        .persist_memory(&make_memory_with_opts(
            "exact_access",
            "memory with access count exactly at max boundary",
            MemoryType::Context,
            None,
            0.05,
            3, // access_count == max_access_count
        ))
        .unwrap();

    let result = engine.consolidate_forget(Some(0.1), None, Some(3)).unwrap();
    assert!(
        result.affected >= 1,
        "memory with access_count == max_access_count should be deleted"
    );
    let m = engine.storage.get_memory("exact_access").unwrap();
    assert!(
        m.is_none(),
        "memory at exact max_access_count should be gone"
    );
}

#[test]
fn forget_with_target_tags_filters_correctly() {
    let engine = CodememEngine::for_testing();

    // Memory with matching tag
    engine
        .persist_memory(&make_tagged_memory(
            "tagged_forget",
            "tagged memory to forget via tag filter",
            0.05,
            0,
            vec!["ephemeral".to_string()],
        ))
        .unwrap();

    // Memory without matching tag (same low importance)
    engine
        .persist_memory(&make_memory_with_opts(
            "untagged_keep",
            "untagged memory should be kept despite low importance",
            MemoryType::Context,
            None,
            0.05,
            0,
        ))
        .unwrap();

    let tags = vec!["ephemeral".to_string()];
    let result = engine
        .consolidate_forget(Some(0.1), Some(&tags), None)
        .unwrap();
    assert!(result.affected >= 1);

    let tagged = engine.storage.get_memory("tagged_forget").unwrap();
    assert!(tagged.is_none(), "tagged memory should be deleted");

    let untagged = engine.storage.get_memory("untagged_keep").unwrap();
    assert!(
        untagged.is_some(),
        "untagged memory should survive tag-filtered forget"
    );
}

#[test]
fn forget_logs_consolidation_status() {
    let engine = CodememEngine::for_testing();

    engine
        .persist_memory(&make_memory_with_opts(
            "forget_log",
            "memory for forget log testing xyz",
            MemoryType::Context,
            None,
            0.01,
            0,
        ))
        .unwrap();

    engine.consolidate_forget(Some(0.1), None, None).unwrap();

    let status = engine.consolidation_status().unwrap();
    assert!(
        status.iter().any(|s| s.cycle_type == "forget"),
        "consolidation status should include forget entry"
    );
}

// ── Cluster: keeps highest importance ────────────────────────────────

#[test]
fn cluster_preserves_all_memories_without_embeddings() {
    let engine = CodememEngine::for_testing();

    // Multiple memories of same type — without embeddings, no cosine similarity
    // can be computed, so fallback to content_hash equality. Different content
    // means different hashes, so no merging.
    for i in 0..5 {
        engine
            .persist_memory(&make_memory_with_opts(
                &format!("cluster_m{i}"),
                &format!("unique memory number {i} with distinct content xyz_{i}"),
                MemoryType::Context,
                None,
                0.5 + (i as f64) * 0.05,
                0,
            ))
            .unwrap();
    }

    let result = engine.consolidate_cluster(Some(0.90)).unwrap();
    assert_eq!(result.affected, 0, "all unique content should be preserved");

    // All 5 memories should still exist
    for i in 0..5 {
        let m = engine.storage.get_memory(&format!("cluster_m{i}")).unwrap();
        assert!(m.is_some(), "memory cluster_m{i} should still exist");
    }
}

#[test]
fn cluster_details_include_algorithm_info() {
    let engine = CodememEngine::for_testing();
    let result = engine.consolidate_cluster(Some(0.95)).unwrap();
    assert_eq!(result.details["algorithm"], "semantic_cosine");
    assert!(result.details["similarity_threshold"].as_f64().is_some());
}

#[test]
fn cluster_logs_consolidation_status() {
    let engine = CodememEngine::for_testing();
    engine.consolidate_cluster(None).unwrap();

    let status = engine.consolidation_status().unwrap();
    assert!(
        status.iter().any(|s| s.cycle_type == "cluster"),
        "consolidation status should include cluster entry"
    );
}

// ── Creative: cross-type connections ─────────────────────────────────

#[test]
fn creative_cross_type_without_embeddings_no_connections() {
    let engine = CodememEngine::for_testing();

    // Cross-type memories, but without an embedding provider, no vector similarity
    // can be computed — so no SHARES_THEME edges should be created.
    engine
        .persist_memory(&make_memory_with_opts(
            "decision1",
            "decided to use async runtime for concurrency",
            MemoryType::Decision,
            None,
            0.7,
            0,
        ))
        .unwrap();
    engine
        .persist_memory(&make_memory_with_opts(
            "pattern1",
            "observed async runtime pattern in concurrency module",
            MemoryType::Pattern,
            None,
            0.7,
            0,
        ))
        .unwrap();
    engine
        .persist_memory(&make_memory_with_opts(
            "insight1",
            "insight about async runtime improving throughput significantly",
            MemoryType::Insight,
            None,
            0.7,
            0,
        ))
        .unwrap();

    let result = engine.consolidate_creative().unwrap();
    assert_eq!(result.cycle, "creative");
    // Without embeddings loaded into the vector index, KNN search returns nothing
    assert_eq!(
        result.affected, 0,
        "without embeddings, no connections should be made"
    );
}

#[test]
fn creative_logs_consolidation_status() {
    let engine = CodememEngine::for_testing();
    engine.consolidate_creative().unwrap();

    let status = engine.consolidation_status().unwrap();
    assert!(
        status.iter().any(|s| s.cycle_type == "creative"),
        "consolidation status should include creative entry"
    );
}

// ── Consolidation status: multiple cycles ────────────────────────────

#[test]
fn consolidation_status_after_multiple_cycles() {
    let engine = CodememEngine::for_testing();

    engine.consolidate_decay(None).unwrap();
    engine.consolidate_forget(None, None, None).unwrap();
    engine.consolidate_cluster(None).unwrap();
    engine.consolidate_creative().unwrap();

    let status = engine.consolidation_status().unwrap();
    let cycle_types: Vec<&str> = status.iter().map(|s| s.cycle_type.as_str()).collect();

    assert!(
        cycle_types.contains(&"decay"),
        "should have decay in status"
    );
    assert!(
        cycle_types.contains(&"forget"),
        "should have forget in status"
    );
    assert!(
        cycle_types.contains(&"cluster"),
        "should have cluster in status"
    );
    assert!(
        cycle_types.contains(&"creative"),
        "should have creative in status"
    );
}

// ── Summarize: error paths ──────────────────────────────────────────

#[test]
fn summarize_with_custom_cluster_size_still_requires_provider() {
    let engine = CodememEngine::for_testing();
    let result = engine.consolidate_summarize(Some(3));
    assert!(
        result.is_err(),
        "summarize should fail without compress provider regardless of cluster size"
    );
}

// ── Forget: find_forgettable_by_tags ────────────────────────────────

#[test]
fn find_forgettable_by_tags_empty_engine_returns_empty() {
    let engine = CodememEngine::for_testing();
    let tags = vec!["ephemeral".to_string()];
    let ids = engine.find_forgettable_by_tags(0.1, &tags, 0).unwrap();
    assert!(ids.is_empty());
}

#[test]
fn find_forgettable_by_tags_filters_by_importance_and_tag() {
    let engine = CodememEngine::for_testing();

    // Low importance with matching tag
    engine
        .persist_memory(&make_tagged_memory(
            "match_tag",
            "low importance with matching tag for forget",
            0.05,
            0,
            vec!["temp".to_string()],
        ))
        .unwrap();

    // High importance with matching tag (should NOT be returned)
    engine
        .persist_memory(&make_tagged_memory(
            "high_imp_tag",
            "high importance with matching tag for keep",
            0.9,
            0,
            vec!["temp".to_string()],
        ))
        .unwrap();

    // Low importance without matching tag (should NOT be returned)
    engine
        .persist_memory(&make_memory_with_opts(
            "no_tag",
            "low importance without matching tag for check",
            MemoryType::Context,
            None,
            0.05,
            0,
        ))
        .unwrap();

    let tags = vec!["temp".to_string()];
    let ids = engine.find_forgettable_by_tags(0.1, &tags, 0).unwrap();

    assert!(ids.contains(&"match_tag".to_string()));
    assert!(
        !ids.contains(&"high_imp_tag".to_string()),
        "high importance should not be forgettable"
    );
    assert!(
        !ids.contains(&"no_tag".to_string()),
        "memory without matching tag should not be forgettable"
    );
}
