use super::*;

// ── VALID_CYCLES constant ─────────────────────────────────────────────────

#[test]
fn valid_cycles_contains_five_entries() {
    assert_eq!(VALID_CYCLES.len(), 5);
}

#[test]
fn valid_cycles_contains_decay() {
    assert!(VALID_CYCLES.contains(&"decay"));
}

#[test]
fn valid_cycles_contains_creative() {
    assert!(VALID_CYCLES.contains(&"creative"));
}

#[test]
fn valid_cycles_contains_cluster() {
    assert!(VALID_CYCLES.contains(&"cluster"));
}

#[test]
fn valid_cycles_contains_forget() {
    assert!(VALID_CYCLES.contains(&"forget"));
}

#[test]
fn valid_cycles_contains_summarize() {
    assert!(VALID_CYCLES.contains(&"summarize"));
}

// ── is_valid_cycle tests ──────────────────────────────────────────────────

#[test]
fn is_valid_cycle_accepts_known_cycles() {
    for name in &["decay", "creative", "cluster", "forget", "summarize"] {
        assert!(is_valid_cycle(name), "{name} should be valid");
    }
}

#[test]
fn is_valid_cycle_rejects_unknown() {
    assert!(!is_valid_cycle("rem"));
    assert!(!is_valid_cycle(""));
    assert!(!is_valid_cycle("DECAY"));
}

// ── Consolidation engine integration ────────────────────────────────

#[test]
fn consolidate_decay_on_empty_engine() {
    let engine = codemem_engine::CodememEngine::for_testing();
    let result = engine.consolidate_decay(None).unwrap();
    assert_eq!(result.cycle, "decay");
    assert_eq!(result.affected, 0);
}

#[test]
fn consolidate_creative_on_empty_engine() {
    let engine = codemem_engine::CodememEngine::for_testing();
    let result = engine.consolidate_creative().unwrap();
    assert_eq!(result.cycle, "creative");
    assert_eq!(result.affected, 0);
}

#[test]
fn consolidate_cluster_on_empty_engine() {
    let engine = codemem_engine::CodememEngine::for_testing();
    let result = engine.consolidate_cluster(None).unwrap();
    assert_eq!(result.cycle, "cluster");
    assert_eq!(result.affected, 0);
}

#[test]
fn consolidate_forget_on_empty_engine() {
    let engine = codemem_engine::CodememEngine::for_testing();
    let result = engine.consolidate_forget(None, None, None).unwrap();
    assert_eq!(result.cycle, "forget");
    assert_eq!(result.affected, 0);
}

#[test]
fn consolidate_forget_with_explicit_thresholds() {
    let engine = codemem_engine::CodememEngine::for_testing();
    // importance_threshold=0.5, no target_tags, max_access_count=10
    let result = engine
        .consolidate_forget(Some(0.5), None, Some(10))
        .unwrap();
    assert_eq!(result.cycle, "forget");
    assert_eq!(result.affected, 0);
}

#[test]
fn consolidate_forget_with_zero_threshold() {
    // threshold=0.0 means nothing qualifies (importance < 0.0 is impossible)
    let engine = codemem_engine::CodememEngine::for_testing();
    let result = engine.consolidate_forget(Some(0.0), None, None).unwrap();
    assert_eq!(result.affected, 0);
}

#[test]
fn consolidate_forget_deletes_low_importance_memory() {
    let engine = codemem_engine::CodememEngine::for_testing();
    let mut memory = codemem_core::MemoryNode::test_default("low importance throwaway");
    memory.id = "forget-me".to_string();
    memory.importance = 0.05;
    memory.confidence = 0.5;
    engine.persist_memory(&memory).unwrap();

    // Default threshold is 0.1; importance=0.05 < 0.1, access_count=0 → should be deleted
    let result = engine.consolidate_forget(None, None, None).unwrap();
    assert_eq!(result.affected, 1);
}

#[test]
fn consolidate_forget_spares_high_importance_memory() {
    let engine = codemem_engine::CodememEngine::for_testing();
    let mut memory =
        codemem_core::MemoryNode::test_default("important decision about architecture");
    memory.id = "keep-me".to_string();
    memory.memory_type = codemem_core::MemoryType::Decision;
    memory.importance = 0.9;
    engine.persist_memory(&memory).unwrap();

    let result = engine.consolidate_forget(None, None, None).unwrap();
    assert_eq!(
        result.affected, 0,
        "high-importance memory should survive forget"
    );
}

#[test]
fn consolidate_decay_with_explicit_days() {
    let engine = codemem_engine::CodememEngine::for_testing();
    let result = engine.consolidate_decay(Some(1)).unwrap();
    assert_eq!(result.cycle, "decay");
    assert_eq!(result.affected, 0);
}

#[test]
fn consolidate_cluster_with_explicit_threshold() {
    let engine = codemem_engine::CodememEngine::for_testing();
    let result = engine.consolidate_cluster(Some(0.95)).unwrap();
    assert_eq!(result.cycle, "cluster");
    assert_eq!(result.affected, 0);
}

#[test]
fn consolidation_status_empty() {
    let engine = codemem_engine::CodememEngine::for_testing();
    let entries = engine.consolidation_status().unwrap();
    assert!(entries.is_empty(), "no runs recorded yet");
}

#[test]
fn consolidation_status_after_decay() {
    let engine = codemem_engine::CodememEngine::for_testing();
    engine.consolidate_decay(None).unwrap();
    let entries = engine.consolidation_status().unwrap();
    assert!(
        entries.iter().any(|e| e.cycle_type == "decay"),
        "decay should be recorded"
    );
}
