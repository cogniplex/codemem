use super::*;

// ── VALID_CYCLES constant ─────────────────────────────────────────────────

#[test]
fn valid_cycles_contains_four_entries() {
    assert_eq!(VALID_CYCLES.len(), 4);
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

// ── is_valid_cycle tests ──────────────────────────────────────────────────

#[test]
fn is_valid_cycle_accepts_known_cycles() {
    for name in &["decay", "creative", "cluster", "forget"] {
        assert!(is_valid_cycle(name), "{name} should be valid");
    }
}

#[test]
fn is_valid_cycle_rejects_unknown() {
    assert!(!is_valid_cycle("summarize"));
    assert!(!is_valid_cycle("rem"));
    assert!(!is_valid_cycle(""));
    assert!(!is_valid_cycle("DECAY"));
}
