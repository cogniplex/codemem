use super::*;

// ── BATCH_WINDOW constant ─────────────────────────────────────────────────

#[test]
fn batch_window_is_five_seconds() {
    assert_eq!(BATCH_WINDOW, std::time::Duration::from_secs(5));
}

// ── batch_importance tests ────────────────────────────────────────────────

#[test]
fn batch_importance_single_file() {
    let imp = batch_importance(1);
    assert!((imp - 0.35).abs() < f64::EPSILON, "expected 0.35, got {imp}");
}

#[test]
fn batch_importance_ten_files() {
    let imp = batch_importance(10);
    assert!((imp - 0.8).abs() < f64::EPSILON, "expected 0.8, got {imp}");
}

#[test]
fn batch_importance_twenty_files_capped() {
    let imp = batch_importance(20);
    assert!((imp - 0.8).abs() < f64::EPSILON, "expected 0.8 (capped), got {imp}");
}

#[test]
fn batch_importance_zero_files() {
    let imp = batch_importance(0);
    assert!((imp - 0.3).abs() < f64::EPSILON, "expected 0.3, got {imp}");
}

#[test]
fn batch_importance_monotonically_increases() {
    let mut prev = batch_importance(0);
    for n in 1..=15 {
        let curr = batch_importance(n);
        assert!(curr >= prev, "importance should not decrease: {prev} > {curr} at n={n}");
        prev = curr;
    }
}

// ── is_trivial_batch tests ────────────────────────────────────────────────

#[test]
fn trivial_batch_two_modified_only() {
    assert!(is_trivial_batch(2, 0, 0));
}

#[test]
fn trivial_batch_one_modified_only() {
    assert!(is_trivial_batch(1, 0, 0));
}

#[test]
fn non_trivial_batch_three_files() {
    assert!(!is_trivial_batch(3, 0, 0));
}

#[test]
fn non_trivial_batch_with_create() {
    assert!(!is_trivial_batch(1, 1, 0));
}

#[test]
fn non_trivial_batch_with_delete() {
    assert!(!is_trivial_batch(2, 0, 1));
}

#[test]
fn non_trivial_batch_large_with_creates() {
    assert!(!is_trivial_batch(10, 3, 2));
}

#[test]
fn trivial_batch_zero_files() {
    // Edge case: no files is technically trivial
    assert!(is_trivial_batch(0, 0, 0));
}
