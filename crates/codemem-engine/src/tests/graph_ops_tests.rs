use crate::CodememEngine;
use chrono::{Duration, Utc};

#[test]
fn what_changed_rejects_reversed_date_range() {
    let engine = CodememEngine::for_testing();
    let now = Utc::now();
    let yesterday = now - Duration::days(1);
    let result = engine.what_changed(now, yesterday, None);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("before"), "Error should mention date ordering");
}

#[test]
fn detect_drift_rejects_reversed_date_range() {
    let engine = CodememEngine::for_testing();
    let now = Utc::now();
    let yesterday = now - Duration::days(1);
    let result = engine.detect_drift(now, yesterday, None);
    assert!(result.is_err());
}

#[test]
fn find_stale_files_clamps_extreme_stale_days() {
    let engine = CodememEngine::for_testing();
    let result = engine.find_stale_files(None, u64::MAX);
    assert!(result.is_ok());
}
