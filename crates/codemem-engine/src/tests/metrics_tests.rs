use super::*;
use codemem_core::Metrics;

#[test]
fn record_and_snapshot() {
    let m = InMemoryMetrics::new();
    m.record_latency("recall", 10.0);
    m.record_latency("recall", 20.0);
    m.record_latency("recall", 30.0);
    m.increment_counter("tool_calls", 3);
    m.record_gauge("memory_count", 42.0);

    let snap = m.snapshot();
    assert_eq!(snap.latencies["recall"].count, 3);
    assert_eq!(snap.counters["tool_calls"], 3);
    assert!((snap.gauges["memory_count"] - 42.0).abs() < f64::EPSILON);
}

#[test]
fn percentile_calculation() {
    let samples: VecDeque<f64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        .into_iter()
        .collect();
    let stats = compute_latency_stats(&samples);
    assert_eq!(stats.count, 10);
    assert!((stats.min_ms - 1.0).abs() < f64::EPSILON);
    assert!((stats.max_ms - 10.0).abs() < f64::EPSILON);
    assert!((stats.p50_ms - 5.5).abs() < 1.5); // roughly median
}
