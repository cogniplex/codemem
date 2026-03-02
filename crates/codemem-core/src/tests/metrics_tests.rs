use super::*;

#[test]
fn noop_metrics_compiles_and_runs() {
    let m = NoopMetrics;
    m.record_latency("test_op", 42.0);
    m.increment_counter("test_count", 1);
    m.record_gauge("test_gauge", 2.72);
}
