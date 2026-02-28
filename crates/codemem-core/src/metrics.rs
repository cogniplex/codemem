//! Metrics trait and default no-op implementation for Codemem.

use std::collections::HashMap;

/// Trait for recording operational metrics.
pub trait Metrics: Send + Sync {
    /// Record a latency measurement for an operation.
    fn record_latency(&self, operation: &str, duration_ms: f64);

    /// Increment a named counter.
    fn increment_counter(&self, name: &str, delta: u64);

    /// Record a gauge (point-in-time) value.
    fn record_gauge(&self, name: &str, value: f64);
}

/// No-op metrics implementation (default).
pub struct NoopMetrics;

impl Metrics for NoopMetrics {
    fn record_latency(&self, _operation: &str, _duration_ms: f64) {}
    fn increment_counter(&self, _name: &str, _delta: u64) {}
    fn record_gauge(&self, _name: &str, _value: f64) {}
}

/// Snapshot of collected metrics.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct MetricsSnapshot {
    /// Latency percentiles per operation (p50, p95, p99, count, total_ms).
    pub latencies: HashMap<String, LatencyStats>,
    /// Cumulative counter values.
    pub counters: HashMap<String, u64>,
    /// Last-recorded gauge values.
    pub gauges: HashMap<String, f64>,
}

/// Latency statistics for a single operation.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct LatencyStats {
    pub count: u64,
    pub total_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noop_metrics_compiles_and_runs() {
        let m = NoopMetrics;
        m.record_latency("test_op", 42.0);
        m.increment_counter("test_count", 1);
        m.record_gauge("test_gauge", 2.72);
    }
}
