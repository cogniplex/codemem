//! In-memory metrics collector for MCP tool calls.

use codemem_core::{LatencyStats, Metrics, MetricsSnapshot};
use std::collections::HashMap;
use std::sync::Mutex;

/// In-memory metrics collector.
///
/// Collects latency samples, counter increments, and gauge values.
/// Thread-safe via internal `Mutex`.
pub struct InMemoryMetrics {
    inner: Mutex<Inner>,
}

struct Inner {
    /// Raw latency samples per operation (kept for percentile calculation).
    latency_samples: HashMap<String, Vec<f64>>,
    /// Cumulative counters.
    counters: HashMap<String, u64>,
    /// Point-in-time gauges.
    gauges: HashMap<String, f64>,
}

impl InMemoryMetrics {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(Inner {
                latency_samples: HashMap::new(),
                counters: HashMap::new(),
                gauges: HashMap::new(),
            }),
        }
    }

    /// Take a snapshot of all collected metrics.
    pub fn snapshot(&self) -> MetricsSnapshot {
        let inner = match self.inner.lock() {
            Ok(guard) => guard,
            Err(e) => {
                tracing::warn!("Metrics lock poisoned: {e}");
                return MetricsSnapshot::default();
            }
        };

        let latencies: HashMap<String, LatencyStats> = inner
            .latency_samples
            .iter()
            .map(|(name, samples)| {
                let stats = compute_latency_stats(samples);
                (name.clone(), stats)
            })
            .collect();

        MetricsSnapshot {
            latencies,
            counters: inner.counters.clone(),
            gauges: inner.gauges.clone(),
        }
    }
}

impl Default for InMemoryMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl Metrics for InMemoryMetrics {
    fn record_latency(&self, operation: &str, duration_ms: f64) {
        if let Ok(mut inner) = self.inner.lock() {
            inner
                .latency_samples
                .entry(operation.to_string())
                .or_default()
                .push(duration_ms);
        }
    }

    fn increment_counter(&self, name: &str, delta: u64) {
        if let Ok(mut inner) = self.inner.lock() {
            *inner.counters.entry(name.to_string()).or_insert(0) += delta;
        }
    }

    fn record_gauge(&self, name: &str, value: f64) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.gauges.insert(name.to_string(), value);
        }
    }
}

/// Compute percentile-based statistics from a slice of samples.
fn compute_latency_stats(samples: &[f64]) -> LatencyStats {
    if samples.is_empty() {
        return LatencyStats::default();
    }

    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let count = sorted.len() as u64;
    let total: f64 = sorted.iter().sum();
    let min = sorted[0];
    let max = sorted[sorted.len() - 1];

    let p50 = percentile(&sorted, 50.0);
    let p95 = percentile(&sorted, 95.0);
    let p99 = percentile(&sorted, 99.0);

    LatencyStats {
        count,
        total_ms: total,
        min_ms: min,
        max_ms: max,
        p50_ms: p50,
        p95_ms: p95,
        p99_ms: p99,
    }
}

fn percentile(sorted: &[f64], pct: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = (pct / 100.0 * (sorted.len() as f64 - 1.0)).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

#[cfg(test)]
mod tests {
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
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let stats = compute_latency_stats(&samples);
        assert_eq!(stats.count, 10);
        assert!((stats.min_ms - 1.0).abs() < f64::EPSILON);
        assert!((stats.max_ms - 10.0).abs() < f64::EPSILON);
        assert!((stats.p50_ms - 5.5).abs() < 1.5); // roughly median
    }
}
