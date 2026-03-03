//! In-memory metrics collector for operational metrics.

use codemem_core::{LatencyStats, Metrics, MetricsSnapshot};
use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;

/// Maximum number of latency samples to retain per operation.
const MAX_SAMPLES: usize = 10_000;

/// In-memory metrics collector.
///
/// Collects latency samples, counter increments, and gauge values.
/// Thread-safe via internal `Mutex`.
pub struct InMemoryMetrics {
    inner: Mutex<Inner>,
}

struct Inner {
    /// Raw latency samples per operation, capped at `MAX_SAMPLES` per key.
    latency_samples: HashMap<String, VecDeque<f64>>,
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
            let samples = inner
                .latency_samples
                .entry(operation.to_string())
                .or_default();
            if samples.len() >= MAX_SAMPLES {
                samples.pop_front();
            }
            samples.push_back(duration_ms);
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

/// Compute percentile-based statistics from a deque of samples.
/// The deque is capped at `MAX_SAMPLES`, so the sort is bounded.
fn compute_latency_stats(samples: &VecDeque<f64>) -> LatencyStats {
    if samples.is_empty() {
        return LatencyStats::default();
    }

    let mut sorted: Vec<f64> = samples.iter().copied().collect();
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
#[path = "tests/metrics_tests.rs"]
mod tests;
