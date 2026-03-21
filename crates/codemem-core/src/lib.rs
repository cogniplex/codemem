//! codemem-core: Shared types, traits, and errors for the Codemem memory engine.

// ── PageRank algorithm constants ──────────────────────────────────────────────
/// Default PageRank damping factor. Higher values give more weight to graph structure,
/// lower values favor uniform distribution (common: 0.85).
pub const PAGERANK_DAMPING_DEFAULT: f64 = 0.85;

/// Default maximum iterations for PageRank power iteration. Higher values increase accuracy
/// but cost; 100 iterations typically suffices for convergence (common: 100).
pub const PAGERANK_ITERATIONS_DEFAULT: usize = 100;

/// Default convergence tolerance for PageRank. Iteration stops when max score delta
/// drops below this threshold (common: 1e-6).
pub const PAGERANK_TOLERANCE_DEFAULT: f64 = 1e-6;

pub mod config;
pub mod error;
pub mod metrics;
pub mod traits;
pub mod types;
pub mod utils;

// ── utils ───────────────────────────────────────────────────────────────────
pub use utils::truncate;

// ── config ──────────────────────────────────────────────────────────────────
pub use config::{
    ChunkingConfig, CodememConfig, EmbeddingConfig, EnrichmentConfig, FanOutLimits, MemoryConfig,
    ScipConfig, ScipIndexersConfig, StorageConfig,
};

// ── error ───────────────────────────────────────────────────────────────────
pub use error::CodememError;

// ── metrics ─────────────────────────────────────────────────────────────────
pub use metrics::{LatencyStats, Metrics, MetricsSnapshot, NoopMetrics};

// ── traits ──────────────────────────────────────────────────────────────────
pub use traits::{
    ConsolidationLogEntry, EmbeddingProvider, GraphBackend, GraphStats, PendingUnresolvedRef,
    StorageBackend, StorageStats, VectorBackend, VectorStats,
};

// ── types ───────────────────────────────────────────────────────────────────
pub use types::{
    content_hash, DetectedPattern, DistanceMetric, Edge, GraphConfig, GraphNode, MemoryNode,
    MemoryType, NodeCoverageEntry, NodeKind, NodeMemoryResult, PatternType, RawGraphMetrics,
    RelationshipType, Repository, ScopeContext, ScoreBreakdown, ScoringWeights, SearchResult,
    Session, SessionActivitySummary, UnresolvedRefData, VectorConfig, ENRICHMENT_ANALYSES,
};
