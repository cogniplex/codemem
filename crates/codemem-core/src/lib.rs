//! codemem-core: Shared types, traits, and errors for the Codemem memory engine.

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
    RelationshipType, Repository, ScoreBreakdown, ScoringWeights, SearchResult, Session,
    SessionActivitySummary, UnresolvedRefData, VectorConfig, ENRICHMENT_ANALYSES,
};
