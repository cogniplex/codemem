//! codemem-core: Shared types, traits, and errors for the Codemem memory engine.

pub mod config;
pub mod error;
pub mod metrics;
pub mod traits;
pub mod types;

// ── config ──────────────────────────────────────────────────────────────────
pub use config::{ChunkingConfig, CodememConfig, EmbeddingConfig, EnrichmentConfig, StorageConfig};

// ── error ───────────────────────────────────────────────────────────────────
pub use error::CodememError;

// ── metrics ─────────────────────────────────────────────────────────────────
pub use metrics::{LatencyStats, Metrics, MetricsSnapshot, NoopMetrics};

// ── traits ──────────────────────────────────────────────────────────────────
pub use traits::{
    ConsolidationLogEntry, EmbeddingProvider, GraphBackend, GraphStats, StorageBackend,
    StorageStats, VectorBackend, VectorStats,
};

// ── types ───────────────────────────────────────────────────────────────────
pub use types::{
    content_hash, DetectedPattern, DistanceMetric, Edge, GraphConfig, GraphNode, MemoryNode,
    MemoryType, NodeCoverageEntry, NodeKind, NodeMemoryResult, PatternType, RelationshipType,
    Repository, ScoreBreakdown, ScoringWeights, SearchResult, Session, SessionActivitySummary,
    VectorConfig,
};
