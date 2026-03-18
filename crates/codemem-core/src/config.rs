//! Persistent configuration for Codemem.
//!
//! Loads/saves a TOML config at `~/.codemem/config.toml`.

use crate::{CodememError, GraphConfig, ScoringWeights, VectorConfig};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Top-level Codemem configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct CodememConfig {
    pub scoring: ScoringWeights,
    pub vector: VectorConfig,
    pub graph: GraphConfig,
    pub embedding: EmbeddingConfig,
    pub storage: StorageConfig,
    pub chunking: ChunkingConfig,
    pub enrichment: EnrichmentConfig,
    pub scip: ScipConfig,
    pub memory: MemoryConfig,
}

impl CodememConfig {
    /// Load configuration from the given path. Validates after loading.
    pub fn load(path: &Path) -> Result<Self, CodememError> {
        let content = std::fs::read_to_string(path)?;
        let config: Self =
            toml::from_str(&content).map_err(|e| CodememError::Config(e.to_string()))?;
        config.validate()?;
        Ok(config)
    }

    /// Validate configuration values.
    ///
    /// Checks that scoring weights are non-negative, dimensions and cache sizes
    /// are positive, chunk size bounds are consistent, and dedup threshold is
    /// in the valid range.
    pub fn validate(&self) -> Result<(), CodememError> {
        // M5: Scoring weights must be finite and non-negative.
        // Check is_finite() first to reject NaN/Inf, then < 0.0 for negatives.
        let w = &self.scoring;
        let weights = [
            w.vector_similarity,
            w.graph_strength,
            w.token_overlap,
            w.temporal,
            w.tag_matching,
            w.importance,
            w.confidence,
            w.recency,
        ];
        if weights.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return Err(CodememError::Config(
                "All scoring weights must be finite and non-negative".to_string(),
            ));
        }

        // Embedding dimensions must be positive
        if self.embedding.dimensions == 0 {
            return Err(CodememError::Config(
                "Embedding dimensions must be > 0".to_string(),
            ));
        }

        // Vector dimensions must be positive
        if self.vector.dimensions == 0 {
            return Err(CodememError::Config(
                "Vector dimensions must be > 0".to_string(),
            ));
        }

        // Cache capacity must be positive
        if self.embedding.cache_capacity == 0 {
            return Err(CodememError::Config(
                "Embedding cache capacity must be > 0".to_string(),
            ));
        }

        // Batch size must be positive
        if self.embedding.batch_size == 0 {
            return Err(CodememError::Config(
                "Embedding batch size must be > 0".to_string(),
            ));
        }

        // Chunk size bounds
        if self.chunking.min_chunk_size >= self.chunking.max_chunk_size {
            return Err(CodememError::Config(
                "min_chunk_size must be less than max_chunk_size".to_string(),
            ));
        }

        // Dedup threshold in [0.0, 1.0] (also rejects NaN via range check)
        if !(0.0..=1.0).contains(&self.enrichment.dedup_similarity_threshold) {
            return Err(CodememError::Config(
                "dedup_similarity_threshold must be between 0.0 and 1.0".to_string(),
            ));
        }

        // Enrichment confidence in [0.0, 1.0]
        if !(0.0..=1.0).contains(&self.enrichment.insight_confidence) {
            return Err(CodememError::Config(
                "insight_confidence must be between 0.0 and 1.0".to_string(),
            ));
        }

        // SCIP max_references_per_symbol must be positive
        if self.scip.max_references_per_symbol == 0 {
            return Err(CodememError::Config(
                "scip.max_references_per_symbol must be > 0".to_string(),
            ));
        }

        // Chunking score thresholds in [0.0, 1.0]
        let thresholds = [
            (
                self.chunking.min_chunk_score_threshold,
                "min_chunk_score_threshold",
            ),
            (
                self.chunking.min_symbol_score_threshold,
                "min_symbol_score_threshold",
            ),
        ];
        for (val, name) in &thresholds {
            if !(0.0..=1.0).contains(val) {
                return Err(CodememError::Config(format!(
                    "{name} must be between 0.0 and 1.0"
                )));
            }
        }

        Ok(())
    }

    /// Save configuration to the given path. Validates before saving.
    pub fn save(&self, path: &Path) -> Result<(), CodememError> {
        // M5: Validate before saving to prevent persisting invalid config.
        self.validate()?;
        let content =
            toml::to_string_pretty(self).map_err(|e| CodememError::Config(e.to_string()))?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Load from the default path, or return defaults if the file doesn't exist.
    pub fn load_or_default() -> Self {
        let path = Self::default_path();
        if path.exists() {
            match Self::load(&path) {
                Ok(config) => config,
                Err(e) => {
                    tracing::warn!("Failed to load config: {e}, using defaults");
                    CodememConfig::default()
                }
            }
        } else {
            Self::default()
        }
    }

    /// Default config path: `~/.codemem/config.toml`.
    pub fn default_path() -> PathBuf {
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".codemem")
            .join("config.toml")
    }
}

/// Embedding provider configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct EmbeddingConfig {
    /// Provider name: "candle" (default), "ollama", or "openai".
    pub provider: String,
    /// Model name (provider-specific). For Candle: HF repo ID (e.g. "BAAI/bge-base-en-v1.5").
    pub model: String,
    /// API URL for remote providers.
    pub url: String,
    /// Embedding dimensions for remote providers (Ollama/OpenAI).
    /// Ignored by Candle — reads `hidden_size` from model's config.json.
    pub dimensions: usize,
    /// LRU cache capacity.
    pub cache_capacity: usize,
    /// Batch size for embedding forward passes (GPU memory trade-off).
    pub batch_size: usize,
    /// Weight dtype: "f32" (default), "f16" (half precision), "bf16".
    /// F16 halves memory and is faster on Metal GPU.
    pub dtype: String,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            provider: "candle".to_string(),
            model: "BAAI/bge-base-en-v1.5".to_string(),
            url: String::new(),
            dimensions: 768,
            cache_capacity: 10_000,
            batch_size: 16,
            dtype: "f32".to_string(),
        }
    }
}

/// Storage configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct StorageConfig {
    /// Backend type: "sqlite" (default) or "postgres".
    #[serde(default = "default_storage_backend")]
    pub backend: String,
    /// Connection URL for remote backends (e.g., "postgres://user:pass@host/db").
    #[serde(default)]
    pub url: Option<String>,
    /// SQLite cache size in MB.
    pub cache_size_mb: u32,
    /// SQLite busy timeout in seconds.
    pub busy_timeout_secs: u64,
}

fn default_storage_backend() -> String {
    "sqlite".to_string()
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            backend: default_storage_backend(),
            url: None,
            cache_size_mb: 64,
            busy_timeout_secs: 5,
        }
    }
}

/// CST-aware code chunking configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ChunkingConfig {
    /// Whether chunking is enabled during indexing.
    pub enabled: bool,
    /// Maximum chunk size in non-whitespace characters.
    pub max_chunk_size: usize,
    /// Minimum chunk size in non-whitespace characters.
    pub min_chunk_size: usize,
    /// Whether to auto-compact the graph after indexing.
    pub auto_compact: bool,
    /// Maximum number of retained chunk graph-nodes per file after compaction.
    pub max_retained_chunks_per_file: usize,
    /// Minimum chunk score (0.0–1.0) to survive compaction.
    pub min_chunk_score_threshold: f64,
    /// Maximum number of retained symbol graph-nodes per file after compaction.
    pub max_retained_symbols_per_file: usize,
    /// Minimum symbol score (0.0–1.0) to survive compaction.
    pub min_symbol_score_threshold: f64,
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_chunk_size: 1500,
            min_chunk_size: 50,
            auto_compact: true,
            max_retained_chunks_per_file: 10,
            min_chunk_score_threshold: 0.2,
            max_retained_symbols_per_file: 15,
            min_symbol_score_threshold: 0.15,
        }
    }
}

/// SCIP integration configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ScipConfig {
    /// Master switch for SCIP integration.
    pub enabled: bool,
    /// Check PATH for available indexers.
    pub auto_detect_indexers: bool,
    /// Cache .scip files between runs.
    pub cache_index: bool,
    /// Re-index if cache older than this many hours.
    pub cache_ttl_hours: u64,
    /// Create ext: nodes for dependency symbols.
    pub create_external_nodes: bool,
    /// Skip utility symbols with excessive fan-out (fallback for kinds without per-kind limits).
    pub max_references_per_symbol: usize,
    /// Attach hover docs as memories to nodes.
    pub store_docs_as_memories: bool,
    /// Build nested containment tree from SCIP descriptor chains.
    /// When true: file→module→class→method. When false: flat file→symbol.
    pub hierarchical_containment: bool,
    /// Collapse intra-class edges into parent metadata.
    pub collapse_intra_class_edges: bool,
    /// Per-kind fan-out limits (0 = use max_references_per_symbol fallback).
    pub fan_out_limits: FanOutLimits,
    /// Per-language indexer command overrides.
    pub indexers: ScipIndexersConfig,
}

/// Per-kind inbound reference limits. A module can be widely imported; a function less so.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct FanOutLimits {
    pub module: usize,
    pub function: usize,
    pub method: usize,
    pub class: usize,
}

impl Default for FanOutLimits {
    fn default() -> Self {
        Self {
            module: 200,
            function: 30,
            method: 30,
            class: 50,
        }
    }
}

impl Default for ScipConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            auto_detect_indexers: true,
            cache_index: true,
            cache_ttl_hours: 24,
            create_external_nodes: true,
            max_references_per_symbol: 100,
            store_docs_as_memories: true,
            hierarchical_containment: true,
            collapse_intra_class_edges: true,
            fan_out_limits: FanOutLimits::default(),
            indexers: ScipIndexersConfig::default(),
        }
    }
}

/// Per-language SCIP indexer command overrides. Empty string means auto-detect from PATH.
///
/// Commands are split on whitespace — paths with spaces are **not** supported.
/// Use symlinks or PATH entries for indexers in directories with spaces.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ScipIndexersConfig {
    pub rust: String,
    pub typescript: String,
    pub python: String,
    pub java: String,
    pub go: String,
}

/// Enrichment pipeline configuration for controlling insight generation thresholds.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct EnrichmentConfig {
    /// Minimum commit count for a file to generate a high-activity insight.
    pub git_min_commit_count: usize,
    /// Minimum co-change count for a file pair to generate a coupling insight.
    pub git_min_co_change_count: usize,
    /// Minimum coupling degree for a node to generate a high-coupling insight.
    pub perf_min_coupling_degree: usize,
    /// Minimum symbol count for a file to generate a complexity insight.
    pub perf_min_symbol_count: usize,
    /// Default confidence for auto-generated insights.
    pub insight_confidence: f64,
    /// Cosine similarity threshold for deduplicating insights.
    pub dedup_similarity_threshold: f64,
    /// Dead code detection settings.
    pub dead_code: DeadCodeConfig,
}

/// Dead code detection configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DeadCodeConfig {
    /// Whether dead code detection is enabled.
    pub enabled: bool,
    /// Decorator/attribute names that exempt a symbol from dead code detection
    /// (e.g., route handlers, test fixtures, CLI commands).
    pub exempt_decorators: Vec<String>,
    /// Symbol kind strings that are exempt (e.g., "constructor", "test").
    pub exempt_kinds: Vec<String>,
    /// Minimum number of symbol nodes before dead code analysis runs
    /// (avoids false positives on tiny graphs).
    pub min_symbols: usize,
}

impl Default for DeadCodeConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            exempt_decorators: vec![
                "app.route".into(),
                "route".into(),
                "pytest.fixture".into(),
                "fixture".into(),
                "click.command".into(),
                "celery.task".into(),
                "property".into(),
                "staticmethod".into(),
                "classmethod".into(),
                "override".into(),
                "abstractmethod".into(),
                "test".into(),
                "tokio::test".into(),
                "async_trait".into(),
            ],
            exempt_kinds: vec!["constructor".into(), "test".into()],
            min_symbols: 10,
        }
    }
}

impl Default for EnrichmentConfig {
    fn default() -> Self {
        Self {
            git_min_commit_count: 25,
            git_min_co_change_count: 5,
            perf_min_coupling_degree: 25,
            perf_min_symbol_count: 30,
            insight_confidence: 0.5,
            dedup_similarity_threshold: 0.90,
            dead_code: DeadCodeConfig::default(),
        }
    }
}

/// Memory expiration settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MemoryConfig {
    /// Default TTL in hours for session-scoped memories (memories with a session_id).
    /// Set to 0 to disable auto-expiry for session memories.
    pub default_session_ttl_hours: u64,
    /// Expire `static-analysis` tagged memories when the underlying file is re-indexed
    /// with a changed content hash.
    pub expire_enrichments_on_reindex: bool,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            default_session_ttl_hours: 168, // 7 days
            expire_enrichments_on_reindex: true,
        }
    }
}

#[cfg(test)]
#[path = "tests/config_tests.rs"]
mod tests;
