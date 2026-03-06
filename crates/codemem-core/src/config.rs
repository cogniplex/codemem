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
    /// Model name (provider-specific).
    pub model: String,
    /// API URL for remote providers.
    pub url: String,
    /// Embedding dimensions.
    pub dimensions: usize,
    /// LRU cache capacity.
    pub cache_capacity: usize,
    /// Batch size for embedding forward passes (GPU memory trade-off).
    pub batch_size: usize,
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
        }
    }
}

/// Storage configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct StorageConfig {
    /// Path to the database file.
    pub db_path: String,
    /// SQLite cache size in MB.
    pub cache_size_mb: u32,
    /// SQLite busy timeout in seconds.
    pub busy_timeout_secs: u64,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            db_path: dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".codemem")
                .join("codemem.db")
                .to_string_lossy()
                .into_owned(),
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
        }
    }
}

#[cfg(test)]
#[path = "tests/config_tests.rs"]
mod tests;
