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
    /// Load configuration from the given path.
    pub fn load(path: &Path) -> Result<Self, CodememError> {
        let content = std::fs::read_to_string(path)?;
        toml::from_str(&content).map_err(|e| CodememError::Config(e.to_string()))
    }

    /// Save configuration to the given path.
    pub fn save(&self, path: &Path) -> Result<(), CodememError> {
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
            Self::load(&path).unwrap_or_default()
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
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            provider: "candle".to_string(),
            model: "BAAI/bge-base-en-v1.5".to_string(),
            url: String::new(),
            dimensions: 768,
            cache_capacity: 10_000,
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
