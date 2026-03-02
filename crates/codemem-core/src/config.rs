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
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_chunk_size: 1500,
            min_chunk_size: 50,
        }
    }
}

#[cfg(test)]
#[path = "tests/config_tests.rs"]
mod tests;
