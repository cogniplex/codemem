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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_roundtrips_through_toml() {
        let config = CodememConfig::default();
        let toml_str =
            toml::to_string_pretty(&config).expect("default config should serialize to TOML");
        let parsed: CodememConfig =
            toml::from_str(&toml_str).expect("serialized TOML should parse back");
        assert!((parsed.scoring.vector_similarity - 0.25).abs() < f64::EPSILON);
        assert_eq!(parsed.vector.dimensions, 768);
        assert_eq!(parsed.embedding.provider, "candle");
    }

    #[test]
    fn load_nonexistent_returns_error() {
        let result = CodememConfig::load(Path::new("/tmp/nonexistent_codemem_config.toml"));
        assert!(result.is_err());
    }

    #[test]
    fn save_and_load_roundtrip() {
        let dir = std::env::temp_dir().join("codemem_config_test");
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("config.toml");

        let mut config = CodememConfig::default();
        config.scoring.vector_similarity = 0.5;
        config.storage.cache_size_mb = 128;

        config.save(&path).expect("save should succeed");
        let loaded = CodememConfig::load(&path).expect("load should succeed");

        assert!((loaded.scoring.vector_similarity - 0.5).abs() < f64::EPSILON);
        assert_eq!(loaded.storage.cache_size_mb, 128);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_or_default_returns_default_when_no_file() {
        let config = CodememConfig::load_or_default();
        assert!((config.scoring.vector_similarity - 0.25).abs() < f64::EPSILON);
    }

    #[test]
    fn default_path_ends_with_config_toml() {
        let path = CodememConfig::default_path();
        assert!(path.ends_with("config.toml"));
    }

    #[test]
    fn partial_toml_uses_defaults_for_missing_fields() {
        let partial = r#"
[scoring]
vector_similarity = 0.4
"#;
        let config: CodememConfig = toml::from_str(partial).expect("partial TOML should parse");
        assert!((config.scoring.vector_similarity - 0.4).abs() < f64::EPSILON);
        // Other fields should use defaults
        assert_eq!(config.vector.dimensions, 768);
        assert_eq!(config.embedding.provider, "candle");
    }
}
