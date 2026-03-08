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
    // Use a path that doesn't exist to test the default fallback
    let nonexistent = std::path::PathBuf::from("/tmp/codemem_test_no_exist/config.toml");
    let _ = std::fs::remove_file(&nonexistent); // ensure it doesn't exist
    let config = CodememConfig::load(&nonexistent).unwrap_or_default();
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

#[test]
fn partial_toml_defaults_for_many_fields() {
    // Only set one embedding field; verify 5+ other fields get correct defaults
    let partial = r#"
[embedding]
provider = "ollama"
"#;
    let config: CodememConfig = toml::from_str(partial).expect("partial TOML should parse");
    assert_eq!(config.embedding.provider, "ollama");
    // scoring defaults
    assert!((config.scoring.vector_similarity - 0.25).abs() < f64::EPSILON);
    assert!((config.scoring.graph_strength - 0.20).abs() < f64::EPSILON);
    assert!((config.scoring.recency - 0.05).abs() < f64::EPSILON);
    // vector defaults
    assert_eq!(config.vector.dimensions, 768);
    assert_eq!(config.vector.m, 16);
    // embedding defaults (non-overridden)
    assert_eq!(config.embedding.dimensions, 768);
    assert_eq!(config.embedding.cache_capacity, 10_000);
    assert_eq!(config.embedding.batch_size, 16);
    // storage defaults
    assert_eq!(config.storage.cache_size_mb, 64);
    // chunking defaults
    assert!(config.chunking.enabled);
    assert_eq!(config.chunking.max_chunk_size, 1500);
    // enrichment defaults
    assert_eq!(config.enrichment.git_min_commit_count, 25);
}

#[test]
fn invalid_toml_syntax_returns_parse_error() {
    let invalid = r#"
[scoring
vector_similarity = 0.4
"#;
    let result: Result<CodememConfig, _> = toml::from_str(invalid);
    assert!(result.is_err());
}

#[test]
fn validate_rejects_nan_scoring_weight() {
    let mut config = CodememConfig::default();
    config.scoring.vector_similarity = f64::NAN;
    assert!(config.validate().is_err());
}

#[test]
fn validate_rejects_infinity_scoring_weight() {
    let mut config = CodememConfig::default();
    config.scoring.graph_strength = f64::INFINITY;
    assert!(config.validate().is_err());
}

#[test]
fn validate_rejects_negative_scoring_weight() {
    let mut config = CodememConfig::default();
    config.scoring.token_overlap = -0.1;
    assert!(config.validate().is_err());
}

#[test]
fn validate_rejects_nan_temporal_weight() {
    let mut config = CodememConfig::default();
    config.scoring.temporal = f64::NAN;
    assert!(config.validate().is_err());
}

#[test]
fn validate_rejects_nan_tag_matching_weight() {
    let mut config = CodememConfig::default();
    config.scoring.tag_matching = f64::NAN;
    assert!(config.validate().is_err());
}

#[test]
fn validate_rejects_nan_importance_weight() {
    let mut config = CodememConfig::default();
    config.scoring.importance = f64::NAN;
    assert!(config.validate().is_err());
}

#[test]
fn validate_rejects_nan_confidence_weight() {
    let mut config = CodememConfig::default();
    config.scoring.confidence = f64::NAN;
    assert!(config.validate().is_err());
}

#[test]
fn validate_rejects_nan_recency_weight() {
    let mut config = CodememConfig::default();
    config.scoring.recency = f64::NAN;
    assert!(config.validate().is_err());
}

#[test]
fn validate_rejects_negative_temporal_weight() {
    let mut config = CodememConfig::default();
    config.scoring.temporal = -0.01;
    assert!(config.validate().is_err());
}

#[test]
fn validate_rejects_negative_tag_matching_weight() {
    let mut config = CodememConfig::default();
    config.scoring.tag_matching = -1.0;
    assert!(config.validate().is_err());
}

#[test]
fn validate_rejects_negative_importance_weight() {
    let mut config = CodememConfig::default();
    config.scoring.importance = -0.5;
    assert!(config.validate().is_err());
}

#[test]
fn validate_rejects_negative_confidence_weight() {
    let mut config = CodememConfig::default();
    config.scoring.confidence = -100.0;
    assert!(config.validate().is_err());
}

#[test]
fn validate_rejects_negative_recency_weight() {
    let mut config = CodememConfig::default();
    config.scoring.recency = -0.001;
    assert!(config.validate().is_err());
}

#[test]
fn validate_rejects_negative_vector_similarity_weight() {
    let mut config = CodememConfig::default();
    config.scoring.vector_similarity = -0.25;
    assert!(config.validate().is_err());
}

#[test]
fn validate_rejects_negative_graph_strength_weight() {
    let mut config = CodememConfig::default();
    config.scoring.graph_strength = -0.1;
    assert!(config.validate().is_err());
}

#[test]
fn validate_rejects_zero_vector_dimensions() {
    let mut config = CodememConfig::default();
    config.vector.dimensions = 0;
    assert!(config.validate().is_err());
}

#[test]
fn validate_rejects_zero_embedding_cache_capacity() {
    let mut config = CodememConfig::default();
    config.embedding.cache_capacity = 0;
    assert!(config.validate().is_err());
}

#[test]
fn validate_rejects_zero_embedding_batch_size() {
    let mut config = CodememConfig::default();
    config.embedding.batch_size = 0;
    assert!(config.validate().is_err());
}

#[test]
fn validate_rejects_zero_embedding_dimensions() {
    let mut config = CodememConfig::default();
    config.embedding.dimensions = 0;
    assert!(config.validate().is_err());
}

#[test]
fn validate_rejects_nan_insight_confidence() {
    let mut config = CodememConfig::default();
    config.enrichment.insight_confidence = f64::NAN;
    assert!(config.validate().is_err());
}

#[test]
fn validate_rejects_nan_chunk_score_threshold() {
    let mut config = CodememConfig::default();
    config.chunking.min_chunk_score_threshold = f64::NAN;
    assert!(config.validate().is_err());
}

#[test]
fn validate_rejects_nan_symbol_score_threshold() {
    let mut config = CodememConfig::default();
    config.chunking.min_symbol_score_threshold = f64::INFINITY;
    assert!(config.validate().is_err());
}

#[test]
fn save_rejects_invalid_config() {
    let dir = std::env::temp_dir().join("codemem_save_validate_test");
    let _ = std::fs::remove_dir_all(&dir);
    let path = dir.join("config.toml");

    let mut config = CodememConfig::default();
    config.scoring.recency = f64::NAN;
    assert!(config.save(&path).is_err());

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn graph_config_defaults() {
    let config = GraphConfig::default();
    assert!((config.contains_edge_weight - 0.1).abs() < f64::EPSILON);
    assert!((config.calls_edge_weight - 1.0).abs() < f64::EPSILON);
    assert!((config.imports_edge_weight - 0.5).abs() < f64::EPSILON);
}

#[test]
fn graph_config_roundtrips_through_toml() {
    let config = GraphConfig::default();
    let toml_str = toml::to_string_pretty(&config).unwrap();
    let parsed: GraphConfig = toml::from_str(&toml_str).unwrap();
    assert!((parsed.calls_edge_weight - config.calls_edge_weight).abs() < f64::EPSILON);
}

#[test]
fn default_scoring_weights_sum_to_one() {
    let w = ScoringWeights::default();
    let sum = w.vector_similarity
        + w.graph_strength
        + w.token_overlap
        + w.temporal
        + w.tag_matching
        + w.importance
        + w.confidence
        + w.recency;
    assert!(
        (sum - 1.0).abs() < 1e-10,
        "Default scoring weights should sum to 1.0, got {sum}"
    );
}
