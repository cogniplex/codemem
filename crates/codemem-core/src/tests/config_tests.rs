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
