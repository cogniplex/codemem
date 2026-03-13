use super::*;
use codemem_core::ScipConfig;

#[test]
fn test_detect_languages_with_cargo_toml() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("Cargo.toml"),
        "[package]\nname = \"test\"\n",
    )
    .unwrap();

    let orchestrator = ScipOrchestrator::new(ScipConfig::default());
    let langs = orchestrator.detect_languages(dir.path());
    assert!(langs.contains(&ScipLanguage::Rust));
}

#[test]
fn test_detect_languages_with_package_json() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("package.json"), "{}").unwrap();

    let orchestrator = ScipOrchestrator::new(ScipConfig::default());
    let langs = orchestrator.detect_languages(dir.path());
    assert!(langs.contains(&ScipLanguage::TypeScript));
}

#[test]
fn test_detect_languages_with_go_mod() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("go.mod"), "module example.com/foo\n").unwrap();

    let orchestrator = ScipOrchestrator::new(ScipConfig::default());
    let langs = orchestrator.detect_languages(dir.path());
    assert!(langs.contains(&ScipLanguage::Go));
}

#[test]
fn test_detect_languages_with_pyproject_toml() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("pyproject.toml"),
        "[project]\nname = \"foo\"\n",
    )
    .unwrap();

    let orchestrator = ScipOrchestrator::new(ScipConfig::default());
    let langs = orchestrator.detect_languages(dir.path());
    assert!(langs.contains(&ScipLanguage::Python));
}

#[test]
fn test_detect_languages_empty_dir() {
    let dir = tempfile::tempdir().unwrap();
    let orchestrator = ScipOrchestrator::new(ScipConfig::default());
    let langs = orchestrator.detect_languages(dir.path());
    assert!(langs.is_empty());
}

#[test]
fn test_detect_languages_multiple() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("Cargo.toml"),
        "[package]\nname = \"test\"\n",
    )
    .unwrap();
    std::fs::write(dir.path().join("package.json"), "{}").unwrap();

    let orchestrator = ScipOrchestrator::new(ScipConfig::default());
    let langs = orchestrator.detect_languages(dir.path());
    assert!(langs.contains(&ScipLanguage::Rust));
    assert!(langs.contains(&ScipLanguage::TypeScript));
}

#[test]
fn test_config_command_override() {
    let mut config = ScipConfig::default();
    config.indexers.rust = "my-custom-rust-analyzer scip .".to_string();

    let orchestrator = ScipOrchestrator::new(config);
    let cmd = orchestrator.config_command_for(ScipLanguage::Rust);
    assert_eq!(
        cmd.map(|s| s.as_str()),
        Some("my-custom-rust-analyzer scip .")
    );
}

#[test]
fn test_config_command_empty_means_autodetect() {
    let config = ScipConfig::default();
    let orchestrator = ScipOrchestrator::new(config);
    let cmd = orchestrator.config_command_for(ScipLanguage::Rust);
    assert_eq!(cmd, None);
}

#[test]
fn test_parse_shell_command() {
    let (prog, args) = parse_shell_command("rust-analyzer scip .").unwrap();
    assert_eq!(prog, "rust-analyzer");
    assert_eq!(args, vec!["scip", "."]);
}

#[test]
fn test_parse_shell_command_empty() {
    assert!(parse_shell_command("").is_err());
}

#[test]
fn test_parse_shell_command_single() {
    let (prog, args) = parse_shell_command("scip-go").unwrap();
    assert_eq!(prog, "scip-go");
    assert!(args.is_empty());
}

#[test]
fn test_run_on_empty_dir_returns_empty_result() {
    let dir = tempfile::tempdir().unwrap();
    let orchestrator = ScipOrchestrator::new(ScipConfig::default());
    let result = orchestrator.run(dir.path(), "test").unwrap();
    assert!(result.indexed_languages.is_empty());
    assert!(result.scip_result.definitions.is_empty());
}

#[test]
fn test_available_indexers_with_auto_detect_disabled() {
    let config = ScipConfig {
        auto_detect_indexers: false,
        ..ScipConfig::default()
    };

    let orchestrator = ScipOrchestrator::new(config);
    let langs = vec![ScipLanguage::Rust, ScipLanguage::Python];
    let available = orchestrator.detect_available_indexers(&langs);
    // With auto_detect disabled and no config overrides, nothing should be available.
    assert!(available.is_empty());
}

#[test]
fn test_available_indexers_with_config_override() {
    let config = ScipConfig {
        auto_detect_indexers: false,
        indexers: codemem_core::ScipIndexersConfig {
            rust: "custom-rust-indexer scip .".to_string(),
            ..Default::default()
        },
        ..ScipConfig::default()
    };

    let orchestrator = ScipOrchestrator::new(config);
    let langs = vec![ScipLanguage::Rust, ScipLanguage::Python];
    let available = orchestrator.detect_available_indexers(&langs);
    assert_eq!(available, vec![ScipLanguage::Rust]);
}

#[test]
fn test_check_cache_nonexistent() {
    let dir = tempfile::tempdir().unwrap();
    let result = check_cache(dir.path(), ScipLanguage::Rust, 24);
    assert!(result.is_none());
}

#[test]
fn test_check_cache_fresh() {
    let dir = tempfile::tempdir().unwrap();
    let cache_path = dir.path().join("index-rust.scip");
    std::fs::write(&cache_path, b"dummy").unwrap();

    let result = check_cache(dir.path(), ScipLanguage::Rust, 24);
    assert!(result.is_some());
    assert!(result.unwrap().valid);
}

#[test]
fn test_language_name() {
    assert_eq!(ScipLanguage::Rust.name(), "rust");
    assert_eq!(ScipLanguage::TypeScript.name(), "typescript");
    assert_eq!(ScipLanguage::Python.name(), "python");
    assert_eq!(ScipLanguage::Java.name(), "java");
    assert_eq!(ScipLanguage::Go.name(), "go");
}

#[test]
fn test_merge_empty_files() {
    let orchestrator = ScipOrchestrator::new(ScipConfig::default());
    let result = orchestrator
        .merge_scip_files(&[], Path::new("/test"))
        .unwrap();
    assert!(result.definitions.is_empty());
    assert!(result.references.is_empty());
    assert!(result.covered_files.is_empty());
}

#[test]
fn test_namespace_substitution() {
    let (prog, args) = parse_shell_command("scip-python index . --project-name=myproject").unwrap();
    assert_eq!(prog, "scip-python");
    assert_eq!(args, vec!["index", ".", "--project-name=myproject"]);

    // Simulate the {namespace} replacement.
    let cmd = "scip-python index . --project-name={namespace}";
    let expanded = cmd.replace("{namespace}", "my-ns");
    let (prog, args) = parse_shell_command(&expanded).unwrap();
    assert_eq!(prog, "scip-python");
    assert_eq!(args, vec!["index", ".", "--project-name=my-ns"]);
}
