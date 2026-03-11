use super::*;

#[test]
fn available_enrichers_returns_empty_when_tools_missing() {
    // On CI/test machines, pyright/tsc might not be installed
    // Just verify the function doesn't panic
    let _ = available_enrichers();
}

#[test]
fn run_enrichment_empty_inputs() {
    let result = run_enrichment(&[], &[]);
    assert!(result.is_empty());
}

#[test]
fn run_enrichment_no_enrichers() {
    let targets = vec![EnrichmentTarget {
        file_path: "test.py".to_string(),
        refs: vec![],
    }];
    let result = run_enrichment(&targets, &[]);
    assert!(result.is_empty());
}

#[test]
fn pyright_detect_project_root_finds_pyproject() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("pyproject.toml"),
        "[project]\nname = \"test\"",
    )
    .unwrap();
    std::fs::create_dir_all(dir.path().join("src")).unwrap();
    let file = dir.path().join("src/main.py");
    std::fs::write(&file, "").unwrap();

    let enricher = pyright::PyrightEnricher::new();
    let root = enricher.detect_project_root(&file);
    assert_eq!(root, Some(dir.path().to_path_buf()));
}

#[test]
fn tsserver_detect_project_root_finds_tsconfig() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("tsconfig.json"), "{}").unwrap();
    std::fs::create_dir_all(dir.path().join("src")).unwrap();
    let file = dir.path().join("src/app.ts");
    std::fs::write(&file, "").unwrap();

    let enricher = tsserver::TsServerEnricher::new();
    let root = enricher.detect_project_root(&file);
    assert_eq!(root, Some(dir.path().to_path_buf()));
}

#[test]
fn detect_project_root_returns_none_for_rootless() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("orphan.py");
    std::fs::write(&file, "").unwrap();

    let enricher = pyright::PyrightEnricher::new();
    let root = enricher.detect_project_root(&file);
    assert!(root.is_none());
}
