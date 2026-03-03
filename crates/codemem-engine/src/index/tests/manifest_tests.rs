use super::*;
use std::fs;

fn write_temp_file(dir: &Path, name: &str, content: &str) -> std::path::PathBuf {
    let path = dir.join(name);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("failed to create parent dirs");
    }
    fs::write(&path, content).expect("failed to write temp file");
    path
}

#[test]
fn parse_simple_cargo_toml() {
    let dir = std::env::temp_dir().join("codemem_test_cargo_simple");
    fs::create_dir_all(&dir).ok();

    let content = r#"
[package]
name = "my-crate"
version = "0.1.0"
edition = "2021"

[dependencies]
serde = "1.0"
tokio = { version = "1", features = ["full"] }

[dev-dependencies]
tempfile = "3"
"#;
    let path = write_temp_file(&dir, "Cargo.toml", content);
    let result = parse_cargo_toml(&path).expect("should parse");

    assert!(
        result.packages.contains_key("my-crate"),
        "Expected my-crate package, got: {:?}",
        result.packages
    );

    let deps: Vec<_> = result.dependencies.iter().filter(|d| !d.dev).collect();
    assert!(
        deps.iter().any(|d| d.name == "serde" && d.version == "1.0"),
        "Expected serde dep, got: {:#?}",
        deps
    );
    assert!(
        deps.iter().any(|d| d.name == "tokio" && d.version == "1"),
        "Expected tokio dep, got: {:#?}",
        deps
    );

    let dev_deps: Vec<_> = result.dependencies.iter().filter(|d| d.dev).collect();
    assert!(
        dev_deps.iter().any(|d| d.name == "tempfile"),
        "Expected tempfile dev dep, got: {:#?}",
        dev_deps
    );

    // Cleanup
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn parse_cargo_workspace() {
    let dir = std::env::temp_dir().join("codemem_test_cargo_workspace");
    fs::create_dir_all(&dir).ok();

    let content = r#"
[workspace]
members = [
    "crates/core",
    "crates/cli",
    "crates/server",
]
resolver = "2"

[workspace.package]
version = "0.1.0"
"#;
    let path = write_temp_file(&dir, "Cargo.toml", content);
    let result = parse_cargo_toml(&path).expect("should parse");

    assert_eq!(result.workspaces.len(), 1);
    let ws = &result.workspaces[0];
    assert_eq!(ws.kind, "cargo");
    assert_eq!(ws.members.len(), 3);
    assert!(ws.members.contains(&"crates/core".to_string()));
    assert!(ws.members.contains(&"crates/cli".to_string()));
    assert!(ws.members.contains(&"crates/server".to_string()));

    // Cleanup
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn parse_simple_package_json() {
    let dir = std::env::temp_dir().join("codemem_test_npm_simple");
    fs::create_dir_all(&dir).ok();

    let content = r#"
{
    "name": "my-app",
    "version": "1.0.0",
    "dependencies": {
        "react": "^18.0.0",
        "express": "^4.18.0"
    },
    "devDependencies": {
        "typescript": "^5.0.0",
        "jest": "^29.0.0"
    }
}
"#;
    let path = write_temp_file(&dir, "package.json", content);
    let result = parse_package_json(&path).expect("should parse");

    assert!(result.packages.contains_key("my-app"));

    let deps: Vec<_> = result.dependencies.iter().filter(|d| !d.dev).collect();
    assert!(
        deps.iter().any(|d| d.name == "react"),
        "Expected react dep, got: {:#?}",
        deps
    );
    assert!(
        deps.iter().any(|d| d.name == "express"),
        "Expected express dep, got: {:#?}",
        deps
    );

    let dev_deps: Vec<_> = result.dependencies.iter().filter(|d| d.dev).collect();
    assert!(
        dev_deps.iter().any(|d| d.name == "typescript"),
        "Expected typescript dev dep, got: {:#?}",
        dev_deps
    );
    assert!(
        dev_deps.iter().any(|d| d.name == "jest"),
        "Expected jest dev dep, got: {:#?}",
        dev_deps
    );

    // Cleanup
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn parse_npm_workspaces() {
    let dir = std::env::temp_dir().join("codemem_test_npm_workspaces");
    fs::create_dir_all(&dir).ok();

    let content = r#"
{
    "name": "my-monorepo",
    "version": "1.0.0",
    "workspaces": [
        "packages/*",
        "apps/*"
    ]
}
"#;
    let path = write_temp_file(&dir, "package.json", content);
    let result = parse_package_json(&path).expect("should parse");

    assert_eq!(result.workspaces.len(), 1);
    let ws = &result.workspaces[0];
    assert_eq!(ws.kind, "npm");
    assert_eq!(ws.members.len(), 2);
    assert!(ws.members.contains(&"packages/*".to_string()));
    assert!(ws.members.contains(&"apps/*".to_string()));

    // Cleanup
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn parse_npm_workspaces_object_form() {
    let dir = std::env::temp_dir().join("codemem_test_npm_workspaces_obj");
    fs::create_dir_all(&dir).ok();

    let content = r#"
{
    "name": "my-monorepo",
    "version": "1.0.0",
    "workspaces": {
        "packages": [
            "packages/*",
            "tools/*"
        ]
    }
}
"#;
    let path = write_temp_file(&dir, "package.json", content);
    let result = parse_package_json(&path).expect("should parse");

    assert_eq!(result.workspaces.len(), 1);
    let ws = &result.workspaces[0];
    assert_eq!(ws.kind, "npm");
    assert!(ws.members.contains(&"packages/*".to_string()));
    assert!(ws.members.contains(&"tools/*".to_string()));

    // Cleanup
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn parse_nonexistent_file_returns_none() {
    let path = Path::new("/nonexistent/path/Cargo.toml");
    assert!(parse_cargo_toml(path).is_none());

    let path = Path::new("/nonexistent/path/package.json");
    assert!(parse_package_json(path).is_none());
}

#[test]
fn parse_invalid_toml_returns_none() {
    let dir = std::env::temp_dir().join("codemem_test_invalid_toml");
    fs::create_dir_all(&dir).ok();

    let path = write_temp_file(&dir, "Cargo.toml", "this is not valid toml {{{{");
    assert!(parse_cargo_toml(&path).is_none());

    // Cleanup
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn parse_invalid_json_returns_none() {
    let dir = std::env::temp_dir().join("codemem_test_invalid_json");
    fs::create_dir_all(&dir).ok();

    let path = write_temp_file(&dir, "package.json", "not valid json {{{");
    assert!(parse_package_json(&path).is_none());

    // Cleanup
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn scan_manifests_finds_cargo_and_npm() {
    let dir = std::env::temp_dir().join("codemem_test_scan_manifests");
    fs::create_dir_all(&dir).ok();

    // Create a Cargo.toml at root
    write_temp_file(
        &dir,
        "Cargo.toml",
        r#"
[workspace]
members = ["crate-a"]

[package]
name = "root-crate"
version = "0.1.0"

[dependencies]
serde = "1"
"#,
    );

    // Create a package.json in a subdirectory
    write_temp_file(
        &dir,
        "frontend/package.json",
        r#"
{
    "name": "frontend-app",
    "dependencies": {
        "react": "^18"
    }
}
"#,
    );

    let result = scan_manifests(&dir);

    // Should find the workspace
    assert!(
        !result.workspaces.is_empty(),
        "Expected at least one workspace, got none"
    );

    // Should find both packages
    assert!(
        result.packages.contains_key("root-crate"),
        "Expected root-crate package"
    );
    assert!(
        result.packages.contains_key("frontend-app"),
        "Expected frontend-app package"
    );

    // Should find dependencies
    assert!(
        result.dependencies.iter().any(|d| d.name == "serde"),
        "Expected serde dependency"
    );
    assert!(
        result.dependencies.iter().any(|d| d.name == "react"),
        "Expected react dependency"
    );

    // Cleanup
    fs::remove_dir_all(&dir).ok();
}
