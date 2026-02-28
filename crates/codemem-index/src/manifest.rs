//! Manifest file parsing for cross-repo/cross-package dependency detection.
//!
//! Parses `Cargo.toml` and `package.json` files to extract workspace definitions,
//! package names, and dependency relationships. This enables Codemem to understand
//! monorepo structure and cross-package relationships.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// A parsed dependency from a manifest file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dependency {
    /// Package name.
    pub name: String,
    /// Version string or spec.
    pub version: String,
    /// Whether this is a dev/test dependency.
    pub dev: bool,
    /// Path to the manifest file this was found in.
    pub manifest_path: String,
}

/// A parsed workspace/monorepo definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Workspace {
    /// Root path of the workspace.
    pub root: String,
    /// Member package names/paths.
    pub members: Vec<String>,
    /// Manifest type (e.g., "cargo", "npm").
    pub kind: String,
}

/// Result of parsing manifests in a directory tree.
#[derive(Debug, Clone)]
pub struct ManifestResult {
    /// Detected workspaces.
    pub workspaces: Vec<Workspace>,
    /// All parsed dependencies.
    pub dependencies: Vec<Dependency>,
    /// Map: package_name -> manifest_path.
    pub packages: HashMap<String, String>,
}

impl ManifestResult {
    /// Create an empty ManifestResult.
    pub fn new() -> Self {
        Self {
            workspaces: Vec::new(),
            dependencies: Vec::new(),
            packages: HashMap::new(),
        }
    }

    /// Merge another ManifestResult into this one.
    pub fn merge(&mut self, other: ManifestResult) {
        self.workspaces.extend(other.workspaces);
        self.dependencies.extend(other.dependencies);
        self.packages.extend(other.packages);
    }
}

impl Default for ManifestResult {
    fn default() -> Self {
        Self::new()
    }
}

// ── Cargo.toml Parsing ───────────────────────────────────────────────────

/// Parse a Cargo.toml file for workspace members, package name, and dependencies.
pub fn parse_cargo_toml(path: &Path) -> Option<ManifestResult> {
    let content = std::fs::read_to_string(path).ok()?;
    let manifest_path = path.to_string_lossy().to_string();

    let toml_value: toml::Value = toml::from_str(&content).ok()?;
    let table = toml_value.as_table()?;

    let mut result = ManifestResult::new();

    // Extract package name
    if let Some(package) = table.get("package").and_then(|v| v.as_table()) {
        if let Some(name) = package.get("name").and_then(|v| v.as_str()) {
            result
                .packages
                .insert(name.to_string(), manifest_path.clone());
        }
    }

    // Extract workspace members
    if let Some(workspace) = table.get("workspace").and_then(|v| v.as_table()) {
        if let Some(members) = workspace.get("members").and_then(|v| v.as_array()) {
            let member_strings: Vec<String> = members
                .iter()
                .filter_map(|m| m.as_str().map(|s| s.to_string()))
                .collect();

            if !member_strings.is_empty() {
                let root = path
                    .parent()
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_default();

                result.workspaces.push(Workspace {
                    root,
                    members: member_strings,
                    kind: "cargo".to_string(),
                });
            }
        }
    }

    // Extract [dependencies]
    if let Some(deps) = table.get("dependencies").and_then(|v| v.as_table()) {
        for (name, value) in deps {
            let version = extract_cargo_dep_version(value);
            result.dependencies.push(Dependency {
                name: name.clone(),
                version,
                dev: false,
                manifest_path: manifest_path.clone(),
            });
        }
    }

    // Extract [dev-dependencies]
    if let Some(deps) = table.get("dev-dependencies").and_then(|v| v.as_table()) {
        for (name, value) in deps {
            let version = extract_cargo_dep_version(value);
            result.dependencies.push(Dependency {
                name: name.clone(),
                version,
                dev: true,
                manifest_path: manifest_path.clone(),
            });
        }
    }

    // Extract [build-dependencies]
    if let Some(deps) = table.get("build-dependencies").and_then(|v| v.as_table()) {
        for (name, value) in deps {
            let version = extract_cargo_dep_version(value);
            result.dependencies.push(Dependency {
                name: name.clone(),
                version,
                dev: false,
                manifest_path: manifest_path.clone(),
            });
        }
    }

    Some(result)
}

/// Extract version string from a Cargo dependency value.
/// Handles both `"1.0"` (string) and `{ version = "1.0", ... }` (table) forms.
fn extract_cargo_dep_version(value: &toml::Value) -> String {
    match value {
        toml::Value::String(s) => s.clone(),
        toml::Value::Table(t) => t
            .get("version")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        _ => String::new(),
    }
}

// ── package.json Parsing ─────────────────────────────────────────────────

/// Parse a package.json file for workspaces and dependencies.
pub fn parse_package_json(path: &Path) -> Option<ManifestResult> {
    let content = std::fs::read_to_string(path).ok()?;
    let manifest_path = path.to_string_lossy().to_string();

    let json: serde_json::Value = serde_json::from_str(&content).ok()?;
    let obj = json.as_object()?;

    let mut result = ManifestResult::new();

    // Extract package name
    if let Some(name) = obj.get("name").and_then(|v| v.as_str()) {
        result
            .packages
            .insert(name.to_string(), manifest_path.clone());
    }

    // Extract workspaces
    if let Some(workspaces) = obj.get("workspaces") {
        let member_strings = match workspaces {
            serde_json::Value::Array(arr) => arr
                .iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect::<Vec<_>>(),
            serde_json::Value::Object(obj) => {
                // npm workspaces can be { "packages": ["pkg/*"] }
                obj.get("packages")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str().map(|s| s.to_string()))
                            .collect()
                    })
                    .unwrap_or_default()
            }
            _ => Vec::new(),
        };

        if !member_strings.is_empty() {
            let root = path
                .parent()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_default();

            result.workspaces.push(Workspace {
                root,
                members: member_strings,
                kind: "npm".to_string(),
            });
        }
    }

    // Extract dependencies
    if let Some(deps) = obj.get("dependencies").and_then(|v| v.as_object()) {
        for (name, value) in deps {
            let version = value.as_str().unwrap_or("").to_string();
            result.dependencies.push(Dependency {
                name: name.clone(),
                version,
                dev: false,
                manifest_path: manifest_path.clone(),
            });
        }
    }

    // Extract devDependencies
    if let Some(deps) = obj.get("devDependencies").and_then(|v| v.as_object()) {
        for (name, value) in deps {
            let version = value.as_str().unwrap_or("").to_string();
            result.dependencies.push(Dependency {
                name: name.clone(),
                version,
                dev: true,
                manifest_path: manifest_path.clone(),
            });
        }
    }

    // Extract peerDependencies
    if let Some(deps) = obj.get("peerDependencies").and_then(|v| v.as_object()) {
        for (name, value) in deps {
            let version = value.as_str().unwrap_or("").to_string();
            result.dependencies.push(Dependency {
                name: name.clone(),
                version,
                dev: false,
                manifest_path: manifest_path.clone(),
            });
        }
    }

    Some(result)
}

// ── Directory Scanning ───────────────────────────────────────────────────

/// Scan a directory for all manifest files and parse them.
///
/// Walks the directory tree looking for `Cargo.toml` and `package.json` files,
/// skipping common ignore directories (node_modules, target, .git, etc.).
pub fn scan_manifests(root: &Path) -> ManifestResult {
    let mut result = ManifestResult::new();

    let walker = walkdir::WalkDir::new(root)
        .follow_links(false)
        .into_iter()
        .filter_entry(|entry| {
            let name = entry.file_name().to_string_lossy();
            // Skip common directories that should be ignored
            if entry.file_type().is_dir() {
                return !matches!(
                    name.as_ref(),
                    "node_modules" | "target" | ".git" | ".hg" | "vendor" | "dist" | "build"
                );
            }
            true
        });

    for entry in walker.flatten() {
        if !entry.file_type().is_file() {
            continue;
        }

        let path = entry.path();
        let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");

        match file_name {
            "Cargo.toml" => {
                if let Some(manifest) = parse_cargo_toml(path) {
                    result.merge(manifest);
                }
            }
            "package.json" => {
                if let Some(manifest) = parse_package_json(path) {
                    result.merge(manifest);
                }
            }
            _ => {}
        }
    }

    result
}

#[cfg(test)]
mod tests {
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
}
