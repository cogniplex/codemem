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
/// respecting `.gitignore` rules (via the `ignore` crate) to match indexer behavior.
pub fn scan_manifests(root: &Path) -> ManifestResult {
    let mut result = ManifestResult::new();

    let walker = ignore::WalkBuilder::new(root)
        .hidden(true) // skip hidden files/dirs
        .git_ignore(true) // respect .gitignore
        .git_global(true) // respect global gitignore
        .git_exclude(true) // respect .git/info/exclude
        .build();

    for entry in walker {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };

        if !entry.file_type().is_some_and(|ft| ft.is_file()) {
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
#[path = "tests/manifest_tests.rs"]
mod tests;
