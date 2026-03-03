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

// ── go.mod Parsing ────────────────────────────────────────────────────────

/// Parse a go.mod file for module name, Go version, and dependencies.
pub fn parse_go_mod(path: &Path) -> Option<ManifestResult> {
    let content = std::fs::read_to_string(path).ok()?;
    let manifest_path = path.to_string_lossy().to_string();

    let mut result = ManifestResult::new();

    // Extract module name: `module github.com/user/repo`
    for line in content.lines() {
        let trimmed = line.trim();
        if let Some(module) = trimmed.strip_prefix("module ") {
            let module = module.trim();
            result
                .packages
                .insert(module.to_string(), manifest_path.clone());
            break;
        }
    }

    // Parse require blocks (both single-line and block form)
    let mut in_require_block = false;
    for line in content.lines() {
        let trimmed = line.trim();

        if trimmed == "require (" {
            in_require_block = true;
            continue;
        }
        if in_require_block && trimmed == ")" {
            in_require_block = false;
            continue;
        }

        // Single-line require: `require github.com/pkg/errors v0.9.1`
        if let Some(rest) = trimmed.strip_prefix("require ") {
            if !rest.starts_with('(') {
                if let Some(dep) = parse_go_require_line(rest, &manifest_path) {
                    result.dependencies.push(dep);
                }
            }
            continue;
        }

        // Inside require block
        if in_require_block {
            // Skip comments and empty lines
            if trimmed.is_empty() || trimmed.starts_with("//") {
                continue;
            }
            if let Some(dep) = parse_go_require_line(trimmed, &manifest_path) {
                result.dependencies.push(dep);
            }
        }
    }

    Some(result)
}

/// Parse a single Go require line like `github.com/pkg/errors v0.9.1`
fn parse_go_require_line(line: &str, manifest_path: &str) -> Option<Dependency> {
    // Remove inline comments
    let line = line.split("//").next()?.trim();
    let mut parts = line.split_whitespace();
    let name = parts.next()?;
    let version = parts.next().unwrap_or("");
    // Skip indirect dependencies marker, but still capture them
    let is_indirect = line.contains("// indirect");
    Some(Dependency {
        name: name.to_string(),
        version: version.to_string(),
        dev: is_indirect,
        manifest_path: manifest_path.to_string(),
    })
}

// ── pyproject.toml Parsing ────────────────────────────────────────────────

/// Parse a pyproject.toml file for project name, version, and dependencies.
pub fn parse_pyproject_toml(path: &Path) -> Option<ManifestResult> {
    let content = std::fs::read_to_string(path).ok()?;
    let manifest_path = path.to_string_lossy().to_string();

    let toml_value: toml::Value = toml::from_str(&content).ok()?;
    let table = toml_value.as_table()?;

    let mut result = ManifestResult::new();

    // PEP 621 [project] table
    if let Some(project) = table.get("project").and_then(|v| v.as_table()) {
        if let Some(name) = project.get("name").and_then(|v| v.as_str()) {
            result
                .packages
                .insert(name.to_string(), manifest_path.clone());
        }

        // [project.dependencies]
        if let Some(deps) = project.get("dependencies").and_then(|v| v.as_array()) {
            for dep in deps {
                if let Some(spec) = dep.as_str() {
                    if let Some(d) = parse_python_dep_spec(spec, false, &manifest_path) {
                        result.dependencies.push(d);
                    }
                }
            }
        }

        // [project.optional-dependencies] — treat as dev
        if let Some(opt_deps) = project
            .get("optional-dependencies")
            .and_then(|v| v.as_table())
        {
            for (_group, deps) in opt_deps {
                if let Some(arr) = deps.as_array() {
                    for dep in arr {
                        if let Some(spec) = dep.as_str() {
                            if let Some(d) = parse_python_dep_spec(spec, true, &manifest_path) {
                                result.dependencies.push(d);
                            }
                        }
                    }
                }
            }
        }
    }

    // Poetry: [tool.poetry]
    if let Some(tool) = table.get("tool").and_then(|v| v.as_table()) {
        if let Some(poetry) = tool.get("poetry").and_then(|v| v.as_table()) {
            if let Some(name) = poetry.get("name").and_then(|v| v.as_str()) {
                result
                    .packages
                    .insert(name.to_string(), manifest_path.clone());
            }

            // [tool.poetry.dependencies]
            if let Some(deps) = poetry.get("dependencies").and_then(|v| v.as_table()) {
                for (name, value) in deps {
                    // Skip python itself
                    if name == "python" {
                        continue;
                    }
                    let version = extract_poetry_version(value);
                    result.dependencies.push(Dependency {
                        name: name.clone(),
                        version,
                        dev: false,
                        manifest_path: manifest_path.clone(),
                    });
                }
            }

            // [tool.poetry.dev-dependencies]
            if let Some(deps) = poetry.get("dev-dependencies").and_then(|v| v.as_table()) {
                for (name, value) in deps {
                    let version = extract_poetry_version(value);
                    result.dependencies.push(Dependency {
                        name: name.clone(),
                        version,
                        dev: true,
                        manifest_path: manifest_path.clone(),
                    });
                }
            }
        }
    }

    Some(result)
}

/// Parse a PEP 508 dependency specifier like `requests>=2.28.0` into name + version.
fn parse_python_dep_spec(spec: &str, dev: bool, manifest_path: &str) -> Option<Dependency> {
    let spec = spec.trim();
    if spec.is_empty() {
        return None;
    }

    // Split on version operators: >=, <=, ==, !=, ~=, >, <
    // Also handle extras like `package[extra]>=1.0`
    let name_end = spec
        .find(['>', '<', '=', '!', '~', ';', '['])
        .unwrap_or(spec.len());
    let name = spec[..name_end].trim();

    // Extract version part (everything after the operator)
    let version_part = &spec[name_end..];
    // Strip extras like [extra] before version
    let version_part = if version_part.starts_with('[') {
        version_part
            .find(']')
            .map(|i| &version_part[i + 1..])
            .unwrap_or(version_part)
    } else {
        version_part
    };
    // Strip environment markers (after `;`)
    let version_part = version_part.split(';').next().unwrap_or("").trim();

    Some(Dependency {
        name: name.to_string(),
        version: version_part.to_string(),
        dev,
        manifest_path: manifest_path.to_string(),
    })
}

/// Extract version from a Poetry dependency value.
fn extract_poetry_version(value: &toml::Value) -> String {
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

// ── pom.xml Parsing ───────────────────────────────────────────────────────

/// Parse a pom.xml file for groupId, artifactId, version, and dependencies.
/// Uses basic regex extraction — no full XML parser needed.
pub fn parse_pom_xml(path: &Path) -> Option<ManifestResult> {
    let content = std::fs::read_to_string(path).ok()?;
    let manifest_path = path.to_string_lossy().to_string();

    let mut result = ManifestResult::new();

    // Extract top-level artifactId (not inside <dependency>)
    // Find the first <artifactId> that isn't inside a <dependencies> block
    if let Some(artifact_id) = extract_xml_tag_before_deps(&content, "artifactId") {
        let group_id = extract_xml_tag_before_deps(&content, "groupId").unwrap_or_default();
        let name = if group_id.is_empty() {
            artifact_id.clone()
        } else {
            format!("{group_id}:{artifact_id}")
        };
        result.packages.insert(name, manifest_path.clone());
    }

    // Extract dependencies from <dependencies> block
    let re_dep = regex::Regex::new(r"(?s)<dependency>(.*?)</dependency>").ok()?;
    for cap in re_dep.captures_iter(&content) {
        let dep_block = &cap[1];
        let group = extract_xml_tag(dep_block, "groupId").unwrap_or_default();
        let artifact = match extract_xml_tag(dep_block, "artifactId") {
            Some(a) => a,
            None => continue,
        };
        let version = extract_xml_tag(dep_block, "version").unwrap_or_default();
        let scope = extract_xml_tag(dep_block, "scope").unwrap_or_default();

        let name = if group.is_empty() {
            artifact
        } else {
            format!("{group}:{artifact}")
        };

        result.dependencies.push(Dependency {
            name,
            version,
            dev: scope == "test",
            manifest_path: manifest_path.clone(),
        });
    }

    Some(result)
}

/// Extract text content of an XML tag using basic regex.
fn extract_xml_tag(content: &str, tag: &str) -> Option<String> {
    let pattern = format!(r"<{tag}>\s*(.*?)\s*</{tag}>");
    let re = regex::Regex::new(&pattern).ok()?;
    re.captures(content).map(|c| c[1].to_string())
}

/// Extract the first occurrence of an XML tag that appears before any `<dependencies>` block.
fn extract_xml_tag_before_deps(content: &str, tag: &str) -> Option<String> {
    let deps_pos = content.find("<dependencies>");
    let search_area = match deps_pos {
        Some(pos) => &content[..pos],
        None => content,
    };
    extract_xml_tag(search_area, tag)
}

// ── .csproj Parsing ───────────────────────────────────────────────────────

/// Parse a .csproj file for PackageReference items.
pub fn parse_csproj(path: &Path) -> Option<ManifestResult> {
    let content = std::fs::read_to_string(path).ok()?;
    let manifest_path = path.to_string_lossy().to_string();

    let mut result = ManifestResult::new();

    // Extract project name from filename
    let name = path
        .file_stem()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown");
    result
        .packages
        .insert(name.to_string(), manifest_path.clone());

    // Match: <PackageReference Include="Name" Version="1.0" />
    // or:    <PackageReference Include="Name" Version="1.0"></PackageReference>
    let re =
        regex::Regex::new(r#"<PackageReference\s+Include="([^"]+)"\s+Version="([^"]*)"[^/]*/>"#)
            .ok()?;
    for cap in re.captures_iter(&content) {
        result.dependencies.push(Dependency {
            name: cap[1].to_string(),
            version: cap[2].to_string(),
            dev: false,
            manifest_path: manifest_path.clone(),
        });
    }

    // Also match the two-line form: Include then Version on separate attributes
    let re2 =
        regex::Regex::new(r#"<PackageReference\s+Include="([^"]+)"\s*Version="([^"]*)"[^>]*>"#)
            .ok()?;
    // Only add if not already captured (avoid duplicates by checking names)
    let existing_names: std::collections::HashSet<String> =
        result.dependencies.iter().map(|d| d.name.clone()).collect();
    for cap in re2.captures_iter(&content) {
        let name = cap[1].to_string();
        if !existing_names.contains(&name) {
            result.dependencies.push(Dependency {
                name,
                version: cap[2].to_string(),
                dev: false,
                manifest_path: manifest_path.clone(),
            });
        }
    }

    Some(result)
}

// ── Gemfile Parsing ───────────────────────────────────────────────────────

/// Parse a Gemfile for gem dependencies.
pub fn parse_gemfile(path: &Path) -> Option<ManifestResult> {
    let content = std::fs::read_to_string(path).ok()?;
    let manifest_path = path.to_string_lossy().to_string();

    let mut result = ManifestResult::new();

    // Match: gem 'name' or gem "name" with optional version
    let re = regex::Regex::new(r#"gem\s+['"]([^'"]+)['"](?:\s*,\s*['"]([^'"]*)['"]\s*)?"#).ok()?;

    let mut in_dev_group = false;
    for line in content.lines() {
        let trimmed = line.trim();

        // Track group :development / :test blocks
        if trimmed.starts_with("group")
            && (trimmed.contains(":development") || trimmed.contains(":test"))
        {
            in_dev_group = true;
            continue;
        }
        if in_dev_group && trimmed == "end" {
            in_dev_group = false;
            continue;
        }

        if let Some(cap) = re.captures(trimmed) {
            let name = cap[1].to_string();
            let version = cap
                .get(2)
                .map(|m| m.as_str().to_string())
                .unwrap_or_default();
            result.dependencies.push(Dependency {
                name,
                version,
                dev: in_dev_group,
                manifest_path: manifest_path.clone(),
            });
        }
    }

    Some(result)
}

// ── composer.json Parsing ─────────────────────────────────────────────────

/// Parse a composer.json file for PHP dependencies.
pub fn parse_composer_json(path: &Path) -> Option<ManifestResult> {
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

    // Extract require
    if let Some(deps) = obj.get("require").and_then(|v| v.as_object()) {
        for (name, value) in deps {
            // Skip php version constraint
            if name == "php" {
                continue;
            }
            let version = value.as_str().unwrap_or("").to_string();
            result.dependencies.push(Dependency {
                name: name.clone(),
                version,
                dev: false,
                manifest_path: manifest_path.clone(),
            });
        }
    }

    // Extract require-dev
    if let Some(deps) = obj.get("require-dev").and_then(|v| v.as_object()) {
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

    Some(result)
}

// ── Directory Scanning ───────────────────────────────────────────────────

/// Scan a directory for all manifest files and parse them.
///
/// Walks the directory tree looking for manifest files (Cargo.toml, package.json,
/// go.mod, pyproject.toml, pom.xml, .csproj, Gemfile, composer.json),
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
            "go.mod" => {
                if let Some(manifest) = parse_go_mod(path) {
                    result.merge(manifest);
                }
            }
            "pyproject.toml" => {
                if let Some(manifest) = parse_pyproject_toml(path) {
                    result.merge(manifest);
                }
            }
            "pom.xml" => {
                if let Some(manifest) = parse_pom_xml(path) {
                    result.merge(manifest);
                }
            }
            "Gemfile" => {
                if let Some(manifest) = parse_gemfile(path) {
                    result.merge(manifest);
                }
            }
            "composer.json" => {
                if let Some(manifest) = parse_composer_json(path) {
                    result.merge(manifest);
                }
            }
            _ => {
                // Check for .csproj files by extension
                if file_name.ends_with(".csproj") {
                    if let Some(manifest) = parse_csproj(path) {
                        result.merge(manifest);
                    }
                }
            }
        }
    }

    result
}

#[cfg(test)]
#[path = "tests/manifest_tests.rs"]
mod tests;
