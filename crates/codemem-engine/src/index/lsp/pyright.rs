//! Pyright-based Python enrichment.
//!
//! Runs `pyright --outputjson` and parses the JSON output to extract:
//! - Resolved reference locations (from hover/definition data in diagnostics)
//! - Type annotations from diagnostic messages
//! - External dependency detection (site-packages, venv paths)

use super::*;
use std::process::Command;
use std::sync::LazyLock;

static RE_TYPE_OF: LazyLock<regex::Regex> =
    LazyLock::new(|| regex::Regex::new(r#"Type of "(\w+)" is "([^"]+)""#).unwrap());

static RE_RETURN_TYPE: LazyLock<regex::Regex> =
    LazyLock::new(|| regex::Regex::new(r#"return type of "(\w+)" is "([^"]+)""#).unwrap());

/// Common paths indicating external packages.
const EXTERNAL_MARKERS: &[&str] = &[
    "site-packages",
    "node_modules",
    "dist-packages",
    ".venv",
    "venv",
    "__pypackages__",
];

pub struct PyrightEnricher;

impl PyrightEnricher {
    pub fn new() -> Self {
        Self
    }

    /// Check if pyright is installed by running `pyright --version`.
    fn check_pyright() -> bool {
        Command::new("pyright")
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    /// Run pyright and parse JSON output.
    fn run_pyright(project_root: &Path) -> Result<PyrightOutput, String> {
        let output = Command::new("pyright")
            .args(["--outputjson", "--level", "basic"])
            .current_dir(project_root)
            .output()
            .map_err(|e| format!("Failed to run pyright: {e}"))?;

        // Pyright returns non-zero on type errors, but still outputs valid JSON
        let stdout = String::from_utf8_lossy(&output.stdout);
        if stdout.is_empty() {
            return Err("Pyright produced no output".to_string());
        }

        serde_json::from_str(&stdout).map_err(|e| format!("Failed to parse pyright output: {e}"))
    }

    /// Extract package name from an external file path.
    ///
    /// e.g., `/venv/lib/python3.11/site-packages/requests/api.py` → `"requests"`
    fn extract_package_from_path(path: &str) -> Option<String> {
        for marker in EXTERNAL_MARKERS {
            if let Some(idx) = path.find(marker) {
                let after = &path[idx + marker.len()..];
                let after = after.trim_start_matches('/').trim_start_matches('\\');
                // First path segment after the marker is the package name
                let pkg = after
                    .split('/')
                    .next()
                    .or_else(|| after.split('\\').next())?;
                if !pkg.is_empty() && !pkg.ends_with(".py") {
                    return Some(pkg.to_string());
                }
            }
        }
        None
    }

    /// Check if a file path is external (in site-packages, venv, etc.)
    fn is_external_path(path: &str) -> bool {
        EXTERNAL_MARKERS.iter().any(|m| path.contains(m))
    }

    /// Build a lookup from (file, line) → source reference info for matching
    /// pyright diagnostics back to our enrichment targets.
    fn build_target_index(targets: &[EnrichmentTarget]) -> HashMap<(String, usize), &RefToResolve> {
        let mut index = HashMap::new();
        for target in targets {
            for r in &target.refs {
                index.insert((r.file_path.clone(), r.line), r);
            }
        }
        index
    }

    /// Process pyright diagnostics that contain definition location info.
    /// Pyright's JSON output includes diagnostics with `definitionRange` or
    /// diagnostics mentioning unresolved imports which we can cross-reference.
    fn process_diagnostics(
        output: &PyrightOutput,
        project_root: &Path,
        target_index: &HashMap<(String, usize), &RefToResolve>,
        result: &mut EnrichmentResult,
    ) {
        for diag in &output.general_diagnostics {
            let Some(ref range) = diag.range else {
                continue;
            };

            // Extract type annotations from diagnostic messages
            if let Some(annotation) =
                Self::extract_type_annotation(&diag.message, &diag.file, range)
            {
                result.type_annotations.push(annotation);
            }

            // Match diagnostic back to a ref target for resolution tracking
            let rel_file = Self::relativize(&diag.file, project_root);
            let line = range.start.line + 1; // pyright uses 0-indexed lines
            if let Some(_ref_target) = target_index.get(&(rel_file.clone(), line)) {
                // If pyright reports an import error for this location, it means
                // the reference is definitively unresolvable within this project.
                // We don't create resolved refs for errors, but we note them.
                if diag.severity == "error" && diag.message.contains("import") {
                    tracing::debug!(
                        "pyright: unresolvable import at {}:{}: {}",
                        rel_file,
                        line,
                        diag.message
                    );
                }
            }
        }
    }

    /// Process pyright's symbol table output when available.
    /// This is the main source of resolved reference data.
    fn process_symbols(
        output: &PyrightOutput,
        project_root: &Path,
        target_index: &HashMap<(String, usize), &RefToResolve>,
        result: &mut EnrichmentResult,
    ) {
        for file_diag in &output.file_diagnostics {
            let source_file = Self::relativize(&file_diag.file, project_root);

            for diag in &file_diag.diagnostics {
                let Some(ref range) = diag.range else {
                    continue;
                };

                // Extract type annotations
                if let Some(annotation) =
                    Self::extract_type_annotation(&diag.message, &source_file, range)
                {
                    result.type_annotations.push(annotation);
                }

                // Check if this diagnostic has definition information
                if let Some(ref def) = diag.definition_range {
                    let line = range.start.line + 1;
                    if target_index.contains_key(&(source_file.clone(), line)) {
                        let target_file_abs = &def.file;
                        let is_ext = Self::is_external_path(target_file_abs);
                        let package_name = if is_ext {
                            Self::extract_package_from_path(target_file_abs)
                        } else {
                            None
                        };

                        let target_file = if is_ext {
                            target_file_abs.to_string()
                        } else {
                            Self::relativize(target_file_abs, project_root)
                        };

                        let target_symbol = diag
                            .symbol_name
                            .clone()
                            .unwrap_or_else(|| def.symbol.clone().unwrap_or_default());

                        if !target_symbol.is_empty() {
                            result.resolved_refs.push(LspResolvedRef {
                                source_file: source_file.clone(),
                                source_line: line,
                                target_file,
                                target_line: def
                                    .range
                                    .as_ref()
                                    .map(|r| r.start.line + 1)
                                    .unwrap_or(0),
                                target_symbol,
                                is_external: is_ext,
                                package_name,
                            });
                        }
                    }
                }
            }
        }
    }

    fn extract_type_annotation(
        message: &str,
        file: &str,
        range: &PyrightRange,
    ) -> Option<TypeAnnotation> {
        // "Type of X is Y"
        if let Some(caps) = RE_TYPE_OF.captures(message) {
            return Some(TypeAnnotation {
                file_path: file.to_string(),
                line: range.start.line + 1,
                symbol_name: caps[1].to_string(),
                resolved_type: caps[2].to_string(),
                return_type: None,
                generic_params: extract_generics(&caps[2]),
            });
        }
        // "return type of X is Y"
        if let Some(caps) = RE_RETURN_TYPE.captures(message) {
            return Some(TypeAnnotation {
                file_path: file.to_string(),
                line: range.start.line + 1,
                symbol_name: caps[1].to_string(),
                resolved_type: String::new(),
                return_type: Some(caps[2].to_string()),
                generic_params: Vec::new(),
            });
        }
        None
    }

    /// Make an absolute path relative to the project root.
    fn relativize(path: &str, root: &Path) -> String {
        let p = Path::new(path);
        p.strip_prefix(root)
            .unwrap_or(p)
            .to_string_lossy()
            .to_string()
    }
}

impl LspEnricher for PyrightEnricher {
    fn name(&self) -> &str {
        "pyright"
    }

    fn is_available(&self) -> bool {
        Self::check_pyright()
    }

    fn extensions(&self) -> &[&str] {
        &[".py", ".pyi"]
    }

    fn enrich(&self, project_root: &Path, targets: &[EnrichmentTarget]) -> EnrichmentResult {
        let mut result = EnrichmentResult::default();
        let target_index = Self::build_target_index(targets);

        match Self::run_pyright(project_root) {
            Ok(output) => {
                // Process general diagnostics (always present)
                Self::process_diagnostics(&output, project_root, &target_index, &mut result);

                // Process per-file diagnostics with definition info (when available)
                Self::process_symbols(&output, project_root, &target_index, &mut result);

                tracing::info!(
                    "pyright: resolved {} refs, {} type annotations, {} errors",
                    result.resolved_refs.len(),
                    result.type_annotations.len(),
                    result.errors.len()
                );
            }
            Err(e) => {
                result.errors.push(e);
            }
        }

        result
    }

    fn detect_project_root(&self, file_path: &Path) -> Option<PathBuf> {
        let mut current = file_path.parent()?;
        loop {
            if current.join("pyproject.toml").exists()
                || current.join("pyrightconfig.json").exists()
                || current.join("setup.py").exists()
                || current.join("setup.cfg").exists()
            {
                return Some(current.to_path_buf());
            }
            current = current.parent()?;
        }
    }
}

impl Default for PyrightEnricher {
    fn default() -> Self {
        Self::new()
    }
}

/// Extract generic type parameters from a type string.
/// e.g., "List[User]" → ["User"], "Dict[str, int]" → ["str", "int"]
fn extract_generics(type_str: &str) -> Vec<String> {
    if let Some(start) = type_str.find('[') {
        if let Some(end) = type_str.rfind(']') {
            let inner = &type_str[start + 1..end];
            return inner
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
        }
    }
    Vec::new()
}

// ── Pyright JSON output structures ──────────────────────────────────────────

#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct PyrightOutput {
    #[serde(default)]
    general_diagnostics: Vec<PyrightDiagnostic>,
    #[serde(default)]
    file_diagnostics: Vec<PyrightFileDiagnostics>,
    #[serde(default)]
    #[allow(dead_code)]
    version: String,
}

/// Per-file diagnostics section from pyright JSON.
#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct PyrightFileDiagnostics {
    file: String,
    diagnostics: Vec<PyrightDiagnostic>,
}

#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct PyrightDiagnostic {
    #[serde(default)]
    file: String,
    message: String,
    severity: String,
    #[serde(default)]
    range: Option<PyrightRange>,
    #[serde(default)]
    #[allow(dead_code)]
    rule: Option<String>,
    /// Definition location when pyright resolves a reference.
    #[serde(default)]
    definition_range: Option<PyrightDefinition>,
    /// Symbol name associated with this diagnostic.
    #[serde(default)]
    symbol_name: Option<String>,
}

/// Definition location from pyright.
#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct PyrightDefinition {
    file: String,
    #[serde(default)]
    range: Option<PyrightRange>,
    /// Symbol name at the definition site.
    #[serde(default)]
    symbol: Option<String>,
}

#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct PyrightRange {
    start: PyrightPosition,
    #[allow(dead_code)]
    end: PyrightPosition,
}

#[derive(Debug, serde::Deserialize)]
struct PyrightPosition {
    line: usize,
    #[allow(dead_code)]
    character: usize,
}

#[cfg(test)]
#[path = "../tests/pyright_tests.rs"]
mod tests;
