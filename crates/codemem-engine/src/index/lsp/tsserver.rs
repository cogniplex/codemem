//! TypeScript/JavaScript enrichment via tsc.
//!
//! Runs `tsc --noEmit --pretty false` and parses diagnostic output.
//! Also uses `tsc --declaration` to extract type information when available.

use super::*;
use std::process::Command;
use std::sync::LazyLock;

static RE_TSC_DIAG: LazyLock<regex::Regex> = LazyLock::new(|| {
    regex::Regex::new(r"^(.+?)\((\d+),(\d+)\): (?:error|warning) (TS\d+): (.+)$").unwrap()
});

/// Diagnostic codes indicating unresolved references in TypeScript.
const UNRESOLVED_CODES: &[&str] = &[
    "TS2304", // Cannot find name
    "TS2305", // Module has no exported member
    "TS2307", // Cannot find module
    "TS2339", // Property does not exist on type
    "TS2552", // Cannot find name (did you mean?)
];

/// Common paths indicating external packages.
#[allow(dead_code)]
const EXTERNAL_MARKERS: &[&str] = &["node_modules", ".yarn", ".pnpm"];

pub struct TsServerEnricher;

impl TsServerEnricher {
    pub fn new() -> Self {
        Self
    }

    fn check_tsc() -> bool {
        Command::new("tsc")
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    /// Run tsc --noEmit and parse diagnostics.
    fn run_tsc(project_root: &Path) -> Result<Vec<TscDiagnostic>, String> {
        let output = Command::new("tsc")
            .args(["--noEmit", "--pretty", "false"])
            .current_dir(project_root)
            .output()
            .map_err(|e| format!("Failed to run tsc: {e}"))?;

        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        let text = if !stdout.is_empty() { stdout } else { stderr };

        Ok(parse_tsc_diagnostics(&text))
    }

    /// Run tsc --declaration to extract type declarations.
    /// Returns the temporary directory path containing .d.ts files.
    /// Caller is responsible for cleaning up the directory.
    fn run_tsc_declaration(project_root: &Path) -> Result<PathBuf, String> {
        let dir = std::env::temp_dir().join(format!("codemem-tsc-{}", std::process::id()));
        std::fs::create_dir_all(&dir).map_err(|e| format!("Failed to create temp dir: {e}"))?;

        let output = Command::new("tsc")
            .args([
                "--declaration",
                "--emitDeclarationOnly",
                "--declarationDir",
                &dir.to_string_lossy(),
            ])
            .current_dir(project_root)
            .output()
            .map_err(|e| format!("Failed to run tsc --declaration: {e}"))?;

        if !output.status.success() {
            // tsc may fail on type errors but still produce some declarations
            let stderr = String::from_utf8_lossy(&output.stderr);
            tracing::debug!("tsc --declaration had errors: {stderr}");
        }

        Ok(dir)
    }

    #[allow(dead_code)]
    fn is_external_path(path: &str) -> bool {
        EXTERNAL_MARKERS.iter().any(|m| path.contains(m))
    }

    /// Extract package name from a node_modules path.
    /// e.g., `node_modules/@acme/shared/dist/index.js` → `@acme/shared`
    /// e.g., `node_modules/lodash/index.js` → `lodash`
    #[allow(dead_code)]
    fn extract_package_from_path(path: &str) -> Option<String> {
        let marker = "node_modules/";
        let idx = path.rfind(marker)?;
        let after = &path[idx + marker.len()..];

        if after.starts_with('@') {
            // Scoped package: @scope/name
            let parts: Vec<&str> = after.splitn(3, '/').collect();
            if parts.len() >= 2 {
                return Some(format!("{}/{}", parts[0], parts[1]));
            }
        } else {
            // Regular package
            let name = after.split('/').next()?;
            if !name.is_empty() {
                return Some(name.to_string());
            }
        }
        None
    }

    /// Parse declaration files for type information.
    fn parse_declarations(decl_dir: &Path, project_root: &Path) -> Vec<TypeAnnotation> {
        let mut annotations = Vec::new();

        let walk = ignore::WalkBuilder::new(decl_dir).hidden(false).build();

        for entry in walk.flatten() {
            let path = entry.path();
            if path.extension().is_none_or(|e| e != "ts") {
                continue;
            }

            let content = match std::fs::read_to_string(path) {
                Ok(c) => c,
                Err(_) => continue,
            };

            // Map .d.ts path back to source file
            let rel = path
                .strip_prefix(decl_dir)
                .unwrap_or(path)
                .to_string_lossy()
                .replace(".d.ts", ".ts");

            Self::extract_type_annotations_from_dts(&content, &rel, &mut annotations);
        }

        // Try to relativize file paths
        annotations.iter_mut().for_each(|a| {
            if let Ok(stripped) = Path::new(&a.file_path).strip_prefix(project_root) {
                a.file_path = stripped.to_string_lossy().to_string();
            }
        });

        annotations
    }

    /// Extract type annotations from a .d.ts file content.
    fn extract_type_annotations_from_dts(
        content: &str,
        source_file: &str,
        annotations: &mut Vec<TypeAnnotation>,
    ) {
        static RE_FUNC_DECL: LazyLock<regex::Regex> = LazyLock::new(|| {
            regex::Regex::new(
                r"(?m)^(?:export )?(?:declare )?function (\w+)(?:<([^>]+)>)?\([^)]*\):\s*(.+);$",
            )
            .unwrap()
        });

        static RE_CONST_DECL: LazyLock<regex::Regex> = LazyLock::new(|| {
            regex::Regex::new(r"(?m)^(?:export )?(?:declare )?(?:const|let|var) (\w+):\s*(.+);$")
                .unwrap()
        });

        for (line_num, line) in content.lines().enumerate() {
            if let Some(caps) = RE_FUNC_DECL.captures(line) {
                let generic_params = caps.get(2).map_or(Vec::new(), |m| {
                    m.as_str()
                        .split(',')
                        .map(|s| s.trim().to_string())
                        .collect()
                });
                annotations.push(TypeAnnotation {
                    file_path: source_file.to_string(),
                    line: line_num + 1,
                    symbol_name: caps[1].to_string(),
                    resolved_type: String::new(),
                    return_type: Some(caps[3].trim().to_string()),
                    generic_params,
                });
            } else if let Some(caps) = RE_CONST_DECL.captures(line) {
                annotations.push(TypeAnnotation {
                    file_path: source_file.to_string(),
                    line: line_num + 1,
                    symbol_name: caps[1].to_string(),
                    resolved_type: caps[2].trim().to_string(),
                    return_type: None,
                    generic_params: Vec::new(),
                });
            }
        }
    }
}

impl LspEnricher for TsServerEnricher {
    fn name(&self) -> &str {
        "tsserver"
    }

    fn is_available(&self) -> bool {
        Self::check_tsc()
    }

    fn extensions(&self) -> &[&str] {
        &[".ts", ".tsx", ".js", ".jsx"]
    }

    fn enrich(&self, project_root: &Path, targets: &[EnrichmentTarget]) -> EnrichmentResult {
        let mut result = EnrichmentResult::default();

        // Note: tsserver enrichment extracts type annotations via tsc --declaration
        // and confirms unresolvable refs via tsc --noEmit. It does NOT resolve
        // references (that would require the TS Language Service API).

        // Build target index for matching diagnostics
        let mut target_index: HashMap<(String, usize), &RefToResolve> = HashMap::new();
        for target in targets {
            for r in &target.refs {
                target_index.insert((r.file_path.clone(), r.line), r);
            }
        }

        // Phase 1: Run tsc --noEmit to get diagnostics
        match Self::run_tsc(project_root) {
            Ok(diagnostics) => {
                for diag in &diagnostics {
                    // Check if this diagnostic corresponds to an unresolved ref
                    if UNRESOLVED_CODES.contains(&diag.code.as_str()) {
                        // If we have a target at this location, mark it as confirmed unresolvable
                        if target_index.contains_key(&(diag.file.clone(), diag.line)) {
                            tracing::debug!(
                                "tsc: confirmed unresolvable at {}:{}: {} ({})",
                                diag.file,
                                diag.line,
                                diag.message,
                                diag.code
                            );
                        }
                    }
                }
            }
            Err(e) => {
                result.errors.push(e);
            }
        }

        // Phase 2: Try to extract type declarations
        match Self::run_tsc_declaration(project_root) {
            Ok(decl_dir) => {
                let annotations = Self::parse_declarations(&decl_dir, project_root);
                result.type_annotations = annotations;
                // Clean up temp directory
                let _ = std::fs::remove_dir_all(&decl_dir);
            }
            Err(e) => {
                // Non-fatal: declaration extraction is best-effort
                tracing::debug!("tsc --declaration failed (non-fatal): {e}");
            }
        }

        tracing::info!(
            "tsserver: {} type annotations, {} errors",
            result.type_annotations.len(),
            result.errors.len()
        );

        result
    }

    fn detect_project_root(&self, file_path: &Path) -> Option<PathBuf> {
        let mut current = file_path.parent()?;
        loop {
            if current.join("tsconfig.json").exists() || current.join("jsconfig.json").exists() {
                return Some(current.to_path_buf());
            }
            current = current.parent()?;
        }
    }
}

impl Default for TsServerEnricher {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
#[allow(dead_code)]
pub(crate) struct TscDiagnostic {
    pub file: String,
    pub line: usize,
    pub col: usize,
    pub message: String,
    pub code: String,
}

/// Parse tsc plain-text diagnostics like:
/// `src/app.ts(10,5): error TS2304: Cannot find name 'foo'.`
pub(crate) fn parse_tsc_diagnostics(output: &str) -> Vec<TscDiagnostic> {
    output
        .lines()
        .filter_map(|line| {
            let caps = RE_TSC_DIAG.captures(line.trim())?;
            Some(TscDiagnostic {
                file: caps[1].to_string(),
                line: caps[2].parse().unwrap_or(0),
                col: caps[3].parse().unwrap_or(0),
                message: caps[5].to_string(),
                code: caps[4].to_string(),
            })
        })
        .collect()
}

#[cfg(test)]
#[path = "../tests/tsserver_tests.rs"]
mod tests;
