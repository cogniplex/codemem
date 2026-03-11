//! LSP-based enrichment for compiler-grade reference resolution.
//!
//! Each enricher wraps a language toolchain (pyright, tsc, etc.) and
//! provides batch resolution of unresolved/low-confidence references.

pub mod pyright;
pub mod tsserver;

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

/// Results of applying LSP enrichment to the graph.
#[derive(Debug, Default)]
pub struct LspEnrichStats {
    pub edges_upgraded: usize,
    pub ext_nodes_created: usize,
    pub type_annotations_applied: usize,
    pub errors: Vec<String>,
}

/// A reference to be resolved by LSP.
#[derive(Debug, Clone)]
pub struct RefToResolve {
    /// Source symbol qualified name.
    pub source_node: String,
    /// Unresolved target name.
    pub target_name: String,
    /// File containing the reference.
    pub file_path: String,
    /// Line of the reference.
    pub line: usize,
    /// Current confidence from ast-grep (None = fully unresolved).
    pub current_confidence: Option<f64>,
}

/// A batch of references to resolve in a single file.
#[derive(Debug, Clone)]
pub struct EnrichmentTarget {
    pub file_path: String,
    pub refs: Vec<RefToResolve>,
}

/// A reference resolved by LSP.
#[derive(Debug, Clone)]
pub struct LspResolvedRef {
    /// Source file path.
    pub source_file: String,
    /// Source line number.
    pub source_line: usize,
    /// Resolved target file path (absolute).
    pub target_file: String,
    /// Resolved target line number.
    pub target_line: usize,
    /// Resolved target symbol name.
    pub target_symbol: String,
    /// Whether target is in external dependencies.
    pub is_external: bool,
    /// Package name if external.
    pub package_name: Option<String>,
}

/// Type annotation enrichment for an existing symbol.
#[derive(Debug, Clone)]
pub struct TypeAnnotation {
    pub file_path: String,
    pub line: usize,
    pub symbol_name: String,
    pub resolved_type: String,
    pub return_type: Option<String>,
    pub generic_params: Vec<String>,
}

/// Combined enrichment results from one enricher run.
#[derive(Debug, Clone, Default)]
pub struct EnrichmentResult {
    pub resolved_refs: Vec<LspResolvedRef>,
    pub type_annotations: Vec<TypeAnnotation>,
    pub errors: Vec<String>,
}

/// Trait for language-specific LSP enrichment.
pub trait LspEnricher: Send + Sync {
    /// Human-readable name (e.g., "pyright", "tsserver").
    fn name(&self) -> &str;

    /// Check if this enricher's tooling is available on PATH.
    fn is_available(&self) -> bool;

    /// File extensions this enricher handles.
    fn extensions(&self) -> &[&str];

    /// Batch resolve references for a project.
    fn enrich(&self, project_root: &Path, targets: &[EnrichmentTarget]) -> EnrichmentResult;

    /// Detect project root by walking up from a file.
    fn detect_project_root(&self, file_path: &Path) -> Option<PathBuf>;
}

/// Discover available enrichers on the system.
pub fn available_enrichers() -> Vec<Box<dyn LspEnricher>> {
    let mut enrichers: Vec<Box<dyn LspEnricher>> = Vec::new();

    let pyright = pyright::PyrightEnricher::new();
    if pyright.is_available() {
        enrichers.push(Box::new(pyright));
    }

    let ts = tsserver::TsServerEnricher::new();
    if ts.is_available() {
        enrichers.push(Box::new(ts));
    }

    enrichers
}

/// Run all available enrichers against grouped targets.
///
/// Groups targets by file extension, finds matching enricher,
/// detects project roots, and runs each enricher.
pub fn run_enrichment(
    targets: &[EnrichmentTarget],
    enrichers: &[Box<dyn LspEnricher>],
) -> Vec<EnrichmentResult> {
    if enrichers.is_empty() || targets.is_empty() {
        return Vec::new();
    }

    // Group targets by extension
    let mut by_ext: HashMap<String, Vec<&EnrichmentTarget>> = HashMap::new();
    for target in targets {
        if let Some(ext) = Path::new(&target.file_path)
            .extension()
            .and_then(|e| e.to_str())
        {
            by_ext.entry(format!(".{ext}")).or_default().push(target);
        }
    }

    let mut results = Vec::new();

    // For each enricher, find matching extension groups and run
    let mut claimed_exts: HashSet<String> = HashSet::new();
    for enricher in enrichers {
        let matching_exts: Vec<String> = enricher
            .extensions()
            .iter()
            .filter(|ext| by_ext.contains_key(**ext) && !claimed_exts.contains(**ext))
            .map(|ext| ext.to_string())
            .collect();

        if matching_exts.is_empty() {
            continue;
        }

        // Collect all targets for this enricher
        let enricher_targets: Vec<EnrichmentTarget> = matching_exts
            .iter()
            .flat_map(|ext| {
                by_ext
                    .get(ext)
                    .into_iter()
                    .flat_map(|targets| targets.iter().map(|t| (*t).clone()))
            })
            .collect();

        // Mark extensions as claimed (no double-enrichment)
        for ext in &matching_exts {
            claimed_exts.insert(ext.clone());
        }

        // Detect project root from first file
        let project_root = enricher_targets
            .first()
            .and_then(|t| enricher.detect_project_root(Path::new(&t.file_path)));

        if let Some(root) = project_root {
            let result = enricher.enrich(&root, &enricher_targets);
            results.push(result);
        }
    }

    results
}

#[cfg(test)]
#[path = "../tests/lsp_tests.rs"]
mod tests;
