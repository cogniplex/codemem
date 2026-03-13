//! Main indexing pipeline orchestrator.
//!
//! Walks a directory, filters supported files, checks for changes,
//! and parses each file using the CodeParser.

use crate::index::chunker::CodeChunk;
use crate::index::incremental::ChangeDetector;
use crate::index::parser::{CodeParser, ParseResult};
use crate::index::resolver::{ReferenceResolver, ResolvedEdge, UnresolvedRef};
use crate::index::symbol::{Reference, Symbol};
use ignore::WalkBuilder;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

/// Progress event emitted during directory indexing.
#[derive(Debug, Clone)]
pub struct IndexProgress {
    /// Number of files scanned so far.
    pub files_scanned: usize,
    /// Number of files parsed so far.
    pub files_parsed: usize,
    /// Total symbols extracted so far.
    pub total_symbols: usize,
    /// Current file being processed (relative path).
    pub current_file: String,
}

/// Result of indexing a directory.
#[derive(Debug)]
pub struct IndexResult {
    /// Total number of files scanned (walked).
    pub files_scanned: usize,
    /// Number of files successfully parsed.
    pub files_parsed: usize,
    /// Number of files skipped (unchanged since last index).
    pub files_skipped: usize,
    /// Total symbols extracted across all files.
    pub total_symbols: usize,
    /// Total references extracted across all files.
    pub total_references: usize,
    /// Total CST-aware chunks extracted across all files.
    pub total_chunks: usize,
    /// Individual parse results for each successfully parsed file.
    pub parse_results: Vec<ParseResult>,
}

/// Result of `index_and_resolve()` — the complete indexing + resolution output.
#[derive(Debug)]
pub struct IndexAndResolveResult {
    /// The underlying index result with per-file parse data.
    pub index: IndexResult,
    /// All symbols collected from all parsed files.
    pub symbols: Vec<Symbol>,
    /// All references collected from all parsed files.
    pub references: Vec<Reference>,
    /// All CST-aware chunks collected from all parsed files.
    pub chunks: Vec<CodeChunk>,
    /// All unique file paths that were parsed (relative to `root_path`).
    pub file_paths: HashSet<String>,
    /// Resolved edges from reference resolution.
    pub edges: Vec<ResolvedEdge>,
    /// References that could not be resolved to known symbols.
    /// Preserved for deferred cross-repo linking.
    pub unresolved: Vec<UnresolvedRef>,
    /// The absolute root path that was indexed. Downstream code can use this
    /// to reconstruct absolute paths (e.g. for `git -C` or file reading).
    pub root_path: PathBuf,
    /// SCIP graph build result (nodes, edges, memories). None if SCIP was skipped.
    pub scip_build: Option<super::scip::graph_builder::ScipBuildResult>,
}

/// The main indexing pipeline.
///
/// Coordinates directory walking, change detection, and parsing
/// to produce a complete index of a codebase.
pub struct Indexer {
    parser: CodeParser,
    change_detector: ChangeDetector,
}

impl Indexer {
    /// Create a new Indexer with default settings.
    pub fn new() -> Self {
        Self {
            parser: CodeParser::new(),
            change_detector: ChangeDetector::new(),
        }
    }

    /// Create a new Indexer with a pre-loaded ChangeDetector.
    pub fn with_change_detector(change_detector: ChangeDetector) -> Self {
        Self {
            parser: CodeParser::new(),
            change_detector,
        }
    }

    /// Get a reference to the change detector for persistence.
    pub fn change_detector(&self) -> &ChangeDetector {
        &self.change_detector
    }

    /// Get a mutable reference to the change detector.
    pub fn change_detector_mut(&mut self) -> &mut ChangeDetector {
        &mut self.change_detector
    }

    /// Index a directory, returning all parse results.
    ///
    /// Walks the directory respecting `.gitignore` rules (via the `ignore` crate),
    /// filters by supported file extensions, checks incremental state, and parses
    /// each changed file.
    pub fn index_directory(
        &mut self,
        root: &Path,
    ) -> Result<IndexResult, codemem_core::CodememError> {
        self.index_directory_inner(root, None)
    }

    /// Index a directory with optional progress reporting.
    ///
    /// If a broadcast sender is provided, progress events are sent as files
    /// are processed. This is useful for SSE streaming to the frontend.
    pub fn index_directory_with_progress(
        &mut self,
        root: &Path,
        tx: Option<&tokio::sync::broadcast::Sender<IndexProgress>>,
    ) -> Result<IndexResult, codemem_core::CodememError> {
        self.index_directory_inner(root, tx)
    }

    /// Common implementation for directory indexing with optional progress callback.
    fn index_directory_inner(
        &mut self,
        root: &Path,
        tx: Option<&tokio::sync::broadcast::Sender<IndexProgress>>,
    ) -> Result<IndexResult, codemem_core::CodememError> {
        let mut files_scanned = 0usize;
        let mut files_parsed = 0usize;
        let mut files_skipped = 0usize;
        let mut total_symbols = 0usize;
        let mut total_references = 0usize;
        let mut total_chunks = 0usize;
        let mut parse_results = Vec::new();

        let walker = WalkBuilder::new(root)
            .hidden(true) // skip hidden files/dirs
            .git_ignore(true) // respect .gitignore
            .git_global(true) // respect global gitignore
            .git_exclude(true) // respect .git/info/exclude
            .build();

        for entry in walker {
            let entry = match entry {
                Ok(e) => e,
                Err(err) => {
                    tracing::warn!("Walk error: {}", err);
                    continue;
                }
            };

            // Skip directories
            if !entry.file_type().is_some_and(|ft| ft.is_file()) {
                continue;
            }

            let path = entry.path();

            // Check if the file extension is supported
            let ext = match path.extension().and_then(|e| e.to_str()) {
                Some(e) => e,
                None => continue,
            };

            if !self.parser.supports_extension(ext) {
                continue;
            }

            files_scanned += 1;

            // Read file content
            let content = match std::fs::read(path) {
                Ok(c) => c,
                Err(err) => {
                    tracing::warn!("Failed to read {}: {}", path.display(), err);
                    continue;
                }
            };

            // Use paths relative to root so node IDs are portable across machines.
            let rel_path = path.strip_prefix(root).unwrap_or(path);
            let path_str = rel_path.to_string_lossy().to_string();

            // Check incremental state (returns pre-computed hash to avoid double-hashing)
            let (changed, hash) = self.change_detector.check_changed(&path_str, &content);
            if !changed {
                files_skipped += 1;
                continue;
            }

            // Parse the file
            match self.parser.parse_file(&path_str, &content) {
                Some(result) => {
                    total_symbols += result.symbols.len();
                    total_references += result.references.len();
                    total_chunks += result.chunks.len();
                    files_parsed += 1;

                    // Record the pre-computed hash (avoids re-hashing)
                    self.change_detector.record_hash(&path_str, hash);

                    parse_results.push(result);

                    // Send progress event if a sender is provided
                    if let Some(tx) = tx {
                        let _ = tx.send(IndexProgress {
                            files_scanned,
                            files_parsed,
                            total_symbols,
                            current_file: path_str.clone(),
                        });
                    }
                }
                None => {
                    tracing::warn!("Failed to parse {}", path_str);
                }
            }
        }

        tracing::info!(
            "Indexed {}: {} scanned, {} parsed, {} skipped, {} symbols, {} references, {} chunks",
            root.display(),
            files_scanned,
            files_parsed,
            files_skipped,
            total_symbols,
            total_references,
            total_chunks,
        );

        Ok(IndexResult {
            files_scanned,
            files_parsed,
            files_skipped,
            total_symbols,
            total_references,
            total_chunks,
            parse_results,
        })
    }

    /// Index a directory, collect all symbols/references/chunks, and resolve
    /// references into graph edges.
    ///
    /// This is the high-level entry point that combines `index_directory()`
    /// with reference resolution — the common pipeline shared by the MCP
    /// `index_codebase` tool and the CLI `index` command.
    pub fn index_and_resolve(
        &mut self,
        root: &Path,
    ) -> Result<IndexAndResolveResult, codemem_core::CodememError> {
        self.index_and_resolve_with_scip(root, None, None)
    }

    /// Index a directory with optional SCIP integration.
    ///
    /// If `scip_covered_files` is provided, symbol/reference extraction is skipped
    /// for those files (SCIP already handled them). Code chunking still runs for ALL files.
    /// The `scip_build` result is attached to the output for persistence.
    pub fn index_and_resolve_with_scip(
        &mut self,
        root: &Path,
        scip_covered_files: Option<&HashSet<String>>,
        scip_build: Option<super::scip::graph_builder::ScipBuildResult>,
    ) -> Result<IndexAndResolveResult, codemem_core::CodememError> {
        let result = self.index_directory(root)?;

        let mut all_symbols = Vec::new();
        let mut all_references = Vec::new();
        let mut all_chunks = Vec::new();
        let mut file_paths = HashSet::new();

        // Consume parse_results by value to avoid cloning symbols/references/chunks
        let IndexResult {
            files_scanned,
            files_parsed,
            files_skipped,
            total_symbols,
            total_references,
            total_chunks,
            parse_results,
        } = result;

        for pr in parse_results {
            file_paths.insert(pr.file_path.clone());
            // For SCIP-covered files, only keep chunks — skip symbols/references
            // since SCIP provides compiler-grade data for those.
            if scip_covered_files.is_some_and(|s| s.contains(&pr.file_path)) {
                all_chunks.extend(pr.chunks);
            } else {
                all_symbols.extend(pr.symbols);
                all_references.extend(pr.references);
                all_chunks.extend(pr.chunks);
            }
        }

        let mut resolver = ReferenceResolver::new();
        resolver.add_symbols(&all_symbols);
        resolver.add_imports(&all_references);
        let resolve_result = resolver.resolve_all_with_unresolved(&all_references);

        // Canonicalize the root so downstream code can reconstruct absolute paths.
        let root_path = std::fs::canonicalize(root).unwrap_or_else(|_| root.to_path_buf());

        Ok(IndexAndResolveResult {
            index: IndexResult {
                files_scanned,
                files_parsed,
                files_skipped,
                total_symbols,
                total_references,
                total_chunks,
                parse_results: Vec::new(),
            },
            symbols: all_symbols,
            references: all_references,
            chunks: all_chunks,
            file_paths,
            edges: resolve_result.edges,
            unresolved: resolve_result.unresolved,
            root_path,
            scip_build,
        })
    }
}

impl Default for Indexer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[path = "tests/indexer_tests.rs"]
mod tests;
