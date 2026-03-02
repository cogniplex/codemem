//! Main indexing pipeline orchestrator.
//!
//! Walks a directory, filters supported files, checks for changes,
//! and parses each file using the CodeParser.

use crate::incremental::ChangeDetector;
use crate::parser::{CodeParser, ParseResult};
use ignore::WalkBuilder;
use std::path::Path;

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

            let path_str = path.to_string_lossy().to_string();

            // Check incremental state
            if !self.change_detector.is_changed(&path_str, &content) {
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

                    // Update the change detector hash
                    self.change_detector.update_hash(&path_str, &content);

                    parse_results.push(result);
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

    /// Index a directory with optional progress reporting.
    ///
    /// If a broadcast sender is provided, progress events are sent as files
    /// are processed. This is useful for SSE streaming to the frontend.
    pub fn index_directory_with_progress(
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
            .hidden(true)
            .git_ignore(true)
            .git_global(true)
            .git_exclude(true)
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

            let path_str = path.to_string_lossy().to_string();

            // Compute a relative path for progress reporting
            let relative_path = path
                .strip_prefix(root)
                .unwrap_or(path)
                .to_string_lossy()
                .to_string();

            // Check incremental state
            if !self.change_detector.is_changed(&path_str, &content) {
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

                    // Update the change detector hash
                    self.change_detector.update_hash(&path_str, &content);

                    parse_results.push(result);

                    // Send progress event if a sender is provided
                    if let Some(tx) = tx {
                        let _ = tx.send(IndexProgress {
                            files_scanned,
                            files_parsed,
                            total_symbols,
                            current_file: relative_path,
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
}

impl Default for Indexer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[path = "tests/indexer_tests.rs"]
mod tests;
