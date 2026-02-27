//! Main indexing pipeline orchestrator.
//!
//! Walks a directory, filters supported files, checks for changes,
//! and parses each file using the CodeParser.

use crate::incremental::ChangeDetector;
use crate::parser::{CodeParser, ParseResult};
use ignore::WalkBuilder;
use std::path::Path;

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
            "Indexed {}: {} scanned, {} parsed, {} skipped, {} symbols, {} references",
            root.display(),
            files_scanned,
            files_parsed,
            files_skipped,
            total_symbols,
            total_references,
        );

        Ok(IndexResult {
            files_scanned,
            files_parsed,
            files_skipped,
            total_symbols,
            total_references,
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
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn index_temp_directory() {
        // Create a temp directory with a Rust file
        let dir = std::env::temp_dir().join("codemem_index_test");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();

        fs::write(
            dir.join("main.rs"),
            b"pub fn hello() { println!(\"hello\"); }\n",
        )
        .unwrap();

        fs::write(
            dir.join("lib.rs"),
            b"pub struct Config { pub debug: bool }\n",
        )
        .unwrap();

        // Also create a non-Rust file that should be skipped
        fs::write(dir.join("readme.txt"), b"This is not Rust").unwrap();

        let mut indexer = Indexer::new();
        let result = indexer.index_directory(&dir).unwrap();

        assert_eq!(result.files_scanned, 2, "Should scan 2 .rs files");
        assert_eq!(result.files_parsed, 2, "Should parse 2 .rs files");
        assert_eq!(
            result.files_skipped, 0,
            "No files should be skipped on first run"
        );
        assert!(
            result.total_symbols >= 2,
            "Should have at least 2 symbols (hello, Config)"
        );

        // Run again - all files should be skipped (incremental)
        let result2 = indexer.index_directory(&dir).unwrap();
        assert_eq!(result2.files_scanned, 2);
        assert_eq!(
            result2.files_parsed, 0,
            "All files should be skipped on second run"
        );
        assert_eq!(result2.files_skipped, 2);

        // Cleanup
        let _ = fs::remove_dir_all(&dir);
    }
}
