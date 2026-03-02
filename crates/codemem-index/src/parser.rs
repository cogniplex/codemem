//! Tree-sitter parsing coordinator.
//!
//! Detects language from file extension, selects the appropriate extractor,
//! and runs symbol + reference extraction.

use crate::chunker::{chunk_file, ChunkConfig, CodeChunk};
use crate::extractor::LanguageExtractor;
use crate::languages;
use crate::symbol::{Reference, Symbol};
use std::path::Path;
use tree_sitter::Parser;

/// Result of parsing a single file.
#[derive(Debug, Clone)]
pub struct ParseResult {
    /// Path to the parsed file.
    pub file_path: String,
    /// Language that was detected and used.
    pub language: String,
    /// All symbols extracted from the file.
    pub symbols: Vec<Symbol>,
    /// All references extracted from the file.
    pub references: Vec<Reference>,
    /// CST-aware code chunks extracted from the file.
    pub chunks: Vec<CodeChunk>,
}

/// Coordinates tree-sitter parsing across multiple languages.
pub struct CodeParser {
    extractors: Vec<Box<dyn LanguageExtractor>>,
    chunk_config: ChunkConfig,
}

impl CodeParser {
    /// Create a new CodeParser with all registered language extractors.
    pub fn new() -> Self {
        Self {
            extractors: languages::all_extractors(),
            chunk_config: ChunkConfig::default(),
        }
    }

    /// Create a new CodeParser with a custom chunk configuration.
    pub fn with_chunk_config(chunk_config: ChunkConfig) -> Self {
        Self {
            extractors: languages::all_extractors(),
            chunk_config,
        }
    }

    /// Parse a single file and extract symbols, references, and chunks.
    ///
    /// Returns `None` if the file extension is not supported or parsing fails.
    pub fn parse_file(&self, path: &str, content: &[u8]) -> Option<ParseResult> {
        let extension = Path::new(path).extension().and_then(|ext| ext.to_str())?;

        let extractor = self.find_extractor(extension)?;

        let mut parser = Parser::new();
        parser
            .set_language(&extractor.tree_sitter_language())
            .ok()?;

        let tree = parser.parse(content, None)?;

        let symbols = extractor.extract_symbols(&tree, content, path);
        let references = extractor.extract_references(&tree, content, path);
        let chunks = chunk_file(&tree, content, path, &symbols, &self.chunk_config);

        Some(ParseResult {
            file_path: path.to_string(),
            language: extractor.language_name().to_string(),
            symbols,
            references,
            chunks,
        })
    }

    /// Returns the list of all supported file extensions.
    pub fn supported_extensions(&self) -> Vec<&str> {
        self.extractors
            .iter()
            .flat_map(|e| e.file_extensions().iter().copied())
            .collect()
    }

    /// Check if a given file extension is supported.
    pub fn supports_extension(&self, ext: &str) -> bool {
        self.extractors
            .iter()
            .any(|e| e.file_extensions().contains(&ext))
    }

    /// Find the extractor for a given file extension.
    fn find_extractor(&self, ext: &str) -> Option<&dyn LanguageExtractor> {
        self.extractors
            .iter()
            .find(|e| e.file_extensions().contains(&ext))
            .map(|e| e.as_ref())
    }
}

impl Default for CodeParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[path = "tests/parser_tests.rs"]
mod tests;
