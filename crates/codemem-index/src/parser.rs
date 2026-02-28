//! Tree-sitter parsing coordinator.
//!
//! Detects language from file extension, selects the appropriate extractor,
//! and runs symbol + reference extraction.

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
}

/// Coordinates tree-sitter parsing across multiple languages.
pub struct CodeParser {
    extractors: Vec<Box<dyn LanguageExtractor>>,
}

impl CodeParser {
    /// Create a new CodeParser with all registered language extractors.
    pub fn new() -> Self {
        Self {
            extractors: languages::all_extractors(),
        }
    }

    /// Parse a single file and extract symbols and references.
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

        Some(ParseResult {
            file_path: path.to_string(),
            language: extractor.language_name().to_string(),
            symbols,
            references,
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
mod tests {
    use super::*;

    #[test]
    fn parse_rust_file() {
        let parser = CodeParser::new();
        let source = b"pub fn hello() { println!(\"hello\"); }";
        let result = parser.parse_file("src/main.rs", source);
        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(result.language, "rust");
        assert!(!result.symbols.is_empty());
    }

    #[test]
    fn unsupported_extension_returns_none() {
        let parser = CodeParser::new();
        let result = parser.parse_file("file.xyz", b"some content");
        assert!(result.is_none());
    }

    #[test]
    fn supported_extensions_includes_rs() {
        let parser = CodeParser::new();
        assert!(parser.supports_extension("rs"));
        assert!(parser.supports_extension("py"));
        assert!(parser.supports_extension("go"));
        assert!(parser.supports_extension("java"));
        assert!(parser.supports_extension("scala"));
        assert!(parser.supports_extension("rb"));
        assert!(parser.supports_extension("cs"));
        assert!(parser.supports_extension("kt"));
        assert!(parser.supports_extension("swift"));
        assert!(parser.supports_extension("php"));
        assert!(parser.supports_extension("tf"));
        assert!(!parser.supports_extension("xyz"));
    }
}
