//! LanguageExtractor trait for per-language symbol extraction from tree-sitter ASTs.

use crate::symbol::{Reference, Symbol};

/// Trait for per-language symbol extraction from tree-sitter ASTs.
///
/// Each language (Rust, TypeScript, Go, etc.) implements this trait to provide
/// language-specific extraction of symbols and references from parsed syntax trees.
pub trait LanguageExtractor: Send + Sync {
    /// Returns the human-readable language name (e.g., "rust", "typescript").
    fn language_name(&self) -> &str;

    /// Returns the file extensions this extractor handles (e.g., &["rs"] for Rust).
    fn file_extensions(&self) -> &[&str];

    /// Returns the tree-sitter Language for configuring the parser.
    fn tree_sitter_language(&self) -> tree_sitter::Language;

    /// Extract all symbol definitions from a parsed tree-sitter AST.
    fn extract_symbols(
        &self,
        tree: &tree_sitter::Tree,
        source: &[u8],
        file_path: &str,
    ) -> Vec<Symbol>;

    /// Extract all references (calls, imports, etc.) from a parsed tree-sitter AST.
    fn extract_references(
        &self,
        tree: &tree_sitter::Tree,
        source: &[u8],
        file_path: &str,
    ) -> Vec<Reference>;
}
