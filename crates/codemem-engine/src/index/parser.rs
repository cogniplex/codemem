//! ast-grep parsing coordinator.
//!
//! Detects language from file extension, selects the appropriate rules,
//! and runs symbol + reference extraction via the unified engine.

use crate::index::chunker::{chunk_file, ChunkConfig, CodeChunk};
use crate::index::engine::AstGrepEngine;
use crate::index::symbol::{Reference, Symbol};
use ast_grep_core::tree_sitter::LanguageExt;
use std::path::Path;

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

/// Coordinates ast-grep parsing across multiple languages.
pub struct CodeParser {
    engine: AstGrepEngine,
    chunk_config: ChunkConfig,
}

impl CodeParser {
    /// Create a new CodeParser with all registered language rules.
    pub fn new() -> Self {
        Self {
            engine: AstGrepEngine::new(),
            chunk_config: ChunkConfig::default(),
        }
    }

    /// Create a new CodeParser with a custom chunk configuration.
    pub fn with_chunk_config(chunk_config: ChunkConfig) -> Self {
        Self {
            engine: AstGrepEngine::new(),
            chunk_config,
        }
    }

    /// Parse a single file and extract symbols, references, and chunks.
    ///
    /// Returns `None` if the file extension is not supported or parsing fails.
    pub fn parse_file(&self, path: &str, content: &[u8]) -> Option<ParseResult> {
        let extension = Path::new(path).extension().and_then(|ext| ext.to_str())?;

        let lang = self.engine.find_language(extension)?;
        let source = std::str::from_utf8(content).ok()?;

        // C1: Parse source once and share the tree across all three passes.
        let root = lang.lang.ast_grep(source);
        let symbols = self
            .engine
            .extract_symbols_from_tree(lang, &root, source, path);
        let references = self
            .engine
            .extract_references_from_tree(lang, &root, source, path);
        let chunks = chunk_file(&root, source, path, &symbols, &self.chunk_config);

        // Map internal language names to the canonical names used by consumers.
        // tsx/javascript share TypeScript extraction rules (same grammar family).
        // Consumers (graph nodes, MCP tools) treat them as "typescript" for uniformity,
        // since JS/TS/TSX/JSX all use the same symbol/reference extraction logic.
        let language_name = match lang.name {
            "tsx" | "javascript" => "typescript",
            other => other,
        };

        Some(ParseResult {
            file_path: path.to_string(),
            language: language_name.to_string(),
            symbols,
            references,
            chunks,
        })
    }

    /// Returns the list of all supported file extensions.
    pub fn supported_extensions(&self) -> Vec<&str> {
        self.engine.supported_extensions()
    }

    /// Check if a given file extension is supported.
    pub fn supports_extension(&self, ext: &str) -> bool {
        self.engine.supports_extension(ext)
    }
}

impl ParseResult {
    /// C5: Generate (id, text) pairs suitable for adding to a BM25 index.
    ///
    /// For symbols: uses `qualified_name + signature + doc_comment` as document text.
    /// For chunks: uses the chunk text content.
    ///
    /// IDs are prefixed with `sym:` and `chunk:` respectively.
    pub fn bm25_documents(&self) -> Vec<(String, String)> {
        let mut docs = Vec::with_capacity(self.symbols.len() + self.chunks.len());

        for sym in &self.symbols {
            let mut text = sym.qualified_name.clone();
            if !sym.signature.is_empty() {
                text.push(' ');
                text.push_str(&sym.signature);
            }
            if let Some(ref doc) = sym.doc_comment {
                text.push(' ');
                text.push_str(doc);
            }
            docs.push((format!("sym:{}", sym.qualified_name), text));
        }

        for chunk in &self.chunks {
            let id = format!("chunk:{}:{}", chunk.file_path, chunk.index);
            docs.push((id, chunk.text.clone()));
        }

        docs
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
