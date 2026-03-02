//! CST-aware code chunking.
//!
//! Splits source files into semantically meaningful chunks using the
//! concrete syntax tree (CST) produced by tree-sitter. The algorithm:
//!
//! 1. If a CST node fits within `max_chunk_size` (non-whitespace chars) -> emit it as a chunk.
//! 2. If too large -> recurse into named children.
//! 3. Adjacent small siblings are merged greedily until the merged size would exceed `max_chunk_size`.
//!
//! Each chunk records its parent symbol (resolved by line-range containment).

use crate::symbol::Symbol;
use serde::{Deserialize, Serialize};

/// A code chunk produced by the CST-aware chunker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeChunk {
    /// 0-based index of this chunk within the file.
    pub index: usize,
    /// The source text of this chunk.
    pub text: String,
    /// The tree-sitter node kind (e.g., "function_item", "impl_item").
    pub node_kind: String,
    /// 0-based starting line.
    pub line_start: usize,
    /// 0-based ending line.
    pub line_end: usize,
    /// Byte offset start.
    pub byte_start: usize,
    /// Byte offset end.
    pub byte_end: usize,
    /// Count of non-whitespace characters.
    pub non_ws_chars: usize,
    /// Qualified name of the innermost containing symbol, if any.
    pub parent_symbol: Option<String>,
    /// Path of the source file.
    pub file_path: String,
}

/// Configuration for the chunker.
#[derive(Debug, Clone)]
pub struct ChunkConfig {
    /// Maximum chunk size in non-whitespace characters.
    pub max_chunk_size: usize,
    /// Minimum chunk size in non-whitespace characters.
    pub min_chunk_size: usize,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            max_chunk_size: 1500,
            min_chunk_size: 50,
        }
    }
}

/// Count non-whitespace characters in a string.
fn count_non_ws(s: &str) -> usize {
    s.chars().filter(|c| !c.is_whitespace()).count()
}

/// Intermediate chunk before index assignment and parent resolution.
struct RawChunk {
    text: String,
    node_kind: String,
    line_start: usize,
    line_end: usize,
    byte_start: usize,
    byte_end: usize,
    non_ws_chars: usize,
}

/// Chunk a file using its CST tree.
///
/// - `tree`: the tree-sitter parse tree for the file.
/// - `source`: the raw source bytes.
/// - `file_path`: path to the source file (stored in each chunk).
/// - `symbols`: symbols extracted from this file (used for parent resolution).
/// - `config`: chunking parameters.
pub fn chunk_file(
    tree: &tree_sitter::Tree,
    source: &[u8],
    file_path: &str,
    symbols: &[Symbol],
    config: &ChunkConfig,
) -> Vec<CodeChunk> {
    let source_str = String::from_utf8_lossy(source);
    if source_str.trim().is_empty() {
        return Vec::new();
    }

    let root = tree.root_node();
    let mut raw_chunks = Vec::new();
    collect_chunks(root, source, config, &mut raw_chunks);

    // Merge adjacent small chunks greedily
    let merged = merge_small_chunks(raw_chunks, source, config);

    // Assign indices and resolve parent symbols
    merged
        .into_iter()
        .enumerate()
        .map(|(idx, raw)| {
            let parent_symbol = resolve_parent_symbol(raw.line_start, raw.line_end, symbols);
            CodeChunk {
                index: idx,
                text: raw.text,
                node_kind: raw.node_kind,
                line_start: raw.line_start,
                line_end: raw.line_end,
                byte_start: raw.byte_start,
                byte_end: raw.byte_end,
                non_ws_chars: raw.non_ws_chars,
                parent_symbol,
                file_path: file_path.to_string(),
            }
        })
        .collect()
}

/// Recursively collect chunks from a CST node.
fn collect_chunks(
    node: tree_sitter::Node<'_>,
    source: &[u8],
    config: &ChunkConfig,
    out: &mut Vec<RawChunk>,
) {
    let text = node.utf8_text(source).unwrap_or("");
    let nws = count_non_ws(text);

    // If the node fits, emit it as a single chunk
    if nws <= config.max_chunk_size {
        out.push(RawChunk {
            text: text.to_string(),
            node_kind: node.kind().to_string(),
            line_start: node.start_position().row,
            line_end: node.end_position().row,
            byte_start: node.start_byte(),
            byte_end: node.end_byte(),
            non_ws_chars: nws,
        });
        return;
    }

    // Too large: recurse into named children
    let mut has_named_children = false;
    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        has_named_children = true;
        collect_chunks(child, source, config, out);
    }

    // If no named children (e.g., a very large string literal), emit as-is
    if !has_named_children {
        out.push(RawChunk {
            text: text.to_string(),
            node_kind: node.kind().to_string(),
            line_start: node.start_position().row,
            line_end: node.end_position().row,
            byte_start: node.start_byte(),
            byte_end: node.end_byte(),
            non_ws_chars: nws,
        });
    }
}

/// Merge adjacent small chunks greedily.
fn merge_small_chunks(chunks: Vec<RawChunk>, source: &[u8], config: &ChunkConfig) -> Vec<RawChunk> {
    if chunks.is_empty() {
        return Vec::new();
    }

    let mut result: Vec<RawChunk> = Vec::new();

    for chunk in chunks {
        if let Some(last) = result.last_mut() {
            // If both the current accumulator and new chunk are small, try merging
            if last.non_ws_chars < config.min_chunk_size
                || chunk.non_ws_chars < config.min_chunk_size
            {
                let combined_nws = last.non_ws_chars + chunk.non_ws_chars;
                if combined_nws <= config.max_chunk_size {
                    // Merge: extract text from byte range covering both
                    let merged_start = last.byte_start;
                    let merged_end = chunk.byte_end;
                    let merged_text =
                        String::from_utf8_lossy(&source[merged_start..merged_end]).to_string();
                    let merged_nws = count_non_ws(&merged_text);

                    last.text = merged_text;
                    last.node_kind = "merged".to_string();
                    last.line_end = chunk.line_end;
                    last.byte_end = merged_end;
                    last.non_ws_chars = merged_nws;
                    continue;
                }
            }
        }
        result.push(chunk);
    }

    result
}

/// Resolve the innermost parent symbol for a given line range.
fn resolve_parent_symbol(line_start: usize, line_end: usize, symbols: &[Symbol]) -> Option<String> {
    let mut best: Option<&Symbol> = None;
    let mut best_span = usize::MAX;

    for sym in symbols {
        // Symbol must contain the chunk's line range
        if sym.line_start <= line_start && sym.line_end >= line_end {
            let span = sym.line_end - sym.line_start;
            if span < best_span {
                best_span = span;
                best = Some(sym);
            }
        }
    }

    best.map(|s| s.qualified_name.clone())
}

#[cfg(test)]
#[path = "tests/chunker_tests.rs"]
mod tests;
