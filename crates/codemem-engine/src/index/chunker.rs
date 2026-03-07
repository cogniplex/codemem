//! CST-aware code chunking.
//!
//! Splits source files into semantically meaningful chunks using the
//! concrete syntax tree (CST) produced by ast-grep/tree-sitter. The algorithm:
//!
//! 1. If a CST node fits within `max_chunk_size` (non-whitespace chars) -> emit it as a chunk.
//! 2. If too large -> recurse into named children, preferring semantic boundaries
//!    (function/class/impl definitions) as split points.
//! 3. Adjacent small siblings are merged greedily, but only when they share the same
//!    semantic category (e.g., imports with imports, declarations with declarations).
//! 4. When a chunk comes from inside a function/class, a truncated signature header
//!    is prepended so the chunk is self-contextualizing for embeddings.
//!
//! Each chunk records its parent symbol (resolved by line-range containment).

use crate::index::symbol::Symbol;
use ast_grep_core::{Doc, Node};
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
    /// Number of lines to overlap between adjacent chunks (0 = no overlap).
    pub overlap_lines: usize,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            max_chunk_size: 1500,
            min_chunk_size: 50,
            overlap_lines: 0,
        }
    }
}

/// Count non-whitespace characters in a string.
fn count_non_ws(s: &str) -> usize {
    s.chars().filter(|c| !c.is_whitespace()).count()
}

/// Semantic category of a CST node, used to decide merge compatibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SemanticCategory {
    /// Import/use statements
    Import,
    /// Function, method, class, struct, enum, impl, trait, interface definitions
    Declaration,
    /// Comments and doc comments
    Comment,
    /// Everything else (expressions, statements, etc.)
    Other,
}

/// Classify a tree-sitter node kind into a semantic category.
fn classify_node(kind: &str) -> SemanticCategory {
    match kind {
        // Imports / use
        k if k.contains("import") || k == "use_declaration" || k == "use_item"
            || k == "extern_crate_declaration" || k == "include_directive"
            || k == "using_declaration" || k == "package_declaration" => SemanticCategory::Import,

        // Comments
        k if k.contains("comment") || k == "line_comment" || k == "block_comment"
            || k == "doc_comment" => SemanticCategory::Comment,

        // Declarations — functions, classes, structs, etc.
        k if k.contains("function") || k.contains("method") || k.contains("class")
            || k.contains("struct") || k.contains("enum") || k.contains("interface")
            || k.contains("trait") || k.contains("impl")
            || k == "const_item" || k == "static_item" || k == "type_alias"
            || k == "type_item" || k == "mod_item" || k == "module"
            || k == "lexical_declaration" || k == "variable_declaration"
            || k == "export_statement" => SemanticCategory::Declaration,

        _ => SemanticCategory::Other,
    }
}

/// Returns true if a node kind represents a semantic boundary — a top-level
/// declaration that should be kept whole when possible.
fn is_semantic_boundary(kind: &str) -> bool {
    matches!(classify_node(kind), SemanticCategory::Declaration)
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
    /// Semantic category for merge compatibility.
    category: SemanticCategory,
}

/// Chunk a file using its CST tree.
///
/// - `root`: the ast-grep parsed root (AstGrep instance).
/// - `source`: the raw source string.
/// - `file_path`: path to the source file (stored in each chunk).
/// - `symbols`: symbols extracted from this file (used for parent resolution).
/// - `config`: chunking parameters.
pub fn chunk_file<D: Doc>(
    root: &ast_grep_core::AstGrep<D>,
    source: &str,
    file_path: &str,
    symbols: &[Symbol],
    config: &ChunkConfig,
) -> Vec<CodeChunk>
where
    D::Lang: ast_grep_core::Language,
{
    if source.trim().is_empty() {
        return Vec::new();
    }

    let root_node = root.root();
    let mut raw_chunks = Vec::new();
    collect_chunks(&root_node, config, &mut raw_chunks);

    // Merge adjacent small chunks greedily
    let merged = merge_small_chunks(raw_chunks, source, config);

    // C3: Apply overlap — prepend trailing lines from previous chunk
    let merged = if config.overlap_lines > 0 {
        apply_overlap(merged, source, config.overlap_lines)
    } else {
        merged
    };

    // C2: Build interval index once for O(log n) parent resolution per chunk
    let interval_index = SymbolIntervalIndex::build(symbols);

    // Assign indices, resolve parent symbols, and inject signature context
    merged
        .into_iter()
        .enumerate()
        .map(|(idx, raw)| {
            let parent = interval_index.resolve(raw.line_start, raw.line_end);
            let parent_symbol = parent.map(|s| s.qualified_name.clone());

            // Signature context injection: if this chunk is strictly inside a
            // symbol (doesn't start at the symbol's first line), prepend a
            // truncated signature so the chunk is self-contextualizing.
            let text = if let Some(sym) = parent {
                if raw.line_start > sym.line_start && !sym.signature.is_empty() {
                    let sig = truncate_signature(&sym.signature, 120);
                    format!("[context: {sig}]\n{}", raw.text)
                } else {
                    raw.text
                }
            } else {
                raw.text
            };

            CodeChunk {
                index: idx,
                non_ws_chars: count_non_ws(&text),
                text,
                node_kind: raw.node_kind,
                line_start: raw.line_start,
                line_end: raw.line_end,
                byte_start: raw.byte_start,
                byte_end: raw.byte_end,
                parent_symbol,
                file_path: file_path.to_string(),
            }
        })
        .collect()
}

/// Recursively collect chunks from a CST node.
fn collect_chunks<D: Doc>(node: &Node<'_, D>, config: &ChunkConfig, out: &mut Vec<RawChunk>)
where
    D::Lang: ast_grep_core::Language,
{
    let text = node.text();
    let nws = count_non_ws(&text);
    let kind = node.kind().to_string();

    // If the node fits, emit it as a single chunk
    if nws <= config.max_chunk_size {
        let range = node.range();
        out.push(RawChunk {
            text: text.to_string(),
            category: classify_node(&kind),
            node_kind: kind,
            line_start: node.start_pos().line(),
            line_end: node.end_pos().line(),
            byte_start: range.start,
            byte_end: range.end,
            non_ws_chars: nws,
        });
        return;
    }

    // Too large: recurse into named children
    let named_children: Vec<_> = node.children().filter(|c| c.is_named()).collect();
    if named_children.is_empty() {
        // No named children (e.g., a very large string literal), emit as-is
        let range = node.range();
        out.push(RawChunk {
            text: text.to_string(),
            category: classify_node(&kind),
            node_kind: kind,
            line_start: node.start_pos().line(),
            line_end: node.end_pos().line(),
            byte_start: range.start,
            byte_end: range.end,
            non_ws_chars: nws,
        });
        return;
    }

    // Semantic-boundary-aware splitting: if this node contains semantic boundaries
    // (e.g., an impl block with methods), split at those boundaries. Group
    // non-boundary children between boundaries together.
    let has_boundaries = named_children
        .iter()
        .any(|c| is_semantic_boundary(&c.kind()));

    if has_boundaries {
        // Collect runs: non-boundary children are grouped, boundary children
        // are recursed individually.
        let mut non_boundary_group: Vec<&Node<'_, D>> = Vec::new();
        for child in &named_children {
            if is_semantic_boundary(&child.kind()) {
                // Flush any accumulated non-boundary nodes as a merged chunk
                if !non_boundary_group.is_empty() {
                    emit_group(&non_boundary_group, config, out);
                    non_boundary_group.clear();
                }
                // Recurse the boundary child on its own
                collect_chunks(child, config, out);
            } else {
                non_boundary_group.push(child);
            }
        }
        // Flush trailing non-boundary nodes
        if !non_boundary_group.is_empty() {
            emit_group(&non_boundary_group, config, out);
        }
    } else {
        for child in &named_children {
            collect_chunks(child, config, out);
        }
    }
}

/// Emit a group of non-boundary sibling nodes. If they fit together, emit as one
/// chunk; otherwise recurse each individually.
fn emit_group<D: Doc>(
    nodes: &[&Node<'_, D>],
    config: &ChunkConfig,
    out: &mut Vec<RawChunk>,
) where
    D::Lang: ast_grep_core::Language,
{
    if nodes.is_empty() {
        return;
    }

    // Check total size of the group
    let total_nws: usize = nodes.iter().map(|n| count_non_ws(&n.text())).sum();
    if total_nws <= config.max_chunk_size {
        // Emit as a single merged chunk
        let first = nodes.first().unwrap();
        let last = nodes.last().unwrap();
        let text: String = nodes.iter().map(|n| n.text().to_string()).collect::<Vec<_>>().join("\n");
        let first_kind = first.kind();
        let kind = nodes
            .iter()
            .map(|n| n.kind().to_string())
            .collect::<Vec<_>>()
            .join(",");
        let range_start = first.range().start;
        let range_end = last.range().end;
        out.push(RawChunk {
            text,
            category: classify_node(&first_kind),
            node_kind: kind,
            line_start: first.start_pos().line(),
            line_end: last.end_pos().line(),
            byte_start: range_start,
            byte_end: range_end,
            non_ws_chars: total_nws,
        });
    } else {
        // Too large together — recurse each individually
        for node in nodes {
            collect_chunks(node, config, out);
        }
    }
}

/// Returns true if two semantic categories are compatible for merging.
/// Only merges chunks of the same category, treating Comment as mergeable
/// with anything (comments often annotate adjacent code).
fn categories_mergeable(a: SemanticCategory, b: SemanticCategory) -> bool {
    a == b || a == SemanticCategory::Comment || b == SemanticCategory::Comment
}

/// Merge adjacent small chunks greedily, respecting semantic categories.
fn merge_small_chunks(chunks: Vec<RawChunk>, source: &str, config: &ChunkConfig) -> Vec<RawChunk> {
    if chunks.is_empty() {
        return Vec::new();
    }

    let mut result: Vec<RawChunk> = Vec::new();

    for chunk in chunks {
        if let Some(last) = result.last_mut() {
            // Only merge if at least one is below min_chunk_size AND categories are compatible
            if (last.non_ws_chars < config.min_chunk_size
                || chunk.non_ws_chars < config.min_chunk_size)
                && categories_mergeable(last.category, chunk.category)
            {
                // Compute actual merged non-whitespace count before deciding to merge
                let merged_start = last.byte_start;
                let merged_end = chunk.byte_end;
                let merged_text = if merged_end <= source.len() {
                    source[merged_start..merged_end].to_string()
                } else {
                    format!("{}\n{}", last.text, chunk.text)
                };
                let merged_nws = count_non_ws(&merged_text);

                if merged_nws <= config.max_chunk_size {
                    last.text = merged_text;
                    // C4: Preserve individual node_kinds as comma-separated
                    if last.node_kind.contains(&chunk.node_kind) {
                        // Already contains this kind, no-op
                    } else {
                        last.node_kind = format!("{},{}", last.node_kind, chunk.node_kind);
                    }
                    last.line_end = chunk.line_end;
                    last.byte_end = merged_end;
                    last.non_ws_chars = merged_nws;
                    // Keep the more specific category (prefer non-Comment)
                    if last.category == SemanticCategory::Comment {
                        last.category = chunk.category;
                    }
                    continue;
                }
            }
        }
        result.push(chunk);
    }

    result
}

/// C3: Apply overlap between adjacent chunks by prepending trailing lines
/// from the previous chunk to the current one.
fn apply_overlap(chunks: Vec<RawChunk>, source: &str, overlap_lines: usize) -> Vec<RawChunk> {
    if chunks.len() <= 1 || overlap_lines == 0 {
        return chunks;
    }

    let source_lines: Vec<&str> = source.lines().collect();
    let mut result = Vec::with_capacity(chunks.len());

    for (i, mut chunk) in chunks.into_iter().enumerate() {
        if i > 0 && chunk.line_start > 0 {
            // Prepend `overlap_lines` lines from before this chunk's start
            let overlap_start = chunk.line_start.saturating_sub(overlap_lines);
            if overlap_start < chunk.line_start && overlap_start < source_lines.len() {
                let end = chunk.line_start.min(source_lines.len());
                let prefix: String = source_lines[overlap_start..end].join("\n");
                chunk.text = format!("{}\n{}", prefix, chunk.text);
                chunk.line_start = overlap_start;
                chunk.non_ws_chars = count_non_ws(&chunk.text);
            }
        }
        result.push(chunk);
    }

    result
}

/// Truncate a signature to at most `max_len` chars, cutting at a word boundary.
fn truncate_signature(sig: &str, max_len: usize) -> &str {
    // Take only the first line of multi-line signatures
    let first_line = sig.lines().next().unwrap_or(sig);
    if first_line.len() <= max_len {
        return first_line;
    }
    // Find last space before max_len
    match first_line[..max_len].rfind(' ') {
        Some(pos) => &first_line[..pos],
        None => &first_line[..max_len],
    }
}

/// Pre-sorted symbol index for O(log n) parent resolution via binary search.
struct SymbolIntervalIndex<'a> {
    /// Symbols sorted by (line_start ASC, line_end DESC) — outermost first at each start.
    sorted: Vec<&'a Symbol>,
}

impl<'a> SymbolIntervalIndex<'a> {
    fn build(symbols: &'a [Symbol]) -> Self {
        let mut sorted: Vec<&Symbol> = symbols.iter().collect();
        sorted.sort_by(|a, b| {
            a.line_start
                .cmp(&b.line_start)
                .then_with(|| b.line_end.cmp(&a.line_end))
        });
        Self { sorted }
    }

    /// Find the innermost symbol containing [line_start, line_end].
    /// Uses binary search to find candidates starting at or before line_start,
    /// then scans forward for the tightest containment.
    fn resolve(&self, line_start: usize, line_end: usize) -> Option<&'a Symbol> {
        if self.sorted.is_empty() {
            return None;
        }

        // Binary search: find the rightmost symbol whose line_start <= line_start
        let idx = match self
            .sorted
            .binary_search_by(|s| s.line_start.cmp(&line_start))
        {
            Ok(i) => i,
            Err(i) => {
                if i == 0 {
                    return None;
                }
                i - 1
            }
        };

        let mut best: Option<&Symbol> = None;
        let mut best_span = usize::MAX;

        // Scan backwards from idx (all candidates have line_start <= line_start)
        for &sym in self.sorted[..=idx].iter().rev() {
            if sym.line_start > line_start {
                continue;
            }
            // Once we pass symbols that start too early and are too short, stop
            if best.is_some() && sym.line_end < line_end {
                // Symbols are sorted with largest span first at each start position,
                // so once we see one that doesn't contain us and we already have
                // a best, earlier symbols with the same start won't either.
                // But symbols with smaller line_start may still contain us.
                continue;
            }
            if sym.line_end >= line_end {
                let span = sym.line_end - sym.line_start;
                if span < best_span {
                    best_span = span;
                    best = Some(sym);
                }
            }
        }

        best
    }
}

#[cfg(test)]
#[path = "tests/chunker_tests.rs"]
mod tests;
