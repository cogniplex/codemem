//! Code search and summary tree domain logic.

use crate::index::symbol::{symbol_from_graph_node, Symbol};
use crate::CodememEngine;
use codemem_core::{CodememError, GraphBackend, NodeKind, RelationshipType, VectorBackend};
use serde::Serialize;

/// A code search result returned by semantic vector search over symbols and chunks.
#[derive(Debug, Clone, Serialize)]
pub struct CodeSearchResult {
    pub id: String,
    /// "chunk" or the graph node kind string (e.g. "function", "class").
    pub kind: String,
    pub label: String,
    pub similarity: f64,
    pub file_path: Option<serde_json::Value>,
    pub line_start: Option<serde_json::Value>,
    pub line_end: Option<serde_json::Value>,
    // Symbol-specific fields
    pub qualified_name: Option<String>,
    pub signature: Option<serde_json::Value>,
    pub doc_comment: Option<serde_json::Value>,
    // Chunk-specific fields
    pub node_kind: Option<serde_json::Value>,
    pub parent_symbol: Option<serde_json::Value>,
    pub non_ws_chars: Option<serde_json::Value>,
}

/// A recursive tree node for the summary tree.
#[derive(Debug, Clone, Serialize)]
pub struct SummaryTreeNode {
    pub id: String,
    pub kind: String,
    pub label: String,
    pub centrality: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub symbol_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunk_count: Option<usize>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub children: Vec<SummaryTreeNode>,
}

/// A filtered symbol match from the in-memory index cache.
#[derive(Debug, Clone, Serialize)]
pub struct SymbolSearchResult {
    pub name: String,
    pub qualified_name: String,
    pub kind: String,
    pub signature: String,
    pub file_path: String,
    pub line_start: usize,
    pub line_end: usize,
    pub visibility: String,
    pub parent: Option<String>,
}

/// Extract code references from free-form text for use in auto-linking.
///
/// Recognizes:
/// - CamelCase identifiers (likely type/class names): `ProcessRequest`, `HashMap`
/// - Backtick-wrapped code: content from \`...\`
/// - Function-like patterns: `word()` or `word::path`
/// - Short file paths: patterns like `src/foo.rs`, `lib/bar.py`
///
/// Returns deduplicated references suitable for matching against graph node labels.
pub fn extract_code_references(text: &str) -> Vec<String> {
    use regex::Regex;
    use std::collections::HashSet;
    use std::sync::OnceLock;

    static RE_CAMEL: OnceLock<Regex> = OnceLock::new();
    static RE_BACKTICK: OnceLock<Regex> = OnceLock::new();
    static RE_FUNC: OnceLock<Regex> = OnceLock::new();
    static RE_PATH: OnceLock<Regex> = OnceLock::new();

    let re_camel = RE_CAMEL.get_or_init(|| {
        // CamelCase: at least two words, each starting uppercase, min 4 chars total
        Regex::new(r"\b([A-Z][a-z]+(?:[A-Z][a-z0-9]*)+)\b").unwrap()
    });
    let re_backtick = RE_BACKTICK.get_or_init(|| {
        // Content inside backticks (non-greedy, at least 2 chars)
        Regex::new(r"`([^`]{2,})`").unwrap()
    });
    let re_func = RE_FUNC.get_or_init(|| {
        // Function calls like word() or qualified paths like word::path
        Regex::new(r"\b([a-zA-Z_]\w*(?:::[a-zA-Z_]\w*)+)\b|\b([a-zA-Z_]\w*)\(\)").unwrap()
    });
    let re_path = RE_PATH.get_or_init(|| {
        // File paths: word/word.ext patterns (2-4 segments, common extensions)
        Regex::new(r"\b([a-zA-Z0-9_\-]+(?:/[a-zA-Z0-9_\-]+)*\.[a-zA-Z]{1,4})\b").unwrap()
    });

    let mut seen = HashSet::new();
    let mut refs = Vec::new();

    let mut add = |s: &str| {
        let trimmed = s.trim();
        if trimmed.len() >= 2 && seen.insert(trimmed.to_string()) {
            refs.push(trimmed.to_string());
        }
    };

    for cap in re_camel.captures_iter(text) {
        if let Some(m) = cap.get(1) {
            add(m.as_str());
        }
    }
    for cap in re_backtick.captures_iter(text) {
        if let Some(m) = cap.get(1) {
            add(m.as_str());
        }
    }
    for cap in re_func.captures_iter(text) {
        // Group 1: qualified path (a::b::c), Group 2: function call (word())
        if let Some(m) = cap.get(1) {
            add(m.as_str());
        }
        if let Some(m) = cap.get(2) {
            add(m.as_str());
        }
    }
    for cap in re_path.captures_iter(text) {
        if let Some(m) = cap.get(1) {
            let path = m.as_str();
            // Only include if it looks like a real file path (has a directory separator)
            if path.contains('/') {
                add(path);
            }
        }
    }

    refs
}

impl CodememEngine {
    /// Estimate the approximate RAM usage of the in-memory graph.
    ///
    /// The graph engine (`GraphEngine`) keeps all nodes and edges in memory via
    /// `HashMap`s plus a `petgraph::DiGraph`. Each node carries a `GraphNode` struct
    /// (~200 bytes: id, kind, label, payload HashMap, centrality, optional memory_id
    /// and namespace) plus a petgraph `NodeIndex`. Each edge carries an `Edge` struct
    /// (~150 bytes: id, src, dst, relationship, weight, properties HashMap, timestamps)
    /// plus petgraph edge weight storage and adjacency list entries.
    ///
    /// Returns an estimate in bytes. Actual usage may be higher due to HashMap overhead,
    /// string heap allocations, and payload contents.
    pub fn graph_memory_estimate(&self) -> usize {
        let (node_count, edge_count) = match self.lock_graph() {
            Ok(g) => (g.node_count(), g.edge_count()),
            Err(_) => (0, 0),
        };
        node_count * 200 + edge_count * 150
    }

    /// Semantic code search: embed query, vector search, filter to sym:/chunk: IDs,
    /// enrich with graph node data.
    pub fn search_code(
        &self,
        query: &str,
        k: usize,
    ) -> Result<Vec<CodeSearchResult>, CodememError> {
        let emb_guard = self
            .lock_embeddings()?
            .ok_or_else(|| CodememError::InvalidInput("Embedding service not available".into()))?;

        let query_embedding = emb_guard
            .embed(query)
            .map_err(|e| CodememError::InvalidInput(format!("Embedding failed: {e}")))?;
        drop(emb_guard);

        let vec = self.lock_vector()?;
        let raw_results: Vec<(String, f32)> = vec
            .search(&query_embedding, k * 3)
            .unwrap_or_default()
            .into_iter()
            .filter(|(id, _)| id.starts_with("sym:") || id.starts_with("chunk:"))
            .take(k)
            .collect();
        drop(vec);

        if raw_results.is_empty() {
            return Ok(Vec::new());
        }

        let mut output = Vec::new();
        for (id, distance) in &raw_results {
            let similarity = 1.0 - *distance as f64;
            if let Ok(Some(node)) = self.storage.get_graph_node(id) {
                if id.starts_with("chunk:") {
                    output.push(CodeSearchResult {
                        id: id.clone(),
                        kind: "chunk".to_string(),
                        label: node.label,
                        similarity,
                        file_path: node.payload.get("file_path").cloned(),
                        line_start: node.payload.get("line_start").cloned(),
                        line_end: node.payload.get("line_end").cloned(),
                        qualified_name: None,
                        signature: None,
                        doc_comment: None,
                        node_kind: node.payload.get("node_kind").cloned(),
                        parent_symbol: node.payload.get("parent_symbol").cloned(),
                        non_ws_chars: node.payload.get("non_ws_chars").cloned(),
                    });
                } else {
                    output.push(CodeSearchResult {
                        id: id.clone(),
                        kind: node.kind.to_string(),
                        label: node.label,
                        similarity,
                        file_path: node.payload.get("file_path").cloned(),
                        line_start: node.payload.get("line_start").cloned(),
                        line_end: node.payload.get("line_end").cloned(),
                        qualified_name: Some(id.strip_prefix("sym:").unwrap_or(id).to_string()),
                        signature: node.payload.get("signature").cloned(),
                        doc_comment: node.payload.get("doc_comment").cloned(),
                        node_kind: None,
                        parent_symbol: None,
                        non_ws_chars: None,
                    });
                }
            }
        }

        Ok(output)
    }

    /// Build a hierarchical summary tree starting from a given node.
    /// Traverses Contains edges: packages -> files -> symbols.
    pub fn summary_tree(
        &self,
        start_id: &str,
        max_depth: usize,
        include_chunks: bool,
    ) -> Result<SummaryTreeNode, CodememError> {
        let graph = self.lock_graph()?;

        fn build_tree(
            graph: &dyn GraphBackend,
            node_id: &str,
            depth: usize,
            max_depth: usize,
            include_chunks: bool,
        ) -> Option<SummaryTreeNode> {
            if depth > max_depth {
                return None;
            }
            let node = match graph.get_node(node_id) {
                Ok(Some(n)) => n,
                _ => return None,
            };

            let mut children: Vec<SummaryTreeNode> = Vec::new();
            if depth < max_depth {
                if let Ok(edges) = graph.get_edges(node_id) {
                    let mut child_ids: Vec<String> = edges
                        .iter()
                        .filter(|e| {
                            e.src == node_id && e.relationship == RelationshipType::Contains
                        })
                        .map(|e| e.dst.clone())
                        .collect();
                    child_ids.sort();

                    for child_id in &child_ids {
                        if !include_chunks && child_id.starts_with("chunk:") {
                            continue;
                        }
                        if let Some(child) =
                            build_tree(graph, child_id, depth + 1, max_depth, include_chunks)
                        {
                            children.push(child);
                        }
                    }
                }
            }

            let (symbol_count, chunk_count) = if node.kind == NodeKind::File {
                let syms = children
                    .iter()
                    .filter(|c| c.kind != "chunk" && c.kind != "package" && c.kind != "file")
                    .count();
                let chunks = if include_chunks {
                    children.iter().filter(|c| c.kind == "chunk").count()
                } else {
                    0
                };
                (Some(syms), Some(chunks))
            } else {
                (None, None)
            };

            Some(SummaryTreeNode {
                id: node.id,
                kind: node.kind.to_string(),
                label: node.label,
                centrality: node.centrality,
                symbol_count,
                chunk_count,
                children,
            })
        }

        build_tree(&*graph, start_id, 0, max_depth, include_chunks)
            .ok_or_else(|| CodememError::NotFound(format!("Node not found: {start_id}")))
    }

    /// Look up a single symbol by qualified name with cache-through to the DB.
    ///
    /// 1. Check the in-memory `IndexCache`.
    /// 2. On miss, query storage for `sym:{qualified_name}` graph node.
    /// 3. Reconstruct a `Symbol`, insert into cache, return.
    pub fn get_symbol(&self, qualified_name: &str) -> Result<Option<Symbol>, CodememError> {
        // 1. Check cache
        {
            let cache = self.lock_index_cache()?;
            if let Some(ref c) = *cache {
                if let Some(sym) = c
                    .symbols
                    .iter()
                    .find(|s| s.qualified_name == qualified_name)
                {
                    return Ok(Some(sym.clone()));
                }
            }
        }

        // 2. Cache miss — query DB
        let node_id = format!("sym:{qualified_name}");
        let node = match self.storage.get_graph_node(&node_id)? {
            Some(n) => n,
            None => return Ok(None),
        };

        // 3. Reconstruct Symbol from GraphNode
        let sym = match symbol_from_graph_node(&node) {
            Some(s) => s,
            None => return Ok(None),
        };

        // 4. Insert into cache (init if None)
        {
            let mut cache = self.lock_index_cache()?;
            if let Some(ref mut c) = *cache {
                // Dedup: only insert if not already present
                if !c.symbols.iter().any(|s| s.qualified_name == qualified_name) {
                    c.symbols.push(sym.clone());
                }
            } else {
                *cache = Some(crate::IndexCache {
                    symbols: vec![sym.clone()],
                    chunks: Vec::new(),
                    root_path: String::new(),
                });
            }
        }

        Ok(Some(sym))
    }

    /// Search symbols by name, with optional kind filter. Falls through to the DB
    /// when the in-memory cache is empty or has no matches.
    pub fn search_symbols(
        &self,
        query: &str,
        limit: usize,
        kind_filter: Option<&str>,
    ) -> Result<Vec<SymbolSearchResult>, CodememError> {
        let query_lower = query.to_lowercase();
        let kind_lower = kind_filter.map(|k| k.to_lowercase());

        // Helper closure: filter + map a Symbol into a SymbolSearchResult
        let matches_filter = |sym: &Symbol| -> bool {
            let name_match = sym.name.to_lowercase().contains(&query_lower)
                || sym.qualified_name.to_lowercase().contains(&query_lower);
            if !name_match {
                return false;
            }
            if let Some(ref kl) = kind_lower {
                return sym.kind.to_string().to_lowercase() == *kl;
            }
            true
        };

        let to_result = |sym: &Symbol| -> SymbolSearchResult {
            SymbolSearchResult {
                name: sym.name.clone(),
                qualified_name: sym.qualified_name.clone(),
                kind: sym.kind.to_string(),
                signature: sym.signature.clone(),
                file_path: sym.file_path.clone(),
                line_start: sym.line_start,
                line_end: sym.line_end,
                visibility: sym.visibility.to_string(),
                parent: sym.parent.clone(),
            }
        };

        // 1. Try cache first — only short-circuit if we got enough results
        let mut results = Vec::new();
        let mut seen_qnames = std::collections::HashSet::new();
        {
            let cache = self.lock_index_cache()?;
            if let Some(ref c) = *cache {
                for sym in c.symbols.iter().filter(|s| matches_filter(s)) {
                    if seen_qnames.insert(sym.qualified_name.clone()) {
                        results.push(to_result(sym));
                        if results.len() >= limit {
                            return Ok(results);
                        }
                    }
                }
            }
        }

        // 2. Cache empty or partial results — top up from DB.
        // Over-fetch by 3× because search_graph_nodes returns all node kinds
        // (files, chunks, packages…) and we filter to sym:* nodes only.
        // NOTE: namespace=None is intentional — the in-memory cache is already
        // cross-namespace, so the DB fallback should match that behavior.
        let db_nodes = self
            .storage
            .search_graph_nodes(&query_lower, None, limit * 3)?;

        let mut new_symbols = Vec::new();

        for node in &db_nodes {
            if !node.id.starts_with("sym:") {
                continue;
            }
            if let Some(sym) = symbol_from_graph_node(node) {
                if matches_filter(&sym) && seen_qnames.insert(sym.qualified_name.clone()) {
                    results.push(to_result(&sym));
                    new_symbols.push(sym);
                    if results.len() >= limit {
                        break;
                    }
                }
            }
        }

        // 3. Populate cache with DB results for future queries
        if !new_symbols.is_empty() {
            let mut cache = self.lock_index_cache()?;
            if let Some(ref mut c) = *cache {
                for sym in new_symbols {
                    if !c
                        .symbols
                        .iter()
                        .any(|s| s.qualified_name == sym.qualified_name)
                    {
                        c.symbols.push(sym);
                    }
                }
            } else {
                *cache = Some(crate::IndexCache {
                    symbols: new_symbols,
                    chunks: Vec::new(),
                    root_path: String::new(),
                });
            }
        }

        Ok(results)
    }
}
