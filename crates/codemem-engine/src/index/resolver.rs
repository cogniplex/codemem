//! Reference resolution into graph edges.
//!
//! Resolves unresolved references (by simple name) to their target symbols
//! and produces typed edges for the knowledge graph.

use crate::index::symbol::{Reference, ReferenceKind, Symbol};
use codemem_core::RelationshipType;
use std::collections::{HashMap, HashSet};

/// A resolved edge connecting two symbols by qualified name.
#[derive(Debug, Clone)]
pub struct ResolvedEdge {
    /// Qualified name of the source symbol.
    pub source_qualified_name: String,
    /// Qualified name of the resolved target symbol.
    pub target_qualified_name: String,
    /// The relationship type for this edge.
    pub relationship: RelationshipType,
    /// File path where the reference occurs.
    pub file_path: String,
    /// Line number of the reference.
    pub line: usize,
    /// R2: Confidence of the resolution (0.0 = guessed, 1.0 = exact match).
    pub resolution_confidence: f64,
}

/// Resolves references to target symbols and produces graph edges.
pub struct ReferenceResolver {
    /// Map of qualified_name -> Symbol for exact resolution.
    symbol_index: HashMap<String, Symbol>,
    /// Map of simple name -> Vec<qualified_name> for ambiguous resolution.
    name_index: HashMap<String, Vec<String>>,
    /// R2: Set of imported qualified names per file for scoring.
    file_imports: HashMap<String, HashSet<String>>,
}

impl ReferenceResolver {
    /// Create a new empty resolver.
    pub fn new() -> Self {
        Self {
            symbol_index: HashMap::new(),
            name_index: HashMap::new(),
            file_imports: HashMap::new(),
        }
    }

    /// Add symbols to the resolver's index.
    pub fn add_symbols(&mut self, symbols: &[Symbol]) {
        for sym in symbols {
            self.symbol_index
                .insert(sym.qualified_name.clone(), sym.clone());

            self.name_index
                .entry(sym.name.clone())
                .or_default()
                .push(sym.qualified_name.clone());
        }
    }

    /// R2: Register import references so the resolver can prefer imported symbols.
    pub fn add_imports(&mut self, references: &[Reference]) {
        for r in references {
            if r.kind == ReferenceKind::Import {
                self.file_imports
                    .entry(r.file_path.clone())
                    .or_default()
                    .insert(r.target_name.clone());
            }
        }
    }

    /// Resolve a single reference to a target symbol with confidence.
    ///
    /// Resolution strategy:
    /// 1. Exact match on qualified name (confidence 1.0)
    /// 2. R4: Cross-module path resolution — strip `crate::` prefix, try partial matches
    /// 3. Simple name match with R2 scoring heuristics (confidence varies)
    /// 4. Unresolved (returns None)
    pub fn resolve_with_confidence(&self, reference: &Reference) -> Option<(&Symbol, f64)> {
        // 1. Exact qualified name match
        if let Some(sym) = self.symbol_index.get(&reference.target_name) {
            return Some((sym, 1.0));
        }

        // R4: Try stripping `crate::` prefix for cross-module resolution
        if reference.target_name.starts_with("crate::") {
            let stripped = &reference.target_name["crate::".len()..];
            if let Some(sym) = self.symbol_index.get(stripped) {
                return Some((sym, 0.95));
            }
            // Try matching against all qualified names ending with this suffix
            for (qn, sym) in &self.symbol_index {
                if qn.ends_with(stripped) {
                    let prefix_len = qn.len() - stripped.len();
                    if prefix_len == 0 || qn.as_bytes()[prefix_len - 1] == b':' {
                        return Some((sym, 0.85));
                    }
                }
            }
        }

        // R4: Try matching `module::function` against `crate::module::function`
        if reference.target_name.contains("::") {
            let with_crate = format!("crate::{}", reference.target_name);
            if let Some(sym) = self.symbol_index.get(&with_crate) {
                return Some((sym, 0.9));
            }
            // Try suffix matching for partial paths
            for (qn, sym) in &self.symbol_index {
                if qn.ends_with(&reference.target_name) {
                    let prefix_len = qn.len() - reference.target_name.len();
                    if prefix_len == 0 || qn.as_bytes()[prefix_len - 1] == b':' {
                        return Some((sym, 0.8));
                    }
                }
            }
        }

        // 2. Simple name match with scoring heuristics
        let simple_name = reference
            .target_name
            .rsplit("::")
            .next()
            .unwrap_or(&reference.target_name);

        if let Some(candidates) = self.name_index.get(simple_name) {
            if candidates.len() == 1 {
                // Unambiguous
                let confidence = if simple_name == reference.target_name {
                    0.9 // Exact simple name match
                } else {
                    0.7 // Matched via last segment only
                };
                return self
                    .symbol_index
                    .get(&candidates[0])
                    .map(|s| (s, confidence));
            }

            // R2: Score candidates with heuristics
            let file_imports = self.file_imports.get(&reference.file_path);
            let mut best: Option<(&Symbol, f64)> = None;

            for qn in candidates {
                if let Some(sym) = self.symbol_index.get(qn) {
                    let mut score: f64 = 0.0;

                    // Prefer symbols imported in the same file
                    if let Some(imports) = file_imports {
                        if imports.contains(&sym.qualified_name)
                            || imports.iter().any(|imp| imp.ends_with(&sym.name))
                        {
                            score += 0.4;
                        }
                    }

                    // Prefer symbols in the same file
                    if sym.file_path == reference.file_path {
                        score += 0.3;
                    }

                    // Prefer exact name match (not just substring/last-segment)
                    if sym.name == reference.target_name {
                        score += 0.2;
                    }

                    // Prefer symbols in the same package/module (share path prefix)
                    let ref_module = extract_module_path(&reference.file_path);
                    let sym_module = extract_module_path(&sym.file_path);
                    if ref_module == sym_module {
                        score += 0.1;
                    }

                    if best.is_none() || score > best.unwrap().1 {
                        best = Some((sym, score));
                    }
                }
            }

            if let Some((sym, score)) = best {
                // Normalize score to a confidence value in [0.3, 0.8]
                let confidence = 0.3 + (score.min(1.0) * 0.5);
                return Some((sym, confidence));
            }
        }

        None
    }

    /// Resolve all references into edges.
    ///
    /// Only produces edges for successfully resolved references.
    pub fn resolve_all(&self, references: &[Reference]) -> Vec<ResolvedEdge> {
        references
            .iter()
            .filter_map(|r| {
                let (target, confidence) = self.resolve_with_confidence(r)?;
                let relationship = match r.kind {
                    ReferenceKind::Call => RelationshipType::Calls,
                    ReferenceKind::Import => RelationshipType::Imports,
                    ReferenceKind::Inherits => RelationshipType::Inherits,
                    ReferenceKind::Implements => RelationshipType::Implements,
                    ReferenceKind::TypeUsage => RelationshipType::DependsOn,
                };

                Some(ResolvedEdge {
                    source_qualified_name: r.source_qualified_name.clone(),
                    target_qualified_name: target.qualified_name.clone(),
                    relationship,
                    file_path: r.file_path.clone(),
                    line: r.line,
                    resolution_confidence: confidence,
                })
            })
            .collect()
    }
}

/// Extract a module path from a file path for same-package heuristic.
/// e.g., "src/index/parser.rs" -> "src/index"
fn extract_module_path(file_path: &str) -> &str {
    file_path.rsplit_once('/').map(|(dir, _)| dir).unwrap_or("")
}

impl Default for ReferenceResolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[path = "tests/resolver_tests.rs"]
mod tests;
