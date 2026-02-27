//! Reference resolution into graph edges.
//!
//! Resolves unresolved references (by simple name) to their target symbols
//! and produces typed edges for the knowledge graph.

use crate::symbol::{Reference, ReferenceKind, Symbol};
use codemem_core::RelationshipType;
use std::collections::HashMap;

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
}

/// Resolves references to target symbols and produces graph edges.
pub struct ReferenceResolver {
    /// Map of qualified_name -> Symbol for exact resolution.
    symbol_index: HashMap<String, Symbol>,
    /// Map of simple name -> Vec<qualified_name> for ambiguous resolution.
    name_index: HashMap<String, Vec<String>>,
}

impl ReferenceResolver {
    /// Create a new empty resolver.
    pub fn new() -> Self {
        Self {
            symbol_index: HashMap::new(),
            name_index: HashMap::new(),
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

    /// Resolve a single reference to a target symbol.
    ///
    /// Resolution strategy:
    /// 1. Exact match on qualified name
    /// 2. Simple name match with locality preference (same file first)
    /// 3. Unresolved (returns None)
    pub fn resolve(&self, reference: &Reference) -> Option<&Symbol> {
        // 1. Exact qualified name match
        if let Some(sym) = self.symbol_index.get(&reference.target_name) {
            return Some(sym);
        }

        // 2. Simple name match
        // Extract the last segment of the target name for lookup
        let simple_name = reference
            .target_name
            .rsplit("::")
            .next()
            .unwrap_or(&reference.target_name);

        if let Some(candidates) = self.name_index.get(simple_name) {
            if candidates.len() == 1 {
                // Unambiguous
                return self.symbol_index.get(&candidates[0]);
            }

            // Prefer candidate in the same file
            for qn in candidates {
                if let Some(sym) = self.symbol_index.get(qn) {
                    if sym.file_path == reference.file_path {
                        return Some(sym);
                    }
                }
            }

            // Fall back to first candidate
            if let Some(qn) = candidates.first() {
                return self.symbol_index.get(qn);
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
                let target = self.resolve(r)?;
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
                })
            })
            .collect()
    }

    /// Get the number of indexed symbols.
    pub fn symbol_count(&self) -> usize {
        self.symbol_index.len()
    }
}

impl Default for ReferenceResolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbol::{SymbolKind, Visibility};

    fn make_symbol(name: &str, qn: &str, file: &str) -> Symbol {
        Symbol {
            name: name.to_string(),
            qualified_name: qn.to_string(),
            kind: SymbolKind::Function,
            signature: format!("fn {}()", name),
            visibility: Visibility::Public,
            file_path: file.to_string(),
            line_start: 0,
            line_end: 0,
            doc_comment: None,
            parent: None,
        }
    }

    #[test]
    fn resolve_exact_match() {
        let mut resolver = ReferenceResolver::new();
        let sym = make_symbol("foo", "module::foo", "lib.rs");
        resolver.add_symbols(&[sym]);

        let reference = Reference {
            source_qualified_name: "bar".to_string(),
            target_name: "module::foo".to_string(),
            kind: ReferenceKind::Call,
            file_path: "lib.rs".to_string(),
            line: 10,
        };

        let result = resolver.resolve(&reference);
        assert!(result.is_some());
        assert_eq!(result.unwrap().qualified_name, "module::foo");
    }

    #[test]
    fn resolve_simple_name_match() {
        let mut resolver = ReferenceResolver::new();
        let sym = make_symbol("foo", "module::foo", "lib.rs");
        resolver.add_symbols(&[sym]);

        let reference = Reference {
            source_qualified_name: "bar".to_string(),
            target_name: "foo".to_string(),
            kind: ReferenceKind::Call,
            file_path: "lib.rs".to_string(),
            line: 10,
        };

        let result = resolver.resolve(&reference);
        assert!(result.is_some());
        assert_eq!(result.unwrap().qualified_name, "module::foo");
    }

    #[test]
    fn resolve_prefers_same_file() {
        let mut resolver = ReferenceResolver::new();
        let sym1 = make_symbol("foo", "a::foo", "a.rs");
        let sym2 = make_symbol("foo", "b::foo", "b.rs");
        resolver.add_symbols(&[sym1, sym2]);

        let reference = Reference {
            source_qualified_name: "caller".to_string(),
            target_name: "foo".to_string(),
            kind: ReferenceKind::Call,
            file_path: "b.rs".to_string(),
            line: 5,
        };

        let result = resolver.resolve(&reference);
        assert!(result.is_some());
        assert_eq!(result.unwrap().qualified_name, "b::foo");
    }

    #[test]
    fn resolve_all_produces_edges() {
        let mut resolver = ReferenceResolver::new();
        let sym = make_symbol("target_fn", "mod::target_fn", "lib.rs");
        resolver.add_symbols(&[sym]);

        let references = vec![Reference {
            source_qualified_name: "caller".to_string(),
            target_name: "target_fn".to_string(),
            kind: ReferenceKind::Call,
            file_path: "lib.rs".to_string(),
            line: 10,
        }];

        let edges = resolver.resolve_all(&references);
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].relationship, RelationshipType::Calls);
        assert_eq!(edges[0].target_qualified_name, "mod::target_fn");
    }

    #[test]
    fn unresolved_reference_skipped() {
        let resolver = ReferenceResolver::new();
        let references = vec![Reference {
            source_qualified_name: "caller".to_string(),
            target_name: "nonexistent".to_string(),
            kind: ReferenceKind::Call,
            file_path: "lib.rs".to_string(),
            line: 10,
        }];

        let edges = resolver.resolve_all(&references);
        assert!(edges.is_empty());
    }
}
