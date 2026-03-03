use super::*;
use crate::index::symbol::{SymbolKind, Visibility};

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
