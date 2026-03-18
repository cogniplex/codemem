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
        parameters: Vec::new(),
        return_type: None,
        is_async: false,
        attributes: Vec::new(),
        throws: Vec::new(),
        generic_params: None,
        is_abstract: false,
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

    let result = resolver.resolve_with_confidence(&reference);
    assert!(result.is_some());
    assert_eq!(result.unwrap().0.qualified_name, "module::foo");
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

    let result = resolver.resolve_with_confidence(&reference);
    assert!(result.is_some());
    assert_eq!(result.unwrap().0.qualified_name, "module::foo");
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

    let result = resolver.resolve_with_confidence(&reference);
    assert!(result.is_some());
    assert_eq!(result.unwrap().0.qualified_name, "b::foo");
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

// ── UnresolvedRef + package hint tests ─────────────────────────────

#[test]
fn unresolved_refs_are_preserved() {
    let resolver = ReferenceResolver::new(); // no symbols registered
    let references = vec![
        Reference {
            source_qualified_name: "app::main".to_string(),
            target_name: "requests.api.get".to_string(),
            kind: ReferenceKind::Import,
            file_path: "app.py".to_string(),
            line: 1,
        },
        Reference {
            source_qualified_name: "app::main".to_string(),
            target_name: "unknown_func".to_string(),
            kind: ReferenceKind::Call,
            file_path: "app.py".to_string(),
            line: 5,
        },
    ];

    let result = resolver.resolve_all_with_unresolved(&references);
    assert!(result.edges.is_empty(), "no symbols → no resolved edges");
    assert_eq!(result.unresolved.len(), 2);

    // Verify first unresolved ref has correct fields
    let first = &result.unresolved[0];
    assert_eq!(first.source_node, "app::main");
    assert_eq!(first.target_name, "requests.api.get");
    assert_eq!(first.ref_kind, "import");
    assert_eq!(first.file_path, "app.py");
    assert_eq!(first.line, 1);
    assert_eq!(first.package_hint, Some("requests".to_string()));

    // Call reference should have no package hint
    let second = &result.unresolved[1];
    assert_eq!(second.package_hint, None);
}

#[test]
fn package_hint_python_import() {
    let hint = extract_package_hint("requests.api.get", ReferenceKind::Import);
    assert_eq!(hint, Some("requests".to_string()));
}

#[test]
fn package_hint_scoped_npm() {
    let hint = extract_package_hint("@acme/shared-lib/utils", ReferenceKind::Import);
    assert_eq!(hint, Some("@acme/shared-lib".to_string()));
}

#[test]
fn package_hint_go_module() {
    let hint = extract_package_hint("github.com/acme/utils", ReferenceKind::Import);
    assert_eq!(hint, Some("github.com/acme/utils".to_string()));
}

#[test]
fn package_hint_rust_crate() {
    let hint = extract_package_hint("serde::Serialize", ReferenceKind::Import);
    assert_eq!(hint, Some("serde".to_string()));
}

#[test]
fn package_hint_local_import_none() {
    let hint = extract_package_hint("crate::module::item", ReferenceKind::Import);
    assert_eq!(hint, None);
}

#[test]
fn package_hint_relative_import_none() {
    let hint = extract_package_hint("./utils", ReferenceKind::Import);
    assert_eq!(hint, None);
}

#[test]
fn package_hint_call_reference_none() {
    let hint = extract_package_hint("something", ReferenceKind::Call);
    assert_eq!(hint, None);
}

#[test]
fn package_hint_single_word_import() {
    let hint = extract_package_hint("flask", ReferenceKind::Import);
    assert_eq!(hint, Some("flask".to_string()));
}

#[test]
fn callback_references_capped_at_0_6() {
    let mut resolver = ReferenceResolver::new();
    let sym = Symbol {
        name: "transform".to_string(),
        qualified_name: "utils.transform".to_string(),
        kind: SymbolKind::Function,
        signature: "def transform(x)".to_string(),
        visibility: Visibility::Public,
        file_path: "utils.py".to_string(),
        line_start: 1,
        line_end: 3,
        doc_comment: None,
        parent: None,
        parameters: Vec::new(),
        return_type: None,
        is_async: false,
        attributes: Vec::new(),
        throws: Vec::new(),
        generic_params: None,
        is_abstract: false,
    };
    resolver.add_symbols(&[sym]);

    let references = vec![Reference {
        source_qualified_name: "main".to_string(),
        target_name: "transform".to_string(),
        kind: ReferenceKind::Callback,
        file_path: "utils.py".to_string(),
        line: 10,
    }];

    let edges = resolver.resolve_all(&references);
    assert_eq!(edges.len(), 1, "callback reference should resolve");
    assert!(
        edges[0].resolution_confidence <= 0.6,
        "callback confidence should be capped at 0.6, got: {}",
        edges[0].resolution_confidence
    );
    assert_eq!(edges[0].relationship, RelationshipType::Calls);
}

#[test]
fn callback_references_capped_in_resolve_all_with_unresolved() {
    let mut resolver = ReferenceResolver::new();
    let sym = Symbol {
        name: "transform".to_string(),
        qualified_name: "utils.transform".to_string(),
        kind: SymbolKind::Function,
        signature: "def transform(x)".to_string(),
        visibility: Visibility::Public,
        file_path: "utils.py".to_string(),
        line_start: 1,
        line_end: 3,
        doc_comment: None,
        parent: None,
        parameters: Vec::new(),
        return_type: None,
        is_async: false,
        attributes: Vec::new(),
        throws: Vec::new(),
        generic_params: None,
        is_abstract: false,
    };
    resolver.add_symbols(&[sym]);

    let references = vec![Reference {
        source_qualified_name: "main".to_string(),
        target_name: "transform".to_string(),
        kind: ReferenceKind::Callback,
        file_path: "utils.py".to_string(),
        line: 10,
    }];

    let result = resolver.resolve_all_with_unresolved(&references);
    assert_eq!(result.edges.len(), 1, "callback reference should resolve");
    assert!(
        result.edges[0].resolution_confidence <= 0.6,
        "callback confidence should be capped at 0.6, got: {}",
        result.edges[0].resolution_confidence
    );
    assert_eq!(result.edges[0].relationship, RelationshipType::Calls);
}

#[test]
fn resolve_all_with_unresolved_splits_correctly() {
    let mut resolver = ReferenceResolver::new();
    let sym = make_symbol("known_fn", "mod::known_fn", "lib.rs");
    resolver.add_symbols(&[sym]);

    let references = vec![
        // This one should resolve
        Reference {
            source_qualified_name: "caller".to_string(),
            target_name: "known_fn".to_string(),
            kind: ReferenceKind::Call,
            file_path: "lib.rs".to_string(),
            line: 10,
        },
        // This one should NOT resolve
        Reference {
            source_qualified_name: "caller".to_string(),
            target_name: "serde::Deserialize".to_string(),
            kind: ReferenceKind::Import,
            file_path: "lib.rs".to_string(),
            line: 1,
        },
        // Another resolvable one (exact qualified name)
        Reference {
            source_qualified_name: "other".to_string(),
            target_name: "mod::known_fn".to_string(),
            kind: ReferenceKind::Call,
            file_path: "other.rs".to_string(),
            line: 20,
        },
    ];

    let result = resolver.resolve_all_with_unresolved(&references);
    assert_eq!(result.edges.len(), 2, "two references should resolve");
    assert_eq!(
        result.unresolved.len(),
        1,
        "one reference should be unresolved"
    );

    // The unresolved one should be the serde import
    assert_eq!(result.unresolved[0].target_name, "serde::Deserialize");
    assert_eq!(result.unresolved[0].package_hint, Some("serde".to_string()));
    assert_eq!(result.unresolved[0].ref_kind, "import");
}
