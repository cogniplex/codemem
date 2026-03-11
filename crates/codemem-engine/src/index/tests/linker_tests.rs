use super::*;
use crate::index::manifest::ManifestResult;
use crate::index::symbol::{Symbol, SymbolKind, Visibility};

fn make_symbol(name: &str, qn: &str, vis: Visibility) -> Symbol {
    Symbol {
        name: name.to_string(),
        qualified_name: qn.to_string(),
        kind: SymbolKind::Function,
        signature: format!("fn {name}()"),
        visibility: vis,
        file_path: "lib.rs".to_string(),
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

fn make_pending_ref(
    id: &str,
    namespace: &str,
    source_node: &str,
    target_name: &str,
    package_hint: Option<&str>,
    ref_kind: &str,
) -> PendingRef {
    PendingRef {
        id: id.to_string(),
        namespace: namespace.to_string(),
        source_node: source_node.to_string(),
        target_name: target_name.to_string(),
        package_hint: package_hint.map(|s| s.to_string()),
        ref_kind: ref_kind.to_string(),
        file_path: None,
        line: None,
    }
}

#[test]
fn extract_packages_from_manifests() {
    let mut manifests = ManifestResult::new();
    manifests
        .packages
        .insert("my-lib".to_string(), "Cargo.toml".to_string());
    manifests
        .packages
        .insert("my-utils".to_string(), "utils/Cargo.toml".to_string());
    manifests
        .dependencies
        .push(crate::index::manifest::Dependency {
            name: "my-lib".to_string(),
            version: "0.1.0".to_string(),
            dev: false,
            manifest_path: "Cargo.toml".to_string(),
        });

    let packages = extract_packages(&manifests, "repo-a");
    assert_eq!(packages.len(), 2);
    assert!(packages.iter().all(|p| p.namespace == "repo-a"));

    let lib_pkg = packages
        .iter()
        .find(|p| p.package_name == "my-lib")
        .unwrap();
    assert_eq!(lib_pkg.version, "0.1.0");
    assert_eq!(lib_pkg.manifest, "Cargo.toml");

    let utils_pkg = packages
        .iter()
        .find(|p| p.package_name == "my-utils")
        .unwrap();
    assert!(utils_pkg.version.is_empty()); // no self-dependency entry
}

#[test]
fn match_symbol_exact() {
    let symbols = vec![make_symbol(
        "validate",
        "utils.validate",
        Visibility::Public,
    )];
    let result = match_symbol("utils.validate", &symbols);
    assert!(result.is_some());
    let (qn, conf) = result.unwrap();
    assert_eq!(qn, "utils.validate");
    assert!(conf >= 1.0);
}

#[test]
fn match_symbol_suffix() {
    let symbols = vec![
        make_symbol("validate", "utils.validate", Visibility::Public),
        make_symbol("check", "other.check", Visibility::Public),
    ];
    // Exact match confidence for comparison
    let (_, exact_conf) = match_symbol("utils.validate", &symbols).unwrap();

    let result = match_symbol("validate", &symbols);
    assert!(result.is_some());
    let (qn, conf) = result.unwrap();
    assert_eq!(qn, "utils.validate");
    // Suffix match should be high confidence but strictly below exact
    assert!(conf > 0.5);
    assert!(conf < exact_conf);
}

#[test]
fn match_symbol_simple_name() {
    let symbols = vec![
        make_symbol("validate", "a.validate", Visibility::Private),
        make_symbol("validate", "b.validate", Visibility::Public),
    ];
    // "validate" suffix-matches both; public one should win
    let result = match_symbol("validate", &symbols);
    assert!(result.is_some());
    let (qn, _conf) = result.unwrap();
    assert_eq!(qn, "b.validate");
}

#[test]
fn match_symbol_no_match() {
    let symbols = vec![make_symbol(
        "validate",
        "utils.validate",
        Visibility::Public,
    )];
    let result = match_symbol("nonexistent", &symbols);
    assert!(result.is_none());
}

#[test]
fn match_symbol_prefers_public_over_private() {
    let symbols = vec![
        make_symbol("process", "internal.process", Visibility::Private),
        make_symbol("process", "api.process", Visibility::Public),
    ];
    let result = match_symbol("process", &symbols);
    assert!(result.is_some());
    let (qn, _conf) = result.unwrap();
    assert_eq!(qn, "api.process");
}

#[test]
fn forward_link_creates_edges() {
    let pending_refs = vec![make_pending_ref(
        "ref-1",
        "repo-a",
        "sym:handler.process",
        "validate",
        Some("shared-lib"),
        "call",
    )];

    let registry = vec![RegisteredPackage {
        package_name: "shared-lib".to_string(),
        namespace: "repo-b".to_string(),
        version: "1.0.0".to_string(),
        manifest: "Cargo.toml".to_string(),
    }];

    let resolve_fn = |_ns: &str, _name: &str| -> Vec<SymbolMatch> {
        vec![SymbolMatch {
            qualified_name: "shared.validate".to_string(),
            visibility: Visibility::Public,
            kind: "function".to_string(),
        }]
    };

    let result = forward_link("repo-a", &pending_refs, &registry, &resolve_fn);
    assert_eq!(result.forward_edges.len(), 1);
    assert_eq!(result.resolved_ref_ids, vec!["ref-1"]);

    let edge = &result.forward_edges[0];
    assert_eq!(edge.source, "sym:handler.process");
    assert_eq!(edge.target, "sym:shared.validate");
    assert_eq!(edge.relationship, "Calls");
    assert_eq!(edge.source_namespace, "repo-a");
    assert_eq!(edge.target_namespace, "repo-b");
    // Forward-linked edge should have meaningful confidence
    assert!(edge.confidence > 0.5);
    assert!(edge.confidence <= 1.0);
}

#[test]
fn forward_link_skips_same_namespace() {
    let pending_refs = vec![make_pending_ref(
        "ref-1",
        "repo-a",
        "sym:handler.process",
        "validate",
        Some("my-lib"),
        "call",
    )];

    // Registry entry is in the SAME namespace
    let registry = vec![RegisteredPackage {
        package_name: "my-lib".to_string(),
        namespace: "repo-a".to_string(),
        version: "1.0.0".to_string(),
        manifest: "Cargo.toml".to_string(),
    }];

    let resolve_fn = |_ns: &str, _name: &str| -> Vec<SymbolMatch> {
        vec![SymbolMatch {
            qualified_name: "my.validate".to_string(),
            visibility: Visibility::Public,
            kind: "function".to_string(),
        }]
    };

    let result = forward_link("repo-a", &pending_refs, &registry, &resolve_fn);
    assert!(result.forward_edges.is_empty());
    assert!(result.resolved_ref_ids.is_empty());
}

#[test]
fn forward_link_skips_no_package_hint() {
    let pending_refs = vec![make_pending_ref(
        "ref-1",
        "repo-a",
        "sym:handler.process",
        "validate",
        None, // no package hint
        "call",
    )];

    let registry = vec![RegisteredPackage {
        package_name: "shared-lib".to_string(),
        namespace: "repo-b".to_string(),
        version: "1.0.0".to_string(),
        manifest: "Cargo.toml".to_string(),
    }];

    let resolve_fn = |_ns: &str, _name: &str| -> Vec<SymbolMatch> {
        panic!("should not be called");
    };

    let result = forward_link("repo-a", &pending_refs, &registry, &resolve_fn);
    assert!(result.forward_edges.is_empty());
    assert!(result.resolved_ref_ids.is_empty());
}

#[test]
fn backward_link_creates_edges() {
    let symbols = vec![
        make_symbol("validate", "utils.validate", Visibility::Public),
        make_symbol("process", "handler.process", Visibility::Public),
    ];

    let pending_refs = vec![make_pending_ref(
        "ref-1",
        "repo-b", // from another namespace
        "sym:caller.run",
        "validate",
        Some("my-lib"),
        "call",
    )];

    let result = backward_link("repo-a", &["my-lib".to_string()], &pending_refs, &symbols);
    assert_eq!(result.backward_edges.len(), 1);
    assert_eq!(result.resolved_ref_ids, vec!["ref-1"]);

    let edge = &result.backward_edges[0];
    assert_eq!(edge.source, "sym:caller.run");
    assert_eq!(edge.target, "sym:utils.validate");
    assert_eq!(edge.relationship, "Calls");
    assert_eq!(edge.source_namespace, "repo-b");
    assert_eq!(edge.target_namespace, "repo-a");
}

#[test]
fn backward_link_skips_self() {
    let symbols = vec![make_symbol(
        "validate",
        "utils.validate",
        Visibility::Public,
    )];

    let pending_refs = vec![make_pending_ref(
        "ref-1",
        "repo-a", // same namespace as target
        "sym:caller.run",
        "validate",
        Some("my-lib"),
        "call",
    )];

    let result = backward_link("repo-a", &["my-lib".to_string()], &pending_refs, &symbols);
    assert!(result.backward_edges.is_empty());
    assert!(result.resolved_ref_ids.is_empty());
}

#[test]
fn make_edge_id_format() {
    let id = make_edge_id("repo-a", "sym:handler", "repo-b", "validate");
    assert_eq!(id, "xref:repo-a/sym:handler->repo-b/validate");
}

#[test]
fn ref_kind_to_relationship_mapping() {
    assert_eq!(ref_kind_to_relationship("call"), "Calls");
    assert_eq!(ref_kind_to_relationship("import"), "Imports");
    assert_eq!(ref_kind_to_relationship("inherits"), "Inherits");
    assert_eq!(ref_kind_to_relationship("implements"), "Implements");
    assert_eq!(ref_kind_to_relationship("type_usage"), "DependsOn");
    assert_eq!(ref_kind_to_relationship("unknown"), "RelatesTo");
    assert_eq!(ref_kind_to_relationship(""), "RelatesTo");
}
