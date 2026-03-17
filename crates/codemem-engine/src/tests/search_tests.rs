use crate::search::extract_code_references;
use crate::CodememEngine;
use codemem_core::{Edge, GraphNode, NodeKind, RelationshipType};
use std::collections::HashMap;

// ── extract_code_references ─────────────────────────────────────────

#[test]
fn extract_camel_case_identifiers() {
    let refs = extract_code_references("The ProcessRequest handler calls BuildResponse");
    assert!(
        refs.contains(&"ProcessRequest".to_string()),
        "should extract CamelCase: got {refs:?}"
    );
    assert!(
        refs.contains(&"BuildResponse".to_string()),
        "should extract CamelCase: got {refs:?}"
    );
}

#[test]
fn extract_backtick_code() {
    let refs = extract_code_references("Use `my_function` and `SomeType` in the code");
    assert!(
        refs.contains(&"my_function".to_string()),
        "should extract backtick content: got {refs:?}"
    );
    assert!(
        refs.contains(&"SomeType".to_string()),
        "should extract backtick content: got {refs:?}"
    );
}

#[test]
fn extract_qualified_paths() {
    let refs = extract_code_references("Call std::collections::HashMap for the map");
    assert!(
        refs.contains(&"std::collections::HashMap".to_string()),
        "should extract qualified path: got {refs:?}"
    );
}

#[test]
fn extract_function_calls() {
    let refs = extract_code_references("Call process() to handle the request");
    assert!(
        refs.contains(&"process".to_string()),
        "should extract function call: got {refs:?}"
    );
}

#[test]
fn extract_file_paths() {
    let refs = extract_code_references("Look at src/main.rs for the entry point");
    assert!(
        refs.contains(&"src/main.rs".to_string()),
        "should extract file path: got {refs:?}"
    );
}

#[test]
fn extract_deduplicates() {
    let refs = extract_code_references("ProcessRequest and ProcessRequest again");
    let count = refs.iter().filter(|r| *r == "ProcessRequest").count();
    assert_eq!(count, 1, "should deduplicate references");
}

#[test]
fn extract_ignores_short_backticks() {
    let refs = extract_code_references("Use `x` in the code");
    assert!(
        !refs.contains(&"x".to_string()),
        "should ignore single-char backtick content"
    );
}

#[test]
fn extract_no_path_without_slash() {
    let refs = extract_code_references("The file main.rs is important");
    // "main.rs" without a directory separator should not be extracted as a path
    assert!(
        !refs.iter().any(|r| r == "main.rs"),
        "should not extract bare filename without slash as a path: got {refs:?}"
    );
}

// ── graph_memory_estimate ───────────────────────────────────────────

#[test]
fn graph_memory_estimate_empty() {
    let engine = CodememEngine::for_testing();
    assert_eq!(
        engine.graph_memory_estimate(),
        0,
        "empty graph should have 0 estimate"
    );
}

#[test]
fn graph_memory_estimate_with_nodes() {
    let engine = CodememEngine::for_testing();
    {
        let mut graph = engine.lock_graph().unwrap();
        let node = GraphNode {
            id: "n1".to_string(),
            kind: NodeKind::File,
            label: "test.rs".to_string(),
            payload: HashMap::new(),
            centrality: 0.0,
            memory_id: None,
            namespace: None,
            valid_from: None,
            valid_to: None,
        };
        graph.add_node(node).unwrap();
    }
    assert_eq!(engine.graph_memory_estimate(), 200, "1 node * 200 bytes");
}

// ── summary_tree ────────────────────────────────────────────────────

#[test]
fn summary_tree_not_found() {
    let engine = CodememEngine::for_testing();
    let result = engine.summary_tree("nonexistent", 3, false);
    assert!(result.is_err(), "should error for missing node");
}

#[test]
fn summary_tree_single_node() {
    let engine = CodememEngine::for_testing();
    {
        let mut graph = engine.lock_graph().unwrap();
        let node = GraphNode {
            id: "pkg:mypackage".to_string(),
            kind: NodeKind::Package,
            label: "mypackage".to_string(),
            payload: HashMap::new(),
            centrality: 0.5,
            memory_id: None,
            namespace: None,
            valid_from: None,
            valid_to: None,
        };
        graph.add_node(node).unwrap();
    }

    let tree = engine.summary_tree("pkg:mypackage", 3, false).unwrap();
    assert_eq!(tree.id, "pkg:mypackage");
    assert_eq!(tree.kind, "package");
    assert_eq!(tree.label, "mypackage");
    assert!(tree.children.is_empty());
}

#[test]
fn summary_tree_with_children() {
    let engine = CodememEngine::for_testing();
    let now = chrono::Utc::now();
    {
        let mut graph = engine.lock_graph().unwrap();
        // Package -> File -> Function
        for (id, kind, label) in [
            ("pkg:root", NodeKind::Package, "root"),
            ("file:src/lib.rs", NodeKind::File, "src/lib.rs"),
            ("sym:main", NodeKind::Function, "main"),
        ] {
            graph
                .add_node(GraphNode {
                    id: id.to_string(),
                    kind,
                    label: label.to_string(),
                    payload: HashMap::new(),
                    centrality: 0.0,
                    memory_id: None,
                    namespace: None,
                    valid_from: None,
                    valid_to: None,
                })
                .unwrap();
        }
        // Contains edges
        for (id, src, dst) in [
            ("e1", "pkg:root", "file:src/lib.rs"),
            ("e2", "file:src/lib.rs", "sym:main"),
        ] {
            graph
                .add_edge(Edge {
                    id: id.to_string(),
                    src: src.to_string(),
                    dst: dst.to_string(),
                    relationship: RelationshipType::Contains,
                    weight: 1.0,
                    properties: HashMap::new(),
                    created_at: now,
                    valid_from: None,
                    valid_to: None,
                })
                .unwrap();
        }
    }

    let tree = engine.summary_tree("pkg:root", 3, false).unwrap();
    assert_eq!(tree.children.len(), 1, "package should have 1 file child");
    assert_eq!(tree.children[0].id, "file:src/lib.rs");
    assert_eq!(
        tree.children[0].children.len(),
        1,
        "file should have 1 symbol child"
    );
    assert_eq!(tree.children[0].children[0].id, "sym:main");
}

#[test]
fn summary_tree_respects_max_depth() {
    let engine = CodememEngine::for_testing();
    let now = chrono::Utc::now();
    {
        let mut graph = engine.lock_graph().unwrap();
        for (id, kind) in [
            ("pkg:a", NodeKind::Package),
            ("file:b", NodeKind::File),
            ("sym:c", NodeKind::Function),
        ] {
            graph
                .add_node(GraphNode {
                    id: id.to_string(),
                    kind,
                    label: id.to_string(),
                    payload: HashMap::new(),
                    centrality: 0.0,
                    memory_id: None,
                    namespace: None,
                    valid_from: None,
                    valid_to: None,
                })
                .unwrap();
        }
        for (id, src, dst) in [("e1", "pkg:a", "file:b"), ("e2", "file:b", "sym:c")] {
            graph
                .add_edge(Edge {
                    id: id.to_string(),
                    src: src.to_string(),
                    dst: dst.to_string(),
                    relationship: RelationshipType::Contains,
                    weight: 1.0,
                    properties: HashMap::new(),
                    created_at: now,
                    valid_from: None,
                    valid_to: None,
                })
                .unwrap();
        }
    }

    // max_depth=1 means we get pkg -> file but file won't expand children
    let tree = engine.summary_tree("pkg:a", 1, false).unwrap();
    assert_eq!(tree.children.len(), 1);
    assert!(
        tree.children[0].children.is_empty(),
        "depth=1 should not recurse into file children"
    );
}

// ── get_symbol / search_symbols ─────────────────────────────────────

#[test]
fn get_symbol_returns_none_for_missing() {
    let engine = CodememEngine::for_testing();
    let sym = engine.get_symbol("nonexistent::symbol").unwrap();
    assert!(sym.is_none());
}

#[test]
fn search_symbols_returns_empty_for_no_match() {
    let engine = CodememEngine::for_testing();
    let results = engine
        .search_symbols("zzz_nonexistent_zzz", 10, None)
        .unwrap();
    assert!(results.is_empty());
}
