use super::*;
use crate::index::scip::{ScipExternal, ScipReadResult, ScipReference, ScipRelationship};

fn make_def(
    scip_symbol: &str,
    qualified_name: &str,
    file_path: &str,
    kind: NodeKind,
    line_start: u32,
    line_end: u32,
) -> ScipDefinition {
    ScipDefinition {
        scip_symbol: scip_symbol.to_string(),
        qualified_name: qualified_name.to_string(),
        file_path: file_path.to_string(),
        line_start,
        line_end,
        col_start: 0,
        col_end: 10,
        kind,
        documentation: vec![],
        relationships: vec![],
        is_test: false,
        is_generated: false,
    }
}

fn make_ref(scip_symbol: &str, file_path: &str, line: u32, role_bitmask: i32) -> ScipReference {
    ScipReference {
        scip_symbol: scip_symbol.to_string(),
        file_path: file_path.to_string(),
        line,
        col_start: 0,
        col_end: 5,
        role_bitmask,
    }
}

#[test]
fn test_build_creates_sym_nodes() {
    let scip = ScipReadResult {
        project_root: String::new(),
        definitions: vec![make_def(
            "rust-analyzer cargo foo 1.0 bar/baz().",
            "bar::baz",
            "src/bar.rs",
            NodeKind::Function,
            10,
            20,
        )],
        references: vec![],
        externals: vec![],
        covered_files: vec!["src/bar.rs".to_string()],
    };

    let result = build_graph(&scip, Some("test-ns"), &ScipConfig::default());
    // Filter for explicit definition nodes (not synthetic intermediates).
    let def_nodes: Vec<_> = result
        .nodes
        .iter()
        .filter(|n| {
            n.id.starts_with("sym:")
                && n.payload.get("source").and_then(|v| v.as_str()) == Some("scip")
        })
        .collect();
    assert_eq!(def_nodes.len(), 1);
    assert_eq!(def_nodes[0].id, "sym:bar::baz");
    assert_eq!(def_nodes[0].kind, NodeKind::Function);
    assert_eq!(def_nodes[0].namespace, Some("test-ns".to_string()));
}

#[test]
fn test_build_creates_flat_contains_edges() {
    let config = ScipConfig {
        hierarchical_containment: false,
        ..ScipConfig::default()
    };

    let scip = ScipReadResult {
        project_root: String::new(),
        definitions: vec![make_def(
            "rust-analyzer cargo foo 1.0 bar/baz().",
            "bar::baz",
            "src/bar.rs",
            NodeKind::Function,
            10,
            20,
        )],
        references: vec![],
        externals: vec![],
        covered_files: vec!["src/bar.rs".to_string()],
    };

    let result = build_graph(&scip, None, &config);
    let contains: Vec<_> = result
        .edges
        .iter()
        .filter(|e| e.relationship == RelationshipType::Contains)
        .collect();
    assert_eq!(contains.len(), 1);
    assert_eq!(contains[0].src, "file:src/bar.rs");
    assert_eq!(contains[0].dst, "sym:bar::baz");
}

#[test]
fn test_hierarchical_containment_chain() {
    let scip = ScipReadResult {
        project_root: String::new(),
        definitions: vec![make_def(
            "rust-analyzer cargo foo 1.0 auth/middleware/validate_token().",
            "auth::middleware::validate_token",
            "src/auth.rs",
            NodeKind::Function,
            10,
            20,
        )],
        references: vec![],
        externals: vec![],
        covered_files: vec!["src/auth.rs".to_string()],
    };

    let result = build_graph(&scip, Some("test-ns"), &ScipConfig::default());

    // Should create: file→sym:auth→sym:auth::middleware→sym:auth::middleware::validate_token
    let contains: Vec<_> = result
        .edges
        .iter()
        .filter(|e| e.relationship == RelationshipType::Contains)
        .collect();
    assert_eq!(contains.len(), 3, "should have 3 CONTAINS edges in chain");

    // Check edge chain
    assert!(contains
        .iter()
        .any(|e| e.src == "file:src/auth.rs" && e.dst == "sym:auth"));
    assert!(contains
        .iter()
        .any(|e| e.src == "sym:auth" && e.dst == "sym:auth::middleware"));
    assert!(contains.iter().any(
        |e| e.src == "sym:auth::middleware" && e.dst == "sym:auth::middleware::validate_token"
    ));

    // Should create synthetic intermediate nodes
    let auth_node = result.nodes.iter().find(|n| n.id == "sym:auth");
    assert!(auth_node.is_some(), "should create synthetic auth node");
    assert_eq!(auth_node.unwrap().kind, NodeKind::Module);
    assert_eq!(
        auth_node
            .unwrap()
            .payload
            .get("source")
            .and_then(|v| v.as_str()),
        Some("scip-synthetic")
    );

    let mw_node = result.nodes.iter().find(|n| n.id == "sym:auth::middleware");
    assert!(mw_node.is_some(), "should create synthetic middleware node");
    assert_eq!(mw_node.unwrap().kind, NodeKind::Module);
}

#[test]
fn test_hierarchical_deduplicates_intermediate_nodes() {
    // Two definitions in the same module should share the intermediate node.
    let scip = ScipReadResult {
        project_root: String::new(),
        definitions: vec![
            make_def(
                "rust-analyzer cargo foo 1.0 auth/login().",
                "auth::login",
                "src/auth.rs",
                NodeKind::Function,
                1,
                10,
            ),
            make_def(
                "rust-analyzer cargo foo 1.0 auth/logout().",
                "auth::logout",
                "src/auth.rs",
                NodeKind::Function,
                20,
                30,
            ),
        ],
        references: vec![],
        externals: vec![],
        covered_files: vec!["src/auth.rs".to_string()],
    };

    let result = build_graph(&scip, None, &ScipConfig::default());

    // Should have exactly one sym:auth node (not duplicated).
    let auth_nodes: Vec<_> = result.nodes.iter().filter(|n| n.id == "sym:auth").collect();
    assert_eq!(auth_nodes.len(), 1, "should deduplicate intermediate nodes");

    // file→auth should appear once, auth→login and auth→logout should each appear once.
    let contains: Vec<_> = result
        .edges
        .iter()
        .filter(|e| e.relationship == RelationshipType::Contains)
        .collect();
    assert_eq!(contains.len(), 3, "file→auth, auth→login, auth→logout");
}

#[test]
fn test_build_creates_pkg_nodes() {
    let scip = ScipReadResult {
        project_root: String::new(),
        definitions: vec![],
        references: vec![],
        externals: vec![
            ScipExternal {
                scip_symbol: "rust-analyzer cargo serde 1.0.0 Serialize#serialize().".to_string(),
                package_manager: "cargo".to_string(),
                package_name: "serde".to_string(),
                package_version: "1.0.0".to_string(),
                kind: NodeKind::Method,
                documentation: vec!["Serialize this value".to_string()],
            },
            ScipExternal {
                scip_symbol: "rust-analyzer cargo serde 1.0.0 Deserialize#deserialize()."
                    .to_string(),
                package_manager: "cargo".to_string(),
                package_name: "serde".to_string(),
                package_version: "1.0.0".to_string(),
                kind: NodeKind::Method,
                documentation: vec![],
            },
        ],
        covered_files: vec![],
    };

    let result = build_graph(&scip, None, &ScipConfig::default());
    // Two symbols from the same package should produce one pkg: node.
    assert_eq!(result.ext_nodes_created, 1);
    assert_eq!(result.nodes.len(), 1);
    assert_eq!(result.nodes[0].kind, NodeKind::External);
    assert_eq!(result.nodes[0].id, "pkg:cargo:serde");
    assert_eq!(result.nodes[0].label, "serde");
    // Package-level nodes don't create per-symbol doc memories.
    assert_eq!(result.doc_memories_created, 0);
}

#[test]
fn test_build_creates_call_edges_from_refs() {
    let scip = ScipReadResult {
        project_root: String::new(),
        definitions: vec![
            make_def(
                "rust-analyzer cargo foo 1.0 bar/caller().",
                "bar::caller",
                "src/bar.rs",
                NodeKind::Function,
                1,
                50,
            ),
            make_def(
                "rust-analyzer cargo foo 1.0 bar/callee().",
                "bar::callee",
                "src/bar.rs",
                NodeKind::Function,
                60,
                80,
            ),
        ],
        references: vec![make_ref(
            "rust-analyzer cargo foo 1.0 bar/callee().",
            "src/bar.rs",
            25, // Inside caller's range
            0,  // No special role flags = generic reference → CALLS
        )],
        externals: vec![],
        covered_files: vec!["src/bar.rs".to_string()],
    };

    let result = build_graph(&scip, None, &ScipConfig::default());
    let calls: Vec<_> = result
        .edges
        .iter()
        .filter(|e| e.relationship == RelationshipType::Calls)
        .collect();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].src, "sym:bar::caller");
    assert_eq!(calls[0].dst, "sym:bar::callee");
}

#[test]
fn test_build_skips_high_fanout_refs() {
    let mut refs = Vec::new();
    // Create 101 references from the same symbol in the same file.
    for i in 0..101 {
        refs.push(make_ref(
            "rust-analyzer cargo foo 1.0 bar/utility().",
            "src/bar.rs",
            i,
            0,
        ));
    }

    let scip = ScipReadResult {
        project_root: String::new(),
        definitions: vec![
            make_def(
                "rust-analyzer cargo foo 1.0 bar/caller().",
                "bar::caller",
                "src/bar.rs",
                NodeKind::Function,
                0,
                200,
            ),
            make_def(
                "rust-analyzer cargo foo 1.0 bar/utility().",
                "bar::utility",
                "src/bar.rs",
                NodeKind::Function,
                300,
                310,
            ),
        ],
        references: refs,
        externals: vec![],
        covered_files: vec!["src/bar.rs".to_string()],
    };

    let result = build_graph(&scip, None, &ScipConfig::default());
    let calls: Vec<_> = result
        .edges
        .iter()
        .filter(|e| e.relationship == RelationshipType::Calls)
        .collect();
    // All 101 refs should be skipped (> 100 threshold).
    assert_eq!(calls.len(), 0);
}

#[test]
fn test_build_import_edges() {
    let scip = ScipReadResult {
        project_root: String::new(),
        definitions: vec![
            make_def(
                "rust-analyzer cargo foo 1.0 bar/func().",
                "bar::func",
                "src/bar.rs",
                NodeKind::Function,
                1,
                50,
            ),
            make_def(
                "rust-analyzer cargo foo 1.0 baz/helper().",
                "baz::helper",
                "src/baz.rs",
                NodeKind::Function,
                1,
                30,
            ),
        ],
        references: vec![make_ref(
            "rust-analyzer cargo foo 1.0 baz/helper().",
            "src/bar.rs",
            5,
            0x2, // IMPORT role
        )],
        externals: vec![],
        covered_files: vec!["src/bar.rs".to_string(), "src/baz.rs".to_string()],
    };

    let result = build_graph(&scip, None, &ScipConfig::default());
    let imports: Vec<_> = result
        .edges
        .iter()
        .filter(|e| e.relationship == RelationshipType::Imports)
        .collect();
    assert_eq!(imports.len(), 1);
    assert_eq!(imports[0].dst, "sym:baz::helper");
}

#[test]
fn test_build_implements_edges() {
    let mut def = make_def(
        "rust-analyzer cargo foo 1.0 MyStruct#.",
        "MyStruct",
        "src/lib.rs",
        NodeKind::Class,
        1,
        50,
    );
    def.relationships.push(ScipRelationship {
        target_symbol: "rust-analyzer cargo foo 1.0 MyTrait#.".to_string(),
        is_implementation: true,
        is_type_definition: false,
        is_reference: false,
        is_definition: false,
    });

    let target_def = make_def(
        "rust-analyzer cargo foo 1.0 MyTrait#.",
        "MyTrait",
        "src/lib.rs",
        NodeKind::Trait,
        60,
        80,
    );

    let scip = ScipReadResult {
        project_root: String::new(),
        definitions: vec![def, target_def],
        references: vec![],
        externals: vec![],
        covered_files: vec!["src/lib.rs".to_string()],
    };

    let result = build_graph(&scip, None, &ScipConfig::default());
    let impls: Vec<_> = result
        .edges
        .iter()
        .filter(|e| e.relationship == RelationshipType::Implements)
        .collect();
    assert_eq!(impls.len(), 1);
    assert_eq!(impls[0].src, "sym:MyStruct");
    assert_eq!(impls[0].dst, "sym:MyTrait");
}

#[test]
fn test_build_doc_memories() {
    let mut def = make_def(
        "rust-analyzer cargo foo 1.0 bar/baz().",
        "bar::baz",
        "src/bar.rs",
        NodeKind::Function,
        10,
        20,
    );
    def.documentation = vec![
        "fn baz() -> Result<(), Error>".to_string(),
        "Does something useful.".to_string(),
    ];

    let scip = ScipReadResult {
        project_root: String::new(),
        definitions: vec![def],
        references: vec![],
        externals: vec![],
        covered_files: vec!["src/bar.rs".to_string()],
    };

    let result = build_graph(&scip, None, &ScipConfig::default());
    assert_eq!(result.doc_memories_created, 1);
    let (mem, related_node) = &result.memories[0];
    assert_eq!(related_node, "sym:bar::baz");
    assert!(mem.content.contains("fn baz()"));
    assert_eq!(mem.memory_type, MemoryType::Context);
    assert!(mem.tags.contains(&"scip-doc".to_string()));
    assert!(mem.tags.contains(&"auto-generated".to_string()));
}

#[test]
fn test_build_test_nodes() {
    let mut def = make_def(
        "rust-analyzer cargo foo 1.0 tests/test_bar().",
        "tests::test_bar",
        "src/tests.rs",
        NodeKind::Function,
        1,
        10,
    );
    def.is_test = true;

    let scip = ScipReadResult {
        project_root: String::new(),
        definitions: vec![def],
        references: vec![],
        externals: vec![],
        covered_files: vec!["src/tests.rs".to_string()],
    };

    let result = build_graph(&scip, None, &ScipConfig::default());
    assert_eq!(result.nodes[0].kind, NodeKind::Test);
}

#[test]
fn test_scip_edge_properties() {
    let props = scip_edge_properties();
    assert_eq!(props.get("source").unwrap(), "scip");
    assert_eq!(props.get("confidence").unwrap(), &serde_json::json!(0.15));
}

#[test]
fn test_edge_deduplication() {
    // Two references from the same source to the same target on different lines
    // but with the same role should produce unique edge IDs.
    let scip = ScipReadResult {
        project_root: String::new(),
        definitions: vec![
            make_def(
                "rust-analyzer cargo foo 1.0 bar/caller().",
                "bar::caller",
                "src/bar.rs",
                NodeKind::Function,
                1,
                100,
            ),
            make_def(
                "rust-analyzer cargo foo 1.0 bar/callee().",
                "bar::callee",
                "src/bar.rs",
                NodeKind::Function,
                110,
                120,
            ),
        ],
        references: vec![
            make_ref(
                "rust-analyzer cargo foo 1.0 bar/callee().",
                "src/bar.rs",
                10,
                0,
            ),
            make_ref(
                "rust-analyzer cargo foo 1.0 bar/callee().",
                "src/bar.rs",
                20,
                0,
            ),
        ],
        externals: vec![],
        covered_files: vec!["src/bar.rs".to_string()],
    };

    let result = build_graph(&scip, None, &ScipConfig::default());
    let calls: Vec<_> = result
        .edges
        .iter()
        .filter(|e| e.relationship == RelationshipType::Calls)
        .collect();
    // Each reference on a different line gets a unique edge ID.
    assert_eq!(calls.len(), 2);
}

#[test]
fn test_read_only_role_treated_as_calls() {
    // scip-go sets all references to READ_ACCESS (0x8) only.
    // When no IMPORT or WRITE flags are set, this should produce CALLS, not READS.
    let scip = ScipReadResult {
        project_root: String::new(),
        definitions: vec![
            make_def(
                "scip-go gomod foo 1.0 bar/caller().",
                "bar.caller",
                "src/bar.go",
                NodeKind::Function,
                1,
                50,
            ),
            make_def(
                "scip-go gomod foo 1.0 bar/callee().",
                "bar.callee",
                "src/bar.go",
                NodeKind::Function,
                60,
                80,
            ),
        ],
        references: vec![make_ref(
            "scip-go gomod foo 1.0 bar/callee().",
            "src/bar.go",
            25,
            0x8, // READ_ACCESS only — should become CALLS
        )],
        externals: vec![],
        covered_files: vec!["src/bar.go".to_string()],
    };

    let result = build_graph(&scip, None, &ScipConfig::default());
    let calls: Vec<_> = result
        .edges
        .iter()
        .filter(|e| e.relationship == RelationshipType::Calls)
        .collect();
    let reads: Vec<_> = result
        .edges
        .iter()
        .filter(|e| e.relationship == RelationshipType::Reads)
        .collect();
    assert_eq!(calls.len(), 1, "READ_ACCESS-only should become CALLS");
    assert_eq!(reads.len(), 0, "should not create READS edges");
}

#[test]
fn test_read_with_import_stays_import() {
    // When IMPORT flag is set alongside READ_ACCESS, should stay IMPORTS.
    let scip = ScipReadResult {
        project_root: String::new(),
        definitions: vec![
            make_def(
                "rust-analyzer cargo foo 1.0 bar/func().",
                "bar::func",
                "src/bar.rs",
                NodeKind::Function,
                1,
                50,
            ),
            make_def(
                "rust-analyzer cargo foo 1.0 baz/helper().",
                "baz::helper",
                "src/baz.rs",
                NodeKind::Function,
                1,
                30,
            ),
        ],
        references: vec![make_ref(
            "rust-analyzer cargo foo 1.0 baz/helper().",
            "src/bar.rs",
            5,
            0x2 | 0x8, // IMPORT + READ_ACCESS
        )],
        externals: vec![],
        covered_files: vec!["src/bar.rs".to_string(), "src/baz.rs".to_string()],
    };

    let result = build_graph(&scip, None, &ScipConfig::default());
    let imports: Vec<_> = result
        .edges
        .iter()
        .filter(|e| e.relationship == RelationshipType::Imports)
        .collect();
    assert_eq!(imports.len(), 1, "IMPORT+READ should stay IMPORTS");
}

#[test]
fn test_stub_pkg_nodes_created_for_missing_targets() {
    // Reference to an external symbol not listed in scip.externals.
    // The graph builder should create a stub pkg: node so the edge doesn't fail FK.
    let scip = ScipReadResult {
        project_root: String::new(),
        definitions: vec![make_def(
            "scip-go gomod foo 1.0 bar/caller().",
            "bar.caller",
            "src/bar.go",
            NodeKind::Function,
            1,
            50,
        )],
        references: vec![make_ref(
            "scip-go gomod golang.org/x/net 0.1.0 http/Get().",
            "src/bar.go",
            10,
            0, // generic reference → CALLS
        )],
        externals: vec![], // No externals listed
        covered_files: vec!["src/bar.go".to_string()],
    };

    let result = build_graph(&scip, None, &ScipConfig::default());

    // Should have created a stub pkg: node for the target package
    let pkg_nodes: Vec<_> = result
        .nodes
        .iter()
        .filter(|n| n.id.starts_with("pkg:"))
        .collect();
    assert_eq!(pkg_nodes.len(), 1, "should create stub pkg: node");
    assert_eq!(pkg_nodes[0].kind, NodeKind::External);
    assert_eq!(pkg_nodes[0].id, "pkg:gomod:golang.org/x/net");
    assert_eq!(
        pkg_nodes[0].payload.get("source").and_then(|v| v.as_str()),
        Some("scip")
    );

    // The CALLS edge should reference this pkg: node
    let calls: Vec<_> = result
        .edges
        .iter()
        .filter(|e| e.relationship == RelationshipType::Calls)
        .collect();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].dst, "pkg:gomod:golang.org/x/net");
}

#[test]
fn test_wildcard_modules_filtered() {
    // TypeScript `declare module '*.css'` should be filtered out.
    let scip = ScipReadResult {
        project_root: String::new(),
        definitions: vec![
            make_def(
                "scip-typescript npm foo 1.0 src/types/global.d.ts/'*.css'.",
                "src.types.global.d.ts.'*.css'",
                "src/types/global.d.ts",
                NodeKind::Module,
                86,
                86,
            ),
            make_def(
                "scip-typescript npm foo 1.0 src/App#.",
                "src.App",
                "src/App.tsx",
                NodeKind::Class,
                1,
                100,
            ),
        ],
        references: vec![],
        externals: vec![],
        covered_files: vec![
            "src/types/global.d.ts".to_string(),
            "src/App.tsx".to_string(),
        ],
    };

    let result = build_graph(&scip, None, &ScipConfig::default());
    let sym_nodes: Vec<_> = result
        .nodes
        .iter()
        .filter(|n| n.id.starts_with("sym:"))
        .collect();
    assert_eq!(sym_nodes.len(), 1, "wildcard module should be filtered");
    assert_eq!(sym_nodes[0].id, "sym:src.App");
}

#[test]
fn test_intra_class_edge_collapsing() {
    // Two methods in the same class calling each other should be collapsed into
    // parent metadata, not produce CALLS edges in the graph.
    let scip = ScipReadResult {
        project_root: String::new(),
        definitions: vec![
            make_def(
                "rust-analyzer cargo foo 1.0 MyClass#method_a().",
                "MyClass::method_a",
                "src/lib.rs",
                NodeKind::Method,
                10,
                30,
            ),
            make_def(
                "rust-analyzer cargo foo 1.0 MyClass#method_b().",
                "MyClass::method_b",
                "src/lib.rs",
                NodeKind::Method,
                40,
                60,
            ),
            make_def(
                "rust-analyzer cargo foo 1.0 MyClass#.",
                "MyClass",
                "src/lib.rs",
                NodeKind::Class,
                1,
                100,
            ),
        ],
        references: vec![
            // method_a calls method_b
            make_ref(
                "rust-analyzer cargo foo 1.0 MyClass#method_b().",
                "src/lib.rs",
                20, // inside method_a
                0,
            ),
            // method_b calls method_a
            make_ref(
                "rust-analyzer cargo foo 1.0 MyClass#method_a().",
                "src/lib.rs",
                50, // inside method_b
                0,
            ),
        ],
        externals: vec![],
        covered_files: vec!["src/lib.rs".to_string()],
    };

    let config = ScipConfig {
        collapse_intra_class_edges: true,
        ..ScipConfig::default()
    };
    let result = build_graph(&scip, None, &config);

    // Intra-class CALLS edges should be removed from the edge set.
    let calls: Vec<_> = result
        .edges
        .iter()
        .filter(|e| e.relationship == RelationshipType::Calls)
        .collect();
    assert_eq!(calls.len(), 0, "intra-class calls should be collapsed");

    // Parent class node should have intra_class_calls metadata.
    let class_node = result
        .nodes
        .iter()
        .find(|n| n.id == "sym:MyClass")
        .expect("class node should exist");
    let intra_calls = class_node
        .payload
        .get("intra_class_calls")
        .expect("should have intra_class_calls metadata");
    let entries = intra_calls.as_array().expect("should be array");
    assert_eq!(entries.len(), 2, "should have 2 intra-class call pairs");
}

#[test]
fn test_blocklist_filters_builtin_calls() {
    // A reference to a blocked builtin (e.g., `clone` in Rust via scip-cargo)
    // should be filtered out, while a reference to a user-defined symbol should
    // pass through.
    let scip = ScipReadResult {
        project_root: String::new(),
        definitions: vec![
            make_def(
                "rust-analyzer cargo foo 1.0 bar/caller().",
                "bar::caller",
                "src/bar.rs",
                NodeKind::Function,
                1,
                50,
            ),
            make_def(
                "rust-analyzer cargo foo 1.0 bar/callee().",
                "bar::callee",
                "src/bar.rs",
                NodeKind::Function,
                60,
                80,
            ),
        ],
        references: vec![
            // Reference to a user-defined symbol → should produce a CALLS edge.
            make_ref(
                "rust-analyzer cargo foo 1.0 bar/callee().",
                "src/bar.rs",
                10,
                0,
            ),
            // Reference to a blocked builtin (clone) via scip-cargo format →
            // should be filtered out by the blocklist.
            make_ref(
                "scip-cargo std 1.0 core/Clone#clone().",
                "src/bar.rs",
                20,
                0,
            ),
        ],
        externals: vec![],
        covered_files: vec!["src/bar.rs".to_string()],
    };

    let result = build_graph(&scip, None, &ScipConfig::default());
    let calls: Vec<_> = result
        .edges
        .iter()
        .filter(|e| e.relationship == RelationshipType::Calls)
        .collect();
    // Only the user-defined call should survive; the blocked builtin should be filtered.
    assert_eq!(calls.len(), 1, "blocked builtin call should be filtered");
    assert_eq!(
        calls[0].dst, "sym:bar::callee",
        "only the user-defined callee should have an edge"
    );
}

#[test]
fn test_intra_module_edges_not_collapsed() {
    // Two functions in the same module calling each other should NOT be collapsed.
    let scip = ScipReadResult {
        project_root: String::new(),
        definitions: vec![
            make_def(
                "rust-analyzer cargo foo 1.0 mymod/func_a().",
                "mymod::func_a",
                "src/lib.rs",
                NodeKind::Function,
                10,
                30,
            ),
            make_def(
                "rust-analyzer cargo foo 1.0 mymod/func_b().",
                "mymod::func_b",
                "src/lib.rs",
                NodeKind::Function,
                40,
                60,
            ),
        ],
        references: vec![make_ref(
            "rust-analyzer cargo foo 1.0 mymod/func_b().",
            "src/lib.rs",
            20,
            0,
        )],
        externals: vec![],
        covered_files: vec!["src/lib.rs".to_string()],
    };

    let config = ScipConfig {
        collapse_intra_class_edges: true,
        ..ScipConfig::default()
    };
    let result = build_graph(&scip, None, &config);

    // Module children calling each other should produce normal CALLS edges.
    let calls: Vec<_> = result
        .edges
        .iter()
        .filter(|e| e.relationship == RelationshipType::Calls)
        .collect();
    assert_eq!(
        calls.len(),
        1,
        "inter-function calls in a module should NOT be collapsed"
    );
}
