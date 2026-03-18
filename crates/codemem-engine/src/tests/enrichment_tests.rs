use crate::enrichment::resolve_path;
use std::path::{Path, PathBuf};

#[test]
fn resolve_path_with_root_joins() {
    let result = resolve_path("src/main.rs", Some(Path::new("/home/user/project")));
    assert_eq!(result, PathBuf::from("/home/user/project/src/main.rs"));
}

#[test]
fn resolve_path_without_root_returns_as_is() {
    let result = resolve_path("src/main.rs", None);
    assert_eq!(result, PathBuf::from("src/main.rs"));
}

#[test]
fn resolve_path_nested_relative() {
    let result = resolve_path("a/b/c.rs", Some(Path::new("/root")));
    assert_eq!(result, PathBuf::from("/root/a/b/c.rs"));
}

#[test]
fn resolve_path_root_only_file() {
    let result = resolve_path("lib.rs", Some(Path::new("/project")));
    assert_eq!(result, PathBuf::from("/project/lib.rs"));
}

// ── Helpers ──────────────────────────────────────────────────────────

use crate::CodememEngine;
use codemem_core::{Edge, GraphNode, NodeKind, RelationshipType};
use serde_json::json;
use std::collections::HashMap;

/// Build a File graph node with the given label (path).
fn file_node(path: &str) -> GraphNode {
    GraphNode {
        id: format!("file:{path}"),
        kind: NodeKind::File,
        label: path.to_string(),
        payload: HashMap::new(),
        centrality: 0.0,
        memory_id: None,
        namespace: None,
        valid_from: None,
        valid_to: None,
    }
}

/// Build a Function graph node.
fn function_node(id: &str, label: &str, file_path: &str) -> GraphNode {
    let mut payload = HashMap::new();
    payload.insert("file_path".into(), json!(file_path));
    GraphNode {
        id: id.to_string(),
        kind: NodeKind::Function,
        label: label.to_string(),
        payload,
        centrality: 0.0,
        memory_id: None,
        namespace: None,
        valid_from: None,
        valid_to: None,
    }
}

/// Build a Method graph node with visibility and optional doc_comment.
fn method_node(
    id: &str,
    label: &str,
    file_path: &str,
    visibility: &str,
    doc_comment: Option<&str>,
) -> GraphNode {
    let mut payload = HashMap::new();
    payload.insert("file_path".into(), json!(file_path));
    payload.insert("visibility".into(), json!(visibility));
    if let Some(doc) = doc_comment {
        payload.insert("doc_comment".into(), json!(doc));
    }
    GraphNode {
        id: id.to_string(),
        kind: NodeKind::Method,
        label: label.to_string(),
        payload,
        centrality: 0.0,
        memory_id: None,
        namespace: None,
        valid_from: None,
        valid_to: None,
    }
}

/// Build an Endpoint graph node.
fn endpoint_node(id: &str, label: &str) -> GraphNode {
    GraphNode {
        id: id.to_string(),
        kind: NodeKind::Endpoint,
        label: label.to_string(),
        payload: HashMap::new(),
        centrality: 0.0,
        memory_id: None,
        namespace: None,
        valid_from: None,
        valid_to: None,
    }
}

/// Build a Package graph node.
fn package_node(id: &str, label: &str) -> GraphNode {
    GraphNode {
        id: id.to_string(),
        kind: NodeKind::Package,
        label: label.to_string(),
        payload: HashMap::new(),
        centrality: 0.0,
        memory_id: None,
        namespace: None,
        valid_from: None,
        valid_to: None,
    }
}

/// Build a Test graph node.
fn test_node(id: &str, label: &str, file_path: &str) -> GraphNode {
    let mut payload = HashMap::new();
    payload.insert("file_path".into(), json!(file_path));
    GraphNode {
        id: id.to_string(),
        kind: NodeKind::Test,
        label: label.to_string(),
        payload,
        centrality: 0.0,
        memory_id: None,
        namespace: None,
        valid_from: None,
        valid_to: None,
    }
}

/// Build an edge between two nodes.
fn make_edge(src: &str, dst: &str, rel: RelationshipType) -> Edge {
    Edge {
        id: format!("{src}-{:?}-{dst}", rel),
        src: src.to_string(),
        dst: dst.to_string(),
        relationship: rel,
        weight: 1.0,
        properties: HashMap::new(),
        created_at: chrono::Utc::now(),
        valid_from: None,
        valid_to: None,
    }
}

// ── store_insight tests ──────────────────────────────────────────────

#[test]
fn store_insight_returns_id_on_success() {
    let engine = CodememEngine::for_testing();
    let id = engine.store_insight(
        "SQL injection risk in user input handler",
        "security",
        &["severity:high"],
        0.8,
        None,
        &[],
    );
    assert!(id.is_some(), "store_insight should return an ID");
}

#[test]
fn store_insight_creates_memory_in_storage() {
    let engine = CodememEngine::for_testing();
    let id = engine
        .store_insight(
            "Architecture uses layered design",
            "architecture",
            &[],
            0.7,
            None,
            &[],
        )
        .unwrap();

    let memory = engine.storage.get_memory(&id).unwrap().unwrap();
    assert_eq!(memory.content, "Architecture uses layered design");
    assert_eq!(memory.memory_type, codemem_core::MemoryType::Insight);
    assert!(memory.tags.contains(&"track:architecture".to_string()));
    assert!(memory.tags.contains(&"static-analysis".to_string()));
}

#[test]
fn store_insight_tags_include_track_and_static_analysis() {
    let engine = CodememEngine::for_testing();
    let id = engine
        .store_insight(
            "test content unique abc",
            "testing",
            &["custom-tag"],
            0.5,
            None,
            &[],
        )
        .unwrap();

    let memory = engine.storage.get_memory(&id).unwrap().unwrap();
    assert!(memory.tags.contains(&"track:testing".to_string()));
    assert!(memory.tags.contains(&"static-analysis".to_string()));
    assert!(memory.tags.contains(&"custom-tag".to_string()));
}

#[test]
fn store_insight_importance_is_clamped() {
    let engine = CodememEngine::for_testing();
    let id = engine
        .store_insight("over limit value xyz", "test", &[], 1.5, None, &[])
        .unwrap();
    let memory = engine.storage.get_memory(&id).unwrap().unwrap();
    assert!(
        memory.importance <= 1.0,
        "importance should be clamped to 1.0"
    );
}

#[test]
fn store_insight_creates_relates_to_edges_for_links() {
    let engine = CodememEngine::for_testing();

    // Add a node to link to
    {
        let mut graph = engine.lock_graph().unwrap();
        graph.add_node(file_node("src/auth.rs")).unwrap();
    }

    let id = engine
        .store_insight(
            "Auth module is sensitive xyz",
            "security",
            &[],
            0.7,
            None,
            &["file:src/auth.rs".to_string()],
        )
        .unwrap();

    let graph = engine.lock_graph().unwrap();
    let edges = graph.get_edges(&id).unwrap();
    let relates_to: Vec<_> = edges
        .iter()
        .filter(|e| e.relationship == RelationshipType::RelatesTo)
        .collect();
    assert!(
        !relates_to.is_empty(),
        "should have RELATES_TO edge to linked node"
    );
}

#[test]
fn store_insight_metadata_includes_generated_by() {
    let engine = CodememEngine::for_testing();
    let id = engine
        .store_insight(
            "Architecture note about module boundaries xyz",
            "architecture",
            &[],
            0.6,
            None,
            &[],
        )
        .unwrap();

    let memory = engine.storage.get_memory(&id).unwrap().unwrap();
    assert_eq!(
        memory.metadata.get("generated_by").and_then(|v| v.as_str()),
        Some("enrichment_pipeline"),
        "metadata should include generated_by field"
    );
    assert_eq!(
        memory.metadata.get("track").and_then(|v| v.as_str()),
        Some("architecture"),
        "metadata should include track field"
    );
}

#[test]
fn store_insight_with_namespace() {
    let engine = CodememEngine::for_testing();
    let id = engine
        .store_insight(
            "namespaced insight content abc",
            "test",
            &[],
            0.5,
            Some("my-project"),
            &[],
        )
        .unwrap();

    let memory = engine.storage.get_memory(&id).unwrap().unwrap();
    assert_eq!(memory.namespace.as_deref(), Some("my-project"));
}

// ── store_pattern_memory tests ───────────────────────────────────────

#[test]
fn store_pattern_memory_returns_id() {
    let engine = CodememEngine::for_testing();
    let id = engine.store_pattern_memory("Code smell: long function detected xyz", None, &[]);
    assert!(id.is_some(), "store_pattern_memory should return an ID");
}

#[test]
fn store_pattern_memory_creates_pattern_type() {
    let engine = CodememEngine::for_testing();
    let id = engine
        .store_pattern_memory("Code smell: too many params abc", None, &[])
        .unwrap();

    let memory = engine.storage.get_memory(&id).unwrap().unwrap();
    assert_eq!(memory.memory_type, codemem_core::MemoryType::Pattern);
    assert_eq!(memory.importance, 0.5);
    assert!(memory.tags.contains(&"static-analysis".to_string()));
    assert!(memory.tags.contains(&"track:code-smell".to_string()));
}

#[test]
fn store_pattern_memory_creates_graph_node() {
    let engine = CodememEngine::for_testing();
    let id = engine
        .store_pattern_memory("Code smell: deep nesting xyz", None, &[])
        .unwrap();

    let graph = engine.lock_graph().unwrap();
    let node = graph.get_node(&id).unwrap();
    assert!(node.is_some(), "should create a Memory graph node");
    let node = node.unwrap();
    assert_eq!(node.kind, NodeKind::Memory);
    assert_eq!(node.memory_id.as_deref(), Some(id.as_str()));
}

#[test]
fn store_pattern_memory_creates_link_edges() {
    let engine = CodememEngine::for_testing();

    {
        let mut graph = engine.lock_graph().unwrap();
        graph
            .add_node(function_node("fn:process", "process", "src/lib.rs"))
            .unwrap();
    }

    let id = engine
        .store_pattern_memory(
            "Code smell: long function process xyz",
            None,
            &["fn:process".to_string()],
        )
        .unwrap();

    let graph = engine.lock_graph().unwrap();
    let edges = graph.get_edges(&id).unwrap();
    let link_edges: Vec<_> = edges
        .iter()
        .filter(|e| e.relationship == RelationshipType::RelatesTo && e.dst == "fn:process")
        .collect();
    assert!(
        !link_edges.is_empty(),
        "should have RELATES_TO edge to linked fn node"
    );
}

#[test]
fn store_pattern_memory_indexes_in_bm25() {
    let engine = CodememEngine::for_testing();
    let id = engine
        .store_pattern_memory("Code smell: extremely deep nesting detected xyz", None, &[])
        .unwrap();

    let bm25 = engine.lock_bm25().unwrap();
    let score = bm25.score("deep nesting", &id);
    assert!(
        score > 0.0,
        "BM25 should score the pattern memory positively for matching terms"
    );
}

// ── enrich_security tests ────────────────────────────────────────────

#[test]
fn enrich_security_empty_graph_returns_zero_insights() {
    let engine = CodememEngine::for_testing();
    let result = engine.enrich_security(None).unwrap();
    assert_eq!(result.insights_stored, 0);
}

#[test]
fn enrich_security_detects_sensitive_files() {
    let engine = CodememEngine::for_testing();
    {
        let mut graph = engine.lock_graph().unwrap();
        graph.add_node(file_node("src/auth.rs")).unwrap();
        graph.add_node(file_node("src/utils.rs")).unwrap();
    }

    let result = engine.enrich_security(None).unwrap();
    assert!(
        result.insights_stored >= 1,
        "should detect auth.rs as sensitive"
    );
    let details = &result.details;
    assert_eq!(details["sensitive_file_count"], 1);
}

#[test]
fn enrich_security_detects_endpoints() {
    let engine = CodememEngine::for_testing();
    {
        let mut graph = engine.lock_graph().unwrap();
        graph
            .add_node(endpoint_node("ep:login", "POST /api/login"))
            .unwrap();
        graph
            .add_node(endpoint_node("ep:health", "GET /api/health"))
            .unwrap();
    }

    let result = engine.enrich_security(None).unwrap();
    assert_eq!(result.details["endpoint_count"], 2);
    assert!(result.insights_stored >= 1);
}

#[test]
fn enrich_security_detects_security_functions() {
    let engine = CodememEngine::for_testing();
    {
        let mut graph = engine.lock_graph().unwrap();
        graph
            .add_node(function_node(
                "fn:verify_password",
                "verify_password",
                "src/auth.rs",
            ))
            .unwrap();
        graph
            .add_node(function_node("fn:process", "process", "src/main.rs"))
            .unwrap();
    }

    let result = engine.enrich_security(None).unwrap();
    assert_eq!(result.details["security_function_count"], 1);
}

#[test]
fn enrich_security_annotates_nodes_with_security_flags() {
    let engine = CodememEngine::for_testing();
    {
        let mut graph = engine.lock_graph().unwrap();
        graph.add_node(file_node("src/credentials.rs")).unwrap();
    }

    engine.enrich_security(None).unwrap();

    let graph = engine.lock_graph().unwrap();
    let node = graph.get_node("file:src/credentials.rs").unwrap().unwrap();
    assert!(
        node.payload.contains_key("security_flags"),
        "sensitive file should be annotated with security_flags"
    );
}

#[test]
fn enrich_security_matches_multiple_patterns() {
    let engine = CodememEngine::for_testing();
    {
        let mut graph = engine.lock_graph().unwrap();
        graph.add_node(file_node("config/.env")).unwrap();
        graph.add_node(file_node("src/jwt_handler.rs")).unwrap();
        graph.add_node(file_node("src/token_validator.rs")).unwrap();
    }

    let result = engine.enrich_security(None).unwrap();
    assert_eq!(
        result.details["sensitive_file_count"], 3,
        ".env, jwt, and token files should all match"
    );
}

// ── enrich_performance tests ─────────────────────────────────────────

#[test]
fn enrich_performance_empty_graph_returns_zero_insights() {
    let engine = CodememEngine::for_testing();
    let result = engine.enrich_performance(10, None).unwrap();
    assert_eq!(result.insights_stored, 0);
}

#[test]
fn enrich_performance_annotates_coupling_scores() {
    let engine = CodememEngine::for_testing();
    {
        let mut graph = engine.lock_graph().unwrap();
        graph.add_node(file_node("src/a.rs")).unwrap();
        graph.add_node(file_node("src/b.rs")).unwrap();
        let edge = make_edge("file:src/a.rs", "file:src/b.rs", RelationshipType::Imports);
        graph.add_edge(edge).unwrap();
    }

    engine.enrich_performance(10, None).unwrap();

    let graph = engine.lock_graph().unwrap();
    let node_a = graph.get_node("file:src/a.rs").unwrap().unwrap();
    assert!(
        node_a.payload.contains_key("coupling_score"),
        "nodes should be annotated with coupling_score"
    );
}

#[test]
fn enrich_performance_computes_dependency_layers() {
    let engine = CodememEngine::for_testing();
    {
        let mut graph = engine.lock_graph().unwrap();
        graph.add_node(file_node("src/a.rs")).unwrap();
        graph.add_node(file_node("src/b.rs")).unwrap();
        graph.add_node(file_node("src/c.rs")).unwrap();
        // a -> b -> c chain
        graph
            .add_edge(make_edge(
                "file:src/a.rs",
                "file:src/b.rs",
                RelationshipType::DependsOn,
            ))
            .unwrap();
        graph
            .add_edge(make_edge(
                "file:src/b.rs",
                "file:src/c.rs",
                RelationshipType::DependsOn,
            ))
            .unwrap();
    }

    let result = engine.enrich_performance(10, None).unwrap();
    let max_depth = result.details["max_depth"].as_u64().unwrap();
    assert!(max_depth >= 2, "should detect dependency depth >= 2");
}

#[test]
fn enrich_performance_counts_symbols_per_file() {
    let engine = CodememEngine::for_testing();
    {
        let mut graph = engine.lock_graph().unwrap();
        graph.add_node(file_node("src/big.rs")).unwrap();
        for i in 0..5 {
            graph
                .add_node(function_node(
                    &format!("fn:func_{i}"),
                    &format!("func_{i}"),
                    "src/big.rs",
                ))
                .unwrap();
        }
    }

    engine.enrich_performance(10, None).unwrap();

    let graph = engine.lock_graph().unwrap();
    let node = graph.get_node("file:src/big.rs").unwrap().unwrap();
    let sym_count = node.payload.get("symbol_count").and_then(|v| v.as_u64());
    assert_eq!(sym_count, Some(5), "should count 5 symbols in the file");
}

// ── enrich_architecture tests ────────────────────────────────────────

#[test]
fn enrich_architecture_empty_graph_returns_zero_insights() {
    let engine = CodememEngine::for_testing();
    let result = engine.enrich_architecture(None).unwrap();
    assert_eq!(result.insights_stored, 0);
}

#[test]
fn enrich_architecture_detects_layered_deps() {
    let engine = CodememEngine::for_testing();
    {
        let mut graph = engine.lock_graph().unwrap();
        graph.add_node(file_node("api/routes.rs")).unwrap();
        graph.add_node(file_node("service/logic.rs")).unwrap();
        graph.add_node(file_node("storage/db.rs")).unwrap();

        // api -> service -> storage (layered)
        graph
            .add_edge(make_edge(
                "file:api/routes.rs",
                "file:service/logic.rs",
                RelationshipType::Imports,
            ))
            .unwrap();
        graph
            .add_edge(make_edge(
                "file:service/logic.rs",
                "file:storage/db.rs",
                RelationshipType::Imports,
            ))
            .unwrap();
    }

    let result = engine.enrich_architecture(None).unwrap();
    assert!(
        result.insights_stored >= 1,
        "should produce architecture insight for layered deps"
    );
    assert_eq!(
        result.details["module_count"], 3,
        "should detect 3 module groups"
    );
}

#[test]
fn enrich_architecture_detects_circular_deps() {
    let engine = CodememEngine::for_testing();
    {
        let mut graph = engine.lock_graph().unwrap();
        graph.add_node(file_node("alpha/mod.rs")).unwrap();
        graph.add_node(file_node("beta/mod.rs")).unwrap();

        // alpha -> beta AND beta -> alpha (circular)
        graph
            .add_edge(make_edge(
                "file:alpha/mod.rs",
                "file:beta/mod.rs",
                RelationshipType::Imports,
            ))
            .unwrap();
        graph
            .add_edge(make_edge(
                "file:beta/mod.rs",
                "file:alpha/mod.rs",
                RelationshipType::Imports,
            ))
            .unwrap();
    }

    let result = engine.enrich_architecture(None).unwrap();
    // Should detect both layered deps AND circular dep
    assert!(
        result.insights_stored >= 2,
        "should produce insights for layers + circular dep; got {}",
        result.insights_stored
    );
}

#[test]
fn enrich_architecture_detects_known_patterns() {
    let engine = CodememEngine::for_testing();
    {
        let mut graph = engine.lock_graph().unwrap();
        graph
            .add_node(package_node("pkg:controllers", "controllers"))
            .unwrap();
        graph
            .add_node(package_node("pkg:models", "models"))
            .unwrap();
        graph.add_node(package_node("pkg:views", "views")).unwrap();
    }

    let result = engine.enrich_architecture(None).unwrap();
    let patterns_detected = result.details["patterns_detected"].as_u64().unwrap();
    assert_eq!(
        patterns_detected, 3,
        "should detect MVC pattern (controllers, models, views)"
    );
}

#[test]
fn enrich_architecture_ignores_non_structural_edges() {
    let engine = CodememEngine::for_testing();
    {
        let mut graph = engine.lock_graph().unwrap();
        graph.add_node(file_node("foo/a.rs")).unwrap();
        graph.add_node(file_node("bar/b.rs")).unwrap();

        // RELATES_TO is not a structural dependency edge
        graph
            .add_edge(make_edge(
                "file:foo/a.rs",
                "file:bar/b.rs",
                RelationshipType::RelatesTo,
            ))
            .unwrap();
    }

    let result = engine.enrich_architecture(None).unwrap();
    assert_eq!(
        result.details["dependency_edges"], 0,
        "RELATES_TO edges should not count as architectural deps"
    );
}

// ── enrich_test_mapping tests ────────────────────────────────────────

#[test]
fn enrich_test_mapping_empty_graph_returns_zero() {
    let engine = CodememEngine::for_testing();
    let result = engine.enrich_test_mapping(None).unwrap();
    assert_eq!(result.insights_stored, 0);
    assert_eq!(result.details["test_edges_created"], 0);
}

#[test]
fn enrich_test_mapping_links_test_to_function_by_name() {
    let engine = CodememEngine::for_testing();
    {
        let mut graph = engine.lock_graph().unwrap();
        graph
            .add_node(function_node("fn:parse", "parse", "src/parser.rs"))
            .unwrap();
        graph
            .add_node(test_node(
                "test:test_parse",
                "test_parse",
                "tests/parser_test.rs",
            ))
            .unwrap();
    }

    let result = engine.enrich_test_mapping(None).unwrap();
    assert!(
        result.details["test_edges_created"].as_u64().unwrap() >= 1,
        "should create edge from test_parse to parse"
    );
}

#[test]
fn enrich_test_mapping_links_test_to_function_by_calls_edge() {
    let engine = CodememEngine::for_testing();
    {
        let mut graph = engine.lock_graph().unwrap();
        graph
            .add_node(function_node("fn:compute", "compute", "src/math.rs"))
            .unwrap();
        graph
            .add_node(test_node("test:my_test", "my_test", "tests/math_test.rs"))
            .unwrap();
        // Test calls compute
        graph
            .add_edge(make_edge(
                "test:my_test",
                "fn:compute",
                RelationshipType::Calls,
            ))
            .unwrap();
    }

    let result = engine.enrich_test_mapping(None).unwrap();
    assert!(
        result.details["test_edges_created"].as_u64().unwrap() >= 1,
        "should create edge from test to called function"
    );
}

#[test]
fn enrich_test_mapping_reports_untested_public_functions() {
    let engine = CodememEngine::for_testing();
    {
        let mut graph = engine.lock_graph().unwrap();
        graph
            .add_node(method_node(
                "fn:do_stuff",
                "do_stuff",
                "src/lib.rs",
                "public",
                None,
            ))
            .unwrap();
    }

    let result = engine.enrich_test_mapping(None).unwrap();
    assert!(
        result.details["files_with_untested"].as_u64().unwrap() >= 1,
        "should flag file with untested public function"
    );
    assert!(
        result.insights_stored >= 1,
        "should produce insight for untested public functions"
    );
}

#[test]
fn enrich_test_mapping_ignores_private_untested_functions() {
    let engine = CodememEngine::for_testing();
    {
        let mut graph = engine.lock_graph().unwrap();
        graph
            .add_node(method_node(
                "fn:helper",
                "helper",
                "src/lib.rs",
                "private",
                None,
            ))
            .unwrap();
    }

    let result = engine.enrich_test_mapping(None).unwrap();
    assert_eq!(
        result.details["files_with_untested"], 0,
        "private functions should not be flagged as untested"
    );
}

// ── enrich_api_surface tests ─────────────────────────────────────────

#[test]
fn enrich_api_surface_empty_graph_returns_zero() {
    let engine = CodememEngine::for_testing();
    let result = engine.enrich_api_surface(None).unwrap();
    assert_eq!(result.insights_stored, 0);
    assert_eq!(result.details["files_analyzed"], 0);
}

#[test]
fn enrich_api_surface_counts_public_and_private() {
    let engine = CodememEngine::for_testing();
    {
        let mut graph = engine.lock_graph().unwrap();
        graph
            .add_node(method_node(
                "fn:pub_fn",
                "pub_fn",
                "src/api.rs",
                "public",
                None,
            ))
            .unwrap();
        graph
            .add_node(method_node(
                "fn:priv_fn",
                "priv_fn",
                "src/api.rs",
                "private",
                None,
            ))
            .unwrap();
        graph
            .add_node(method_node(
                "fn:pub_fn2",
                "pub_fn2",
                "src/api.rs",
                "public",
                None,
            ))
            .unwrap();
    }

    let result = engine.enrich_api_surface(None).unwrap();
    assert_eq!(result.details["files_analyzed"], 1);
    assert_eq!(result.details["total_public_symbols"], 2);
    assert_eq!(result.details["total_private_symbols"], 1);
}

#[test]
fn enrich_api_surface_stores_insight_for_files_with_public_symbols() {
    let engine = CodememEngine::for_testing();
    {
        let mut graph = engine.lock_graph().unwrap();
        graph
            .add_node(method_node(
                "fn:handler",
                "handler",
                "src/routes.rs",
                "public",
                None,
            ))
            .unwrap();
    }

    let result = engine.enrich_api_surface(None).unwrap();
    assert!(
        result.insights_stored >= 1,
        "should store insight for file with public symbols"
    );
}

#[test]
fn enrich_api_surface_skips_nodes_without_file_path() {
    let engine = CodememEngine::for_testing();
    {
        let mut graph = engine.lock_graph().unwrap();
        let node = GraphNode {
            id: "fn:orphan".to_string(),
            kind: NodeKind::Function,
            label: "orphan".to_string(),
            payload: HashMap::from([("visibility".into(), json!("public"))]),
            centrality: 0.0,
            memory_id: None,
            namespace: None,
            valid_from: None,
            valid_to: None,
        };
        graph.add_node(node).unwrap();
    }

    let result = engine.enrich_api_surface(None).unwrap();
    assert_eq!(
        result.details["files_analyzed"], 0,
        "nodes without file_path should be skipped"
    );
}

// ── enrich_doc_coverage tests ────────────────────────────────────────

#[test]
fn enrich_doc_coverage_empty_graph_returns_zero() {
    let engine = CodememEngine::for_testing();
    let result = engine.enrich_doc_coverage(None).unwrap();
    assert_eq!(result.insights_stored, 0);
    assert_eq!(result.details["files_analyzed"], 0);
}

#[test]
fn enrich_doc_coverage_full_coverage_no_insight() {
    let engine = CodememEngine::for_testing();
    {
        let mut graph = engine.lock_graph().unwrap();
        graph
            .add_node(method_node(
                "fn:well_documented",
                "well_documented",
                "src/lib.rs",
                "public",
                Some("This function does something."),
            ))
            .unwrap();
    }

    let result = engine.enrich_doc_coverage(None).unwrap();
    assert_eq!(
        result.insights_stored, 0,
        "fully documented file should not generate insight"
    );
    assert_eq!(result.details["total_public_documented"], 1);
    assert_eq!(result.details["total_public_undocumented"], 0);
}

#[test]
fn enrich_doc_coverage_low_coverage_generates_insight() {
    let engine = CodememEngine::for_testing();
    {
        let mut graph = engine.lock_graph().unwrap();
        for i in 0..3 {
            graph
                .add_node(method_node(
                    &format!("fn:func_{i}"),
                    &format!("func_{i}"),
                    "src/bad.rs",
                    "public",
                    None,
                ))
                .unwrap();
        }
    }

    let result = engine.enrich_doc_coverage(None).unwrap();
    assert!(
        result.insights_stored >= 1,
        "0% doc coverage should generate insight"
    );
    assert_eq!(result.details["total_public_undocumented"], 3);
}

#[test]
fn enrich_doc_coverage_ignores_private_symbols() {
    let engine = CodememEngine::for_testing();
    {
        let mut graph = engine.lock_graph().unwrap();
        graph
            .add_node(method_node(
                "fn:internal",
                "internal",
                "src/lib.rs",
                "private",
                None,
            ))
            .unwrap();
    }

    let result = engine.enrich_doc_coverage(None).unwrap();
    assert_eq!(
        result.details["files_analyzed"], 0,
        "private symbols should not count"
    );
}

#[test]
fn enrich_doc_coverage_mixed_coverage() {
    let engine = CodememEngine::for_testing();
    {
        let mut graph = engine.lock_graph().unwrap();
        graph
            .add_node(method_node(
                "fn:documented",
                "documented",
                "src/mixed.rs",
                "public",
                Some("has docs"),
            ))
            .unwrap();
        for i in 0..3 {
            graph
                .add_node(method_node(
                    &format!("fn:undoc_{i}"),
                    &format!("undoc_{i}"),
                    "src/mixed.rs",
                    "public",
                    None,
                ))
                .unwrap();
        }
    }

    let result = engine.enrich_doc_coverage(None).unwrap();
    assert_eq!(result.details["total_public_documented"], 1);
    assert_eq!(result.details["total_public_undocumented"], 3);
    assert!(
        result.insights_stored >= 1,
        "25% coverage should trigger insight"
    );
}

// ── enrich_hot_complex tests ─────────────────────────────────────────

#[test]
fn enrich_hot_complex_empty_graph_returns_zero() {
    let engine = CodememEngine::for_testing();
    let result = engine.enrich_hot_complex(None).unwrap();
    assert_eq!(result.insights_stored, 0);
    assert_eq!(result.details["hot_complex_files"], 0);
}

#[test]
fn enrich_hot_complex_no_overlap_no_insight() {
    let engine = CodememEngine::for_testing();
    {
        let mut graph = engine.lock_graph().unwrap();

        let mut churn_node = file_node("src/churny.rs");
        churn_node
            .payload
            .insert("git_churn_rate".into(), json!(5.0));
        graph.add_node(churn_node).unwrap();

        let mut complex_fn = function_node("fn:complex", "complex", "src/stable.rs");
        complex_fn
            .payload
            .insert("cyclomatic_complexity".into(), json!(15));
        graph.add_node(complex_fn).unwrap();
    }

    let result = engine.enrich_hot_complex(None).unwrap();
    assert_eq!(
        result.details["hot_complex_files"], 0,
        "no overlap between churn and complexity => no hot_complex files"
    );
}

#[test]
fn enrich_hot_complex_detects_overlap() {
    let engine = CodememEngine::for_testing();
    {
        let mut graph = engine.lock_graph().unwrap();

        let mut file = file_node("src/risky.rs");
        file.payload.insert("git_churn_rate".into(), json!(10.0));
        graph.add_node(file).unwrap();

        let mut func = function_node("fn:risky_fn", "risky_fn", "src/risky.rs");
        func.payload
            .insert("cyclomatic_complexity".into(), json!(20));
        graph.add_node(func).unwrap();
    }

    let result = engine.enrich_hot_complex(None).unwrap();
    assert_eq!(
        result.details["hot_complex_files"], 1,
        "should detect 1 hot+complex file"
    );
    assert!(
        result.insights_stored >= 1,
        "should store insight for hot+complex file"
    );
}

#[test]
fn enrich_hot_complex_ignores_low_complexity() {
    let engine = CodememEngine::for_testing();
    {
        let mut graph = engine.lock_graph().unwrap();

        let mut file = file_node("src/simple.rs");
        file.payload.insert("git_churn_rate".into(), json!(8.0));
        graph.add_node(file).unwrap();

        let mut func = function_node("fn:easy", "easy", "src/simple.rs");
        func.payload
            .insert("cyclomatic_complexity".into(), json!(3));
        graph.add_node(func).unwrap();
    }

    let result = engine.enrich_hot_complex(None).unwrap();
    assert_eq!(
        result.details["hot_complex_files"], 0,
        "complexity <= 5 should be ignored"
    );
}

// ── enrich_quality_stratification tests ──────────────────────────────

#[test]
fn enrich_quality_stratification_empty_returns_zero() {
    let engine = CodememEngine::for_testing();
    let result = engine.enrich_quality_stratification(None).unwrap();
    assert_eq!(result.details["total_analyzed"], 0);
    assert_eq!(result.insights_stored, 0);
}

#[test]
fn enrich_quality_stratification_classifies_critical() {
    let engine = CodememEngine::for_testing();

    engine.store_insight(
        "SQL injection detected in user handler — security risk abc",
        "security",
        &["severity:critical"],
        0.5,
        None,
        &[],
    );

    let result = engine.enrich_quality_stratification(None).unwrap();
    assert_eq!(
        result.details["critical"],
        json!(1),
        "security insight should be classified as critical"
    );
}

#[test]
fn enrich_quality_stratification_classifies_signal() {
    let engine = CodememEngine::for_testing();

    engine.store_insight(
        "High complexity in module parser cyclomatic 15 xyz",
        "complexity",
        &[],
        0.3,
        None,
        &[],
    );

    let result = engine.enrich_quality_stratification(None).unwrap();
    assert_eq!(
        result.details["signal"],
        json!(1),
        "complexity insight should be classified as signal"
    );
}

#[test]
fn enrich_quality_stratification_classifies_noise() {
    let engine = CodememEngine::for_testing();

    engine.store_insight(
        "File count: 42 files analyzed in repository xyz",
        "stats",
        &[],
        0.6,
        None,
        &[],
    );

    let result = engine.enrich_quality_stratification(None).unwrap();
    assert_eq!(
        result.details["noise"],
        json!(1),
        "generic insight should be classified as noise"
    );
}

#[test]
fn enrich_quality_stratification_reclassifies_importance() {
    let engine = CodememEngine::for_testing();

    let id = engine
        .store_insight(
            "Credential leak detected critical vulnerability xyz",
            "security",
            &[],
            0.3,
            None,
            &[],
        )
        .unwrap();

    engine.enrich_quality_stratification(None).unwrap();

    let memory = engine.storage.get_memory(&id).unwrap().unwrap();
    assert!(
        memory.importance >= 0.8,
        "critical insight importance should be raised to >= 0.8, got {}",
        memory.importance
    );
}

#[test]
fn enrich_quality_stratification_respects_namespace() {
    let engine = CodememEngine::for_testing();

    engine.store_insight(
        "SQL injection detected critical vulnerability abc",
        "security",
        &[],
        0.3,
        Some("proj-a"),
        &[],
    );

    let result = engine
        .enrich_quality_stratification(Some("proj-b"))
        .unwrap();
    assert_eq!(
        result.details["total_analyzed"], 0,
        "should not analyze insights from a different namespace"
    );
}

// ── enrich_change_impact tests ───────────────────────────────────────

#[test]
fn enrich_change_impact_missing_file_returns_error() {
    let engine = CodememEngine::for_testing();
    let result = engine.enrich_change_impact("nonexistent.rs", None);
    assert!(result.is_err(), "should error when file node not found");
}

#[test]
fn enrich_change_impact_isolated_file_returns_zero_impact() {
    let engine = CodememEngine::for_testing();
    {
        let mut graph = engine.lock_graph().unwrap();
        graph.add_node(file_node("src/isolated.rs")).unwrap();
    }

    let result = engine
        .enrich_change_impact("src/isolated.rs", None)
        .unwrap();
    assert_eq!(result.details["impact_score"], 0);
    assert_eq!(result.insights_stored, 0);
}

#[test]
fn enrich_change_impact_counts_co_changed_files() {
    let engine = CodememEngine::for_testing();
    {
        let mut graph = engine.lock_graph().unwrap();
        graph.add_node(file_node("src/main.rs")).unwrap();
        graph.add_node(file_node("src/config.rs")).unwrap();

        let edge = make_edge(
            "file:src/main.rs",
            "file:src/config.rs",
            RelationshipType::CoChanged,
        );
        graph.add_edge(edge).unwrap();
    }

    let result = engine.enrich_change_impact("src/main.rs", None).unwrap();
    assert_eq!(result.details["co_changed"], 1);
    assert!(result.details["impact_score"].as_u64().unwrap() >= 1);
}

#[test]
fn enrich_change_impact_counts_callers_from_symbol_edges() {
    let engine = CodememEngine::for_testing();
    {
        let mut graph = engine.lock_graph().unwrap();
        graph.add_node(file_node("src/target.rs")).unwrap();
        graph
            .add_node(function_node("fn:target_fn", "target_fn", "src/target.rs"))
            .unwrap();
        graph
            .add_node(function_node("fn:caller_fn", "caller_fn", "src/caller.rs"))
            .unwrap();
        graph
            .add_edge(make_edge(
                "fn:caller_fn",
                "fn:target_fn",
                RelationshipType::Calls,
            ))
            .unwrap();
    }

    let result = engine.enrich_change_impact("src/target.rs", None).unwrap();
    assert!(
        result.details["callers"].as_u64().unwrap() >= 1,
        "should detect caller from another file"
    );
}

#[test]
fn enrich_change_impact_stores_insight_for_nonzero_impact() {
    let engine = CodememEngine::for_testing();
    {
        let mut graph = engine.lock_graph().unwrap();
        graph.add_node(file_node("src/hub.rs")).unwrap();
        graph.add_node(file_node("src/dep1.rs")).unwrap();
        graph.add_node(file_node("src/dep2.rs")).unwrap();

        graph
            .add_edge(make_edge(
                "file:src/hub.rs",
                "file:src/dep1.rs",
                RelationshipType::CoChanged,
            ))
            .unwrap();
        graph
            .add_edge(make_edge(
                "file:src/hub.rs",
                "file:src/dep2.rs",
                RelationshipType::CoChanged,
            ))
            .unwrap();
    }

    let result = engine.enrich_change_impact("src/hub.rs", None).unwrap();
    assert!(
        result.insights_stored >= 1,
        "should store insight for file with impact"
    );
}

// ── enrich_complexity / enrich_code_smells (empty graph) ─────────────

#[test]
fn enrich_complexity_empty_graph_returns_zero() {
    let engine = CodememEngine::for_testing();
    let result = engine.enrich_complexity(None, None).unwrap();
    assert_eq!(result.insights_stored, 0);
    assert_eq!(result.details["symbols_analyzed"], 0);
}

#[test]
fn enrich_code_smells_empty_graph_returns_zero() {
    let engine = CodememEngine::for_testing();
    let result = engine.enrich_code_smells(None, None).unwrap();
    assert_eq!(result.insights_stored, 0);
    assert_eq!(result.details["smells_detected"], 0);
}

#[test]
fn enrich_code_smells_detects_long_function() {
    let engine = CodememEngine::for_testing();
    {
        let mut graph = engine.lock_graph().unwrap();
        let mut func = function_node("fn:big_func", "big_func", "src/nonexistent.rs");
        func.payload.insert("line_start".into(), json!(1));
        func.payload.insert("line_end".into(), json!(101));
        graph.add_node(func).unwrap();
    }

    let result = engine.enrich_code_smells(None, None).unwrap();
    assert!(
        result.insights_stored >= 1,
        "should detect long function (100 lines > 50 threshold)"
    );
}

#[test]
fn enrich_code_smells_detects_too_many_parameters() {
    let engine = CodememEngine::for_testing();
    {
        let mut graph = engine.lock_graph().unwrap();
        let mut func = function_node("fn:many_params", "many_params", "src/nonexistent.rs");
        func.payload.insert("line_start".into(), json!(1));
        func.payload.insert("line_end".into(), json!(10));
        func.payload.insert(
            "signature".into(),
            json!("fn many_params(a: i32, b: i32, c: i32, d: i32, e: i32, f: i32)"),
        );
        graph.add_node(func).unwrap();
    }

    let result = engine.enrich_code_smells(None, None).unwrap();
    assert!(
        result.insights_stored >= 1,
        "should detect function with 6 parameters > 5 threshold"
    );
}

// ── enrich_git_history / enrich_blame (graceful error) ───────────────

#[test]
fn enrich_git_history_fails_gracefully_on_non_git_dir() {
    let engine = CodememEngine::for_testing();
    let result = engine.enrich_git_history("/nonexistent/path", 30, None);
    assert!(
        result.is_err(),
        "should return error for non-existent git repo"
    );
}

#[test]
fn enrich_blame_empty_graph_stores_zero() {
    let engine = CodememEngine::for_testing();
    let result = engine.enrich_blame("/tmp", None).unwrap();
    assert_eq!(result.details["files_annotated"], 0);
    assert_eq!(result.insights_stored, 0);
}

// ── Namespace filtering across enrichment methods ────────────────────

#[test]
fn enrich_security_respects_namespace_in_insights() {
    let engine = CodememEngine::for_testing();
    {
        let mut graph = engine.lock_graph().unwrap();
        graph.add_node(file_node("src/secret_key.rs")).unwrap();
    }

    let result = engine.enrich_security(Some("my-ns")).unwrap();
    assert!(result.insights_stored >= 1);

    let all_ids = engine.storage.list_memory_ids().unwrap_or_default();
    for id in &all_ids {
        if let Ok(Some(mem)) = engine.storage.get_memory(id) {
            if mem.tags.contains(&"static-analysis".to_string()) {
                assert_eq!(
                    mem.namespace.as_deref(),
                    Some("my-ns"),
                    "insight should have the correct namespace"
                );
            }
        }
    }
}

// ── run_enrichments pipeline tests ────────────────────────────────────

#[test]
fn run_enrichments_with_empty_graph_produces_zero_insights() {
    let engine = CodememEngine::for_testing();
    let tmp = tempfile::TempDir::new().unwrap();
    let path = tmp.path().to_str().unwrap();

    let result = engine.run_enrichments(path, &[], 30, None, None);
    assert_eq!(
        result.total_insights, 0,
        "empty graph should produce zero enrichment insights"
    );
}

#[test]
fn run_enrichments_selected_analyses_only_runs_requested() {
    let engine = CodememEngine::for_testing();
    let tmp = tempfile::TempDir::new().unwrap();
    let path = tmp.path().to_str().unwrap();

    // Only run doc_coverage and api_surface
    let analyses = vec!["doc_coverage".to_string(), "api_surface".to_string()];
    let result = engine.run_enrichments(path, &analyses, 30, None, None);

    // These keys should be present in results
    assert!(
        result.results.get("doc_coverage").is_some(),
        "doc_coverage result should be present"
    );
    assert!(
        result.results.get("api_surface").is_some(),
        "api_surface result should be present"
    );

    // Analyses not requested should NOT be present
    assert!(
        result.results.get("security").is_none(),
        "security should not be present when not requested"
    );
    assert!(
        result.results.get("architecture").is_none(),
        "architecture should not be present when not requested"
    );
}

#[test]
fn run_enrichments_complexity_with_real_source_file() {
    let engine = CodememEngine::for_testing();
    let tmp = tempfile::TempDir::new().unwrap();
    let src_dir = tmp.path().join("src");
    std::fs::create_dir_all(&src_dir).unwrap();

    // Write a source file with a very complex function (many decision points)
    // Need > 10 cyclomatic complexity: 1 base + each if/for/while/match/&&/|| adds 1
    let source = r#"fn complex_handler(request: Request) -> Response {
    if request.method == "GET" {
        if request.authenticated {
            if request.has_permission("read") {
                for item in request.items() {
                    if item.is_valid() {
                        if item.is_active() {
                            match item.kind() {
                                Kind::A => { process_a(item); }
                                Kind::B => { process_b(item); }
                                _ => { handle_default(item); }
                            }
                        }
                    }
                }
            }
        }
    } else if request.method == "POST" {
        if request.body.is_some() && request.content_type == "json" {
            if request.authorized && request.valid {
                process_post(request);
            }
        }
    } else if request.method == "DELETE" {
        if request.is_admin() || request.has_permission("delete") {
            delete_handler(request);
        }
    }
    Response::ok()
}
"#;
    std::fs::write(src_dir.join("handler.rs"), source).unwrap();

    // Add a function node pointing to our file
    {
        let mut graph = engine.lock_graph().unwrap();
        let mut func = function_node("fn:complex_handler", "complex_handler", "src/handler.rs");
        func.payload.insert("line_start".into(), json!(1));
        func.payload.insert("line_end".into(), json!(32));
        graph.add_node(func).unwrap();
    }

    let analyses = vec!["complexity".to_string()];
    let result = engine.run_enrichments(tmp.path().to_str().unwrap(), &analyses, 30, None, None);

    let complexity = &result.results["complexity"];
    assert_eq!(
        complexity["symbols_analyzed"], 1,
        "should analyze 1 function"
    );
    // The function has many if/else if/match/for/&&/|| — cyclomatic > 10
    let high_count = complexity["high_complexity_count"].as_u64().unwrap_or(0);
    assert!(
        high_count >= 1,
        "complex function should be flagged as high complexity"
    );
}

#[test]
fn run_enrichments_code_smells_detects_deep_nesting_in_real_file() {
    let engine = CodememEngine::for_testing();
    let tmp = tempfile::TempDir::new().unwrap();
    let src_dir = tmp.path().join("src");
    std::fs::create_dir_all(&src_dir).unwrap();

    // Write a file with deeply nested code (>4 levels)
    let source = r#"fn deeply_nested() {
    if true {
        if true {
            if true {
                if true {
                    if true {
                        println!("deep");
                    }
                }
            }
        }
    }
}
"#;
    std::fs::write(src_dir.join("nested.rs"), source).unwrap();

    {
        let mut graph = engine.lock_graph().unwrap();
        let mut func = function_node("fn:deeply_nested", "deeply_nested", "src/nested.rs");
        func.payload.insert("line_start".into(), json!(1));
        func.payload.insert("line_end".into(), json!(13));
        graph.add_node(func).unwrap();
    }

    let analyses = vec!["code_smells".to_string()];
    let result = engine.run_enrichments(tmp.path().to_str().unwrap(), &analyses, 30, None, None);

    assert!(
        result.total_insights >= 1,
        "deeply nested function should trigger code smell insight"
    );
}

#[test]
fn run_enrichments_produces_static_analysis_tagged_insights() {
    let engine = CodememEngine::for_testing();
    let tmp = tempfile::TempDir::new().unwrap();

    // Set up graph with security-relevant files
    {
        let mut graph = engine.lock_graph().unwrap();
        graph.add_node(file_node("src/auth.rs")).unwrap();
        graph
            .add_node(endpoint_node("ep:login", "POST /api/login"))
            .unwrap();
    }

    let analyses = vec!["security".to_string()];
    let result = engine.run_enrichments(tmp.path().to_str().unwrap(), &analyses, 30, None, None);

    assert!(
        result.total_insights >= 1,
        "security analysis should produce insights"
    );

    // Verify that stored insights have the static-analysis tag
    let all_ids = engine.storage.list_memory_ids().unwrap_or_default();
    let mut found_static_analysis = false;
    for id in &all_ids {
        if let Ok(Some(mem)) = engine.storage.get_memory(id) {
            if mem.tags.contains(&"static-analysis".to_string()) {
                found_static_analysis = true;
                assert_eq!(
                    mem.memory_type,
                    codemem_core::MemoryType::Insight,
                    "static-analysis tagged memory should be Insight type"
                );
            }
        }
    }
    assert!(
        found_static_analysis,
        "should find at least one memory tagged with static-analysis"
    );
}

#[test]
fn run_enrichments_with_namespace_passes_to_insights() {
    let engine = CodememEngine::for_testing();
    let tmp = tempfile::TempDir::new().unwrap();

    {
        let mut graph = engine.lock_graph().unwrap();
        graph.add_node(file_node("src/credentials.rs")).unwrap();
    }

    let analyses = vec!["security".to_string()];
    let result = engine.run_enrichments(
        tmp.path().to_str().unwrap(),
        &analyses,
        30,
        Some("test-ns"),
        None,
    );

    assert!(result.total_insights >= 1);

    let all_ids = engine.storage.list_memory_ids().unwrap_or_default();
    for id in &all_ids {
        if let Ok(Some(mem)) = engine.storage.get_memory(id) {
            if mem.tags.contains(&"static-analysis".to_string()) {
                assert_eq!(
                    mem.namespace.as_deref(),
                    Some("test-ns"),
                    "insight namespace should match the one passed to run_enrichments"
                );
            }
        }
    }
}

#[test]
fn run_enrichments_change_impact_requires_file_path() {
    let engine = CodememEngine::for_testing();
    let tmp = tempfile::TempDir::new().unwrap();

    let analyses = vec!["change_impact".to_string()];
    let result = engine.run_enrichments(
        tmp.path().to_str().unwrap(),
        &analyses,
        30,
        None,
        None, // no file_path provided
    );

    let change_impact = &result.results["change_impact"];
    assert!(
        change_impact.get("error").is_some(),
        "change_impact without file_path should produce an error in results"
    );
}

#[test]
fn run_enrichments_git_on_non_git_dir_reports_error_in_results() {
    let engine = CodememEngine::for_testing();
    let tmp = tempfile::TempDir::new().unwrap();

    let analyses = vec!["git".to_string()];
    let result = engine.run_enrichments(tmp.path().to_str().unwrap(), &analyses, 30, None, None);

    let git_result = &result.results["git"];
    assert!(
        git_result.get("error").is_some(),
        "git enrichment on non-git dir should report error: {:?}",
        git_result
    );
}

// ── enrich_complexity with real files ────────────────────────────────

#[test]
fn enrich_complexity_annotates_graph_nodes() {
    let engine = CodememEngine::for_testing();
    let tmp = tempfile::TempDir::new().unwrap();
    let src_dir = tmp.path().join("src");
    std::fs::create_dir_all(&src_dir).unwrap();

    let source = r#"fn simple_function() {
    let x = 1;
    let y = 2;
    println!("{}", x + y);
}
"#;
    std::fs::write(src_dir.join("simple.rs"), source).unwrap();

    {
        let mut graph = engine.lock_graph().unwrap();
        let mut func = function_node("fn:simple_function", "simple_function", "src/simple.rs");
        func.payload.insert("line_start".into(), json!(1));
        func.payload.insert("line_end".into(), json!(5));
        graph.add_node(func).unwrap();
    }

    engine.enrich_complexity(None, Some(tmp.path())).unwrap();

    let graph = engine.lock_graph().unwrap();
    let node = graph.get_node("fn:simple_function").unwrap().unwrap();
    assert!(
        node.payload.contains_key("cyclomatic_complexity"),
        "function should be annotated with cyclomatic_complexity"
    );
    assert!(
        node.payload.contains_key("cognitive_complexity"),
        "function should be annotated with cognitive_complexity"
    );

    let cyc = node.payload["cyclomatic_complexity"].as_u64().unwrap();
    assert!(
        cyc <= 5,
        "simple function should have low cyclomatic complexity, got {}",
        cyc
    );
}

// ── enrich_doc_coverage: insight content validation ──────────────────

#[test]
fn enrich_doc_coverage_insight_mentions_coverage_percentage() {
    let engine = CodememEngine::for_testing();
    {
        let mut graph = engine.lock_graph().unwrap();
        // 0 documented out of 2 public = 0% coverage
        for i in 0..2 {
            graph
                .add_node(method_node(
                    &format!("fn:undoc_{i}"),
                    &format!("undoc_{i}"),
                    "src/poorly_covered.rs",
                    "public",
                    None,
                ))
                .unwrap();
        }
    }

    engine.enrich_doc_coverage(None).unwrap();

    let all_ids = engine.storage.list_memory_ids().unwrap_or_default();
    let mut found_coverage_insight = false;
    for id in &all_ids {
        if let Ok(Some(mem)) = engine.storage.get_memory(id) {
            if mem.content.contains("coverage") && mem.content.contains("poorly_covered.rs") {
                found_coverage_insight = true;
                assert!(
                    mem.content.contains('%'),
                    "doc coverage insight should mention percentage"
                );
            }
        }
    }
    assert!(
        found_coverage_insight,
        "should produce a coverage insight for undocumented file"
    );
}

// ── Multiple enrichments compose correctly ───────────────────────────

#[test]
fn multiple_enrichments_accumulate_insights() {
    let engine = CodememEngine::for_testing();
    {
        let mut graph = engine.lock_graph().unwrap();
        graph.add_node(file_node("src/auth.rs")).unwrap();
        graph
            .add_node(endpoint_node("ep:login", "POST /login"))
            .unwrap();
        graph
            .add_node(method_node(
                "fn:public_api",
                "public_api",
                "src/auth.rs",
                "public",
                None,
            ))
            .unwrap();
    }

    let sec_result = engine.enrich_security(None).unwrap();
    let api_result = engine.enrich_api_surface(None).unwrap();
    let doc_result = engine.enrich_doc_coverage(None).unwrap();

    let total =
        sec_result.insights_stored + api_result.insights_stored + doc_result.insights_stored;
    assert!(
        total >= 2,
        "running multiple enrichments should accumulate insights; got {}",
        total
    );
}

// ── add_edges_with_placeholders: placeholder node creation ───────────

#[test]
fn temporal_edge_insertion_creates_placeholder_nodes() {
    let engine = CodememEngine::for_testing();

    // Create a commit node and add it to in-memory graph + storage
    let commit_node = GraphNode {
        id: "commit:abc123".to_string(),
        kind: NodeKind::Commit,
        label: "abc123 feat: test commit".to_string(),
        payload: {
            let mut p = HashMap::new();
            p.insert("hash".into(), json!("abc123"));
            p
        },
        centrality: 0.0,
        memory_id: None,
        namespace: None,
        valid_from: None,
        valid_to: None,
    };
    engine
        .storage
        .insert_graph_node(&commit_node)
        .expect("insert commit to storage");
    {
        let mut graph = engine.lock_graph().expect("lock graph for setup");
        graph.add_node(commit_node).expect("add commit node");
    }

    // Create a ModifiedBy edge referencing a non-existent file node
    let edge = Edge {
        id: "modby:file:src/main.rs:abc123".to_string(),
        src: "file:src/main.rs".to_string(),
        dst: "commit:abc123".to_string(),
        relationship: RelationshipType::ModifiedBy,
        weight: 0.4,
        properties: HashMap::new(),
        created_at: chrono::Utc::now(),
        valid_from: None,
        valid_to: None,
    };

    // The file node does NOT exist yet — add_edges_with_placeholders should create it
    {
        let mut graph = engine.lock_graph().expect("lock graph for edges");
        engine
            .add_edges_with_placeholders(&mut **graph, &[edge])
            .expect("add edges with placeholders");
    }

    // Assert: placeholder file node was created with correct kind
    let graph = engine.lock_graph().expect("lock graph for assertions");
    let file_node = graph
        .get_node("file:src/main.rs")
        .expect("get file node")
        .expect("placeholder file node should exist");
    assert_eq!(file_node.kind, NodeKind::File);

    // Assert: edge exists in the in-memory graph
    let edges = graph.get_edges("file:src/main.rs").expect("get edges");
    assert!(
        edges
            .iter()
            .any(|e| e.id == "modby:file:src/main.rs:abc123"),
        "ModifiedBy edge should exist in the in-memory graph"
    );

    // Assert: placeholder was also persisted to storage
    drop(graph);
    let stored = engine
        .storage
        .get_graph_node("file:src/main.rs")
        .expect("storage query");
    assert!(
        stored.is_some(),
        "placeholder should be persisted to storage"
    );
}

// ── expire_deleted_symbols: delete-then-recreate bug ─────────────────

#[test]
fn expire_deleted_symbols_skips_files_that_exist_on_disk() {
    // The fix filters out deleted-file paths that currently exist on disk.
    // We test this by creating a temp dir with a git repo containing a file,
    // then adding a File graph node and calling expire_deleted_symbols.
    // Since the file exists on disk, it should NOT be expired even if
    // git log --diff-filter=D reports a past deletion.
    let tmp = tempfile::TempDir::new().expect("create temp dir");
    let tmp_path = tmp.path();

    // Create a file that simulates a "resurrected" file
    let src_dir = tmp_path.join("src");
    std::fs::create_dir_all(&src_dir).expect("create src dir");
    std::fs::write(src_dir.join("resurrected.rs"), "fn main() {}").expect("write resurrected.rs");

    // Initialize a git repo so git log commands don't fail
    std::process::Command::new("git")
        .args(["-C", tmp_path.to_str().expect("tmp path to str"), "init"])
        .output()
        .expect("git init");
    std::process::Command::new("git")
        .args([
            "-C",
            tmp_path.to_str().expect("tmp path to str"),
            "add",
            ".",
        ])
        .output()
        .expect("git add");
    std::process::Command::new("git")
        .args([
            "-C",
            tmp_path.to_str().expect("tmp path to str"),
            "commit",
            "-m",
            "initial",
        ])
        .output()
        .expect("git commit");

    let engine = CodememEngine::for_testing();

    // Insert a File graph node with valid_to: None (alive)
    let file_node = GraphNode {
        id: "file:src/resurrected.rs".to_string(),
        kind: NodeKind::File,
        label: "src/resurrected.rs".to_string(),
        payload: HashMap::new(),
        centrality: 0.0,
        memory_id: None,
        namespace: Some("test".to_string()),
        valid_from: None,
        valid_to: None,
    };
    engine
        .storage
        .insert_graph_node(&file_node)
        .expect("insert file node to storage");
    {
        let mut graph = engine.lock_graph().expect("lock graph for setup");
        graph.add_node(file_node).expect("add file node to graph");
    }

    // Call expire_deleted_symbols — git log won't find deletions in this
    // fresh repo, AND the new filter would catch it if it did.
    let result = engine.expire_deleted_symbols(
        tmp_path.to_str().expect("tmp path to str"),
        &[], // empty commits list
        "test",
    );
    assert_eq!(
        result.expect("expire_deleted_symbols should succeed"),
        0,
        "no files should be expired when the file exists on disk"
    );

    // Verify the node still has valid_to: None
    let graph = engine.lock_graph().expect("lock graph for assertion");
    let node = graph
        .get_node("file:src/resurrected.rs")
        .expect("get file node")
        .expect("file node should still exist");
    assert!(
        node.valid_to.is_none(),
        "resurrected file should NOT have valid_to set"
    );
}
