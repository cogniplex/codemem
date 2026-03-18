use crate::CodememEngine;
use chrono::{Duration, Utc};
use codemem_core::{Edge, GraphNode, NodeKind, RelationshipType};
use serde_json::json;
use std::collections::HashMap;

#[test]
fn what_changed_rejects_reversed_date_range() {
    let engine = CodememEngine::for_testing();
    let now = Utc::now();
    let yesterday = now - Duration::days(1);
    let result = engine.what_changed(now, yesterday, None);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("before"), "Error should mention date ordering");
}

#[test]
fn detect_drift_rejects_reversed_date_range() {
    let engine = CodememEngine::for_testing();
    let now = Utc::now();
    let yesterday = now - Duration::days(1);
    let result = engine.detect_drift(now, yesterday, None);
    assert!(result.is_err());
}

#[test]
fn find_stale_files_clamps_extreme_stale_days() {
    let engine = CodememEngine::for_testing();
    let result = engine.find_stale_files(None, u64::MAX);
    assert!(result.is_ok());
}

#[test]
fn symbol_history_returns_commits_with_all_changed_files() {
    let engine = CodememEngine::for_testing();
    let now = Utc::now();

    // Create file1, file2, and a commit node
    let file1 = GraphNode {
        id: "file:src/a.rs".to_string(),
        kind: NodeKind::File,
        label: "src/a.rs".to_string(),
        payload: HashMap::new(),
        centrality: 0.0,
        memory_id: None,
        namespace: None,
        valid_from: None,
        valid_to: None,
    };
    let file2 = GraphNode {
        id: "file:src/b.rs".to_string(),
        kind: NodeKind::File,
        label: "src/b.rs".to_string(),
        payload: HashMap::new(),
        centrality: 0.0,
        memory_id: None,
        namespace: None,
        valid_from: None,
        valid_to: None,
    };
    let commit = GraphNode {
        id: "commit:abc123".to_string(),
        kind: NodeKind::Commit,
        label: "abc123 feat: stuff".to_string(),
        payload: {
            let mut p = HashMap::new();
            p.insert("hash".into(), serde_json::json!("abc123"));
            p.insert("author".into(), serde_json::json!("dev"));
            p.insert("date".into(), serde_json::json!(now.to_rfc3339()));
            p.insert("subject".into(), serde_json::json!("feat: stuff"));
            p
        },
        centrality: 0.0,
        memory_id: None,
        namespace: None,
        valid_from: Some(now),
        valid_to: None,
    };

    // ModifiedBy edges: both files modified by the same commit
    let edge1 = Edge {
        id: "modby:file:src/a.rs:abc123".to_string(),
        src: "file:src/a.rs".to_string(),
        dst: "commit:abc123".to_string(),
        relationship: RelationshipType::ModifiedBy,
        weight: 0.4,
        properties: HashMap::new(),
        created_at: now,
        valid_from: Some(now),
        valid_to: None,
    };
    let edge2 = Edge {
        id: "modby:file:src/b.rs:abc123".to_string(),
        src: "file:src/b.rs".to_string(),
        dst: "commit:abc123".to_string(),
        relationship: RelationshipType::ModifiedBy,
        weight: 0.4,
        properties: HashMap::new(),
        created_at: now,
        valid_from: Some(now),
        valid_to: None,
    };

    // Add nodes and edges to in-memory graph
    {
        let mut graph = engine.lock_graph().expect("lock graph");
        graph.add_node(file1.clone()).expect("add file1");
        graph.add_node(file2.clone()).expect("add file2");
        graph.add_node(commit.clone()).expect("add commit");
        graph.add_edge(edge1.clone()).expect("add edge1");
        graph.add_edge(edge2.clone()).expect("add edge2");
    }

    // Persist edges to storage (symbol_history reads from storage)
    engine
        .storage()
        .insert_graph_node(&file1)
        .expect("persist file1");
    engine
        .storage()
        .insert_graph_node(&file2)
        .expect("persist file2");
    engine
        .storage()
        .insert_graph_node(&commit)
        .expect("persist commit");
    engine
        .storage()
        .insert_graph_edge(&edge1)
        .expect("persist edge1");
    engine
        .storage()
        .insert_graph_edge(&edge2)
        .expect("persist edge2");

    // Query history for file1 — should show commit with BOTH files in changed_files
    let history = engine
        .symbol_history("file:src/a.rs")
        .expect("symbol_history");
    assert_eq!(history.len(), 1, "should find exactly one commit");
    let entry = &history[0];
    assert_eq!(entry.commit_id, "commit:abc123");
    assert!(
        entry.changed_files.contains(&"src/a.rs".to_string()),
        "should contain file a: {:?}",
        entry.changed_files
    );
    assert!(
        entry.changed_files.contains(&"src/b.rs".to_string()),
        "should contain sibling file b: {:?}",
        entry.changed_files
    );
}

// ── Test Impact Analysis ─────────────────────────────────────────────────

/// Helper: create a symbol node with optional is_test flag.
fn make_symbol_node(id: &str, label: &str, is_test: bool) -> GraphNode {
    let mut payload = HashMap::new();
    if is_test {
        payload.insert("is_test".to_string(), json!(true));
    }
    payload.insert("file_path".to_string(), json!(format!("src/{label}.rs")));
    GraphNode {
        id: id.to_string(),
        kind: if is_test {
            NodeKind::Test
        } else {
            NodeKind::Function
        },
        label: label.to_string(),
        payload,
        centrality: 0.0,
        memory_id: None,
        namespace: None,
        valid_from: None,
        valid_to: None,
    }
}

/// Helper: create a Calls edge from caller to callee.
fn make_calls_edge(caller_id: &str, callee_id: &str) -> Edge {
    Edge {
        id: format!("calls:{caller_id}:{callee_id}"),
        src: caller_id.to_string(),
        dst: callee_id.to_string(),
        relationship: RelationshipType::Calls,
        weight: 1.0,
        properties: HashMap::new(),
        created_at: Utc::now(),
        valid_from: None,
        valid_to: None,
    }
}

#[test]
fn test_impact_classifies_direct_and_transitive() {
    let engine = CodememEngine::for_testing();

    // Build call chain:
    //   test_validate -> validate          (depth 1 from validate)
    //   test_login -> login -> validate    (depth 2 from validate)
    //   test_middleware -> middleware -> login -> validate  (depth 3 from validate)
    let validate = make_symbol_node("sym:validate", "validate", false);
    let login = make_symbol_node("sym:login", "login", false);
    let middleware = make_symbol_node("sym:middleware", "middleware", false);
    let test_validate = make_symbol_node("sym:test_validate", "test_validate", true);
    let test_login = make_symbol_node("sym:test_login", "test_login", true);
    let test_middleware = make_symbol_node("sym:test_middleware", "test_middleware", true);

    let edge1 = make_calls_edge("sym:test_validate", "sym:validate");
    let edge2 = make_calls_edge("sym:login", "sym:validate");
    let edge3 = make_calls_edge("sym:test_login", "sym:login");
    let edge4 = make_calls_edge("sym:middleware", "sym:login");
    let edge5 = make_calls_edge("sym:test_middleware", "sym:middleware");

    {
        let mut graph = engine.lock_graph().expect("lock graph");
        graph.add_node(validate).expect("add validate");
        graph.add_node(login).expect("add login");
        graph.add_node(middleware).expect("add middleware");
        graph
            .add_node(test_validate.clone())
            .expect("add test_validate");
        graph.add_node(test_login.clone()).expect("add test_login");
        graph
            .add_node(test_middleware.clone())
            .expect("add test_middleware");
        graph.add_edge(edge1).expect("add edge1");
        graph.add_edge(edge2).expect("add edge2");
        graph.add_edge(edge3).expect("add edge3");
        graph.add_edge(edge4).expect("add edge4");
        graph.add_edge(edge5).expect("add edge5");
    }

    let result = engine
        .test_impact(&["sym:validate"], 4)
        .expect("test_impact should succeed");

    // test_validate: depth 1 (direct)
    // test_login: depth 2 (direct)
    // test_middleware: depth 3 (transitive)
    assert_eq!(
        result.direct_tests.len(),
        2,
        "expected 2 direct tests, got: {:?}",
        result.direct_tests
    );
    assert_eq!(
        result.transitive_tests.len(),
        1,
        "expected 1 transitive test, got: {:?}",
        result.transitive_tests
    );

    // Check specific depths
    let direct_ids: Vec<&str> = result
        .direct_tests
        .iter()
        .map(|h| h.test_symbol.as_str())
        .collect();
    assert!(
        direct_ids.contains(&"sym:test_validate"),
        "test_validate should be direct"
    );
    assert!(
        direct_ids.contains(&"sym:test_login"),
        "test_login should be direct"
    );

    let transitive_ids: Vec<&str> = result
        .transitive_tests
        .iter()
        .map(|h| h.test_symbol.as_str())
        .collect();
    assert!(
        transitive_ids.contains(&"sym:test_middleware"),
        "test_middleware should be transitive"
    );

    // Check depths
    let tv = result
        .direct_tests
        .iter()
        .find(|h| h.test_symbol == "sym:test_validate")
        .unwrap();
    assert_eq!(tv.depth, 1, "test_validate should be at depth 1");

    let tl = result
        .direct_tests
        .iter()
        .find(|h| h.test_symbol == "sym:test_login")
        .unwrap();
    assert_eq!(tl.depth, 2, "test_login should be at depth 2");

    let tm = &result.transitive_tests[0];
    assert_eq!(tm.depth, 3, "test_middleware should be at depth 3");
}

#[test]
fn test_impact_deduplicates_across_symbols() {
    let engine = CodememEngine::for_testing();

    // Both sym:a and sym:b are called by test_both
    let a = make_symbol_node("sym:a", "a", false);
    let b = make_symbol_node("sym:b", "b", false);
    let test_both = make_symbol_node("sym:test_both", "test_both", true);

    let edge1 = make_calls_edge("sym:test_both", "sym:a");
    let edge2 = make_calls_edge("sym:test_both", "sym:b");

    {
        let mut graph = engine.lock_graph().expect("lock graph");
        graph.add_node(a).expect("add a");
        graph.add_node(b).expect("add b");
        graph.add_node(test_both).expect("add test_both");
        graph.add_edge(edge1).expect("add edge1");
        graph.add_edge(edge2).expect("add edge2");
    }

    let result = engine
        .test_impact(&["sym:a", "sym:b"], 4)
        .expect("test_impact");

    // test_both should appear only once despite being reachable from both symbols
    assert_eq!(
        result.direct_tests.len(),
        1,
        "should deduplicate: {:?}",
        result.direct_tests
    );
    assert_eq!(result.direct_tests[0].test_symbol, "sym:test_both");
    assert_eq!(result.direct_tests[0].depth, 1);
}

#[test]
fn test_impact_empty_graph() {
    let engine = CodememEngine::for_testing();
    let result = engine
        .test_impact(&["sym:nonexistent"], 4)
        .expect("test_impact on empty graph");
    assert!(result.direct_tests.is_empty());
    assert!(result.transitive_tests.is_empty());
}
