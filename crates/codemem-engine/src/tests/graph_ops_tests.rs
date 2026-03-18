use crate::CodememEngine;
use chrono::{Duration, Utc};
use codemem_core::{Edge, GraphNode, NodeKind, RelationshipType};
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
