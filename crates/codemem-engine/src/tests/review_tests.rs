use crate::review::parse_diff;
use crate::CodememEngine;
use codemem_core::{Edge, GraphNode, NodeKind, RelationshipType};
use std::collections::HashMap;

fn make_sym_node(id: &str, file_path: &str, line_start: u32, line_end: u32) -> GraphNode {
    let mut payload = HashMap::new();
    payload.insert("file_path".to_string(), serde_json::json!(file_path));
    payload.insert("line_start".to_string(), serde_json::json!(line_start));
    payload.insert("line_end".to_string(), serde_json::json!(line_end));
    GraphNode {
        id: format!("sym:{id}"),
        kind: NodeKind::Function,
        label: id.to_string(),
        payload,
        centrality: 0.0,
        memory_id: None,
        namespace: None,
        valid_from: None,
        valid_to: None,
    }
}

fn make_edge(src: &str, dst: &str, rel: RelationshipType) -> Edge {
    Edge {
        id: format!("{src}->{dst}"),
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

const SAMPLE_DIFF: &str = r#"diff --git a/src/auth.rs b/src/auth.rs
index abc1234..def5678 100644
--- a/src/auth.rs
+++ b/src/auth.rs
@@ -10,6 +10,7 @@ fn validate_token(token: &str) -> bool {
     let decoded = decode(token);
     if decoded.is_expired() {
         return false;
+        log::warn!("expired token");
     }
     true
 }
"#;

// ── parse_diff tests ──────────────────────────────────────────────────

#[test]
fn parse_diff_extracts_file_and_lines() {
    let hunks = parse_diff(SAMPLE_DIFF);
    assert_eq!(hunks.len(), 1);
    assert_eq!(hunks[0].file_path, "src/auth.rs");
    assert_eq!(hunks[0].added_lines, vec![13]); // Line 13 was added
}

#[test]
fn parse_diff_multiple_files() {
    let diff = r#"diff --git a/src/a.rs b/src/a.rs
--- a/src/a.rs
+++ b/src/a.rs
@@ -1,3 +1,4 @@
 fn foo() {
+    bar();
 }
diff --git a/src/b.rs b/src/b.rs
--- a/src/b.rs
+++ b/src/b.rs
@@ -5,3 +5,4 @@
 fn baz() {
+    qux();
 }
"#;
    let hunks = parse_diff(diff);
    assert_eq!(hunks.len(), 2);
    assert_eq!(hunks[0].file_path, "src/a.rs");
    assert_eq!(hunks[1].file_path, "src/b.rs");
}

#[test]
fn parse_diff_empty() {
    let hunks = parse_diff("");
    assert!(hunks.is_empty());
}

// ── diff_to_symbols tests ─────────────────────────────────────────────

#[test]
fn diff_to_symbols_finds_changed_symbol() {
    let engine = CodememEngine::for_testing();

    // Add a symbol node covering lines 10-16 in src/auth.rs
    {
        let mut graph = engine.lock_graph().unwrap();
        let node = make_sym_node("auth::validate_token", "src/auth.rs", 10, 16);
        graph.add_node(node).unwrap();
    }

    let mapping = engine.diff_to_symbols(SAMPLE_DIFF).unwrap();
    assert!(
        mapping
            .changed_symbols
            .contains(&"sym:auth::validate_token".to_string()),
        "Should find validate_token as changed"
    );
    assert!(
        mapping
            .changed_files
            .contains(&"file:src/auth.rs".to_string()),
        "Should include the changed file"
    );
}

#[test]
fn diff_to_symbols_skips_unrelated_symbol() {
    let engine = CodememEngine::for_testing();

    {
        let mut graph = engine.lock_graph().unwrap();
        // Symbol on lines 50-60 — not touched by diff (which modifies line 13)
        let node = make_sym_node("auth::other_func", "src/auth.rs", 50, 60);
        graph.add_node(node).unwrap();
    }

    let mapping = engine.diff_to_symbols(SAMPLE_DIFF).unwrap();
    assert!(
        !mapping
            .changed_symbols
            .contains(&"sym:auth::other_func".to_string()),
        "Should not include unrelated symbol"
    );
}

#[test]
fn diff_to_symbols_finds_containing_parent() {
    let engine = CodememEngine::for_testing();

    {
        let mut graph = engine.lock_graph().unwrap();
        // Child symbol on changed line
        let child = make_sym_node("auth::validate_token", "src/auth.rs", 10, 16);
        graph.add_node(child).unwrap();
        // Parent module containing the child
        let mut parent = make_sym_node("auth", "src/auth.rs", 1, 50);
        parent.kind = NodeKind::Module;
        graph.add_node(parent).unwrap();
        // CONTAINS edge
        let edge = make_edge(
            "sym:auth",
            "sym:auth::validate_token",
            RelationshipType::Contains,
        );
        graph.add_edge(edge).unwrap();
    }

    let mapping = engine.diff_to_symbols(SAMPLE_DIFF).unwrap();
    // The parent module (lines 1-50) also covers the changed line 13, so it
    // appears in changed_symbols directly. containing_symbols only holds parents
    // that DON'T overlap the changed lines themselves.
    assert!(
        mapping.changed_symbols.contains(&"sym:auth".to_string())
            || mapping.containing_symbols.contains(&"sym:auth".to_string()),
        "Should find parent module as changed or containing symbol"
    );
}

// ── blast_radius tests ────────────────────────────────────────────────

#[test]
fn blast_radius_finds_direct_dependents() {
    let engine = CodememEngine::for_testing();

    {
        let mut graph = engine.lock_graph().unwrap();
        // Changed symbol
        let changed = make_sym_node("auth::validate_token", "src/auth.rs", 10, 16);
        graph.add_node(changed).unwrap();
        // Dependent that calls the changed symbol
        let caller = make_sym_node("api::handler", "src/api.rs", 20, 30);
        graph.add_node(caller).unwrap();
        // CALLS edge: api::handler -> auth::validate_token
        let edge = make_edge(
            "sym:api::handler",
            "sym:auth::validate_token",
            RelationshipType::Calls,
        );
        graph.add_edge(edge).unwrap();
    }

    let report = engine.blast_radius(SAMPLE_DIFF, 2).unwrap();
    assert!(
        !report.changed_symbols.is_empty(),
        "Should have changed symbols"
    );
    assert!(
        report
            .direct_dependents
            .iter()
            .any(|d| d.id == "sym:api::handler"),
        "Should find api::handler as direct dependent"
    );
    assert!(
        report.affected_files.contains(&"src/api.rs".to_string()),
        "Should include dependent's file in affected files"
    );
    assert!(
        report.risk_score >= 0.0,
        "Risk score should be non-negative"
    );
}

#[test]
fn blast_radius_empty_diff() {
    let engine = CodememEngine::for_testing();
    let report = engine.blast_radius("", 2).unwrap();
    assert!(report.changed_symbols.is_empty());
    assert!(report.direct_dependents.is_empty());
    assert_eq!(report.risk_score, 0.0);
}
