use crate::cross_repo::{ApiEndpointEntry, UnresolvedRefEntry};
use crate::Storage;
use codemem_core::{Edge, GraphNode, NodeKind, RelationshipType};
use std::collections::HashMap;

// ── Migration Tests ──────────────────────────────────────────────────

#[test]
fn migration_creates_cross_repo_tables() {
    let storage = Storage::open_in_memory().unwrap();
    let conn = storage.conn().unwrap();

    // Verify all three tables exist
    for table in &["package_registry", "unresolved_refs", "api_endpoints"] {
        let exists: bool = conn
            .prepare(&format!("SELECT 1 FROM {table} LIMIT 0"))
            .is_ok();
        assert!(exists, "Table '{table}' should exist after migration");
    }
}

#[test]
fn migration_creates_indexes() {
    let storage = Storage::open_in_memory().unwrap();
    let conn = storage.conn().unwrap();

    let mut stmt = conn
        .prepare("SELECT name FROM sqlite_master WHERE type = 'index' AND sql IS NOT NULL")
        .unwrap();
    let index_names: Vec<String> = stmt
        .query_map([], |row| row.get(0))
        .unwrap()
        .filter_map(|r| r.ok())
        .collect();

    let expected = [
        "idx_unresolved_refs_pkg",
        "idx_unresolved_refs_ns",
        "idx_package_registry_ns",
        "idx_api_endpoints_path",
        "idx_api_endpoints_ns",
    ];
    for idx in &expected {
        assert!(
            index_names.contains(&idx.to_string()),
            "Index '{idx}' should exist. Found: {index_names:?}"
        );
    }
}

// ── Package Registry Tests ───────────────────────────────────────────

#[test]
fn upsert_package_registry_creates_and_updates() {
    let storage = Storage::open_in_memory().unwrap();

    storage
        .upsert_package_registry("serde", "my-lib", "1.0.0", "Cargo.toml")
        .unwrap();

    let entries = storage.get_packages_for_namespace("my-lib").unwrap();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].package_name, "serde");
    assert_eq!(entries[0].version, "1.0.0");

    // Update version
    storage
        .upsert_package_registry("serde", "my-lib", "2.0.0", "Cargo.toml")
        .unwrap();

    let entries = storage.get_packages_for_namespace("my-lib").unwrap();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].version, "2.0.0");
}

#[test]
fn find_namespace_for_package_returns_matches() {
    let storage = Storage::open_in_memory().unwrap();

    storage
        .upsert_package_registry("shared-utils", "repo-a", "1.0.0", "")
        .unwrap();
    storage
        .upsert_package_registry("shared-utils", "repo-b", "2.0.0", "")
        .unwrap();
    storage
        .upsert_package_registry("other-pkg", "repo-c", "0.1.0", "")
        .unwrap();

    let matches = storage.find_namespace_for_package("shared-utils").unwrap();
    assert_eq!(matches.len(), 2);

    let namespaces: Vec<&str> = matches.iter().map(|e| e.namespace.as_str()).collect();
    assert!(namespaces.contains(&"repo-a"));
    assert!(namespaces.contains(&"repo-b"));
}

#[test]
fn delete_package_registry_for_namespace_returns_count() {
    let storage = Storage::open_in_memory().unwrap();

    storage
        .upsert_package_registry("pkg-a", "ns1", "1.0", "")
        .unwrap();
    storage
        .upsert_package_registry("pkg-b", "ns1", "1.0", "")
        .unwrap();
    storage
        .upsert_package_registry("pkg-c", "ns2", "1.0", "")
        .unwrap();

    let deleted = storage
        .delete_package_registry_for_namespace("ns1")
        .unwrap();
    assert_eq!(deleted, 2);

    let remaining = storage.get_packages_for_namespace("ns1").unwrap();
    assert!(remaining.is_empty());

    // ns2 should be untouched
    let ns2 = storage.get_packages_for_namespace("ns2").unwrap();
    assert_eq!(ns2.len(), 1);
}

// ── Unresolved Refs Tests ────────────────────────────────────────────

fn make_ref(id: &str, namespace: &str, package_hint: Option<&str>) -> UnresolvedRefEntry {
    UnresolvedRefEntry {
        id: id.to_string(),
        namespace: namespace.to_string(),
        source_node: format!("node:{id}"),
        target_name: format!("Target{id}"),
        package_hint: package_hint.map(|s| s.to_string()),
        ref_kind: "IMPORTS".to_string(),
        file_path: Some("src/main.rs".to_string()),
        line: Some(42),
        created_at: 1700000000,
    }
}

#[test]
fn insert_and_get_unresolved_ref() {
    let storage = Storage::open_in_memory().unwrap();
    let entry = make_ref("ref1", "ns1", Some("serde"));

    storage.insert_unresolved_ref(&entry).unwrap();

    let refs = storage.get_unresolved_refs_for_namespace("ns1").unwrap();
    assert_eq!(refs.len(), 1);
    assert_eq!(refs[0].id, "ref1");
    assert_eq!(refs[0].package_hint.as_deref(), Some("serde"));
}

#[test]
fn get_unresolved_refs_for_package_hint() {
    let storage = Storage::open_in_memory().unwrap();

    storage
        .insert_unresolved_ref(&make_ref("r1", "ns1", Some("tokio")))
        .unwrap();
    storage
        .insert_unresolved_ref(&make_ref("r2", "ns2", Some("tokio")))
        .unwrap();
    storage
        .insert_unresolved_ref(&make_ref("r3", "ns1", Some("serde")))
        .unwrap();

    let refs = storage
        .get_unresolved_refs_for_package_hint("tokio")
        .unwrap();
    assert_eq!(refs.len(), 2);
}

#[test]
fn insert_unresolved_refs_batch_handles_many_entries() {
    let storage = Storage::open_in_memory().unwrap();

    // Create 50 entries — enough to verify batch splitting works
    // (9 cols * 50 = 450 params, under 999 but tests the path)
    let refs: Vec<UnresolvedRefEntry> = (0..50)
        .map(|i| make_ref(&format!("batch-{i}"), "ns1", Some("pkg")))
        .collect();

    storage.insert_unresolved_refs_batch(&refs).unwrap();

    let retrieved = storage.get_unresolved_refs_for_namespace("ns1").unwrap();
    assert_eq!(retrieved.len(), 50);
}

#[test]
fn insert_unresolved_refs_batch_splits_correctly() {
    let storage = Storage::open_in_memory().unwrap();

    // 9 cols per row, batch size = 999/9 = 111
    // Create 250 entries to force at least 3 batches (111 + 111 + 28)
    let refs: Vec<UnresolvedRefEntry> = (0..250)
        .map(|i| make_ref(&format!("split-{i}"), "ns1", None))
        .collect();

    storage.insert_unresolved_refs_batch(&refs).unwrap();

    let retrieved = storage.get_unresolved_refs_for_namespace("ns1").unwrap();
    assert_eq!(retrieved.len(), 250);
}

#[test]
fn delete_unresolved_refs_batch_removes_correct_entries() {
    let storage = Storage::open_in_memory().unwrap();

    let refs: Vec<UnresolvedRefEntry> = (0..5)
        .map(|i| make_ref(&format!("del-{i}"), "ns1", None))
        .collect();
    storage.insert_unresolved_refs_batch(&refs).unwrap();

    let to_delete: Vec<String> = vec![
        "del-0".to_string(),
        "del-2".to_string(),
        "del-4".to_string(),
    ];
    storage.delete_unresolved_refs_batch(&to_delete).unwrap();

    let remaining = storage.get_unresolved_refs_for_namespace("ns1").unwrap();
    assert_eq!(remaining.len(), 2);

    let remaining_ids: Vec<&str> = remaining.iter().map(|r| r.id.as_str()).collect();
    assert!(remaining_ids.contains(&"del-1"));
    assert!(remaining_ids.contains(&"del-3"));
}

#[test]
fn delete_unresolved_refs_for_namespace_returns_count() {
    let storage = Storage::open_in_memory().unwrap();

    let refs: Vec<UnresolvedRefEntry> = (0..3)
        .map(|i| make_ref(&format!("ns-del-{i}"), "target-ns", None))
        .collect();
    storage.insert_unresolved_refs_batch(&refs).unwrap();
    storage
        .insert_unresolved_ref(&make_ref("other", "other-ns", None))
        .unwrap();

    let deleted = storage
        .delete_unresolved_refs_for_namespace("target-ns")
        .unwrap();
    assert_eq!(deleted, 3);

    let remaining = storage
        .get_unresolved_refs_for_namespace("other-ns")
        .unwrap();
    assert_eq!(remaining.len(), 1);
}

// ── API Endpoint Tests ───────────────────────────────────────────────

fn make_endpoint(id: &str, namespace: &str, method: &str, path: &str) -> ApiEndpointEntry {
    ApiEndpointEntry {
        id: id.to_string(),
        namespace: namespace.to_string(),
        method: Some(method.to_string()),
        path: path.to_string(),
        handler: Some(format!("handler_{id}")),
        schema: "{}".to_string(),
    }
}

#[test]
fn upsert_api_endpoint_creates_and_updates() {
    let storage = Storage::open_in_memory().unwrap();

    let ep = make_endpoint("ep1", "svc-a", "GET", "/api/users");
    storage.upsert_api_endpoint(&ep).unwrap();

    let eps = storage.get_api_endpoints_for_namespace("svc-a").unwrap();
    assert_eq!(eps.len(), 1);
    assert_eq!(eps[0].path, "/api/users");

    // Update handler
    let updated = ApiEndpointEntry {
        handler: Some("new_handler".to_string()),
        ..ep
    };
    storage.upsert_api_endpoint(&updated).unwrap();

    let eps = storage.get_api_endpoints_for_namespace("svc-a").unwrap();
    assert_eq!(eps.len(), 1);
    assert_eq!(eps[0].handler.as_deref(), Some("new_handler"));
}

#[test]
fn get_api_endpoints_for_path() {
    let storage = Storage::open_in_memory().unwrap();

    storage
        .upsert_api_endpoint(&make_endpoint("e1", "svc-a", "GET", "/api/users"))
        .unwrap();
    storage
        .upsert_api_endpoint(&make_endpoint("e2", "svc-b", "POST", "/api/users"))
        .unwrap();
    storage
        .upsert_api_endpoint(&make_endpoint("e3", "svc-a", "GET", "/api/items"))
        .unwrap();

    let eps = storage.get_api_endpoints_for_path("/api/users").unwrap();
    assert_eq!(eps.len(), 2);
}

#[test]
fn find_api_endpoints_by_path_pattern() {
    let storage = Storage::open_in_memory().unwrap();

    storage
        .upsert_api_endpoint(&make_endpoint("e1", "svc-a", "GET", "/api/users"))
        .unwrap();
    storage
        .upsert_api_endpoint(&make_endpoint("e2", "svc-a", "GET", "/api/users/123"))
        .unwrap();
    storage
        .upsert_api_endpoint(&make_endpoint("e3", "svc-a", "GET", "/health"))
        .unwrap();

    let eps = storage
        .find_api_endpoints_by_path_pattern("/api/users%")
        .unwrap();
    assert_eq!(eps.len(), 2);

    let eps = storage
        .find_api_endpoints_by_path_pattern("/health")
        .unwrap();
    assert_eq!(eps.len(), 1);
}

#[test]
fn delete_api_endpoints_for_namespace_returns_count() {
    let storage = Storage::open_in_memory().unwrap();

    storage
        .upsert_api_endpoint(&make_endpoint("e1", "svc-a", "GET", "/a"))
        .unwrap();
    storage
        .upsert_api_endpoint(&make_endpoint("e2", "svc-a", "POST", "/b"))
        .unwrap();
    storage
        .upsert_api_endpoint(&make_endpoint("e3", "svc-b", "GET", "/c"))
        .unwrap();

    let deleted = storage.delete_api_endpoints_for_namespace("svc-a").unwrap();
    assert_eq!(deleted, 2);

    let remaining = storage.get_api_endpoints_for_namespace("svc-b").unwrap();
    assert_eq!(remaining.len(), 1);
}

// ── Cross-namespace Edge Tests ───────────────────────────────────────

#[test]
fn get_cross_namespace_edges_returns_spanning_edges() {
    let storage = Storage::open_in_memory().unwrap();

    // Create nodes in two different namespaces
    let node_a = GraphNode {
        id: "node:a".to_string(),
        kind: NodeKind::Function,
        label: "func_a".to_string(),
        payload: HashMap::new(),
        centrality: 0.0,
        memory_id: None,
        namespace: Some("ns-alpha".to_string()),
        valid_from: None,
        valid_to: None,
    };
    let node_b = GraphNode {
        id: "node:b".to_string(),
        kind: NodeKind::Function,
        label: "func_b".to_string(),
        payload: HashMap::new(),
        centrality: 0.0,
        memory_id: None,
        namespace: Some("ns-beta".to_string()),
        valid_from: None,
        valid_to: None,
    };
    let node_c = GraphNode {
        id: "node:c".to_string(),
        kind: NodeKind::Function,
        label: "func_c".to_string(),
        payload: HashMap::new(),
        centrality: 0.0,
        memory_id: None,
        namespace: Some("ns-alpha".to_string()),
        valid_from: None,
        valid_to: None,
    };

    storage.insert_graph_node(&node_a).unwrap();
    storage.insert_graph_node(&node_b).unwrap();
    storage.insert_graph_node(&node_c).unwrap();

    // Edge within ns-alpha (should NOT appear)
    let intra_edge = Edge {
        id: "edge:intra".to_string(),
        src: "node:a".to_string(),
        dst: "node:c".to_string(),
        relationship: RelationshipType::Calls,
        weight: 1.0,
        properties: HashMap::new(),
        created_at: chrono::Utc::now(),
        valid_from: None,
        valid_to: None,
    };

    // Edge from ns-alpha to ns-beta with cross_namespace property (should appear)
    let cross_edge = Edge {
        id: "edge:cross".to_string(),
        src: "node:a".to_string(),
        dst: "node:b".to_string(),
        relationship: RelationshipType::DependsOn,
        weight: 1.0,
        properties: {
            let mut props = HashMap::new();
            props.insert("cross_namespace".to_string(), serde_json::Value::Bool(true));
            props
        },
        created_at: chrono::Utc::now(),
        valid_from: None,
        valid_to: None,
    };

    storage.insert_graph_edge(&intra_edge).unwrap();
    storage.insert_graph_edge(&cross_edge).unwrap();

    let cross_edges = storage.get_cross_namespace_edges("ns-alpha").unwrap();
    assert_eq!(cross_edges.len(), 1);
    assert_eq!(cross_edges[0].id, "edge:cross");

    // Also visible from the other namespace
    let cross_edges_beta = storage.get_cross_namespace_edges("ns-beta").unwrap();
    assert_eq!(cross_edges_beta.len(), 1);
    assert_eq!(cross_edges_beta[0].id, "edge:cross");
}

#[test]
fn get_cross_namespace_edges_empty_when_no_cross_edges() {
    let storage = Storage::open_in_memory().unwrap();

    let node = GraphNode {
        id: "node:solo".to_string(),
        kind: NodeKind::File,
        label: "solo.rs".to_string(),
        payload: HashMap::new(),
        centrality: 0.0,
        memory_id: None,
        namespace: Some("isolated".to_string()),
        valid_from: None,
        valid_to: None,
    };
    storage.insert_graph_node(&node).unwrap();

    let edges = storage.get_cross_namespace_edges("isolated").unwrap();
    assert!(edges.is_empty());
}
