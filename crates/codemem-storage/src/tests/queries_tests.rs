use crate::Storage;
use codemem_core::{MemoryNode, MemoryType};
use std::collections::HashMap;

fn test_memory_with_metadata(
    content: &str,
    tool: &str,
    extra: HashMap<String, serde_json::Value>,
) -> MemoryNode {
    let now = chrono::Utc::now();
    let mut metadata = extra;
    metadata.insert(
        "tool".to_string(),
        serde_json::Value::String(tool.to_string()),
    );
    MemoryNode {
        id: uuid::Uuid::new_v4().to_string(),
        content: content.to_string(),
        memory_type: MemoryType::Context,
        importance: 0.5,
        confidence: 1.0,
        access_count: 0,
        content_hash: Storage::content_hash(content),
        tags: vec![],
        metadata,
        namespace: None,
        created_at: now,
        updated_at: now,
        last_accessed_at: now,
    }
}

#[test]
fn stats() {
    let storage = Storage::open_in_memory().unwrap();
    let stats = storage.stats().unwrap();
    assert_eq!(stats.memory_count, 0);
}

#[test]
fn get_repeated_searches_groups_by_pattern() {
    let storage = Storage::open_in_memory().unwrap();

    for i in 0..3 {
        let mut extra = HashMap::new();
        extra.insert(
            "pattern".to_string(),
            serde_json::Value::String("error".to_string()),
        );
        let mem = test_memory_with_metadata(&format!("grep search {i} for error"), "Grep", extra);
        storage.insert_memory(&mem).unwrap();
    }

    let mut extra = HashMap::new();
    extra.insert(
        "pattern".to_string(),
        serde_json::Value::String("*.rs".to_string()),
    );
    let mem = test_memory_with_metadata("glob search for rs files", "Glob", extra);
    storage.insert_memory(&mem).unwrap();

    let results = storage.get_repeated_searches(2, None).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, "error");
    assert_eq!(results[0].1, 3);
    assert_eq!(results[0].2.len(), 3);

    let results = storage.get_repeated_searches(1, None).unwrap();
    assert_eq!(results.len(), 2);
}

#[test]
fn get_file_hotspots_groups_by_file_path() {
    let storage = Storage::open_in_memory().unwrap();

    for i in 0..4 {
        let mut extra = HashMap::new();
        extra.insert(
            "file_path".to_string(),
            serde_json::Value::String("src/main.rs".to_string()),
        );
        let mem = test_memory_with_metadata(&format!("read main.rs attempt {i}"), "Read", extra);
        storage.insert_memory(&mem).unwrap();
    }

    let mut extra = HashMap::new();
    extra.insert(
        "file_path".to_string(),
        serde_json::Value::String("src/lib.rs".to_string()),
    );
    let mem = test_memory_with_metadata("read lib.rs", "Read", extra);
    storage.insert_memory(&mem).unwrap();

    let results = storage.get_file_hotspots(3, None).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, "src/main.rs");
    assert_eq!(results[0].1, 4);
}

#[test]
fn get_tool_usage_stats_counts_by_tool() {
    let storage = Storage::open_in_memory().unwrap();

    for i in 0..5 {
        let mem = test_memory_with_metadata(&format!("read file {i}"), "Read", HashMap::new());
        storage.insert_memory(&mem).unwrap();
    }
    for i in 0..3 {
        let mem = test_memory_with_metadata(&format!("grep search {i}"), "Grep", HashMap::new());
        storage.insert_memory(&mem).unwrap();
    }
    let mem = test_memory_with_metadata("edit file", "Edit", HashMap::new());
    storage.insert_memory(&mem).unwrap();

    let stats = storage.get_tool_usage_stats(None).unwrap();
    assert_eq!(stats.get("Read"), Some(&5));
    assert_eq!(stats.get("Grep"), Some(&3));
    assert_eq!(stats.get("Edit"), Some(&1));
}

#[test]
fn get_decision_chains_groups_edits_by_file() {
    let storage = Storage::open_in_memory().unwrap();

    for i in 0..3 {
        let mut extra = HashMap::new();
        extra.insert(
            "file_path".to_string(),
            serde_json::Value::String("src/main.rs".to_string()),
        );
        let mem = test_memory_with_metadata(&format!("edit main.rs {i}"), "Edit", extra);
        storage.insert_memory(&mem).unwrap();
    }

    let mut extra = HashMap::new();
    extra.insert(
        "file_path".to_string(),
        serde_json::Value::String("src/new.rs".to_string()),
    );
    let mem = test_memory_with_metadata("write new.rs", "Write", extra);
    storage.insert_memory(&mem).unwrap();

    let results = storage.get_decision_chains(2, None).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, "src/main.rs");
    assert_eq!(results[0].1, 3);
}

#[test]
fn pattern_queries_empty_db() {
    let storage = Storage::open_in_memory().unwrap();

    let searches = storage.get_repeated_searches(1, None).unwrap();
    assert!(searches.is_empty());

    let hotspots = storage.get_file_hotspots(1, None).unwrap();
    assert!(hotspots.is_empty());

    let stats = storage.get_tool_usage_stats(None).unwrap();
    assert!(stats.is_empty());

    let chains = storage.get_decision_chains(1, None).unwrap();
    assert!(chains.is_empty());
}

#[test]
fn pattern_queries_with_namespace_filter() {
    let storage = Storage::open_in_memory().unwrap();

    for i in 0..3 {
        let mut extra = HashMap::new();
        extra.insert(
            "pattern".to_string(),
            serde_json::Value::String("error".to_string()),
        );
        let mut mem = test_memory_with_metadata(&format!("ns-a grep {i}"), "Grep", extra);
        mem.namespace = Some("project-a".to_string());
        storage.insert_memory(&mem).unwrap();
    }

    for i in 0..2 {
        let mut extra = HashMap::new();
        extra.insert(
            "pattern".to_string(),
            serde_json::Value::String("error".to_string()),
        );
        let mut mem = test_memory_with_metadata(&format!("ns-b grep {i}"), "Grep", extra);
        mem.namespace = Some("project-b".to_string());
        storage.insert_memory(&mem).unwrap();
    }

    let results = storage.get_repeated_searches(1, None).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].1, 5);

    let results = storage.get_repeated_searches(1, Some("project-a")).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].1, 3);
}

// ── Session Management Tests ────────────────────────────────────────

#[test]
fn session_lifecycle() {
    let storage = Storage::open_in_memory().unwrap();

    storage.start_session("sess-1", Some("my-project")).unwrap();

    let sessions = storage.list_sessions(Some("my-project")).unwrap();
    assert_eq!(sessions.len(), 1);
    assert_eq!(sessions[0].id, "sess-1");
    assert_eq!(sessions[0].namespace, Some("my-project".to_string()));
    assert!(sessions[0].ended_at.is_none());

    storage
        .end_session("sess-1", Some("Explored the codebase"))
        .unwrap();

    let sessions = storage.list_sessions(None).unwrap();
    assert_eq!(sessions.len(), 1);
    assert!(sessions[0].ended_at.is_some());
    assert_eq!(
        sessions[0].summary,
        Some("Explored the codebase".to_string())
    );
}

#[test]
fn list_sessions_filters_by_namespace() {
    let storage = Storage::open_in_memory().unwrap();

    storage.start_session("sess-a", Some("project-a")).unwrap();
    storage.start_session("sess-b", Some("project-b")).unwrap();
    storage.start_session("sess-c", None).unwrap();

    let all = storage.list_sessions(None).unwrap();
    assert_eq!(all.len(), 3);

    let proj_a = storage.list_sessions(Some("project-a")).unwrap();
    assert_eq!(proj_a.len(), 1);
    assert_eq!(proj_a[0].id, "sess-a");
}

#[test]
fn start_session_ignores_duplicate() {
    let storage = Storage::open_in_memory().unwrap();
    storage.start_session("sess-1", Some("ns")).unwrap();
    storage.start_session("sess-1", Some("ns")).unwrap();

    let sessions = storage.list_sessions(None).unwrap();
    assert_eq!(sessions.len(), 1);
}
