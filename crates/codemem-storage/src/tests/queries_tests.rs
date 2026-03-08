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
    let stats_map: HashMap<String, usize> = stats.into_iter().collect();
    assert_eq!(stats_map.get("Read"), Some(&5));
    assert_eq!(stats_map.get("Grep"), Some(&3));
    assert_eq!(stats_map.get("Edit"), Some(&1));
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

// ── find_memory_ids_by_tag Tests ────────────────────────────────────

fn tagged_memory(content: &str, tags: Vec<String>, namespace: Option<String>) -> MemoryNode {
    let now = chrono::Utc::now();
    MemoryNode {
        id: uuid::Uuid::new_v4().to_string(),
        content: content.to_string(),
        memory_type: MemoryType::Context,
        importance: 0.5,
        confidence: 1.0,
        access_count: 0,
        content_hash: Storage::content_hash(content),
        tags,
        metadata: HashMap::new(),
        namespace,
        created_at: now,
        updated_at: now,
        last_accessed_at: now,
    }
}

#[test]
fn find_memory_ids_by_tag_basic_match() {
    let storage = Storage::open_in_memory().unwrap();
    let m1 = tagged_memory("tagged memory one", vec!["alpha".to_string()], None);
    let m2 = tagged_memory(
        "tagged memory two",
        vec!["alpha".to_string(), "beta".to_string()],
        None,
    );
    let m3 = tagged_memory("tagged memory three", vec!["beta".to_string()], None);
    storage.insert_memory(&m1).unwrap();
    storage.insert_memory(&m2).unwrap();
    storage.insert_memory(&m3).unwrap();

    let ids = storage
        .find_memory_ids_by_tag("alpha", None, "nonexistent")
        .unwrap();
    assert_eq!(ids.len(), 2);
    assert!(ids.contains(&m1.id));
    assert!(ids.contains(&m2.id));
}

#[test]
fn find_memory_ids_by_tag_empty_tags_not_matched() {
    let storage = Storage::open_in_memory().unwrap();
    let m1 = tagged_memory("no tags memory", vec![], None);
    storage.insert_memory(&m1).unwrap();

    let ids = storage
        .find_memory_ids_by_tag("anything", None, "nonexistent")
        .unwrap();
    assert!(ids.is_empty());
}

#[test]
fn find_memory_ids_by_tag_special_characters() {
    let storage = Storage::open_in_memory().unwrap();
    // Tags with %, _, and " which would break LIKE-based queries
    let m_percent = tagged_memory("percent tag", vec!["100%".to_string()], None);
    let m_underscore = tagged_memory("underscore tag", vec!["my_tag".to_string()], None);
    let m_quote = tagged_memory("quote tag", vec!["say\"hello".to_string()], None);
    storage.insert_memory(&m_percent).unwrap();
    storage.insert_memory(&m_underscore).unwrap();
    storage.insert_memory(&m_quote).unwrap();

    // json_each should handle these correctly
    let ids = storage
        .find_memory_ids_by_tag("100%", None, "nonexistent")
        .unwrap();
    assert_eq!(ids.len(), 1);
    assert_eq!(ids[0], m_percent.id);

    let ids = storage
        .find_memory_ids_by_tag("my_tag", None, "nonexistent")
        .unwrap();
    assert_eq!(ids.len(), 1);
    assert_eq!(ids[0], m_underscore.id);

    let ids = storage
        .find_memory_ids_by_tag("say\"hello", None, "nonexistent")
        .unwrap();
    assert_eq!(ids.len(), 1);
    assert_eq!(ids[0], m_quote.id);
}

#[test]
fn find_memory_ids_by_tag_limit_50_boundary() {
    let storage = Storage::open_in_memory().unwrap();
    let mut all_ids = Vec::new();
    for i in 0..55 {
        let m = tagged_memory(
            &format!("memory number {i}"),
            vec!["shared-tag".to_string()],
            None,
        );
        all_ids.push(m.id.clone());
        storage.insert_memory(&m).unwrap();
    }

    let ids = storage
        .find_memory_ids_by_tag("shared-tag", None, "nonexistent")
        .unwrap();
    assert_eq!(ids.len(), 50, "LIMIT 50 should cap results at 50");
}

#[test]
fn find_memory_ids_by_tag_namespace_isolation() {
    let storage = Storage::open_in_memory().unwrap();
    let m_ns_a = tagged_memory(
        "ns-a memory",
        vec!["shared".to_string()],
        Some("ns-a".to_string()),
    );
    let m_ns_b = tagged_memory(
        "ns-b memory",
        vec!["shared".to_string()],
        Some("ns-b".to_string()),
    );
    storage.insert_memory(&m_ns_a).unwrap();
    storage.insert_memory(&m_ns_b).unwrap();

    let ids = storage
        .find_memory_ids_by_tag("shared", Some("ns-a"), "nonexistent")
        .unwrap();
    assert_eq!(ids.len(), 1);
    assert_eq!(ids[0], m_ns_a.id);

    let ids = storage
        .find_memory_ids_by_tag("shared", Some("ns-b"), "nonexistent")
        .unwrap();
    assert_eq!(ids.len(), 1);
    assert_eq!(ids[0], m_ns_b.id);
}

#[test]
fn find_memory_ids_by_tag_exclude_id() {
    let storage = Storage::open_in_memory().unwrap();
    let m1 = tagged_memory("first tagged", vec!["mytag".to_string()], None);
    let m2 = tagged_memory("second tagged", vec!["mytag".to_string()], None);
    storage.insert_memory(&m1).unwrap();
    storage.insert_memory(&m2).unwrap();

    // Exclude m1's id
    let ids = storage
        .find_memory_ids_by_tag("mytag", None, &m1.id)
        .unwrap();
    assert_eq!(ids.len(), 1);
    assert_eq!(ids[0], m2.id);
}

#[test]
fn find_memory_ids_by_tag_null_namespace_handling() {
    let storage = Storage::open_in_memory().unwrap();
    let m_null_ns = tagged_memory("null ns memory", vec!["tag-x".to_string()], None);
    let m_with_ns = tagged_memory(
        "with ns memory",
        vec!["tag-x".to_string()],
        Some("proj".to_string()),
    );
    storage.insert_memory(&m_null_ns).unwrap();
    storage.insert_memory(&m_with_ns).unwrap();

    // When namespace is None, the query uses `namespace IS NULL`, so only null-ns memory matches
    let ids = storage
        .find_memory_ids_by_tag("tag-x", None, "nonexistent")
        .unwrap();
    assert_eq!(ids.len(), 1);
    assert_eq!(ids[0], m_null_ns.id);
}
