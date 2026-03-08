use super::*;

// ── namespace_from_cwd ──────────────────────────────────────────────

#[test]
fn namespace_from_cwd_absolute_path() {
    assert_eq!(namespace_from_cwd("/Users/me/project"), "project");
}

#[test]
fn namespace_from_cwd_nested_path() {
    assert_eq!(namespace_from_cwd("/home/user/deep/nested/repo"), "repo");
}

#[test]
fn namespace_from_cwd_single_component() {
    assert_eq!(namespace_from_cwd("myproject"), "myproject");
}

#[test]
fn namespace_from_cwd_trailing_slash() {
    // Path::file_name() strips trailing slashes on Unix
    assert_eq!(namespace_from_cwd("/Users/me/project/"), "project");
}

#[test]
fn namespace_from_cwd_root() {
    // Path::new("/").file_name() returns None → unwrap_or(cwd) returns "/"
    // This is a known edge case: "/" is a poor namespace but won't panic.
    assert_eq!(namespace_from_cwd("/"), "/");
}

// ── short_path ──────────────────────────────────────────────────────

#[test]
fn short_path_absolute() {
    assert_eq!(short_path("/home/user/project/src/main.rs"), "src/main.rs");
}

#[test]
fn short_path_relative() {
    assert_eq!(short_path("src/main.rs"), "src/main.rs");
}

#[test]
fn short_path_single_component() {
    assert_eq!(short_path("main.rs"), "main.rs");
}

#[test]
fn short_path_empty() {
    assert_eq!(short_path(""), "");
}

#[test]
fn short_path_deeply_nested() {
    assert_eq!(short_path("/a/b/c/d/e/f.rs"), "e/f.rs");
}

// ── cmd_sessions_list ───────────────────────────────────────────────

#[test]
fn sessions_list_empty() {
    let storage = codemem_storage::Storage::open_in_memory().unwrap();
    // Verify the storage query returns empty (cmd_sessions_list prints "No sessions recorded")
    let sessions = storage.list_sessions(None).unwrap();
    assert!(sessions.is_empty());
    cmd_sessions_list(&storage, None).unwrap();
}

#[test]
fn sessions_list_after_start() {
    let storage = codemem_storage::Storage::open_in_memory().unwrap();
    storage.start_session("sess-1", Some("test-ns")).unwrap();

    let sessions = storage.list_sessions(None).unwrap();
    assert_eq!(sessions.len(), 1);
    assert_eq!(sessions[0].id, "sess-1");
    cmd_sessions_list(&storage, None).unwrap();
}

#[test]
fn sessions_list_with_namespace_filter() {
    let storage = codemem_storage::Storage::open_in_memory().unwrap();
    storage.start_session("sess-a", Some("ns-a")).unwrap();
    storage.start_session("sess-b", Some("ns-b")).unwrap();

    // Verify namespace filtering: only ns-a sessions returned
    let filtered = storage.list_sessions(Some("ns-a")).unwrap();
    assert_eq!(filtered.len(), 1);
    assert_eq!(filtered[0].id, "sess-a");

    // Verify unfiltered returns both
    let all = storage.list_sessions(None).unwrap();
    assert_eq!(all.len(), 2);

    cmd_sessions_list(&storage, Some("ns-a")).unwrap();
}

// ── cmd_sessions_start ──────────────────────────────────────────────

#[test]
fn sessions_start_creates_session() {
    let storage = codemem_storage::Storage::open_in_memory().unwrap();
    cmd_sessions_start(&storage, Some("my-project")).unwrap();

    let sessions = storage.list_sessions(Some("my-project")).unwrap();
    assert_eq!(sessions.len(), 1);
    assert_eq!(sessions[0].namespace.as_deref(), Some("my-project"));
}

#[test]
fn sessions_start_no_namespace() {
    let storage = codemem_storage::Storage::open_in_memory().unwrap();
    cmd_sessions_start(&storage, None).unwrap();

    let sessions = storage.list_sessions(None).unwrap();
    assert!(!sessions.is_empty());
}

// ── cmd_sessions_end ────────────────────────────────────────────────

#[test]
fn sessions_end_with_summary() {
    let storage = codemem_storage::Storage::open_in_memory().unwrap();
    storage.start_session("end-test", Some("ns")).unwrap();

    cmd_sessions_end(&storage, "end-test", Some("all done")).unwrap();

    let sessions = storage.list_sessions(Some("ns")).unwrap();
    assert_eq!(sessions.len(), 1);
    assert!(sessions[0].ended_at.is_some());
    assert_eq!(sessions[0].summary.as_deref(), Some("all done"));
}

#[test]
fn sessions_end_without_summary() {
    let storage = codemem_storage::Storage::open_in_memory().unwrap();
    storage.start_session("end-no-sum", None).unwrap();

    cmd_sessions_end(&storage, "end-no-sum", None).unwrap();

    let sessions = storage.list_sessions(None).unwrap();
    let session = sessions.iter().find(|s| s.id == "end-no-sum").unwrap();
    assert!(session.ended_at.is_some());
}

// ── cmd_context ─────────────────────────────────────────────────────

#[test]
fn context_empty_storage_outputs_empty_json() {
    let storage = codemem_storage::Storage::open_in_memory().unwrap();
    // cmd_context reads from stdin, so we can't easily call it directly.
    // But we can verify the storage-layer queries it depends on return empty.
    let sessions = storage.list_sessions(None).unwrap();
    assert!(sessions.is_empty());
    let ids = storage.list_memory_ids().unwrap();
    assert!(ids.is_empty());
}

#[test]
fn context_with_sessions_and_memories() {
    let storage = codemem_storage::Storage::open_in_memory().unwrap();

    // Create a session with summary
    storage
        .start_session("ctx-sess-1", Some("myproject"))
        .unwrap();
    storage
        .end_session("ctx-sess-1", Some("explored auth module"))
        .unwrap();

    // Create Decision and Insight memories that cmd_context looks for
    let now = chrono::Utc::now();
    for (i, mtype) in [
        codemem_core::MemoryType::Decision,
        codemem_core::MemoryType::Insight,
        codemem_core::MemoryType::Pattern,
        codemem_core::MemoryType::Context, // not a high-signal type — excluded from the count below
    ]
    .iter()
    .enumerate()
    {
        let memory = codemem_core::MemoryNode {
            id: format!("ctx-mem-{i}"),
            content: format!("memory {i} of type {mtype}"),
            memory_type: *mtype,
            importance: 0.7,
            confidence: 0.9,
            access_count: 0,
            content_hash: format!("hash-{i}"),
            tags: vec!["test".to_string()],
            metadata: std::collections::HashMap::new(),
            namespace: Some("myproject".to_string()),
            created_at: now,
            updated_at: now,
            last_accessed_at: now,
        };
        storage.insert_memory(&memory).unwrap();
    }

    // Verify the queries cmd_context uses work correctly
    let sessions = storage.list_sessions(Some("myproject")).unwrap();
    assert_eq!(sessions.len(), 1);
    assert_eq!(sessions[0].summary.as_deref(), Some("explored auth module"));

    let ids = storage.list_memory_ids_for_namespace("myproject").unwrap();
    assert_eq!(ids.len(), 4);

    // Verify the type filtering logic that cmd_context applies
    let mut decision_insight_count = 0;
    for id in &ids {
        if let Some(m) = storage.get_memory_no_touch(id).unwrap() {
            if matches!(
                m.memory_type,
                codemem_core::MemoryType::Decision
                    | codemem_core::MemoryType::Insight
                    | codemem_core::MemoryType::Pattern
            ) {
                decision_insight_count += 1;
            }
        }
    }
    assert_eq!(
        decision_insight_count, 3,
        "should find 3 high-signal memories"
    );
}

#[test]
fn context_pending_analysis_detection() {
    let storage = codemem_storage::Storage::open_in_memory().unwrap();

    let now = chrono::Utc::now();
    let memory = codemem_core::MemoryNode {
        id: "pending-1".to_string(),
        content: "Files modified in session abc: src/main.rs, src/lib.rs".to_string(),
        memory_type: codemem_core::MemoryType::Context,
        importance: 0.4,
        confidence: 1.0,
        access_count: 0,
        content_hash: "hash-pending".to_string(),
        tags: vec!["pending-analysis".to_string(), "file-changes".to_string()],
        metadata: {
            let mut m = std::collections::HashMap::new();
            m.insert(
                "files".to_string(),
                serde_json::json!(["src/main.rs", "src/lib.rs"]),
            );
            m
        },
        namespace: Some("myproject".to_string()),
        created_at: now,
        updated_at: now,
        last_accessed_at: now,
    };
    storage.insert_memory(&memory).unwrap();

    // Verify the pending analysis extraction logic
    let ids = storage.list_memory_ids_for_namespace("myproject").unwrap();
    let mut pending_files: Vec<String> = Vec::new();
    for id in ids.iter().rev().take(100) {
        if let Ok(Some(m)) = storage.get_memory_no_touch(id) {
            if m.tags.contains(&"pending-analysis".to_string()) {
                if let Some(files) = m.metadata.get("files").and_then(|v| v.as_array()) {
                    for f in files {
                        if let Some(fp) = f.as_str() {
                            if !pending_files.contains(&fp.to_string()) {
                                pending_files.push(fp.to_string());
                            }
                        }
                    }
                }
            }
        }
    }
    assert_eq!(pending_files.len(), 2);
    assert!(pending_files.contains(&"src/main.rs".to_string()));
    assert!(pending_files.contains(&"src/lib.rs".to_string()));
}

// ── categorize_memories ─────────────────────────────────────────────

fn make_tool_memory(id: &str, tool: &str, file_path: &str) -> codemem_core::MemoryNode {
    let now = chrono::Utc::now();
    let mut metadata = std::collections::HashMap::new();
    metadata.insert("tool".into(), serde_json::json!(tool));
    if !file_path.is_empty() {
        metadata.insert("file_path".into(), serde_json::json!(file_path));
    }
    codemem_core::MemoryNode {
        id: id.into(),
        content: format!("{tool} {file_path}"),
        memory_type: codemem_core::MemoryType::Context,
        importance: 0.3,
        confidence: 1.0,
        access_count: 0,
        content_hash: format!("h-{id}"),
        tags: vec![],
        metadata,
        namespace: None,
        created_at: now,
        updated_at: now,
        last_accessed_at: now,
    }
}

#[test]
fn categorize_memories_read_edit_grep() {
    let memories = vec![
        make_tool_memory("m1", "Read", "src/main.rs"),
        make_tool_memory("m2", "Edit", "src/lib.rs"),
        {
            let mut m = make_tool_memory("m3", "Grep", "");
            m.metadata
                .insert("pattern".into(), serde_json::json!("TODO"));
            m
        },
    ];

    let cat = categorize_memories(&memories);
    assert_eq!(cat.files_read, vec!["src/main.rs"]);
    assert_eq!(cat.files_edited, vec!["src/lib.rs"]);
    assert_eq!(cat.searches, vec!["TODO"]);
    assert!(cat.decisions.is_empty());
    assert!(cat.prompts.is_empty());
}

#[test]
fn categorize_memories_write_tool() {
    let memories = vec![make_tool_memory("m1", "Write", "src/new.rs")];
    let cat = categorize_memories(&memories);
    assert_eq!(cat.files_edited, vec!["src/new.rs"]);
}

#[test]
fn categorize_memories_glob_tool() {
    let mut m = make_tool_memory("m1", "Glob", "");
    m.metadata
        .insert("pattern".into(), serde_json::json!("**/*.rs"));
    let cat = categorize_memories(&[m]);
    assert_eq!(cat.searches, vec!["**/*.rs"]);
}

#[test]
fn categorize_memories_decisions() {
    let now = chrono::Utc::now();
    let memory = codemem_core::MemoryNode {
        id: "d1".into(),
        content: "Decided to use Postgres".into(),
        memory_type: codemem_core::MemoryType::Decision,
        importance: 0.8,
        confidence: 0.9,
        access_count: 0,
        content_hash: "hd".into(),
        tags: vec![],
        metadata: std::collections::HashMap::new(),
        namespace: None,
        created_at: now,
        updated_at: now,
        last_accessed_at: now,
    };
    let cat = categorize_memories(&[memory]);
    assert_eq!(cat.decisions, vec!["Decided to use Postgres"]);
}

#[test]
fn categorize_memories_prompts() {
    let now = chrono::Utc::now();
    let mut metadata = std::collections::HashMap::new();
    metadata.insert("source".into(), serde_json::json!("UserPromptSubmit"));
    let memory = codemem_core::MemoryNode {
        id: "p1".into(),
        content: "User prompt: fix the auth bug".into(),
        memory_type: codemem_core::MemoryType::Context,
        importance: 0.3,
        confidence: 1.0,
        access_count: 0,
        content_hash: "hp".into(),
        tags: vec!["prompt".to_string()],
        metadata,
        namespace: None,
        created_at: now,
        updated_at: now,
        last_accessed_at: now,
    };
    let cat = categorize_memories(&[memory]);
    assert_eq!(cat.prompts, vec!["fix the auth bug"]);
}

#[test]
fn categorize_memories_deduplicates_files() {
    let memories = vec![
        make_tool_memory("m1", "Read", "src/main.rs"),
        make_tool_memory("m2", "Read", "src/main.rs"), // duplicate
        make_tool_memory("m3", "Edit", "src/lib.rs"),
        make_tool_memory("m4", "Edit", "src/lib.rs"), // duplicate
    ];
    let cat = categorize_memories(&memories);
    assert_eq!(cat.files_read.len(), 1);
    assert_eq!(cat.files_edited.len(), 1);
}

#[test]
fn categorize_memories_empty_file_path_skipped() {
    let memories = vec![make_tool_memory("m1", "Read", "")];
    let cat = categorize_memories(&memories);
    assert!(cat.files_read.is_empty());
}

#[test]
fn categorize_memories_decision_and_prompt_overlap() {
    // A memory that is both Decision type AND has source=UserPromptSubmit
    // should appear in both decisions and prompts.
    let now = chrono::Utc::now();
    let mut metadata = std::collections::HashMap::new();
    metadata.insert("source".into(), serde_json::json!("UserPromptSubmit"));
    let memory = codemem_core::MemoryNode {
        id: "overlap".into(),
        content: "User prompt: decided to use Postgres for persistence".into(),
        memory_type: codemem_core::MemoryType::Decision,
        importance: 0.8,
        confidence: 0.9,
        access_count: 0,
        content_hash: "ho".into(),
        tags: vec![],
        metadata,
        namespace: None,
        created_at: now,
        updated_at: now,
        last_accessed_at: now,
    };
    let cat = categorize_memories(&[memory]);
    assert_eq!(cat.prompts.len(), 1, "should appear as a prompt");
    assert_eq!(cat.decisions.len(), 1, "should also appear as a decision");
}

#[test]
fn categorize_memories_duplicate_prompts_not_deduped() {
    // Unlike files_read/files_edited/searches, prompts and decisions are not
    // deduplicated. This documents the current behavior.
    let now = chrono::Utc::now();
    let make_prompt = |id: &str| {
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("source".into(), serde_json::json!("UserPromptSubmit"));
        codemem_core::MemoryNode {
            id: id.into(),
            content: "User prompt: fix the bug".into(),
            memory_type: codemem_core::MemoryType::Context,
            importance: 0.3,
            confidence: 1.0,
            access_count: 0,
            content_hash: format!("h-{id}"),
            tags: vec![],
            metadata,
            namespace: None,
            created_at: now,
            updated_at: now,
            last_accessed_at: now,
        }
    };
    let memories = vec![make_prompt("p1"), make_prompt("p2")];
    let cat = categorize_memories(&memories);
    // Two identical prompts from different memories → both kept (no dedup)
    assert_eq!(cat.prompts.len(), 2);
}

#[test]
fn categorize_memories_prompt_without_prefix() {
    // If content doesn't start with "User prompt: ", strip_prefix returns None
    // and unwrap_or falls back to the full content.
    let now = chrono::Utc::now();
    let mut metadata = std::collections::HashMap::new();
    metadata.insert("source".into(), serde_json::json!("UserPromptSubmit"));
    let memory = codemem_core::MemoryNode {
        id: "np".into(),
        content: "fix the auth module".into(),
        memory_type: codemem_core::MemoryType::Context,
        importance: 0.3,
        confidence: 1.0,
        access_count: 0,
        content_hash: "hnp".into(),
        tags: vec![],
        metadata,
        namespace: None,
        created_at: now,
        updated_at: now,
        last_accessed_at: now,
    };
    let cat = categorize_memories(&[memory]);
    assert_eq!(cat.prompts.len(), 1);
    // Full content used since "User prompt: " prefix is absent
    assert!(cat.prompts[0].starts_with("fix the auth module"));
}

#[test]
fn categorize_memories_truncates_long_prompt() {
    let now = chrono::Utc::now();
    let long_prompt = "x".repeat(200);
    let mut metadata = std::collections::HashMap::new();
    metadata.insert("source".into(), serde_json::json!("UserPromptSubmit"));
    let memory = codemem_core::MemoryNode {
        id: "lp".into(),
        content: format!("User prompt: {long_prompt}"),
        memory_type: codemem_core::MemoryType::Context,
        importance: 0.3,
        confidence: 1.0,
        access_count: 0,
        content_hash: "hlp".into(),
        tags: vec![],
        metadata,
        namespace: None,
        created_at: now,
        updated_at: now,
        last_accessed_at: now,
    };
    let cat = categorize_memories(&[memory]);
    assert_eq!(cat.prompts.len(), 1);
    // truncate_str(text, 120) should truncate and append "..."
    assert!(cat.prompts[0].len() <= 123); // 120 + "..."
    assert!(cat.prompts[0].ends_with("..."));
}

#[test]
fn categorize_memories_truncates_long_decision() {
    let now = chrono::Utc::now();
    let long_decision = "d".repeat(200);
    let memory = codemem_core::MemoryNode {
        id: "ld".into(),
        content: long_decision,
        memory_type: codemem_core::MemoryType::Decision,
        importance: 0.8,
        confidence: 0.9,
        access_count: 0,
        content_hash: "hld".into(),
        tags: vec![],
        metadata: std::collections::HashMap::new(),
        namespace: None,
        created_at: now,
        updated_at: now,
        last_accessed_at: now,
    };
    let cat = categorize_memories(&[memory]);
    assert_eq!(cat.decisions.len(), 1);
    assert!(cat.decisions[0].len() <= 123);
    assert!(cat.decisions[0].ends_with("..."));
}

// ── build_session_summary ───────────────────────────────────────────

#[test]
fn build_summary_all_categories() {
    let cat = SessionCategories {
        files_read: vec!["/home/user/project/src/main.rs".into()],
        files_edited: vec!["/home/user/project/src/lib.rs".into()],
        searches: vec!["TODO".into()],
        decisions: vec!["use Postgres".into()],
        prompts: vec!["fix the auth bug".into()],
    };
    let summary = build_session_summary(&cat);
    assert!(summary.contains("Requests: fix the auth bug"));
    // Verify short_path integration: absolute paths are shortened
    assert!(summary.contains("Investigated 1 file(s): src/main.rs"));
    assert!(summary.contains("Modified 1 file(s): src/lib.rs"));
    assert!(summary.contains("Decisions: use Postgres"));
    assert!(summary.contains("Searched: TODO"));
}

#[test]
fn build_summary_multiple_files_listed() {
    let cat = SessionCategories {
        files_read: vec!["/a/b/c/foo.rs".into(), "/a/b/c/bar.rs".into()],
        files_edited: vec![],
        searches: vec![],
        decisions: vec![],
        prompts: vec![],
    };
    let summary = build_session_summary(&cat);
    assert!(summary.contains("Investigated 2 file(s): c/foo.rs, c/bar.rs"));
}

#[test]
fn build_summary_empty_returns_empty() {
    let cat = SessionCategories {
        files_read: vec![],
        files_edited: vec![],
        searches: vec![],
        decisions: vec![],
        prompts: vec![],
    };
    assert!(build_session_summary(&cat).is_empty());
}

#[test]
fn build_summary_only_reads() {
    let cat = SessionCategories {
        files_read: vec!["a.rs".into(), "b.rs".into()],
        files_edited: vec![],
        searches: vec![],
        decisions: vec![],
        prompts: vec![],
    };
    let summary = build_session_summary(&cat);
    assert!(summary.contains("Investigated 2 file(s)"));
    assert!(!summary.contains("Modified"));
    assert!(!summary.contains("Decisions"));
}

// ── has_substance ───────────────────────────────────────────────────

#[test]
fn substance_with_edits() {
    let cat = SessionCategories {
        files_read: vec![],
        files_edited: vec!["src/lib.rs".into()],
        searches: vec![],
        decisions: vec![],
        prompts: vec![],
    };
    assert!(has_substance(&cat));
}

#[test]
fn substance_with_decisions() {
    let cat = SessionCategories {
        files_read: vec![],
        files_edited: vec![],
        searches: vec![],
        decisions: vec!["use Postgres".into()],
        prompts: vec![],
    };
    assert!(has_substance(&cat));
}

#[test]
fn substance_with_many_reads() {
    let cat = SessionCategories {
        files_read: (0..5).map(|i| format!("file{i}.rs")).collect(),
        files_edited: vec![],
        searches: vec![],
        decisions: vec![],
        prompts: vec![],
    };
    assert!(has_substance(&cat));
}

#[test]
fn no_substance_trivial_session() {
    // has_substance only checks files_edited, decisions, and files_read.len() >= 5.
    // searches and prompts don't factor in.
    let cat = SessionCategories {
        files_read: vec!["a.rs".into()],
        files_edited: vec![],
        searches: vec![],
        decisions: vec![],
        prompts: vec![],
    };
    assert!(!has_substance(&cat));
}
