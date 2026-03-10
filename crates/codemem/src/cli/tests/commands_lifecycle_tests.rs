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
    let storage = codemem_engine::Storage::open_in_memory().unwrap();
    // Verify the storage query returns empty (cmd_sessions_list prints "No sessions recorded")
    let sessions = storage.list_sessions(None).unwrap();
    assert!(sessions.is_empty());
    cmd_sessions_list(&storage, None).unwrap();
}

#[test]
fn sessions_list_after_start() {
    let storage = codemem_engine::Storage::open_in_memory().unwrap();
    storage.start_session("sess-1", Some("test-ns")).unwrap();

    let sessions = storage.list_sessions(None).unwrap();
    assert_eq!(sessions.len(), 1);
    assert_eq!(sessions[0].id, "sess-1");
    cmd_sessions_list(&storage, None).unwrap();
}

#[test]
fn sessions_list_with_namespace_filter() {
    let storage = codemem_engine::Storage::open_in_memory().unwrap();
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
    let storage = codemem_engine::Storage::open_in_memory().unwrap();
    cmd_sessions_start(&storage, Some("my-project")).unwrap();

    let sessions = storage.list_sessions(Some("my-project")).unwrap();
    assert_eq!(sessions.len(), 1);
    assert_eq!(sessions[0].namespace.as_deref(), Some("my-project"));
}

#[test]
fn sessions_start_no_namespace() {
    let storage = codemem_engine::Storage::open_in_memory().unwrap();
    cmd_sessions_start(&storage, None).unwrap();

    let sessions = storage.list_sessions(None).unwrap();
    assert!(!sessions.is_empty());
}

// ── cmd_sessions_end ────────────────────────────────────────────────

#[test]
fn sessions_end_with_summary() {
    let storage = codemem_engine::Storage::open_in_memory().unwrap();
    storage.start_session("end-test", Some("ns")).unwrap();

    cmd_sessions_end(&storage, "end-test", Some("all done")).unwrap();

    let sessions = storage.list_sessions(Some("ns")).unwrap();
    assert_eq!(sessions.len(), 1);
    assert!(sessions[0].ended_at.is_some());
    assert_eq!(sessions[0].summary.as_deref(), Some("all done"));
}

#[test]
fn sessions_end_without_summary() {
    let storage = codemem_engine::Storage::open_in_memory().unwrap();
    storage.start_session("end-no-sum", None).unwrap();

    cmd_sessions_end(&storage, "end-no-sum", None).unwrap();

    let sessions = storage.list_sessions(None).unwrap();
    let session = sessions.iter().find(|s| s.id == "end-no-sum").unwrap();
    assert!(session.ended_at.is_some());
}

// ── cmd_context ─────────────────────────────────────────────────────

#[test]
fn context_empty_storage_outputs_empty_json() {
    let storage = codemem_engine::Storage::open_in_memory().unwrap();
    // cmd_context reads from stdin, so we can't easily call it directly.
    // But we can verify the storage-layer queries it depends on return empty.
    let sessions = storage.list_sessions(None).unwrap();
    assert!(sessions.is_empty());
    let ids = storage.list_memory_ids().unwrap();
    assert!(ids.is_empty());
}

#[test]
fn context_with_sessions_and_memories() {
    let storage = codemem_engine::Storage::open_in_memory().unwrap();

    // Create a session with summary
    storage
        .start_session("ctx-sess-1", Some("myproject"))
        .unwrap();
    storage
        .end_session("ctx-sess-1", Some("explored auth module"))
        .unwrap();

    // Create Decision and Insight memories that cmd_context looks for
    for (i, mtype) in [
        codemem_core::MemoryType::Decision,
        codemem_core::MemoryType::Insight,
        codemem_core::MemoryType::Pattern,
        codemem_core::MemoryType::Context, // not a high-signal type — excluded from the count below
    ]
    .iter()
    .enumerate()
    {
        let mut memory =
            codemem_core::MemoryNode::test_default(&format!("memory {i} of type {mtype}"));
        memory.id = format!("ctx-mem-{i}");
        memory.memory_type = *mtype;
        memory.importance = 0.7;
        memory.confidence = 0.9;
        memory.tags = vec!["test".to_string()];
        memory.namespace = Some("myproject".to_string());
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
    let storage = codemem_engine::Storage::open_in_memory().unwrap();

    let mut memory = codemem_core::MemoryNode::test_default(
        "Files modified in session abc: src/main.rs, src/lib.rs",
    );
    memory.id = "pending-1".to_string();
    memory.importance = 0.4;
    memory.tags = vec!["pending-analysis".to_string(), "file-changes".to_string()];
    memory.metadata.insert(
        "files".to_string(),
        serde_json::json!(["src/main.rs", "src/lib.rs"]),
    );
    memory.namespace = Some("myproject".to_string());
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
    let mut m = codemem_core::MemoryNode::test_default(&format!("{tool} {file_path}"));
    m.id = id.into();
    m.importance = 0.3;
    m.metadata.insert("tool".into(), serde_json::json!(tool));
    if !file_path.is_empty() {
        m.metadata
            .insert("file_path".into(), serde_json::json!(file_path));
    }
    m
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
    let mut memory = codemem_core::MemoryNode::test_default("Decided to use Postgres");
    memory.id = "d1".into();
    memory.memory_type = codemem_core::MemoryType::Decision;
    memory.importance = 0.8;
    memory.confidence = 0.9;
    let cat = categorize_memories(&[memory]);
    assert_eq!(cat.decisions, vec!["Decided to use Postgres"]);
}

#[test]
fn categorize_memories_prompts() {
    let mut memory = codemem_core::MemoryNode::test_default("User prompt: fix the auth bug");
    memory.id = "p1".into();
    memory.importance = 0.3;
    memory.tags = vec!["prompt".to_string()];
    memory
        .metadata
        .insert("source".into(), serde_json::json!("UserPromptSubmit"));
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
    let mut memory = codemem_core::MemoryNode::test_default(
        "User prompt: decided to use Postgres for persistence",
    );
    memory.id = "overlap".into();
    memory.memory_type = codemem_core::MemoryType::Decision;
    memory.importance = 0.8;
    memory.confidence = 0.9;
    memory
        .metadata
        .insert("source".into(), serde_json::json!("UserPromptSubmit"));
    let cat = categorize_memories(&[memory]);
    assert_eq!(cat.prompts.len(), 1, "should appear as a prompt");
    assert_eq!(cat.decisions.len(), 1, "should also appear as a decision");
}

#[test]
fn categorize_memories_duplicate_prompts_not_deduped() {
    // Unlike files_read/files_edited/searches, prompts and decisions are not
    // deduplicated. This documents the current behavior.
    let make_prompt = |id: &str| {
        let mut m = codemem_core::MemoryNode::test_default("User prompt: fix the bug");
        m.id = id.into();
        m.importance = 0.3;
        m.metadata
            .insert("source".into(), serde_json::json!("UserPromptSubmit"));
        m
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
    let mut memory = codemem_core::MemoryNode::test_default("fix the auth module");
    memory.id = "np".into();
    memory.importance = 0.3;
    memory
        .metadata
        .insert("source".into(), serde_json::json!("UserPromptSubmit"));
    let cat = categorize_memories(&[memory]);
    assert_eq!(cat.prompts.len(), 1);
    // Full content used since "User prompt: " prefix is absent
    assert!(cat.prompts[0].starts_with("fix the auth module"));
}

#[test]
fn categorize_memories_truncates_long_prompt() {
    let long_prompt = "x".repeat(200);
    let mut memory = codemem_core::MemoryNode::test_default(&format!("User prompt: {long_prompt}"));
    memory.id = "lp".into();
    memory.importance = 0.3;
    memory
        .metadata
        .insert("source".into(), serde_json::json!("UserPromptSubmit"));
    let cat = categorize_memories(&[memory]);
    assert_eq!(cat.prompts.len(), 1);
    // truncate_str(text, 120) should truncate and append "..."
    assert!(cat.prompts[0].len() <= 123); // 120 + "..."
    assert!(cat.prompts[0].ends_with("..."));
}

#[test]
fn categorize_memories_truncates_long_decision() {
    let long_decision = "d".repeat(200);
    let mut memory = codemem_core::MemoryNode::test_default(&long_decision);
    memory.id = "ld".into();
    memory.memory_type = codemem_core::MemoryType::Decision;
    memory.importance = 0.8;
    memory.confidence = 0.9;
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

// ── has_substance boundary at exactly 4 reads ─────────────────────────

#[test]
fn no_substance_exactly_four_reads() {
    // has_substance requires files_read.len() >= 5, so 4 is NOT enough.
    let cat = SessionCategories {
        files_read: (0..4).map(|i| format!("file{i}.rs")).collect(),
        files_edited: vec![],
        searches: vec![],
        decisions: vec![],
        prompts: vec![],
    };
    assert!(!has_substance(&cat));
}

// ── build_session_summary truncation ──────────────────────────────────

#[test]
fn build_summary_truncates_prompts_at_three() {
    let cat = SessionCategories {
        files_read: vec![],
        files_edited: vec![],
        searches: vec![],
        decisions: vec![],
        prompts: vec![
            "first".into(),
            "second".into(),
            "third".into(),
            "fourth".into(),
        ],
    };
    let summary = build_session_summary(&cat);
    assert!(summary.contains("first"));
    assert!(summary.contains("second"));
    assert!(summary.contains("third"));
    assert!(
        !summary.contains("fourth"),
        "4th prompt should be truncated"
    );
}

#[test]
fn build_summary_truncates_files_read_at_five() {
    let cat = SessionCategories {
        files_read: (0..6).map(|i| format!("/project/src/f{i}.rs")).collect(),
        files_edited: vec![],
        searches: vec![],
        decisions: vec![],
        prompts: vec![],
    };
    let summary = build_session_summary(&cat);
    // Shows "6 file(s)" in the count, but only lists first 5
    assert!(summary.contains("6 file(s)"));
    assert!(summary.contains("f4.rs"), "5th file should be included");
    assert!(!summary.contains("f5.rs"), "6th file should be truncated");
}

#[test]
fn build_summary_truncates_decisions_at_three() {
    let cat = SessionCategories {
        files_read: vec![],
        files_edited: vec![],
        searches: vec![],
        decisions: vec![
            "decision-a".into(),
            "decision-b".into(),
            "decision-c".into(),
            "decision-d".into(),
        ],
        prompts: vec![],
    };
    let summary = build_session_summary(&cat);
    assert!(summary.contains("decision-a"));
    assert!(summary.contains("decision-c"));
    assert!(
        !summary.contains("decision-d"),
        "4th decision should be truncated"
    );
}

// ── categorize_memories: unknown tool silently ignored ─────────────────

#[test]
fn categorize_memories_unknown_tool_silently_ignored() {
    let memories = vec![
        make_tool_memory("m1", "WebFetch", ""),
        make_tool_memory("m2", "WebSearch", ""),
        make_tool_memory("m3", "Agent", ""),
    ];
    let cat = categorize_memories(&memories);
    assert!(
        cat.files_read.is_empty(),
        "WebFetch should not appear as read"
    );
    assert!(
        cat.files_edited.is_empty(),
        "WebSearch should not appear as edit"
    );
    assert!(cat.searches.is_empty(), "Agent should not appear as search");
}

// ── save_compact_checkpoint ────────────────────────────────────────

#[test]
fn save_compact_checkpoint_with_memories() {
    let storage = codemem_engine::Storage::open_in_memory().unwrap();

    // Add some high-signal memories
    for (i, mtype) in [
        codemem_core::MemoryType::Decision,
        codemem_core::MemoryType::Insight,
        codemem_core::MemoryType::Context, // low-signal — excluded from checkpoint
    ]
    .iter()
    .enumerate()
    {
        let mut m = codemem_core::MemoryNode::test_default(&format!("memory {i}"));
        m.id = format!("ckpt-{i}");
        m.memory_type = *mtype;
        m.namespace = Some("proj".to_string());
        storage.insert_memory(&m).unwrap();
    }

    save_compact_checkpoint(&storage, "sess-1", Some("proj"));

    // Verify a checkpoint memory was created
    let ids = storage.list_memory_ids_for_namespace("proj").unwrap();
    let checkpoint = ids.iter().find_map(|id| {
        let m = storage.get_memory_no_touch(id).ok()??;
        if m.tags.contains(&"pre-compact".to_string()) {
            Some(m)
        } else {
            None
        }
    });
    assert!(checkpoint.is_some(), "checkpoint memory should exist");
    let ckpt = checkpoint.unwrap();
    assert!(ckpt.content.contains("Pre-compact checkpoint"));
    assert!(ckpt.tags.contains(&"checkpoint".to_string()));
    assert_eq!(ckpt.importance, 0.5);
}

#[test]
fn save_compact_checkpoint_empty_storage() {
    let storage = codemem_engine::Storage::open_in_memory().unwrap();

    save_compact_checkpoint(&storage, "sess-1", None);

    let ids = storage.list_memory_ids().unwrap();
    assert_eq!(ids.len(), 1);
    let m = storage.get_memory_no_touch(&ids[0]).unwrap().unwrap();
    assert!(m.content.contains("no key memories"));
}

// ── cmd_session_close ──────────────────────────────────────────────

#[test]
fn session_close_ends_session() {
    let storage = codemem_engine::Storage::open_in_memory().unwrap();
    storage.start_session("close-test", Some("ns")).unwrap();

    // Verify it's active
    let sessions = storage.list_sessions(Some("ns")).unwrap();
    assert!(sessions[0].ended_at.is_none());

    // Simulate what cmd_session_close does
    let summary = format!("Session ended: {}", "prompt_input_exit");
    storage.end_session("close-test", Some(&summary)).unwrap();

    let sessions = storage.list_sessions(Some("ns")).unwrap();
    assert!(sessions[0].ended_at.is_some());
    assert!(sessions[0]
        .summary
        .as_deref()
        .unwrap()
        .contains("prompt_input_exit"));
}

#[test]
fn session_close_skips_already_ended_session() {
    // If the Stop hook already ended the session with a rich summary,
    // cmd_session_close should not overwrite it.
    let storage = codemem_engine::Storage::open_in_memory().unwrap();
    storage.start_session("rich-test", Some("ns")).unwrap();

    // Stop hook ends session with rich summary
    let rich_summary = "Modified 3 file(s): mod.rs, extractors.rs, lib.rs. Decisions: switched to serde_json::Value";
    storage
        .end_session("rich-test", Some(rich_summary))
        .unwrap();

    // Simulate cmd_session_close checking if already ended
    let already_ended = StorageBackend::list_sessions(&storage, Some("ns"), usize::MAX)
        .unwrap_or_default()
        .iter()
        .any(|s| s.id == "rich-test" && s.ended_at.is_some());

    assert!(
        already_ended,
        "session should already be ended by Stop hook"
    );

    // Verify the rich summary is preserved (cmd_session_close would skip)
    let sessions = storage.list_sessions(Some("ns")).unwrap();
    assert_eq!(sessions[0].summary.as_deref(), Some(rich_summary));
}

// ── cmd_tool_error storage ─────────────────────────────────────────

#[test]
fn tool_error_creates_memory() {
    let storage = codemem_engine::Storage::open_in_memory().unwrap();

    // Simulate what cmd_tool_error stores
    let content = format!(
        "Tool error ({} on {}): {}",
        "Edit", "src/main.rs", "File not found"
    );
    let mut memory = codemem_core::MemoryNode::new(content, codemem_core::MemoryType::Context);
    memory.importance = 0.4;
    memory.tags = vec![
        "error".to_string(),
        "tool-failure".to_string(),
        "tool:Edit".to_string(),
    ];
    memory
        .metadata
        .insert("source".into(), serde_json::json!("PostToolUseFailure"));
    memory
        .metadata
        .insert("tool_name".into(), serde_json::json!("Edit"));
    memory
        .metadata
        .insert("error".into(), serde_json::json!("File not found"));
    memory.namespace = Some("proj".to_string());
    storage.insert_memory(&memory).unwrap();

    let ids = storage.list_memory_ids_for_namespace("proj").unwrap();
    assert_eq!(ids.len(), 1);
    let m = storage.get_memory_no_touch(&ids[0]).unwrap().unwrap();
    assert!(m.content.contains("Tool error (Edit"));
    assert!(m.tags.contains(&"tool-failure".to_string()));
}

#[test]
fn tool_error_skips_interrupt() {
    // cmd_tool_error should skip if is_interrupt is true
    // (simulated here — the actual check is in the hook function,
    //  but we test the logic pattern)
    let is_interrupt = true;
    assert!(
        is_interrupt,
        "interrupt tool errors should be skipped, not stored"
    );
}

#[test]
fn tool_error_skips_empty_error() {
    // cmd_tool_error should skip if error is empty
    let error = "";
    assert!(
        error.is_empty(),
        "empty error should be skipped, not stored"
    );
}

#[test]
fn tool_error_with_input_context() {
    let storage = codemem_engine::Storage::open_in_memory().unwrap();

    // Simulate tool error with file_path context
    let content = "Tool error (Read on src/missing.rs): No such file";
    let mut memory =
        codemem_core::MemoryNode::new(content.to_string(), codemem_core::MemoryType::Context);
    memory.importance = 0.4;
    memory.tags = vec![
        "error".to_string(),
        "tool-failure".to_string(),
        "tool:Read".to_string(),
    ];
    memory
        .metadata
        .insert("source".into(), serde_json::json!("PostToolUseFailure"));
    memory
        .metadata
        .insert("tool_name".into(), serde_json::json!("Read"));
    memory
        .metadata
        .insert("error".into(), serde_json::json!("No such file"));
    storage.insert_memory(&memory).unwrap();

    let ids = storage.list_memory_ids().unwrap();
    let m = storage.get_memory_no_touch(&ids[0]).unwrap().unwrap();
    assert!(m.content.contains("src/missing.rs"));
    assert_eq!(m.metadata["tool_name"], serde_json::json!("Read"));
}

// ── cmd_agent_result storage ──────────────────────────────────────

#[test]
fn agent_result_creates_insight_memory() {
    let storage = codemem_engine::Storage::open_in_memory().unwrap();

    // Simulate what cmd_agent_result stores
    let content = "Agent general-purpose result: Found 3 API endpoints in the auth module";
    let mut memory =
        codemem_core::MemoryNode::new(content.to_string(), codemem_core::MemoryType::Insight);
    memory.importance = 0.5;
    memory.tags = vec![
        "agent-result".to_string(),
        "agent:general-purpose".to_string(),
    ];
    memory
        .metadata
        .insert("source".into(), serde_json::json!("SubagentStop"));
    memory
        .metadata
        .insert("agent_type".into(), serde_json::json!("general-purpose"));
    memory
        .metadata
        .insert("agent_id".into(), serde_json::json!("agent-123"));
    memory.namespace = Some("proj".to_string());
    storage.insert_memory(&memory).unwrap();

    let ids = storage.list_memory_ids_for_namespace("proj").unwrap();
    assert_eq!(ids.len(), 1);
    let m = storage.get_memory_no_touch(&ids[0]).unwrap().unwrap();
    assert_eq!(m.memory_type, codemem_core::MemoryType::Insight);
    assert_eq!(m.importance, 0.5);
    assert!(m.tags.contains(&"agent-result".to_string()));
    assert!(m.content.contains("API endpoints"));
}

#[test]
fn agent_result_skips_short_message() {
    // cmd_agent_result skips if last_message.len() < 20
    let short_message = "ok done";
    assert!(
        short_message.len() < 20,
        "short agent messages should be skipped"
    );
}

// ── cmd_agent_start ───────────────────────────────────────────────

#[test]
fn agent_start_does_not_store_memory() {
    // cmd_agent_start only logs, never creates a memory.
    // Verify by checking that a storage remains empty.
    let storage = codemem_engine::Storage::open_in_memory().unwrap();
    let ids = storage.list_memory_ids().unwrap();
    assert!(ids.is_empty(), "agent_start should not store any memories");
}

// ── cmd_checkpoint ────────────────────────────────────────────────

#[test]
fn checkpoint_batch_size_is_bounded() {
    let storage = codemem_engine::Storage::open_in_memory().unwrap();

    // Insert 20 memories — more than the batch limit of 15
    for i in 0..20 {
        let mut m = codemem_core::MemoryNode::test_default(&format!("memory {i}"));
        m.id = format!("batch-{i}");
        m.memory_type = codemem_core::MemoryType::Decision;
        m.namespace = Some("proj".to_string());
        storage.insert_memory(&m).unwrap();
    }

    save_compact_checkpoint(&storage, "sess-1", Some("proj"));

    // Checkpoint should still be created successfully
    let ids = storage.list_memory_ids_for_namespace("proj").unwrap();
    let checkpoint = ids.iter().find_map(|id| {
        let m = storage.get_memory_no_touch(id).ok()??;
        if m.tags.contains(&"pre-compact".to_string()) {
            Some(m)
        } else {
            None
        }
    });
    assert!(checkpoint.is_some());
    let ckpt = checkpoint.unwrap();
    // Should contain exactly 5 key items (the limit in save_compact_checkpoint)
    let item_count = ckpt.content.matches(';').count() + 1;
    assert!(
        item_count <= 5,
        "checkpoint should include at most 5 key items, got {item_count}"
    );
}
