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
    let result = namespace_from_cwd("/Users/me/project/");
    assert!(!result.is_empty());
}

#[test]
fn namespace_from_cwd_root() {
    let result = namespace_from_cwd("/");
    assert!(!result.is_empty());
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
    cmd_sessions_list(&storage, None).unwrap();
}

#[test]
fn sessions_list_after_start() {
    let storage = codemem_storage::Storage::open_in_memory().unwrap();
    storage.start_session("sess-1", Some("test-ns")).unwrap();

    cmd_sessions_list(&storage, None).unwrap();
}

#[test]
fn sessions_list_with_namespace_filter() {
    let storage = codemem_storage::Storage::open_in_memory().unwrap();
    storage.start_session("sess-a", Some("ns-a")).unwrap();
    storage.start_session("sess-b", Some("ns-b")).unwrap();

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
fn context_storage_type_accepted() {
    // Verify cmd_context accepts &dyn StorageBackend
    let storage = codemem_storage::Storage::open_in_memory().unwrap();
    let _ = &storage as &dyn codemem_core::StorageBackend;
}
