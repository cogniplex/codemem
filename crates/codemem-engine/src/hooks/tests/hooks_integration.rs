use super::extractors::*;
use super::*;

// ── Grep tool extraction ────────────────────────────────────────────────────

#[test]
fn extract_grep_extracts_pattern() {
    let json = r#"{
        "tool_name": "Grep",
        "tool_input": {"pattern": "fn\\s+main"},
        "tool_response": "src/main.rs:1:fn main() {}"
    }"#;

    let payload = parse_payload(json).unwrap();
    let extracted = extract(&payload).unwrap().unwrap();

    assert_eq!(extracted.memory_type, MemoryType::Pattern);
    assert!(extracted.content.contains("Grep search: fn\\s+main"));
    assert!(extracted.content.contains("src/main.rs:1:fn main() {}"));
    assert!(extracted.tags.contains(&"pattern:fn\\s+main".to_string()));
    assert!(extracted.tags.contains(&"search".to_string()));
}

#[test]
fn extract_grep_metadata_has_pattern_and_tool() {
    let json = r#"{
        "tool_name": "Grep",
        "tool_input": {"pattern": "TODO"},
        "tool_response": "src/lib.rs:42:// TODO: fix this"
    }"#;

    let payload = parse_payload(json).unwrap();
    let extracted = extract(&payload).unwrap().unwrap();

    assert_eq!(
        extracted.metadata["pattern"],
        serde_json::Value::String("TODO".to_string())
    );
    assert_eq!(
        extracted.metadata["tool"],
        serde_json::Value::String("Grep".to_string())
    );
}

#[test]
fn extract_grep_no_graph_node() {
    let json = r#"{
        "tool_name": "Grep",
        "tool_input": {"pattern": "test"},
        "tool_response": "found something"
    }"#;

    let payload = parse_payload(json).unwrap();
    let extracted = extract(&payload).unwrap().unwrap();
    assert!(extracted.graph_node.is_none());
}

#[test]
fn extract_grep_empty_pattern_graceful() {
    let json = r#"{
        "tool_name": "Grep",
        "tool_input": {},
        "tool_response": "some results"
    }"#;

    let payload = parse_payload(json).unwrap();
    let extracted = extract(&payload).unwrap().unwrap();

    // Should default to empty string pattern without panicking
    assert_eq!(extracted.memory_type, MemoryType::Pattern);
    assert!(extracted.tags.contains(&"pattern:".to_string()));
}

// ── Edit tool extraction ────────────────────────────────────────────────────

#[test]
fn extract_edit_creates_decision_memory() {
    let json = r#"{
        "tool_name": "Edit",
        "tool_input": {"file_path": "src/lib.rs", "old_string": "let x = 1;", "new_string": "let x = 2;"},
        "tool_response": "OK"
    }"#;

    let payload = parse_payload(json).unwrap();
    let extracted = extract(&payload).unwrap().unwrap();

    assert_eq!(extracted.memory_type, MemoryType::Decision);
    assert!(extracted.content.contains("Edit: src/lib.rs"));
    assert!(extracted.content.contains("let x = 1;"));
    assert!(extracted.content.contains("let x = 2;"));
}

#[test]
fn extract_edit_has_file_graph_node() {
    let json = r#"{
        "tool_name": "Edit",
        "tool_input": {"file_path": "src/handler.rs", "old_string": "a", "new_string": "b"},
        "tool_response": "OK"
    }"#;

    let payload = parse_payload(json).unwrap();
    let extracted = extract(&payload).unwrap().unwrap();

    let node = extracted.graph_node.as_ref().unwrap();
    assert_eq!(node.id, "file:src/handler.rs");
    assert_eq!(node.kind, codemem_core::NodeKind::File);
}

#[test]
fn extract_edit_metadata_has_file_path_and_tool() {
    let json = r#"{
        "tool_name": "Edit",
        "tool_input": {"file_path": "src/lib.rs", "old_string": "a", "new_string": "b"},
        "tool_response": "OK"
    }"#;

    let payload = parse_payload(json).unwrap();
    let extracted = extract(&payload).unwrap().unwrap();

    assert_eq!(
        extracted.metadata["file_path"],
        serde_json::Value::String("src/lib.rs".to_string())
    );
    assert_eq!(
        extracted.metadata["tool"],
        serde_json::Value::String("Edit".to_string())
    );
}

#[test]
fn extract_edit_missing_fields_graceful() {
    let json = r#"{
        "tool_name": "Edit",
        "tool_input": {},
        "tool_response": "OK"
    }"#;

    let payload = parse_payload(json).unwrap();
    let extracted = extract(&payload).unwrap().unwrap();

    // Should default to "unknown" file_path without panicking
    assert_eq!(extracted.memory_type, MemoryType::Decision);
    assert!(extracted.content.contains("Edit: unknown"));
}

#[test]
fn extract_multiedit_same_as_edit() {
    let json = r#"{
        "tool_name": "MultiEdit",
        "tool_input": {"file_path": "src/foo.rs", "old_string": "x", "new_string": "y"},
        "tool_response": "OK"
    }"#;

    let payload = parse_payload(json).unwrap();
    let extracted = extract(&payload).unwrap().unwrap();

    assert_eq!(extracted.memory_type, MemoryType::Decision);
    assert!(extracted.content.contains("Edit: src/foo.rs"));
}

// ── Write tool extraction ───────────────────────────────────────────────────

#[test]
fn extract_write_creates_decision_memory() {
    let json = r#"{
        "tool_name": "Write",
        "tool_input": {"file_path": "src/new_module.rs"},
        "tool_response": "File written successfully"
    }"#;

    let payload = parse_payload(json).unwrap();
    let extracted = extract(&payload).unwrap().unwrap();

    assert_eq!(extracted.memory_type, MemoryType::Decision);
    assert!(extracted
        .content
        .contains("File written: src/new_module.rs"));
}

#[test]
fn extract_write_has_file_graph_node() {
    let json = r#"{
        "tool_name": "Write",
        "tool_input": {"file_path": "src/config.rs"},
        "tool_response": "OK"
    }"#;

    let payload = parse_payload(json).unwrap();
    let extracted = extract(&payload).unwrap().unwrap();

    let node = extracted.graph_node.as_ref().unwrap();
    assert_eq!(node.id, "file:src/config.rs");
    assert_eq!(node.kind, codemem_core::NodeKind::File);
}

#[test]
fn extract_write_metadata_has_tool_name() {
    let json = r#"{
        "tool_name": "Write",
        "tool_input": {"file_path": "src/out.rs"},
        "tool_response": "OK"
    }"#;

    let payload = parse_payload(json).unwrap();
    let extracted = extract(&payload).unwrap().unwrap();

    assert_eq!(
        extracted.metadata["tool"],
        serde_json::Value::String("Write".to_string())
    );
}

#[test]
fn extract_write_missing_file_path_graceful() {
    let json = r#"{
        "tool_name": "Write",
        "tool_input": {},
        "tool_response": "OK"
    }"#;

    let payload = parse_payload(json).unwrap();
    let extracted = extract(&payload).unwrap().unwrap();

    assert_eq!(extracted.memory_type, MemoryType::Decision);
    assert!(extracted.content.contains("File written: unknown"));
}

// ── Bash tool extraction ────────────────────────────────────────────────────

#[test]
fn extract_bash_extracts_command() {
    let json = r#"{
        "tool_name": "Bash",
        "tool_input": {"command": "cargo test --workspace"},
        "tool_response": "test result: ok. 42 passed"
    }"#;

    let payload = parse_payload(json).unwrap();
    let extracted = extract(&payload).unwrap().unwrap();

    assert_eq!(extracted.memory_type, MemoryType::Context);
    assert!(extracted
        .content
        .contains("Bash command: cargo test --workspace"));
    assert!(extracted.content.contains("test result: ok"));
}

#[test]
fn extract_bash_tags_include_command_prefix() {
    let json = r#"{
        "tool_name": "Bash",
        "tool_input": {"command": "git status"},
        "tool_response": "nothing to commit"
    }"#;

    let payload = parse_payload(json).unwrap();
    let extracted = extract(&payload).unwrap().unwrap();

    assert!(extracted.tags.contains(&"bash".to_string()));
    assert!(extracted.tags.contains(&"command:git".to_string()));
}

#[test]
fn extract_bash_with_cwd_adds_dir_tag() {
    let json = r#"{
        "tool_name": "Bash",
        "tool_input": {"command": "ls"},
        "tool_response": "file.txt",
        "cwd": "/home/user/project"
    }"#;

    let payload = parse_payload(json).unwrap();
    let extracted = extract(&payload).unwrap().unwrap();

    assert!(extracted
        .tags
        .contains(&"dir:/home/user/project".to_string()));
}

#[test]
fn extract_bash_with_input_cwd_adds_dir_tag() {
    let json = r#"{
        "tool_name": "Bash",
        "tool_input": {"command": "ls", "cwd": "/custom/dir"},
        "tool_response": "file.txt"
    }"#;

    let payload = parse_payload(json).unwrap();
    let extracted = extract(&payload).unwrap().unwrap();

    assert!(extracted.tags.contains(&"dir:/custom/dir".to_string()));
}

#[test]
fn extract_bash_error_detection() {
    let json = r#"{
        "tool_name": "Bash",
        "tool_input": {"command": "cargo build"},
        "tool_response": "error: could not compile"
    }"#;

    let payload = parse_payload(json).unwrap();
    let extracted = extract(&payload).unwrap().unwrap();

    assert!(extracted.tags.contains(&"error".to_string()));
}

#[test]
fn extract_bash_metadata_has_command_and_tool() {
    let json = r#"{
        "tool_name": "Bash",
        "tool_input": {"command": "echo hello"},
        "tool_response": "hello"
    }"#;

    let payload = parse_payload(json).unwrap();
    let extracted = extract(&payload).unwrap().unwrap();

    assert_eq!(
        extracted.metadata["tool"],
        serde_json::Value::String("Bash".to_string())
    );
    assert_eq!(
        extracted.metadata["command"],
        serde_json::Value::String("echo hello".to_string())
    );
}

#[test]
fn extract_bash_empty_command_graceful() {
    let json = r#"{
        "tool_name": "Bash",
        "tool_input": {},
        "tool_response": ""
    }"#;

    let payload = parse_payload(json).unwrap();
    let extracted = extract(&payload).unwrap().unwrap();

    // Should not panic with empty/missing command
    assert_eq!(extracted.memory_type, MemoryType::Context);
    assert!(extracted.tags.contains(&"bash".to_string()));
}

// ── Correct memory_type and tags ────────────────────────────────────────────

#[test]
fn extracted_memory_types_are_correct() {
    // Read -> Context
    let read_json = r#"{
        "tool_name": "Read",
        "tool_input": {"file_path": "f.rs"},
        "tool_response": "content"
    }"#;
    let read_payload = parse_payload(read_json).unwrap();
    assert_eq!(
        extract(&read_payload).unwrap().unwrap().memory_type,
        MemoryType::Context
    );

    // Glob -> Pattern
    let glob_json = r#"{
        "tool_name": "Glob",
        "tool_input": {"pattern": "*.rs"},
        "tool_response": "file.rs"
    }"#;
    let glob_payload = parse_payload(glob_json).unwrap();
    assert_eq!(
        extract(&glob_payload).unwrap().unwrap().memory_type,
        MemoryType::Pattern
    );

    // Grep -> Pattern
    let grep_json = r#"{
        "tool_name": "Grep",
        "tool_input": {"pattern": "test"},
        "tool_response": "match"
    }"#;
    let grep_payload = parse_payload(grep_json).unwrap();
    assert_eq!(
        extract(&grep_payload).unwrap().unwrap().memory_type,
        MemoryType::Pattern
    );

    // Edit -> Decision
    let edit_json = r#"{
        "tool_name": "Edit",
        "tool_input": {"file_path": "f.rs", "old_string": "a", "new_string": "b"},
        "tool_response": "OK"
    }"#;
    let edit_payload = parse_payload(edit_json).unwrap();
    assert_eq!(
        extract(&edit_payload).unwrap().unwrap().memory_type,
        MemoryType::Decision
    );

    // Write -> Decision
    let write_json = r#"{
        "tool_name": "Write",
        "tool_input": {"file_path": "f.rs"},
        "tool_response": "OK"
    }"#;
    let write_payload = parse_payload(write_json).unwrap();
    assert_eq!(
        extract(&write_payload).unwrap().unwrap().memory_type,
        MemoryType::Decision
    );

    // Bash -> Context
    let bash_json = r#"{
        "tool_name": "Bash",
        "tool_input": {"command": "ls"},
        "tool_response": "files"
    }"#;
    let bash_payload = parse_payload(bash_json).unwrap();
    assert_eq!(
        extract(&bash_payload).unwrap().unwrap().memory_type,
        MemoryType::Context
    );
}

#[test]
fn extract_tags_from_path_produces_correct_tags() {
    let tags = extract_tags_from_path("src/hooks/mod.rs");
    assert!(tags.contains(&"ext:rs".to_string()));
    assert!(tags.contains(&"dir:hooks".to_string()));
    assert!(tags.contains(&"file:mod.rs".to_string()));
}

#[test]
fn extract_tags_from_path_no_extension() {
    let tags = extract_tags_from_path("Makefile");
    // Should not contain ext: tag
    assert!(!tags.iter().any(|t| t.starts_with("ext:")));
    assert!(tags.contains(&"file:Makefile".to_string()));
}

// ── Unknown tool returns None ───────────────────────────────────────────────

#[test]
fn unknown_tool_returns_none() {
    let json = r#"{
        "tool_name": "SomeUnknownTool",
        "tool_input": {},
        "tool_response": "whatever"
    }"#;

    let payload = parse_payload(json).unwrap();
    assert!(extract(&payload).unwrap().is_none());
}

// ── Session ID propagation ──────────────────────────────────────────────────

#[test]
fn session_id_propagated_to_extracted_memory() {
    let json = r#"{
        "tool_name": "Read",
        "tool_input": {"file_path": "test.rs"},
        "tool_response": "content",
        "session_id": "session-abc-123"
    }"#;

    let payload = parse_payload(json).unwrap();
    let extracted = extract(&payload).unwrap().unwrap();

    assert_eq!(extracted.session_id.as_deref(), Some("session-abc-123"));
}

// ── Truncation ──────────────────────────────────────────────────────────────

#[test]
fn truncate_respects_max_length() {
    let short = "hello";
    assert_eq!(truncate(short, 100), "hello");

    let long = "a".repeat(3000);
    let result = truncate(&long, 2000);
    assert_eq!(result.len(), 2000);
}

#[test]
fn truncate_respects_char_boundaries() {
    // Multi-byte UTF-8
    let s = "héllo wörld";
    let result = truncate(s, 3);
    // Should not panic and should be valid UTF-8
    assert!(result.len() <= 3);
    assert!(std::str::from_utf8(result.as_bytes()).is_ok());
}
