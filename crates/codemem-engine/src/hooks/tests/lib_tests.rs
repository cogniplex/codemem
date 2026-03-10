use super::extractors::*;
use super::*;

// ── relativize_path tests ───────────────────────────────────────────────────

#[test]
fn relativize_path_strips_cwd_prefix() {
    let result = relativize_path("/home/user/project/src/main.rs", Some("/home/user/project"));
    assert_eq!(result, "src/main.rs");
}

#[test]
fn relativize_path_cwd_with_trailing_slash() {
    let result = relativize_path(
        "/home/user/project/src/main.rs",
        Some("/home/user/project/"),
    );
    assert_eq!(result, "src/main.rs");
}

#[test]
fn relativize_path_no_cwd_returns_original() {
    let result = relativize_path("/home/user/project/src/main.rs", None);
    assert_eq!(result, "/home/user/project/src/main.rs");
}

#[test]
fn relativize_path_no_match_returns_original() {
    let result = relativize_path("/other/path/file.rs", Some("/home/user/project"));
    assert_eq!(result, "/other/path/file.rs");
}

#[test]
fn relativize_path_exact_cwd_no_trailing_file() {
    // Path equals cwd exactly — no stripping possible (no trailing file)
    let result = relativize_path("/home/user/project", Some("/home/user/project"));
    assert_eq!(result, "/home/user/project");
}

// ── build_file_extraction with cwd ──────────────────────────────────────────

#[test]
fn build_file_extraction_relativizes_with_cwd() {
    let payload = HookPayload {
        tool_name: "Read".to_string(),
        tool_input: serde_json::json!({"file_path": "/home/user/project/src/lib.rs"}),
        tool_response: serde_json::Value::String("fn foo() {}".to_string()),
        session_id: None,
        cwd: Some("/home/user/project".to_string()),
        hook_event_name: None,
        transcript_path: None,
        permission_mode: None,
        tool_use_id: None,
    };

    let extracted = build_file_extraction(
        &payload,
        "/home/user/project/src/lib.rs",
        "Read file src/lib.rs".to_string(),
        MemoryType::Context,
        "Read",
    );

    // Graph node ID should use relative path
    let node = extracted.graph_node.unwrap();
    assert_eq!(node.id, "file:src/lib.rs");
    assert_eq!(node.label, "src/lib.rs");

    // Metadata should also store relative path
    assert_eq!(
        extracted.metadata["file_path"],
        serde_json::Value::String("src/lib.rs".to_string())
    );
}

#[test]
fn build_file_extraction_no_cwd_keeps_absolute() {
    let payload = HookPayload {
        tool_name: "Read".to_string(),
        tool_input: serde_json::json!({"file_path": "/home/user/project/src/lib.rs"}),
        tool_response: serde_json::Value::String("fn foo() {}".to_string()),
        session_id: None,
        cwd: None,
        hook_event_name: None,
        transcript_path: None,
        permission_mode: None,
        tool_use_id: None,
    };

    let extracted = build_file_extraction(
        &payload,
        "/home/user/project/src/lib.rs",
        "Read file".to_string(),
        MemoryType::Context,
        "Read",
    );

    let node = extracted.graph_node.unwrap();
    assert_eq!(node.id, "file:/home/user/project/src/lib.rs");
}

// ── extract_bash with cwd relativization ────────────────────────────────────

#[test]
fn extract_bash_relativizes_file_path_with_cwd() {
    let json = r#"{
        "tool_name": "Bash",
        "tool_input": {"command": "cat /home/user/project/src/main.rs"},
        "tool_response": "fn main() {}",
        "cwd": "/home/user/project"
    }"#;

    let payload = parse_payload(json).unwrap();
    let extracted = extract(&payload).unwrap().unwrap();

    let node = extracted.graph_node.unwrap();
    assert_eq!(node.id, "file:src/main.rs");
    assert_eq!(node.label, "src/main.rs");
}

#[test]
fn extract_bash_no_cwd_keeps_absolute_path() {
    let json = r#"{
        "tool_name": "Bash",
        "tool_input": {"command": "cat /home/user/project/src/main.rs"},
        "tool_response": "fn main() {}"
    }"#;

    let payload = parse_payload(json).unwrap();
    let extracted = extract(&payload).unwrap().unwrap();

    let node = extracted.graph_node.unwrap();
    assert_eq!(node.id, "file:/home/user/project/src/main.rs");
}

// ── Existing tests ──────────────────────────────────────────────────────────

#[test]
fn parse_read_payload() {
    let json = r#"{
        "tool_name": "Read",
        "tool_input": {"file_path": "src/main.rs"},
        "tool_response": "fn main() { println!(\"hello\"); }"
    }"#;

    let payload = parse_payload(json).unwrap();
    assert_eq!(payload.tool_name, "Read");

    let extracted = extract(&payload).unwrap().unwrap();
    assert_eq!(extracted.memory_type, MemoryType::Context);
    assert!(extracted.tags.contains(&"ext:rs".to_string()));
}

#[test]
fn parse_edit_payload() {
    let json = r#"{
        "tool_name": "Edit",
        "tool_input": {"file_path": "src/lib.rs", "old_string": "foo", "new_string": "bar"},
        "tool_response": "OK"
    }"#;

    let payload = parse_payload(json).unwrap();
    let extracted = extract(&payload).unwrap().unwrap();
    assert_eq!(extracted.memory_type, MemoryType::Decision);
}

#[test]
fn skip_large_response() {
    let large_response = "x".repeat(MAX_CONTENT_SIZE + 1);
    let json = format!(
        r#"{{"tool_name": "Read", "tool_input": {{"file_path": "big.txt"}}, "tool_response": "{large_response}"}}"#
    );

    let payload = parse_payload(&json).unwrap();
    assert!(extract(&payload).unwrap().is_none());
}

#[test]
fn content_hash_deterministic() {
    let h1 = content_hash("hello");
    let h2 = content_hash("hello");
    assert_eq!(h1, h2);
}

#[test]
fn resolve_edges_edit_after_read_creates_evolved_into() {
    // Simulate: file was previously Read (node exists), now being Edited
    let json = r#"{
        "tool_name": "Edit",
        "tool_input": {"file_path": "src/lib.rs", "old_string": "foo", "new_string": "bar"},
        "tool_response": "OK"
    }"#;

    let payload = parse_payload(json).unwrap();
    let mut extracted = extract(&payload).unwrap().unwrap();
    assert!(extracted.graph_edges.is_empty());

    // The file node "file:src/lib.rs" already exists from a prior Read
    let mut existing = std::collections::HashSet::new();
    existing.insert("file:src/lib.rs".to_string());

    resolve_edges(&mut extracted, &existing);

    assert_eq!(extracted.graph_edges.len(), 1);
    assert_eq!(extracted.graph_edges[0].src_id, "file:src/lib.rs");
    assert_eq!(
        extracted.graph_edges[0].relationship,
        RelationshipType::EvolvedInto
    );
}

#[test]
fn resolve_edges_write_after_read_creates_evolved_into() {
    let json = r#"{
        "tool_name": "Write",
        "tool_input": {"file_path": "src/new.rs"},
        "tool_response": "File written"
    }"#;

    let payload = parse_payload(json).unwrap();
    let mut extracted = extract(&payload).unwrap().unwrap();

    // The file node exists from a prior Read
    let mut existing = std::collections::HashSet::new();
    existing.insert("file:src/new.rs".to_string());

    resolve_edges(&mut extracted, &existing);

    assert_eq!(extracted.graph_edges.len(), 1);
    assert_eq!(
        extracted.graph_edges[0].relationship,
        RelationshipType::EvolvedInto
    );
}

#[test]
fn resolve_edges_edit_no_prior_read_no_edges() {
    let json = r#"{
        "tool_name": "Edit",
        "tool_input": {"file_path": "src/lib.rs", "old_string": "foo", "new_string": "bar"},
        "tool_response": "OK"
    }"#;

    let payload = parse_payload(json).unwrap();
    let mut extracted = extract(&payload).unwrap().unwrap();

    // No prior file nodes exist
    let existing = std::collections::HashSet::new();
    resolve_edges(&mut extracted, &existing);

    assert!(extracted.graph_edges.is_empty());
}

#[test]
fn resolve_edges_read_never_creates_edges() {
    let json = r#"{
        "tool_name": "Read",
        "tool_input": {"file_path": "src/main.rs"},
        "tool_response": "fn main() {}"
    }"#;

    let payload = parse_payload(json).unwrap();
    let mut extracted = extract(&payload).unwrap().unwrap();

    let mut existing = std::collections::HashSet::new();
    existing.insert("file:src/main.rs".to_string());

    resolve_edges(&mut extracted, &existing);

    // Read events should not create edges
    assert!(extracted.graph_edges.is_empty());
}

#[test]
fn resolve_edges_glob_no_graph_node_no_edges() {
    let json = r#"{
        "tool_name": "Glob",
        "tool_input": {"pattern": "**/*.rs"},
        "tool_response": "src/main.rs\nsrc/lib.rs"
    }"#;

    let payload = parse_payload(json).unwrap();
    let mut extracted = extract(&payload).unwrap().unwrap();

    let existing = std::collections::HashSet::new();
    resolve_edges(&mut extracted, &existing);

    // Glob has no graph_node, so no edges
    assert!(extracted.graph_edges.is_empty());
}

#[test]
fn materialize_edges_self_reference() {
    let pending = vec![PendingEdge {
        src_id: "file:src/lib.rs".to_string(),
        dst_id: String::new(),
        relationship: RelationshipType::EvolvedInto,
    }];

    let edges = materialize_edges(&pending, "memory-123");

    assert_eq!(edges.len(), 1);
    assert_eq!(edges[0].src, "file:src/lib.rs");
    assert_eq!(edges[0].dst, "file:src/lib.rs");
    assert_eq!(edges[0].relationship, RelationshipType::EvolvedInto);
    assert!(edges[0].properties.contains_key("triggered_by"));
    assert_eq!(
        edges[0].properties["triggered_by"],
        serde_json::Value::String("memory-123".to_string())
    );
}

#[test]
fn materialize_edges_explicit_src_dst() {
    let pending = vec![PendingEdge {
        src_id: "file:src/a.rs".to_string(),
        dst_id: "file:src/b.rs".to_string(),
        relationship: RelationshipType::RelatesTo,
    }];

    let edges = materialize_edges(&pending, "memory-456");

    assert_eq!(edges.len(), 1);
    assert_eq!(edges[0].src, "file:src/a.rs");
    assert_eq!(edges[0].dst, "file:src/b.rs");
    assert_eq!(edges[0].relationship, RelationshipType::RelatesTo);
}

#[test]
fn materialize_edges_empty_pending() {
    let edges = materialize_edges(&[], "memory-789");
    assert!(edges.is_empty());
}

// ── Object-valued tool_response (issue #27) ────────────────────────────────

#[test]
fn parse_write_payload_with_object_tool_response() {
    let json = r#"{
        "tool_name": "Write",
        "tool_input": {"file_path": "/path/to/file.txt", "content": "hello"},
        "tool_response": {"filePath": "/path/to/file.txt", "success": true}
    }"#;

    let payload = parse_payload(json).unwrap();
    assert_eq!(payload.tool_name, "Write");
    assert!(payload.tool_response.is_object());
    // No known content field → falls back to JSON serialization
    let text = payload.tool_response_text();
    assert!(text.contains("filePath"));
    assert!(text.contains("success"));

    let extracted = extract(&payload).unwrap().unwrap();
    assert_eq!(extracted.memory_type, MemoryType::Decision);
    assert!(extracted.content.contains("File written:"));
}

#[test]
fn parse_read_payload_with_object_tool_response() {
    let json = r#"{
        "tool_name": "Read",
        "tool_input": {"file_path": "/path/to/file.rs"},
        "tool_response": {"type": "text", "file": {"filePath": "/path/to/file.rs", "content": "fn main() {}", "numLines": 1, "startLine": 1, "totalLines": 1}}
    }"#;

    let payload = parse_payload(json).unwrap();
    assert!(payload.tool_response.is_object());
    // tool_response_text() extracts file.content for Read responses
    let text = payload.tool_response_text();
    assert_eq!(text, "fn main() {}");

    let extracted = extract(&payload).unwrap().unwrap();
    assert_eq!(extracted.memory_type, MemoryType::Context);
    assert!(extracted.content.contains("File read:"));
    assert!(extracted.content.contains("fn main()"));
}

#[test]
fn tool_response_text_extracts_text_field() {
    let json = r#"{
        "tool_name": "Bash",
        "tool_input": {"command": "echo hello"},
        "tool_response": {"text": "hello\n"}
    }"#;

    let payload = parse_payload(json).unwrap();
    assert_eq!(payload.tool_response_text(), "hello\n");
}

#[test]
fn tool_response_text_extracts_stdout_field() {
    let json = r#"{
        "tool_name": "Bash",
        "tool_input": {"command": "ls"},
        "tool_response": {"stdout": "file1.rs\nfile2.rs", "exitCode": 0}
    }"#;

    let payload = parse_payload(json).unwrap();
    assert_eq!(payload.tool_response_text(), "file1.rs\nfile2.rs");
}

#[test]
fn tool_response_text_returns_inner_string_for_string_value() {
    let json = r#"{
        "tool_name": "Read",
        "tool_input": {"file_path": "test.rs"},
        "tool_response": "plain text response"
    }"#;

    let payload = parse_payload(json).unwrap();
    assert_eq!(payload.tool_response_text(), "plain text response");
}

#[test]
fn tool_response_text_returns_empty_for_null() {
    let json = r#"{
        "tool_name": "Read",
        "tool_input": {"file_path": "test.rs"},
        "tool_response": null
    }"#;

    let payload = parse_payload(json).unwrap();
    assert_eq!(payload.tool_response_text(), "");
}

#[test]
fn payload_with_extra_hook_fields_parses() {
    let json = r#"{
        "tool_name": "Read",
        "tool_input": {"file_path": "test.rs"},
        "tool_response": "content",
        "hook_event_name": "PostToolUse",
        "transcript_path": "/tmp/transcript.json",
        "permission_mode": "default",
        "tool_use_id": "toolu_abc123"
    }"#;

    let payload = parse_payload(json).unwrap();
    assert_eq!(payload.hook_event_name.as_deref(), Some("PostToolUse"));
    assert_eq!(
        payload.transcript_path.as_deref(),
        Some("/tmp/transcript.json")
    );
    assert_eq!(payload.permission_mode.as_deref(), Some("default"));
    assert_eq!(payload.tool_use_id.as_deref(), Some("toolu_abc123"));
}
