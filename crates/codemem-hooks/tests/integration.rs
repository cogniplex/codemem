//! Integration tests for codemem-hooks: parsing real-world-like
//! PostToolUse payloads and verifying extracted memories.

use codemem_core::MemoryType;
use codemem_hooks::{extract, parse_payload};

// ── Read Tool Tests ────────────────────────────────────────────────────────

#[test]
fn read_tool_extracts_context_memory() {
    let payload_json = r#"{
        "tool_name": "Read",
        "tool_input": {"file_path": "crates/codemem-core/src/lib.rs"},
        "tool_response": "//! codemem-core: Shared types\nuse serde::Serialize;\npub struct MemoryNode { ... }",
        "session_id": "session-123",
        "cwd": "/home/user/project"
    }"#;

    let payload = parse_payload(payload_json).unwrap();
    assert_eq!(payload.tool_name, "Read");
    assert_eq!(payload.session_id, Some("session-123".to_string()));
    assert_eq!(payload.cwd, Some("/home/user/project".to_string()));

    let extracted = extract(&payload).unwrap().unwrap();
    assert_eq!(extracted.memory_type, MemoryType::Context);
    assert!(extracted
        .content
        .contains("File read: crates/codemem-core/src/lib.rs"));
    assert!(extracted.content.contains("codemem-core"));

    // Tags should include extension and directory
    assert!(extracted.tags.contains(&"ext:rs".to_string()));
    assert!(extracted.tags.contains(&"dir:src".to_string()));
    assert!(extracted.tags.contains(&"file:lib.rs".to_string()));

    // Metadata
    assert_eq!(
        extracted.metadata["file_path"],
        serde_json::Value::String("crates/codemem-core/src/lib.rs".to_string())
    );
    assert_eq!(
        extracted.metadata["tool"],
        serde_json::Value::String("Read".to_string())
    );

    // Graph node
    let graph_node = extracted.graph_node.unwrap();
    assert_eq!(graph_node.id, "file:crates/codemem-core/src/lib.rs");
    assert_eq!(graph_node.kind, codemem_core::NodeKind::File);
}

#[test]
fn read_tool_with_nested_path() {
    let payload_json = r#"{
        "tool_name": "Read",
        "tool_input": {"file_path": "src/components/auth/login.tsx"},
        "tool_response": "export function Login() { return <div>Login</div>; }"
    }"#;

    let payload = parse_payload(payload_json).unwrap();
    let extracted = extract(&payload).unwrap().unwrap();

    assert!(extracted.tags.contains(&"ext:tsx".to_string()));
    assert!(extracted.tags.contains(&"dir:auth".to_string()));
    assert!(extracted.tags.contains(&"file:login.tsx".to_string()));
}

// ── Glob Tool Tests ────────────────────────────────────────────────────────

#[test]
fn glob_tool_extracts_pattern_memory() {
    let payload_json = r#"{
        "tool_name": "Glob",
        "tool_input": {"pattern": "**/*.rs"},
        "tool_response": "src/main.rs\nsrc/lib.rs\nsrc/utils/mod.rs"
    }"#;

    let payload = parse_payload(payload_json).unwrap();
    let extracted = extract(&payload).unwrap().unwrap();

    assert_eq!(extracted.memory_type, MemoryType::Pattern);
    assert!(extracted.content.contains("Glob search: **/*.rs"));
    assert!(extracted.content.contains("src/main.rs"));

    assert!(extracted.tags.contains(&"glob:**/*.rs".to_string()));
    assert!(extracted.tags.contains(&"discovery".to_string()));

    assert_eq!(
        extracted.metadata["pattern"],
        serde_json::Value::String("**/*.rs".to_string())
    );
    assert_eq!(
        extracted.metadata["tool"],
        serde_json::Value::String("Glob".to_string())
    );

    // Glob has no graph node
    assert!(extracted.graph_node.is_none());
}

// ── Grep Tool Tests ────────────────────────────────────────────────────────

#[test]
fn grep_tool_extracts_pattern_memory() {
    let payload_json = r#"{
        "tool_name": "Grep",
        "tool_input": {"pattern": "fn main"},
        "tool_response": "src/main.rs:1:fn main() {\nsrc/bin/server.rs:5:fn main() {"
    }"#;

    let payload = parse_payload(payload_json).unwrap();
    let extracted = extract(&payload).unwrap().unwrap();

    assert_eq!(extracted.memory_type, MemoryType::Pattern);
    assert!(extracted.content.contains("Grep search: fn main"));
    assert!(extracted.content.contains("src/main.rs"));

    assert!(extracted.tags.contains(&"pattern:fn main".to_string()));
    assert!(extracted.tags.contains(&"search".to_string()));

    assert_eq!(
        extracted.metadata["tool"],
        serde_json::Value::String("Grep".to_string())
    );
}

// ── Edit Tool Tests ────────────────────────────────────────────────────────

#[test]
fn edit_tool_extracts_decision_memory() {
    let payload_json = r#"{
        "tool_name": "Edit",
        "tool_input": {
            "file_path": "src/handler.rs",
            "old_string": "fn handle(req: Request) -> Response {",
            "new_string": "async fn handle(req: Request) -> Result<Response, Error> {"
        },
        "tool_response": "OK"
    }"#;

    let payload = parse_payload(payload_json).unwrap();
    let extracted = extract(&payload).unwrap().unwrap();

    assert_eq!(extracted.memory_type, MemoryType::Decision);
    assert!(extracted.content.contains("Edit: src/handler.rs"));
    assert!(extracted
        .content
        .contains("fn handle(req: Request) -> Response"));
    assert!(extracted
        .content
        .contains("async fn handle(req: Request) -> Result<Response, Error>"));

    assert!(extracted.tags.contains(&"ext:rs".to_string()));
    assert!(extracted.tags.contains(&"dir:src".to_string()));
    assert!(extracted.tags.contains(&"file:handler.rs".to_string()));

    // Edit creates a graph node
    let graph_node = extracted.graph_node.unwrap();
    assert_eq!(graph_node.id, "file:src/handler.rs");
}

#[test]
fn multiedit_tool_extracts_decision_memory() {
    let payload_json = r#"{
        "tool_name": "MultiEdit",
        "tool_input": {
            "file_path": "src/config.rs",
            "old_string": "const MAX: usize = 100;",
            "new_string": "const MAX: usize = 1000;"
        },
        "tool_response": "OK"
    }"#;

    let payload = parse_payload(payload_json).unwrap();
    let extracted = extract(&payload).unwrap().unwrap();

    assert_eq!(extracted.memory_type, MemoryType::Decision);
    assert!(extracted.content.contains("Edit: src/config.rs"));
}

// ── Write Tool Tests ───────────────────────────────────────────────────────

#[test]
fn write_tool_extracts_decision_memory() {
    let payload_json = r#"{
        "tool_name": "Write",
        "tool_input": {"file_path": "tests/integration_test.rs"},
        "tool_response": "File written successfully (42 bytes)"
    }"#;

    let payload = parse_payload(payload_json).unwrap();
    let extracted = extract(&payload).unwrap().unwrap();

    assert_eq!(extracted.memory_type, MemoryType::Decision);
    assert!(extracted
        .content
        .contains("File written: tests/integration_test.rs"));

    assert!(extracted.tags.contains(&"ext:rs".to_string()));
    assert!(extracted.tags.contains(&"dir:tests".to_string()));
    assert!(extracted
        .tags
        .contains(&"file:integration_test.rs".to_string()));

    let graph_node = extracted.graph_node.unwrap();
    assert_eq!(graph_node.id, "file:tests/integration_test.rs");
    assert_eq!(graph_node.kind, codemem_core::NodeKind::File);
}

// ── Unknown Tool Tests ─────────────────────────────────────────────────────

#[test]
fn unknown_tool_returns_none() {
    let payload_json = r#"{
        "tool_name": "Bash",
        "tool_input": {"command": "ls"},
        "tool_response": "file1.txt\nfile2.txt"
    }"#;

    let payload = parse_payload(payload_json).unwrap();
    let extracted = extract(&payload).unwrap();
    assert!(extracted.is_none());
}

// ── Large Response Skipping ────────────────────────────────────────────────

#[test]
fn large_response_skipped() {
    let large = "x".repeat(100 * 1024 + 1); // > 100KB
    let payload_json = format!(
        r#"{{"tool_name": "Read", "tool_input": {{"file_path": "big.bin"}}, "tool_response": "{large}"}}"#
    );

    let payload = parse_payload(&payload_json).unwrap();
    let extracted = extract(&payload).unwrap();
    assert!(extracted.is_none());
}

// ── Malformed Payload ──────────────────────────────────────────────────────

#[test]
fn malformed_json_returns_error() {
    let result = parse_payload("not json at all");
    assert!(result.is_err());
}

#[test]
fn missing_required_fields_returns_error() {
    // Missing tool_response
    let result = parse_payload(r#"{"tool_name": "Read", "tool_input": {}}"#);
    assert!(result.is_err());
}

// ── Content Hash ───────────────────────────────────────────────────────────

#[test]
fn content_hash_is_deterministic() {
    let h1 = codemem_hooks::content_hash("hello world");
    let h2 = codemem_hooks::content_hash("hello world");
    assert_eq!(h1, h2);
}

#[test]
fn content_hash_differs_for_different_input() {
    let h1 = codemem_hooks::content_hash("hello");
    let h2 = codemem_hooks::content_hash("world");
    assert_ne!(h1, h2);
}

// ── Optional Fields ────────────────────────────────────────────────────────

#[test]
fn payload_without_optional_fields_parses() {
    let payload_json = r#"{
        "tool_name": "Read",
        "tool_input": {"file_path": "test.rs"},
        "tool_response": "fn test() {}"
    }"#;

    let payload = parse_payload(payload_json).unwrap();
    assert!(payload.session_id.is_none());
    assert!(payload.cwd.is_none());

    let extracted = extract(&payload).unwrap().unwrap();
    assert_eq!(extracted.memory_type, MemoryType::Context);
}

// ── Edge Cases ─────────────────────────────────────────────────────────────

#[test]
fn read_tool_missing_file_path_uses_unknown() {
    let payload_json = r#"{
        "tool_name": "Read",
        "tool_input": {},
        "tool_response": "some content"
    }"#;

    let payload = parse_payload(payload_json).unwrap();
    let extracted = extract(&payload).unwrap().unwrap();
    assert!(extracted.content.contains("File read: unknown"));
}

#[test]
fn glob_tool_missing_pattern_uses_star() {
    let payload_json = r#"{
        "tool_name": "Glob",
        "tool_input": {},
        "tool_response": "file1.txt"
    }"#;

    let payload = parse_payload(payload_json).unwrap();
    let extracted = extract(&payload).unwrap().unwrap();
    assert!(extracted.content.contains("Glob search: *"));
    assert!(extracted.tags.contains(&"glob:*".to_string()));
}

#[test]
fn grep_tool_empty_pattern() {
    let payload_json = r#"{
        "tool_name": "Grep",
        "tool_input": {},
        "tool_response": "match1\nmatch2"
    }"#;

    let payload = parse_payload(payload_json).unwrap();
    let extracted = extract(&payload).unwrap().unwrap();
    assert!(extracted.content.contains("Grep search:"));
    assert!(extracted.tags.contains(&"pattern:".to_string()));
}

#[test]
fn edit_tool_missing_strings_uses_empty() {
    let payload_json = r#"{
        "tool_name": "Edit",
        "tool_input": {"file_path": "src/main.rs"},
        "tool_response": "OK"
    }"#;

    let payload = parse_payload(payload_json).unwrap();
    let extracted = extract(&payload).unwrap().unwrap();
    assert_eq!(extracted.memory_type, MemoryType::Decision);
    assert!(extracted.content.contains("Edit: src/main.rs"));
}
