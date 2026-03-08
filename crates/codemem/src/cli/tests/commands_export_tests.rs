use super::*;

// ── csv_escape ──────────────────────────────────────────────────────

#[test]
fn csv_escape_plain_text() {
    assert_eq!(csv_escape("hello world"), "hello world");
}

#[test]
fn csv_escape_with_commas() {
    assert_eq!(csv_escape("hello,world"), "\"hello,world\"");
}

#[test]
fn csv_escape_with_quotes() {
    assert_eq!(csv_escape("he said \"hi\""), "\"he said \"\"hi\"\"\"");
}

#[test]
fn csv_escape_with_newlines() {
    assert_eq!(csv_escape("line1\nline2"), "\"line1\nline2\"");
}

#[test]
fn csv_escape_with_carriage_return() {
    assert_eq!(csv_escape("line1\rline2"), "\"line1\rline2\"");
}

#[test]
fn csv_escape_with_crlf() {
    assert_eq!(csv_escape("line1\r\nline2"), "\"line1\r\nline2\"");
}

#[test]
fn csv_escape_combined() {
    assert_eq!(csv_escape("a,b\n\"c\""), "\"a,b\n\"\"c\"\"\"");
}

// ── Export format rendering tests ───────────────────────────────────
//
// These call the extracted write_* functions directly, so any drift
// between the format logic and these tests is impossible.

fn sample_records() -> Vec<serde_json::Value> {
    vec![
        serde_json::json!({
            "id": "mem-1",
            "content": "decided to use Postgres",
            "memory_type": "decision",
            "importance": 0.8,
            "confidence": 0.9,
            "tags": ["arch", "db"],
            "namespace": "myproject",
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
            "edges": [],
        }),
        serde_json::json!({
            "id": "mem-2",
            "content": "auth module uses JWT tokens",
            "memory_type": "insight",
            "importance": 0.6,
            "confidence": 1.0,
            "tags": ["auth"],
            "namespace": "myproject",
            "created_at": "2026-01-02T00:00:00Z",
            "updated_at": "2026-01-02T00:00:00Z",
            "edges": [],
        }),
    ]
}

#[test]
fn export_jsonl_format_rendering() {
    let records = sample_records();
    let mut buf = Vec::new();
    write_jsonl(&mut buf, &records).unwrap();

    let output = String::from_utf8(buf).unwrap();
    let lines: Vec<&str> = output.lines().collect();
    assert_eq!(lines.len(), 2);

    let first: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
    assert_eq!(first["id"], "mem-1");
    assert_eq!(first["memory_type"], "decision");
}

#[test]
fn export_json_format_rendering() {
    let records = sample_records();
    let mut buf = Vec::new();
    write_json(&mut buf, &records).unwrap();

    let output = String::from_utf8(buf).unwrap();
    let parsed: Vec<serde_json::Value> = serde_json::from_str(&output).unwrap();
    assert_eq!(parsed.len(), 2);
    assert_eq!(parsed[0]["id"], "mem-1");
    assert_eq!(parsed[1]["id"], "mem-2");
}

#[test]
fn export_csv_format_rendering() {
    let records = sample_records();
    let mut buf = Vec::new();
    write_csv(&mut buf, &records).unwrap();

    let output = String::from_utf8(buf).unwrap();
    let lines: Vec<&str> = output.lines().collect();
    assert_eq!(lines.len(), 3, "header + 2 data rows");
    assert!(lines[0].starts_with("id,content,memory_type"));
    assert!(lines[1].contains("mem-1"));
    assert!(lines[1].contains("decision"));
    assert!(lines[1].contains("arch;db")); // tags joined with semicolon
}

#[test]
fn export_markdown_format_rendering() {
    let records = sample_records();
    let mut buf = Vec::new();
    write_markdown(&mut buf, &records, Some("myproject"), None).unwrap();

    let output = String::from_utf8(buf).unwrap();
    assert!(output.contains("# Codemem Export"));
    assert!(output.contains("**Total memories:** 2"));
    assert!(output.contains("**Namespace:** myproject"));
    assert!(output.contains("## decision (1 memories)"));
    assert!(output.contains("## insight (1 memories)"));
    assert!(output.contains("decided to use Postgres"));
    assert!(output.contains("auth module uses JWT tokens"));
    // Verify tag rendering
    assert!(output.contains("**Tags:** arch, db"));
    assert!(output.contains("**Tags:** auth"));
}

#[test]
fn export_markdown_with_type_filter() {
    let records = sample_records();
    let mut buf = Vec::new();
    write_markdown(&mut buf, &records, None, Some("decision")).unwrap();

    let output = String::from_utf8(buf).unwrap();
    assert!(output.contains("**Type filter:** decision"));
    assert!(!output.contains("**Namespace:**"));
}

#[test]
fn export_jsonl_empty_records() {
    let mut buf = Vec::new();
    write_jsonl(&mut buf, &[]).unwrap();
    assert!(buf.is_empty());
}

#[test]
fn export_csv_null_namespace_record() {
    let records = vec![serde_json::json!({
        "id": "mem-null-ns",
        "content": "no namespace",
        "memory_type": "context",
        "importance": 0.5,
        "confidence": 1.0,
        "tags": [],
        "namespace": null,
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
        "edges": [],
    })];
    // CSV: null namespace renders as empty string via .as_str().unwrap_or("")
    let mut buf = Vec::new();
    write_csv(&mut buf, &records).unwrap();
    let output = String::from_utf8(buf).unwrap();
    let data_line = output.lines().nth(1).unwrap();
    assert!(data_line.contains("mem-null-ns"));
    // Namespace field (column 7) should be empty, not literal "null"
    assert!(!data_line.contains(",null,"));
}

#[test]
fn export_markdown_null_namespace_record() {
    let records = vec![serde_json::json!({
        "id": "mem-null-ns",
        "content": "no namespace memory",
        "memory_type": "context",
        "importance": 0.5,
        "confidence": 1.0,
        "tags": [],
        "namespace": null,
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
        "edges": [],
    })];
    let mut buf = Vec::new();
    write_markdown(&mut buf, &records, None, None).unwrap();
    let output = String::from_utf8(buf).unwrap();
    assert!(output.contains("no namespace memory"));
    assert!(output.contains("## context (1 memories)"));
}

#[test]
fn export_csv_empty_records() {
    let mut buf = Vec::new();
    write_csv(&mut buf, &[]).unwrap();
    let output = String::from_utf8(buf).unwrap();
    // Should have only the header line
    assert_eq!(output.lines().count(), 1);
    assert!(output.starts_with("id,content,memory_type"));
}

#[test]
fn export_markdown_empty_records() {
    let mut buf = Vec::new();
    write_markdown(&mut buf, &[], None, None).unwrap();
    let output = String::from_utf8(buf).unwrap();
    assert!(output.contains("# Codemem Export"));
    assert!(output.contains("**Total memories:** 0"));
    // No type sections when there are no records
    assert!(!output.contains("## "));
}

// ── Import parsing tests ────────────────────────────────────────────

#[test]
fn import_line_parsing_valid_json() {
    let line = r#"{"content":"test memory","memory_type":"decision","importance":0.8,"confidence":0.9,"tags":["arch"],"namespace":"myns"}"#;
    let val: serde_json::Value = serde_json::from_str(line).unwrap();

    assert_eq!(val["content"].as_str().unwrap(), "test memory");
    assert_eq!(val["memory_type"].as_str().unwrap(), "decision");
    assert_eq!(val["importance"].as_f64().unwrap(), 0.8);
    assert_eq!(val["tags"].as_array().unwrap().len(), 1);
}

#[test]
fn import_line_parsing_minimal_fields() {
    // cmd_import only requires "content" — everything else has defaults
    let line = r#"{"content":"minimal memory"}"#;
    let val: serde_json::Value = serde_json::from_str(line).unwrap();

    let content = val.get("content").and_then(|v| v.as_str());
    assert_eq!(content, Some("minimal memory"));

    // Defaults that cmd_import would apply
    let memory_type: codemem_core::MemoryType = val
        .get("memory_type")
        .and_then(|v| v.as_str())
        .and_then(|s| s.parse().ok())
        .unwrap_or(codemem_core::MemoryType::Context);
    assert_eq!(memory_type, codemem_core::MemoryType::Context);

    let importance = val
        .get("importance")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.5);
    assert_eq!(importance, 0.5);

    let confidence = val
        .get("confidence")
        .and_then(|v| v.as_f64())
        .unwrap_or(1.0);
    assert_eq!(confidence, 1.0);
}

#[test]
fn import_line_parsing_invalid_json() {
    let line = "this is not json";
    let result: Result<serde_json::Value, _> = serde_json::from_str(line);
    assert!(result.is_err());
}

#[test]
fn import_line_parsing_missing_content() {
    let line = r#"{"memory_type":"decision","importance":0.8}"#;
    let val: serde_json::Value = serde_json::from_str(line).unwrap();
    let content = val.get("content").and_then(|v| v.as_str());
    assert!(content.is_none(), "should have no content field");
}

#[test]
fn import_line_parsing_empty_content() {
    let line = r#"{"content":"","memory_type":"decision"}"#;
    let val: serde_json::Value = serde_json::from_str(line).unwrap();
    let content = val.get("content").and_then(|v| v.as_str());
    assert_eq!(content, Some(""));
    // cmd_import skips empty content
    let should_skip = !matches!(content, Some(c) if !c.is_empty());
    assert!(should_skip);
}
