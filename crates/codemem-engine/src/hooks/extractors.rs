//! Extractor functions for each tool type in the PostToolUse hook.

use codemem_core::{CodememError, GraphNode, MemoryType, NodeKind};
use std::collections::HashMap;

use super::diff::compute_diff;
use super::{ExtractedMemory, HookPayload};

/// Relativize an absolute file path against the hook's cwd.
/// Returns the relative path if cwd is set and the path starts with it,
/// otherwise returns the original path.
pub(super) fn relativize_path<'a>(path: &'a str, cwd: Option<&str>) -> &'a str {
    if let Some(root) = cwd {
        let root_slash = if root.ends_with('/') {
            std::borrow::Cow::Borrowed(root)
        } else {
            std::borrow::Cow::Owned(format!("{root}/"))
        };
        if let Some(rel) = path.strip_prefix(root_slash.as_ref()) {
            return rel;
        }
    }
    path
}

/// Build an `ExtractedMemory` for file-based tools (Read, Edit, Write).
pub(super) fn build_file_extraction(
    payload: &HookPayload,
    file_path: &str,
    content: String,
    memory_type: MemoryType,
    tool_name: &str,
) -> ExtractedMemory {
    let rel_path = relativize_path(file_path, payload.cwd.as_deref());
    let tags = extract_tags_from_path(rel_path);
    let graph_node = Some(GraphNode {
        id: format!("file:{rel_path}"),
        kind: NodeKind::File,
        label: rel_path.to_string(),
        payload: HashMap::new(),
        centrality: 0.0,
        memory_id: None,
        namespace: None,
    });
    let mut metadata = HashMap::new();
    metadata.insert(
        "file_path".to_string(),
        serde_json::Value::String(rel_path.to_string()),
    );
    metadata.insert(
        "tool".to_string(),
        serde_json::Value::String(tool_name.to_string()),
    );
    ExtractedMemory {
        content,
        memory_type,
        tags,
        metadata,
        graph_node,
        graph_edges: vec![],
        session_id: payload.session_id.clone(),
    }
}

/// Extract memory from a Read tool use.
pub(super) fn extract_read(payload: &HookPayload) -> Result<Option<ExtractedMemory>, CodememError> {
    let file_path = payload
        .tool_input
        .get("file_path")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");

    let content = format!(
        "File read: {}\n\n{}",
        file_path,
        truncate(&payload.tool_response, 2000)
    );

    Ok(Some(build_file_extraction(
        payload,
        file_path,
        content,
        MemoryType::Context,
        "Read",
    )))
}

/// Extract memory from a Glob tool use.
pub(super) fn extract_glob(payload: &HookPayload) -> Result<Option<ExtractedMemory>, CodememError> {
    let pattern = payload
        .tool_input
        .get("pattern")
        .and_then(|v| v.as_str())
        .unwrap_or("*");

    let content = format!(
        "Glob search: {}\nResults:\n{}",
        pattern,
        truncate(&payload.tool_response, 2000)
    );

    let tags = vec![format!("glob:{pattern}"), "discovery".to_string()];

    Ok(Some(ExtractedMemory {
        content,
        memory_type: MemoryType::Pattern,
        tags,
        metadata: {
            let mut m = HashMap::new();
            m.insert(
                "pattern".to_string(),
                serde_json::Value::String(pattern.to_string()),
            );
            m.insert(
                "tool".to_string(),
                serde_json::Value::String("Glob".to_string()),
            );
            m
        },
        graph_node: None,
        graph_edges: vec![],
        session_id: payload.session_id.clone(),
    }))
}

/// Extract memory from a Grep tool use.
pub(super) fn extract_grep(payload: &HookPayload) -> Result<Option<ExtractedMemory>, CodememError> {
    let pattern = payload
        .tool_input
        .get("pattern")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    let content = format!(
        "Grep search: {}\nMatches:\n{}",
        pattern,
        truncate(&payload.tool_response, 2000)
    );

    let tags = vec![format!("pattern:{pattern}"), "search".to_string()];

    Ok(Some(ExtractedMemory {
        content,
        memory_type: MemoryType::Pattern,
        tags,
        metadata: {
            let mut m = HashMap::new();
            m.insert(
                "pattern".to_string(),
                serde_json::Value::String(pattern.to_string()),
            );
            m.insert(
                "tool".to_string(),
                serde_json::Value::String("Grep".to_string()),
            );
            m
        },
        graph_node: None,
        graph_edges: vec![],
        session_id: payload.session_id.clone(),
    }))
}

/// Extract memory from an Edit/MultiEdit tool use.
pub(super) fn extract_edit(payload: &HookPayload) -> Result<Option<ExtractedMemory>, CodememError> {
    let file_path = payload
        .tool_input
        .get("file_path")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");

    let old_string = payload
        .tool_input
        .get("old_string")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    let new_string = payload
        .tool_input
        .get("new_string")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    // Compute a semantic diff summary from the old/new strings.
    let diff_summary = compute_diff(file_path, old_string, new_string);

    let content = format!(
        "Edit: {}\nSemantic summary: {}\nChanged:\n  - {}\n  + {}",
        file_path,
        diff_summary.semantic_summary,
        truncate(old_string, 500),
        truncate(new_string, 500)
    );

    let mut extraction =
        build_file_extraction(payload, file_path, content, MemoryType::Decision, "Edit");
    extraction.metadata.insert(
        "semantic_summary".to_string(),
        serde_json::Value::String(diff_summary.semantic_summary),
    );
    extraction.metadata.insert(
        "lines_added".to_string(),
        serde_json::json!(diff_summary.lines_added),
    );
    extraction.metadata.insert(
        "lines_removed".to_string(),
        serde_json::json!(diff_summary.lines_removed),
    );
    extraction.metadata.insert(
        "change_type".to_string(),
        serde_json::Value::String(diff_summary.change_type.to_string()),
    );

    Ok(Some(extraction))
}

/// Extract memory from a Write tool use.
pub(super) fn extract_write(
    payload: &HookPayload,
) -> Result<Option<ExtractedMemory>, CodememError> {
    let file_path = payload
        .tool_input
        .get("file_path")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");

    let content = format!(
        "File written: {}\n\n{}",
        file_path,
        truncate(&payload.tool_response, 2000)
    );

    Ok(Some(build_file_extraction(
        payload,
        file_path,
        content,
        MemoryType::Decision,
        "Write",
    )))
}

/// Extract memory from a Bash tool use.
pub(super) fn extract_bash(payload: &HookPayload) -> Result<Option<ExtractedMemory>, CodememError> {
    let command = payload
        .tool_input
        .get("command")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    let first_word = command.split_whitespace().next().unwrap_or("unknown");
    let response = truncate(&payload.tool_response, 2000);

    let content = format!("Bash command: {}\nOutput:\n{}", command, response);

    let mut tags = vec!["bash".to_string(), format!("command:{first_word}")];

    // Add directory tag if present in input
    if let Some(dir) = payload.tool_input.get("cwd").and_then(|v| v.as_str()) {
        tags.push(format!("dir:{dir}"));
    } else if let Some(dir) = payload.cwd.as_deref() {
        tags.push(format!("dir:{dir}"));
    }

    // Detect error indicators
    let response_lower = payload.tool_response.to_lowercase();
    if response_lower.contains("error:")
        || response_lower.contains("failed")
        || payload
            .tool_input
            .get("exit_code")
            .and_then(|v| v.as_i64())
            .is_some_and(|c| c != 0)
    {
        tags.push("error".to_string());
    }

    let mut metadata = HashMap::new();
    metadata.insert(
        "tool".to_string(),
        serde_json::Value::String("Bash".to_string()),
    );
    metadata.insert(
        "command".to_string(),
        serde_json::Value::String(command.to_string()),
    );

    // Try to detect a file path reference in the command for graph node creation
    let graph_node = extract_file_path_from_command(command).map(|fp| {
        let rel = relativize_path(fp, payload.cwd.as_deref());
        GraphNode {
            id: format!("file:{rel}"),
            kind: NodeKind::File,
            label: rel.to_string(),
            payload: HashMap::new(),
            centrality: 0.0,
            memory_id: None,
            namespace: None,
        }
    });

    Ok(Some(ExtractedMemory {
        content,
        memory_type: MemoryType::Context,
        tags,
        metadata,
        graph_node,
        graph_edges: vec![],
        session_id: payload.session_id.clone(),
    }))
}

/// Try to extract a recognizable file path from a bash command string.
/// Looks for arguments that look like file paths (contain `/` or `.` with an extension).
fn extract_file_path_from_command(command: &str) -> Option<&str> {
    for token in command.split_whitespace() {
        // Skip flags
        if token.starts_with('-') {
            continue;
        }
        // Check for path-like tokens: contains a slash or has a file extension
        let path = std::path::Path::new(token);
        if token.contains('/') || path.extension().is_some() {
            // Validate it looks like a real path (not a URL scheme, not just a dot)
            if !token.starts_with("http://") && !token.starts_with("https://") && token.len() > 1 {
                return Some(token);
            }
        }
    }
    None
}

/// Extract memory from a WebFetch/WebSearch tool use.
pub(super) fn extract_web(payload: &HookPayload) -> Result<Option<ExtractedMemory>, CodememError> {
    let url = payload
        .tool_input
        .get("url")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    let query = payload
        .tool_input
        .get("query")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    let response = truncate(&payload.tool_response, 2000);

    let content = if !url.is_empty() {
        format!("Web fetch: {url}\nResponse:\n{response}")
    } else {
        format!("Web search: {query}\nResults:\n{response}")
    };

    let mut tags = vec!["web-research".to_string()];

    // Extract domain from URL
    if !url.is_empty() {
        if let Some(domain) = extract_domain(url) {
            tags.push(format!("url:{domain}"));
        }
    }

    if !query.is_empty() {
        tags.push(format!("query:{query}"));
    }

    let mut metadata = HashMap::new();
    metadata.insert(
        "tool".to_string(),
        serde_json::Value::String(payload.tool_name.clone()),
    );
    if !url.is_empty() {
        metadata.insert(
            "url".to_string(),
            serde_json::Value::String(url.to_string()),
        );
    }
    if !query.is_empty() {
        metadata.insert(
            "query".to_string(),
            serde_json::Value::String(query.to_string()),
        );
    }

    Ok(Some(ExtractedMemory {
        content,
        memory_type: MemoryType::Context,
        tags,
        metadata,
        graph_node: None,
        graph_edges: vec![],
        session_id: payload.session_id.clone(),
    }))
}

/// Extract domain from a URL string.
fn extract_domain(url: &str) -> Option<&str> {
    let after_scheme = url
        .strip_prefix("https://")
        .or_else(|| url.strip_prefix("http://"))
        .unwrap_or(url);
    let domain = after_scheme.split('/').next()?;
    if domain.is_empty() {
        None
    } else {
        Some(domain)
    }
}

/// Extract memory from Agent/SendMessage tool uses.
pub(super) fn extract_agent_communication(
    payload: &HookPayload,
) -> Result<Option<ExtractedMemory>, CodememError> {
    let response = truncate(&payload.tool_response, 2000);

    let content = format!("Agent communication ({}): {}", payload.tool_name, response);

    let mut metadata = HashMap::new();
    metadata.insert(
        "tool".to_string(),
        serde_json::Value::String(payload.tool_name.clone()),
    );

    Ok(Some(ExtractedMemory {
        content,
        memory_type: MemoryType::Context,
        tags: vec!["agent-communication".to_string()],
        metadata,
        graph_node: None,
        graph_edges: vec![],
        session_id: payload.session_id.clone(),
    }))
}

/// Extract memory from ListFiles/ListDir tool uses.
pub(super) fn extract_list_dir(
    payload: &HookPayload,
) -> Result<Option<ExtractedMemory>, CodememError> {
    let directory = payload
        .tool_input
        .get("path")
        .or_else(|| payload.tool_input.get("directory"))
        .and_then(|v| v.as_str())
        .unwrap_or(".");

    let response = truncate(&payload.tool_response, 2000);
    let content = format!("Listed directory: {directory}\n{response}");

    let mut tags = vec!["discovery".to_string()];
    // Add the directory basename as a tag
    if let Some(name) = std::path::Path::new(directory)
        .file_name()
        .and_then(|f| f.to_str())
    {
        tags.push(format!("dir:{name}"));
    }

    let mut metadata = HashMap::new();
    metadata.insert(
        "tool".to_string(),
        serde_json::Value::String(payload.tool_name.clone()),
    );
    metadata.insert(
        "directory".to_string(),
        serde_json::Value::String(directory.to_string()),
    );

    Ok(Some(ExtractedMemory {
        content,
        memory_type: MemoryType::Context,
        tags,
        metadata,
        graph_node: None,
        graph_edges: vec![],
        session_id: payload.session_id.clone(),
    }))
}

/// Extract entity tags from a file path.
pub(super) fn extract_tags_from_path(path: &str) -> Vec<String> {
    let mut tags = Vec::new();

    // Add file extension tag
    if let Some(ext) = std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
    {
        tags.push(format!("ext:{ext}"));
    }

    // Add directory path components as tags
    let parts: Vec<&str> = path.split('/').collect();
    if parts.len() > 1 {
        // Add parent directory
        if let Some(parent) = parts.get(parts.len() - 2) {
            tags.push(format!("dir:{parent}"));
        }
    }

    // Add filename
    if let Some(filename) = std::path::Path::new(path)
        .file_name()
        .and_then(|f| f.to_str())
    {
        tags.push(format!("file:{filename}"));
    }

    tags
}

/// Truncate string to max length, respecting UTF-8 char boundaries.
pub(super) fn truncate(s: &str, max_len: usize) -> &str {
    if s.len() <= max_len {
        s
    } else {
        let mut end = max_len;
        while end > 0 && !s.is_char_boundary(end) {
            end -= 1;
        }
        &s[..end]
    }
}
