//! Hooks module: PostToolUse hook handler for passive capture from AI coding assistants.
//!
//! Parses PostToolUse JSON payloads from stdin, extracts relevant information
//! based on tool type, and creates appropriate memories with auto-tagging.

pub mod diff;

use codemem_core::{CodememError, GraphNode, MemoryType, NodeKind, RelationshipType};
use serde::Deserialize;
use std::collections::HashMap;

/// Maximum file size to process (100KB).
const MAX_CONTENT_SIZE: usize = 100 * 1024;

/// PostToolUse hook payload from an AI coding assistant.
#[derive(Debug, Deserialize)]
pub struct HookPayload {
    pub tool_name: String,
    pub tool_input: serde_json::Value,
    pub tool_response: String,
    pub session_id: Option<String>,
    pub cwd: Option<String>,
}

/// Extracted memory from a hook payload.
#[derive(Debug)]
pub struct ExtractedMemory {
    pub content: String,
    pub memory_type: MemoryType,
    pub tags: Vec<String>,
    pub metadata: HashMap<String, serde_json::Value>,
    pub graph_node: Option<GraphNode>,
    pub graph_edges: Vec<PendingEdge>,
    pub session_id: Option<String>,
}

/// A pending edge to be created once both nodes exist.
#[derive(Debug)]
pub struct PendingEdge {
    pub src_id: String,
    pub dst_id: String,
    pub relationship: RelationshipType,
}

/// Parse a hook payload from JSON string.
pub fn parse_payload(json: &str) -> Result<HookPayload, CodememError> {
    serde_json::from_str(json)
        .map_err(|e| CodememError::Hook(format!("Failed to parse payload: {e}")))
}

/// Extract memory from a hook payload.
pub fn extract(payload: &HookPayload) -> Result<Option<ExtractedMemory>, CodememError> {
    // Skip large responses
    if payload.tool_response.len() > MAX_CONTENT_SIZE {
        tracing::debug!(
            "Skipping large response ({} bytes)",
            payload.tool_response.len()
        );
        return Ok(None);
    }

    match payload.tool_name.as_str() {
        "Read" => extract_read(payload),
        "Glob" => extract_glob(payload),
        "Grep" => extract_grep(payload),
        "Edit" | "MultiEdit" => extract_edit(payload),
        "Write" => extract_write(payload),
        "Bash" => extract_bash(payload),
        "WebFetch" | "WebSearch" => extract_web(payload),
        "Agent" | "SendMessage" => extract_agent_communication(payload),
        "ListFiles" | "ListDir" => extract_list_dir(payload),
        _ => {
            tracing::debug!("Unknown tool: {}", payload.tool_name);
            Ok(None)
        }
    }
}

/// Populate `graph_edges` on an `ExtractedMemory` by checking which file graph
/// nodes already exist in the database.  This creates edges between files that
/// were previously Read and are now being Edited or Written, capturing the
/// common explore-then-modify workflow.
///
/// `existing_node_ids` should be the set of graph-node IDs already persisted
/// (e.g. from `storage.all_graph_nodes()`).
pub fn resolve_edges(
    extracted: &mut ExtractedMemory,
    existing_node_ids: &std::collections::HashSet<String>,
) {
    // Only file-level tools produce a graph_node with id "file:<path>"
    let current_node_id = match &extracted.graph_node {
        Some(node) => node.id.clone(),
        None => return,
    };

    // Determine the tool that produced this memory
    let tool = extracted
        .metadata
        .get("tool")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    // Only Edit and Write events create edges back to previously-seen files.
    // If the same file was previously Read, the file node already exists.
    // An edit/write after a read represents an evolution of understanding.
    match tool {
        "Edit" | "Write" => {
            if existing_node_ids.contains(&current_node_id) {
                extracted.graph_edges.push(PendingEdge {
                    src_id: current_node_id,
                    dst_id: String::new(), // self-edge marker
                    relationship: RelationshipType::EvolvedInto,
                });
            }
        }
        _ => {}
    }
}

/// Resolve pending edges into concrete `Edge` values, given the memory ID that
/// was just stored and the set of existing graph-node IDs.
///
/// Self-edge markers (dst_id == "") use the same node as both src and dst,
/// representing a file that evolved (was read then edited/written).
pub fn materialize_edges(pending: &[PendingEdge], memory_id: &str) -> Vec<codemem_core::Edge> {
    let now = chrono::Utc::now();
    pending
        .iter()
        .map(|pe| {
            // Self-edge marker: dst_id is empty, so create a self-referencing edge.
            if pe.dst_id.is_empty() {
                // For an EVOLVED_INTO self-reference, the src node already exists
                // from the prior Read; we create an edge from the existing node
                // to itself, annotated with the memory that triggered it.
                let edge_id = format!("{}-{}-{}", pe.src_id, pe.relationship, memory_id);
                let mut props = HashMap::new();
                props.insert(
                    "triggered_by".to_string(),
                    serde_json::Value::String(memory_id.to_string()),
                );
                codemem_core::Edge {
                    id: edge_id,
                    src: pe.src_id.clone(),
                    dst: pe.src_id.clone(),
                    relationship: pe.relationship,
                    weight: 1.0,
                    properties: props,
                    created_at: now,
                    valid_from: None,
                    valid_to: None,
                }
            } else {
                let edge_id = format!("{}-{}-{}", pe.src_id, pe.relationship, pe.dst_id);
                codemem_core::Edge {
                    id: edge_id,
                    src: pe.src_id.clone(),
                    dst: pe.dst_id.clone(),
                    relationship: pe.relationship,
                    weight: 1.0,
                    properties: HashMap::new(),
                    created_at: now,
                    valid_from: None,
                    valid_to: None,
                }
            }
        })
        .collect()
}

/// Content hash for deduplication.
pub use codemem_core::content_hash;

/// Relativize an absolute file path against the hook's cwd.
/// Returns the relative path if cwd is set and the path starts with it,
/// otherwise returns the original path.
fn relativize_path<'a>(path: &'a str, cwd: Option<&str>) -> &'a str {
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
fn build_file_extraction(
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
fn extract_read(payload: &HookPayload) -> Result<Option<ExtractedMemory>, CodememError> {
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
fn extract_glob(payload: &HookPayload) -> Result<Option<ExtractedMemory>, CodememError> {
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
fn extract_grep(payload: &HookPayload) -> Result<Option<ExtractedMemory>, CodememError> {
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
fn extract_edit(payload: &HookPayload) -> Result<Option<ExtractedMemory>, CodememError> {
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

    let content = format!(
        "Edit: {}\nChanged:\n  - {}\n  + {}",
        file_path,
        truncate(old_string, 500),
        truncate(new_string, 500)
    );

    Ok(Some(build_file_extraction(
        payload,
        file_path,
        content,
        MemoryType::Decision,
        "Edit",
    )))
}

/// Extract memory from a Write tool use.
fn extract_write(payload: &HookPayload) -> Result<Option<ExtractedMemory>, CodememError> {
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
fn extract_bash(payload: &HookPayload) -> Result<Option<ExtractedMemory>, CodememError> {
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
fn extract_web(payload: &HookPayload) -> Result<Option<ExtractedMemory>, CodememError> {
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
fn extract_agent_communication(
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
fn extract_list_dir(payload: &HookPayload) -> Result<Option<ExtractedMemory>, CodememError> {
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
fn extract_tags_from_path(path: &str) -> Vec<String> {
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
fn truncate(s: &str, max_len: usize) -> &str {
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

// ── Trigger-Based Auto-Insights ─────────────────────────────────────────

/// An auto-insight generated by trigger-based analysis during PostToolUse.
#[derive(Debug, Clone)]
pub struct AutoInsight {
    /// The insight content to store as a memory.
    pub content: String,
    /// Tags to attach to the insight memory.
    pub tags: Vec<String>,
    /// Importance score for the insight.
    pub importance: f64,
    /// Unique tag used for deduplication within a session.
    pub dedup_tag: String,
}

/// Check trigger conditions against session activity and return any auto-insights.
///
/// Three triggers are evaluated:
/// 1. **Directory focus**: 3+ files read from the same directory suggests deep exploration.
/// 2. **Edit after read**: Editing a file that was previously read indicates an informed change.
/// 3. **Repeated search**: Same search pattern used 2+ times suggests a recurring need.
///
/// Each trigger checks `has_auto_insight()` to avoid duplicate insights within the same session.
pub fn check_triggers(
    storage: &dyn codemem_core::StorageBackend,
    session_id: &str,
    tool_name: &str,
    file_path: Option<&str>,
    pattern: Option<&str>,
) -> Vec<AutoInsight> {
    let mut insights = Vec::new();

    // Trigger 1: 3+ files read from the same directory
    if tool_name == "Read" {
        if let Some(fp) = file_path {
            let directory = std::path::Path::new(fp)
                .parent()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_default();
            if !directory.is_empty() {
                let dedup_tag = format!("dir_focus:{}", directory);
                let already_exists = storage
                    .has_auto_insight(session_id, &dedup_tag)
                    .unwrap_or(true);
                if !already_exists {
                    let count = storage
                        .count_directory_reads(session_id, &directory)
                        .unwrap_or(0);
                    if count >= 3 {
                        insights.push(AutoInsight {
                            content: format!(
                                "Deep exploration of directory '{}': {} files read in this session. \
                                 This area may be a focus of the current task.",
                                directory, count
                            ),
                            tags: vec![
                                "auto-insight".to_string(),
                                "directory-focus".to_string(),
                                format!("dir:{}", directory),
                            ],
                            importance: 0.6,
                            dedup_tag,
                        });
                    }
                }
            }
        }
    }

    // Trigger 2: Edit after read — an informed change
    if matches!(tool_name, "Edit" | "Write") {
        if let Some(fp) = file_path {
            let dedup_tag = format!("edit_after_read:{}", fp);
            let already_exists = storage
                .has_auto_insight(session_id, &dedup_tag)
                .unwrap_or(true);
            if !already_exists {
                let was_read = storage
                    .was_file_read_in_session(session_id, fp)
                    .unwrap_or(false);
                if was_read {
                    insights.push(AutoInsight {
                        content: format!(
                            "File '{}' was read and then modified in this session, \
                             indicating an informed change based on code review.",
                            fp
                        ),
                        tags: vec![
                            "auto-insight".to_string(),
                            "edit-after-read".to_string(),
                            format!(
                                "file:{}",
                                std::path::Path::new(fp)
                                    .file_name()
                                    .and_then(|f| f.to_str())
                                    .unwrap_or("unknown")
                            ),
                        ],
                        importance: 0.5,
                        dedup_tag,
                    });
                }
            }
        }
    }

    // Trigger 3: "Understanding module" — 3+ files read from the same directory
    // (reuses the directory focus data but generates a module-level insight)
    if tool_name == "Read" {
        if let Some(fp) = file_path {
            let directory = std::path::Path::new(fp)
                .parent()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_default();
            if !directory.is_empty() {
                let module_name = std::path::Path::new(&directory)
                    .file_name()
                    .and_then(|f| f.to_str())
                    .unwrap_or("unknown");
                let dedup_tag = format!("exploring_module:{}", directory);
                let already_exists = storage
                    .has_auto_insight(session_id, &dedup_tag)
                    .unwrap_or(true);
                if !already_exists {
                    let count = storage
                        .count_directory_reads(session_id, &directory)
                        .unwrap_or(0);
                    if count >= 3 {
                        insights.push(AutoInsight {
                            content: format!(
                                "Exploring '{}' module: {} files read. Building understanding of this area.",
                                module_name, count
                            ),
                            tags: vec![
                                "auto-insight".to_string(),
                                "exploring-module".to_string(),
                                format!("module:{}", module_name),
                            ],
                            importance: 0.55,
                            dedup_tag,
                        });
                    }
                }
            }
        }
    }

    // Trigger 4: "Debugging" — Bash with error output followed by file reads
    if tool_name == "Bash" {
        let has_error = storage
            .count_search_pattern_in_session(session_id, "error")
            .unwrap_or(0)
            > 0;
        if has_error {
            let area = file_path
                .and_then(|fp| {
                    std::path::Path::new(fp)
                        .parent()
                        .and_then(|p| p.file_name())
                        .and_then(|f| f.to_str())
                })
                .unwrap_or("project");
            let dedup_tag = format!("debugging:{}", area);
            let already_exists = storage
                .has_auto_insight(session_id, &dedup_tag)
                .unwrap_or(true);
            if !already_exists {
                insights.push(AutoInsight {
                    content: format!(
                        "Debugging in '{}': error output detected in bash commands during this session.",
                        area
                    ),
                    tags: vec![
                        "auto-insight".to_string(),
                        "debugging".to_string(),
                        format!("area:{}", area),
                    ],
                    importance: 0.6,
                    dedup_tag,
                });
            }
        }
    }

    // Trigger 5: Same search pattern used 2+ times
    if matches!(tool_name, "Grep" | "Glob") {
        if let Some(pat) = pattern {
            let dedup_tag = format!("repeated_search:{}", pat);
            let already_exists = storage
                .has_auto_insight(session_id, &dedup_tag)
                .unwrap_or(true);
            if !already_exists {
                let count = storage
                    .count_search_pattern_in_session(session_id, pat)
                    .unwrap_or(0);
                if count >= 2 {
                    insights.push(AutoInsight {
                        content: format!(
                            "Search pattern '{}' used {} times in this session. \
                             Consider storing a permanent memory for this recurring lookup.",
                            pat, count
                        ),
                        tags: vec![
                            "auto-insight".to_string(),
                            "repeated-search".to_string(),
                            format!("pattern:{}", pat),
                        ],
                        importance: 0.5,
                        dedup_tag,
                    });
                }
            }
        }
    }

    insights
}

#[cfg(test)]
#[path = "tests/lib_tests.rs"]
mod tests;
