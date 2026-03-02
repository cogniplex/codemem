//! codemem-hooks: PostToolUse hook handler for passive capture from AI coding assistants.
//!
//! Parses PostToolUse JSON payloads from stdin, extracts relevant information
//! based on tool type, and creates appropriate memories with auto-tagging.

pub mod diff;

use codemem_core::{CodememError, GraphNode, MemoryType, NodeKind, RelationshipType};
use serde::Deserialize;
use sha2::{Digest, Sha256};
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

    // Only Edit and Write events create edges back to previously-seen files
    match tool {
        "Edit" => {
            // If the same file was previously Read, the file node already exists.
            // An edit after a read represents an evolution of understanding.
            if existing_node_ids.contains(&current_node_id) {
                extracted.graph_edges.push(PendingEdge {
                    src_id: current_node_id,
                    dst_id: String::new(), // self-edge marker; will be skipped
                    relationship: RelationshipType::EvolvedInto,
                });
            }
        }
        "Write" => {
            // A Write to a previously-seen file is also an evolution.
            if existing_node_ids.contains(&current_node_id) {
                extracted.graph_edges.push(PendingEdge {
                    src_id: current_node_id,
                    dst_id: String::new(),
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
            // Skip self-edge markers where src == dst would be meaningless
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
pub fn content_hash(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Extract memory from a Read tool use.
fn extract_read(payload: &HookPayload) -> Result<Option<ExtractedMemory>, CodememError> {
    let file_path = payload
        .tool_input
        .get("file_path")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");

    // Create a summary of the file content
    let content = format!(
        "File read: {}\n\n{}",
        file_path,
        truncate(&payload.tool_response, 2000)
    );

    let tags = extract_tags_from_path(file_path);

    let graph_node = Some(GraphNode {
        id: format!("file:{file_path}"),
        kind: NodeKind::File,
        label: file_path.to_string(),
        payload: HashMap::new(),
        centrality: 0.0,
        memory_id: None,
        namespace: None,
    });

    Ok(Some(ExtractedMemory {
        content,
        memory_type: MemoryType::Context,
        tags,
        metadata: {
            let mut m = HashMap::new();
            m.insert(
                "file_path".to_string(),
                serde_json::Value::String(file_path.to_string()),
            );
            m.insert(
                "tool".to_string(),
                serde_json::Value::String("Read".to_string()),
            );
            m
        },
        graph_node,
        graph_edges: vec![],
        session_id: payload.session_id.clone(),
    }))
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

    let tags = extract_tags_from_path(file_path);

    let graph_node = Some(GraphNode {
        id: format!("file:{file_path}"),
        kind: NodeKind::File,
        label: file_path.to_string(),
        payload: HashMap::new(),
        centrality: 0.0,
        memory_id: None,
        namespace: None,
    });

    Ok(Some(ExtractedMemory {
        content,
        memory_type: MemoryType::Decision,
        tags,
        metadata: {
            let mut m = HashMap::new();
            m.insert(
                "file_path".to_string(),
                serde_json::Value::String(file_path.to_string()),
            );
            m.insert(
                "tool".to_string(),
                serde_json::Value::String("Edit".to_string()),
            );
            m
        },
        graph_node,
        graph_edges: vec![],
        session_id: payload.session_id.clone(),
    }))
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

    let tags = extract_tags_from_path(file_path);

    let graph_node = Some(GraphNode {
        id: format!("file:{file_path}"),
        kind: NodeKind::File,
        label: file_path.to_string(),
        payload: HashMap::new(),
        centrality: 0.0,
        memory_id: None,
        namespace: None,
    });

    Ok(Some(ExtractedMemory {
        content,
        memory_type: MemoryType::Decision,
        tags,
        metadata: {
            let mut m = HashMap::new();
            m.insert(
                "file_path".to_string(),
                serde_json::Value::String(file_path.to_string()),
            );
            m.insert(
                "tool".to_string(),
                serde_json::Value::String("Write".to_string()),
            );
            m
        },
        graph_node,
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

/// Truncate string to max length.
fn truncate(s: &str, max_len: usize) -> &str {
    if s.len() <= max_len {
        s
    } else {
        &s[..max_len]
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

    // Trigger 3: Same search pattern used 2+ times
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
