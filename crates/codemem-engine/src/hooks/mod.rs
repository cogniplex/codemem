//! Hooks module: PostToolUse hook handler for passive capture from AI coding assistants.
//!
//! Parses PostToolUse JSON payloads from stdin, extracts relevant information
//! based on tool type, and creates appropriate memories with auto-tagging.

pub mod diff;
mod extractors;
pub mod triggers;

use codemem_core::{CodememError, MemoryType, RelationshipType};
use serde::Deserialize;
use std::collections::HashMap;

pub use triggers::{check_triggers, AutoInsight};

use extractors::{
    extract_agent_communication, extract_bash, extract_edit, extract_glob, extract_grep,
    extract_list_dir, extract_read, extract_web, extract_write,
};

/// Maximum file size to process (100KB).
const MAX_CONTENT_SIZE: usize = 100 * 1024;

/// PostToolUse hook payload from an AI coding assistant.
///
/// `tool_response` is `serde_json::Value` because Claude Code sends it as a
/// JSON object (not a plain string). String-valued responses still deserialize
/// correctly into `Value::String`.
#[derive(Debug, Deserialize)]
pub struct HookPayload {
    pub tool_name: String,
    pub tool_input: serde_json::Value,
    pub tool_response: serde_json::Value,
    pub session_id: Option<String>,
    pub cwd: Option<String>,
    /// Name of the hook event (e.g. "PostToolUse").
    pub hook_event_name: Option<String>,
    /// Path to the conversation transcript file.
    pub transcript_path: Option<String>,
    /// Permission mode the assistant is running in.
    pub permission_mode: Option<String>,
    /// Unique ID of the tool use that triggered this hook.
    pub tool_use_id: Option<String>,
}

impl HookPayload {
    /// Extract meaningful text content from the tool response.
    ///
    /// Handles known Claude Code response shapes before falling back to
    /// raw JSON serialization:
    ///
    /// - `Value::String` → inner text (legacy / simple tools)
    /// - Read tool: `{file: {content: "..."}}` → the file content
    /// - Text-bearing: `{text: "..."}` → the text value
    /// - Stdout-bearing: `{stdout: "..."}` → stdout value
    /// - `Value::Null` → empty string
    /// - anything else → compact JSON serialization
    pub fn tool_response_text(&self) -> String {
        match &self.tool_response {
            serde_json::Value::String(s) => s.clone(),
            serde_json::Value::Null => String::new(),
            serde_json::Value::Object(obj) => {
                // Read tool: {file: {content: "..."}}
                if let Some(content) = obj
                    .get("file")
                    .and_then(|f| f.get("content"))
                    .and_then(|c| c.as_str())
                {
                    return content.to_string();
                }
                // Text-bearing responses: {text: "..."}
                if let Some(text) = obj.get("text").and_then(|t| t.as_str()) {
                    return text.to_string();
                }
                // Stdout-bearing responses: {stdout: "..."}
                if let Some(stdout) = obj.get("stdout").and_then(|s| s.as_str()) {
                    return stdout.to_string();
                }
                // Fallback: compact JSON
                serde_json::to_string(&self.tool_response).unwrap_or_default()
            }
            other => other.to_string(),
        }
    }
}

/// Extracted memory from a hook payload.
#[derive(Debug)]
pub struct ExtractedMemory {
    pub content: String,
    pub memory_type: MemoryType,
    pub tags: Vec<String>,
    pub metadata: HashMap<String, serde_json::Value>,
    pub graph_node: Option<codemem_core::GraphNode>,
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
    // Check response size to skip very large payloads.
    // For strings, check directly. For objects, check the extracted text content
    // since that's what we actually store (avoids double-serialization).
    let response_text = payload.tool_response_text();
    if response_text.len() > MAX_CONTENT_SIZE {
        tracing::debug!("Skipping large response ({} bytes)", response_text.len());
        return Ok(None);
    }

    match payload.tool_name.as_str() {
        "Read" => extract_read(payload, &response_text),
        "Glob" => extract_glob(payload, &response_text),
        "Grep" => extract_grep(payload, &response_text),
        "Edit" | "MultiEdit" => extract_edit(payload),
        "Write" => extract_write(payload, &response_text),
        "Bash" => extract_bash(payload, &response_text),
        "WebFetch" | "WebSearch" => extract_web(payload, &response_text),
        "Agent" | "SendMessage" => extract_agent_communication(payload, &response_text),
        "ListFiles" | "ListDir" => extract_list_dir(payload, &response_text),
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

#[cfg(test)]
#[path = "tests/lib_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "tests/hooks_integration.rs"]
mod hooks_integration_tests;
