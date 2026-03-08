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

#[cfg(test)]
#[path = "tests/lib_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "tests/hooks_integration.rs"]
mod hooks_integration_tests;
