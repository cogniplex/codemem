//! Memory CRUD tools: store, recall (unified), delete, associate, refine (with destructive mode), split, merge.

use super::args::{parse_memory_type, parse_opt_string, parse_string_array};
use super::scoring::format_recall_results;
use super::types::ToolResult;
use super::McpServer;
use codemem_core::{CodememError, Edge, GraphBackend, MemoryType, RelationshipType};
use codemem_engine::SplitPart;
use codemem_storage::Storage;
use serde_json::{json, Value};
use std::collections::HashMap;

impl McpServer {
    pub(crate) fn tool_store_memory(&self, args: &Value) -> ToolResult {
        let content = match args.get("content").and_then(|v| v.as_str()) {
            Some(c) if !c.is_empty() => c,
            _ => return ToolResult::tool_error("Missing or empty 'content' parameter"),
        };

        let memory_type = parse_memory_type(args, MemoryType::Context);
        let importance = args
            .get("importance")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.5);
        let tags = parse_string_array(args, "tags");
        let namespace = parse_opt_string(args, "namespace");

        let now = chrono::Utc::now();
        let id = uuid::Uuid::new_v4().to_string();
        let hash = Storage::content_hash(content);

        let memory = codemem_core::MemoryNode {
            id: id.clone(),
            content: content.to_string(),
            memory_type,
            importance,
            confidence: 1.0,
            access_count: 0,
            content_hash: hash,
            tags,
            metadata: HashMap::new(),
            namespace,
            session_id: None,
            created_at: now,
            updated_at: now,
            last_accessed_at: now,
        };

        match self.engine.persist_memory(&memory) {
            Ok(()) => {}
            Err(CodememError::Duplicate(h)) => {
                return ToolResult::text(format!("Memory already exists (hash: {h})"));
            }
            Err(e) => return ToolResult::tool_error(format!("Storage error: {e}")),
        }

        // Handle optional `links` parameter: create RELATES_TO edges to linked nodes
        if let Some(links) = args.get("links").and_then(|v| v.as_array()) {
            let mut graph = match self.lock_graph() {
                Ok(g) => g,
                Err(e) => return ToolResult::tool_error(format!("Lock error: {e}")),
            };
            for link_val in links {
                if let Some(link_id) = link_val.as_str() {
                    let edge = Edge {
                        id: format!("{id}-RELATES_TO-{link_id}"),
                        src: id.clone(),
                        dst: link_id.to_string(),
                        relationship: RelationshipType::RelatesTo,
                        weight: 1.0,
                        properties: HashMap::new(),
                        created_at: now,
                        valid_from: None,
                        valid_to: None,
                    };
                    if let Err(e) = self.engine.storage().insert_graph_edge(&edge) {
                        tracing::warn!("Failed to persist link edge to {link_id}: {e}");
                    }
                    if let Err(e) = graph.add_edge(edge) {
                        tracing::warn!("Failed to add link edge to {link_id}: {e}");
                    }
                }
            }
        }

        // Auto-link memory to code nodes mentioned in content
        let explicit_links = parse_string_array(args, "links");
        self.engine
            .auto_link_to_code_nodes(&id, content, &explicit_links);

        // Persist vector index to disk
        self.engine.save_index();

        ToolResult::text(
            serde_json::to_string_pretty(&json!({
                "id": id,
                "memory_type": memory_type.to_string(),
                "importance": importance,
                "embedded": self.engine.has_embeddings(),
            }))
            .expect("JSON serialization of literal"),
        )
    }

    /// Unified recall tool: basic recall, with optional graph expansion and impact data.
    pub(crate) fn tool_recall(&self, args: &Value) -> ToolResult {
        let query = match args.get("query").and_then(|v| v.as_str()) {
            Some(q) if !q.is_empty() => q,
            _ => return ToolResult::tool_error("Missing or empty 'query' parameter"),
        };

        let k = args.get("k").and_then(|v| v.as_u64()).unwrap_or(10) as usize;
        let memory_type_filter: Option<MemoryType> = args
            .get("memory_type")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse().ok());
        let namespace_filter: Option<&str> = args.get("namespace").and_then(|v| v.as_str());
        let exclude_tags = parse_string_array(args, "exclude_tags");
        let min_importance: Option<f64> = args.get("min_importance").and_then(|v| v.as_f64());
        let min_confidence: Option<f64> = args.get("min_confidence").and_then(|v| v.as_f64());
        let expand = args
            .get("expand")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let expansion_depth = args
            .get("expansion_depth")
            .and_then(|v| v.as_u64())
            .unwrap_or(1) as usize;
        let include_impact = args
            .get("include_impact")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        // If include_impact is requested, use recall_with_impact path
        if include_impact {
            match self.engine.recall_with_impact(query, k, namespace_filter) {
                Ok(results) if results.is_empty() => {
                    ToolResult::text("No matching memories found.")
                }
                Ok(results) => {
                    let output: Vec<Value> = results
                        .iter()
                        .map(|r| {
                            json!({
                                "id": r.search_result.memory.id,
                                "content": r.search_result.memory.content,
                                "memory_type": r.search_result.memory.memory_type.to_string(),
                                "score": format!("{:.4}", r.search_result.score),
                                "importance": r.search_result.memory.importance,
                                "tags": r.search_result.memory.tags,
                                "access_count": r.search_result.memory.access_count,
                                "impact": {
                                    "pagerank": format!("{:.6}", r.pagerank),
                                    "centrality": format!("{:.6}", r.centrality),
                                    "connected_decisions": r.connected_decisions,
                                    "dependent_files": r.dependent_files,
                                    "modification_count": r.search_result.memory.access_count,
                                }
                            })
                        })
                        .collect();
                    ToolResult::text(
                        serde_json::to_string_pretty(&output)
                            .expect("JSON serialization of literal"),
                    )
                }
                Err(e) => ToolResult::tool_error(format!("Recall error: {e}")),
            }
        } else if expand {
            // Graph expansion path
            match self
                .engine
                .recall_with_expansion(query, k, expansion_depth, namespace_filter)
            {
                Ok(results) if results.is_empty() => {
                    ToolResult::text("No matching memories found.")
                }
                Ok(results) => {
                    let output: Vec<Value> = results
                        .iter()
                        .map(|er| {
                            json!({
                                "id": er.result.memory.id,
                                "content": er.result.memory.content,
                                "memory_type": er.result.memory.memory_type.to_string(),
                                "score": format!("{:.4}", er.result.score),
                                "importance": er.result.memory.importance,
                                "tags": er.result.memory.tags,
                                "access_count": er.result.memory.access_count,
                                "expansion_path": er.expansion_path,
                            })
                        })
                        .collect();
                    ToolResult::text(
                        serde_json::to_string_pretty(&output)
                            .expect("JSON serialization of literal"),
                    )
                }
                Err(e) => ToolResult::tool_error(format!("Recall error: {e}")),
            }
        } else {
            // Standard recall
            match self.engine.recall(
                query,
                k,
                memory_type_filter,
                namespace_filter,
                &exclude_tags,
                min_importance,
                min_confidence,
            ) {
                Ok(results) => format_recall_results(&results, None),
                Err(e) => ToolResult::tool_error(format!("Recall error: {e}")),
            }
        }
    }

    pub(crate) fn tool_delete_memory(&self, args: &Value) -> ToolResult {
        let id = match args.get("id").and_then(|v| v.as_str()) {
            Some(id) => id,
            None => return ToolResult::tool_error("Missing 'id' parameter"),
        };

        match self.engine.delete_memory(id) {
            Ok(true) => ToolResult::text(json!({"id": id, "deleted": true}).to_string()),
            Ok(false) => ToolResult::tool_error(format!("Memory not found: {id}")),
            Err(e) => ToolResult::tool_error(format!("Delete failed: {e}")),
        }
    }

    /// Refine a memory. By default creates a new version linked via EVOLVED_INTO.
    /// With `destructive: true`, updates in-place (like the old update_memory tool).
    pub(crate) fn tool_refine_memory(&self, args: &Value) -> ToolResult {
        let id = match args.get("id").and_then(|v| v.as_str()) {
            Some(id) => id,
            None => return ToolResult::tool_error("Missing 'id' parameter"),
        };

        let content = args.get("content").and_then(|v| v.as_str());
        let tags: Option<Vec<String>> = args.get("tags").and_then(|v| v.as_array()).map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        });
        let importance = args.get("importance").and_then(|v| v.as_f64());
        let destructive = args
            .get("destructive")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        if destructive {
            // In-place update (former update_memory behavior)
            let update_content = match content {
                Some(c) => c,
                None => return ToolResult::tool_error("Missing 'content' for destructive refine"),
            };
            match self.engine.update_memory(id, update_content, importance) {
                Ok(()) => ToolResult::text(json!({"id": id, "updated": true}).to_string()),
                Err(e) => ToolResult::tool_error(format!("Update failed: {e}")),
            }
        } else {
            // Default: create new version with EVOLVED_INTO provenance
            match self.engine.refine_memory(id, content, tags, importance) {
                Ok((_memory, new_id)) => ToolResult::text(
                    serde_json::to_string_pretty(&json!({
                        "old_id": id,
                        "new_id": new_id,
                        "relationship": "EVOLVED_INTO",
                    }))
                    .expect("JSON serialization of literal"),
                ),
                Err(e) => ToolResult::tool_error(format!("{e}")),
            }
        }
    }

    pub(crate) fn tool_split_memory(&self, args: &Value) -> ToolResult {
        let source_id = match args.get("id").and_then(|v| v.as_str()) {
            Some(id) => id,
            None => return ToolResult::tool_error("Missing 'id' parameter"),
        };

        let parts_json = match args.get("parts").and_then(|v| v.as_array()) {
            Some(arr) if !arr.is_empty() => arr,
            Some(_) => return ToolResult::tool_error("'parts' array must not be empty"),
            None => return ToolResult::tool_error("Missing 'parts' parameter"),
        };

        let mut parts: Vec<SplitPart> = Vec::new();
        for part in parts_json {
            let content = match part.get("content").and_then(|v| v.as_str()) {
                Some(c) if !c.is_empty() => c.to_string(),
                _ => {
                    return ToolResult::tool_error(
                        "Each part must have a non-empty 'content' field",
                    )
                }
            };
            let tags: Option<Vec<String>> =
                part.get("tags").and_then(|v| v.as_array()).map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect()
                });
            let importance = part.get("importance").and_then(|v| v.as_f64());
            parts.push(SplitPart {
                content,
                tags,
                importance,
            });
        }

        match self.engine.split_memory(source_id, &parts) {
            Ok(child_ids) => ToolResult::text(
                serde_json::to_string_pretty(&json!({
                    "source_id": source_id,
                    "parts": child_ids,
                    "relationship": "PART_OF",
                }))
                .expect("JSON serialization of literal"),
            ),
            Err(e) => ToolResult::tool_error(format!("{e}")),
        }
    }

    pub(crate) fn tool_merge_memories(&self, args: &Value) -> ToolResult {
        let source_ids = parse_string_array(args, "source_ids");

        if source_ids.len() < 2 {
            return ToolResult::tool_error("'source_ids' must contain at least 2 IDs");
        }

        let content = match args.get("content").and_then(|v| v.as_str()) {
            Some(c) if !c.is_empty() => c,
            _ => return ToolResult::tool_error("Missing or empty 'content' parameter"),
        };

        let memory_type = parse_memory_type(args, MemoryType::Insight);
        let importance = args
            .get("importance")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.7);
        let tags = parse_string_array(args, "tags");

        match self
            .engine
            .merge_memories(&source_ids, content, memory_type, importance, tags)
        {
            Ok(merged_id) => ToolResult::text(
                serde_json::to_string_pretty(&json!({
                    "merged_id": merged_id,
                    "source_ids": source_ids,
                    "relationship": "SUMMARIZES",
                }))
                .expect("JSON serialization of literal"),
            ),
            Err(e) => ToolResult::tool_error(format!("{e}")),
        }
    }

    pub(crate) fn tool_associate_memories(&self, args: &Value) -> ToolResult {
        let src = match args.get("source_id").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => return ToolResult::tool_error("Missing 'source_id' parameter"),
        };
        let dst = match args.get("target_id").and_then(|v| v.as_str()) {
            Some(d) => d,
            None => return ToolResult::tool_error("Missing 'target_id' parameter"),
        };
        let rel_str = args
            .get("relationship")
            .and_then(|v| v.as_str())
            .unwrap_or("RELATES_TO");
        let weight = args.get("weight").and_then(|v| v.as_f64()).unwrap_or(1.0);

        let relationship: RelationshipType = match rel_str.parse() {
            Ok(r) => r,
            Err(e) => return ToolResult::tool_error(format!("Invalid relationship: {e}")),
        };

        let edge = Edge {
            id: format!("{src}-{}-{dst}", rel_str),
            src: src.to_string(),
            dst: dst.to_string(),
            relationship,
            weight,
            properties: HashMap::new(),
            created_at: chrono::Utc::now(),
            valid_from: None,
            valid_to: None,
        };

        match self.engine.add_edge(edge) {
            Ok(()) => ToolResult::text(
                json!({
                    "source": src,
                    "target": dst,
                    "relationship": rel_str,
                    "weight": weight,
                })
                .to_string(),
            ),
            Err(e) => ToolResult::tool_error(format!("Failed to store edge: {e}")),
        }
    }
}

#[cfg(test)]
#[path = "tests/tools_memory_crud_tests.rs"]
mod crud_tests;

#[cfg(test)]
#[path = "tests/tools_memory_advanced_tests.rs"]
mod advanced_tests;

#[cfg(test)]
#[path = "tests/tools_memory_quality_tests.rs"]
mod quality_tests;
