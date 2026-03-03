//! Advanced recall & namespace tools: recall_with_expansion, list_namespaces,
//! namespace_stats, delete_namespace, export_memories, import_memories.

use super::args::{parse_opt_string, parse_string_array};
use super::types::ToolResult;
use super::McpServer;
use codemem_core::{MemoryNode, MemoryType};
use codemem_storage::Storage;
use serde_json::{json, Value};
use std::collections::HashMap;

impl McpServer {
    /// MCP tool: recall_with_expansion -- vector search + graph expansion.
    pub(crate) fn tool_recall_with_expansion(&self, args: &Value) -> ToolResult {
        let query = match args.get("query").and_then(|v| v.as_str()) {
            Some(q) if !q.is_empty() => q,
            _ => return ToolResult::tool_error("Missing or empty 'query' parameter"),
        };

        let k = args.get("k").and_then(|v| v.as_u64()).unwrap_or(5) as usize;
        let expansion_depth = args
            .get("expansion_depth")
            .and_then(|v| v.as_u64())
            .unwrap_or(1) as usize;
        let namespace_filter: Option<&str> = args.get("namespace").and_then(|v| v.as_str());

        let results =
            match self
                .engine
                .recall_with_expansion(query, k, expansion_depth, namespace_filter)
            {
                Ok(r) => r,
                Err(e) => return ToolResult::tool_error(format!("Recall error: {e}")),
            };

        if results.is_empty() {
            return ToolResult::text("No matching memories found.");
        }

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
            serde_json::to_string_pretty(&output).expect("JSON serialization of literal"),
        )
    }

    /// MCP tool: list_namespaces -- list all namespaces with memory counts.
    pub(crate) fn tool_list_namespaces(&self) -> ToolResult {
        let namespaces = match self.engine.storage.list_namespaces() {
            Ok(ns) => ns,
            Err(e) => return ToolResult::tool_error(format!("Storage error: {e}")),
        };

        let mut ns_list: Vec<Value> = Vec::new();
        for ns in &namespaces {
            let count = match self.engine.storage.list_memory_ids_for_namespace(ns) {
                Ok(ids) => ids.len(),
                Err(_) => 0,
            };
            ns_list.push(json!({
                "name": ns,
                "memory_count": count,
            }));
        }

        let response = json!({ "namespaces": ns_list });
        ToolResult::text(
            serde_json::to_string_pretty(&response).expect("JSON serialization of literal"),
        )
    }

    /// MCP tool: namespace_stats -- detailed stats for a single namespace.
    pub(crate) fn tool_namespace_stats(&self, args: &Value) -> ToolResult {
        let namespace = match args.get("namespace").and_then(|v| v.as_str()) {
            Some(ns) if !ns.is_empty() => ns,
            _ => return ToolResult::tool_error("Missing or empty 'namespace' parameter"),
        };

        let stats = match self.engine.namespace_stats(namespace) {
            Ok(s) => s,
            Err(e) => return ToolResult::tool_error(format!("Storage error: {e}")),
        };

        if stats.count == 0 {
            return ToolResult::text(
                serde_json::to_string_pretty(&json!({
                    "namespace": namespace,
                    "count": 0,
                    "message": "No memories found in this namespace"
                }))
                .expect("JSON serialization of literal"),
            );
        }

        let response = json!({
            "namespace": stats.namespace,
            "count": stats.count,
            "avg_importance": format!("{:.4}", stats.avg_importance),
            "avg_confidence": format!("{:.4}", stats.avg_confidence),
            "type_distribution": stats.type_distribution,
            "tag_frequency": stats.tag_frequency,
            "oldest": stats.oldest.map(|d| d.to_rfc3339()),
            "newest": stats.newest.map(|d| d.to_rfc3339()),
        });

        ToolResult::text(
            serde_json::to_string_pretty(&response).expect("JSON serialization of literal"),
        )
    }

    /// MCP tool: delete_namespace -- delete all memories in a namespace.
    pub(crate) fn tool_delete_namespace(&self, args: &Value) -> ToolResult {
        let namespace = match args.get("namespace").and_then(|v| v.as_str()) {
            Some(ns) if !ns.is_empty() => ns,
            _ => return ToolResult::tool_error("Missing or empty 'namespace' parameter"),
        };

        let confirm = args
            .get("confirm")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        if !confirm {
            return ToolResult::tool_error(
                "Destructive operation requires 'confirm': true parameter",
            );
        }

        let deleted = match self.engine.delete_namespace(namespace) {
            Ok(d) => d,
            Err(e) => return ToolResult::tool_error(format!("Delete error: {e}")),
        };

        let response = json!({
            "deleted": deleted,
            "namespace": namespace,
        });

        ToolResult::text(
            serde_json::to_string_pretty(&response).expect("JSON serialization of literal"),
        )
    }

    // ── Export/Import Tools ─────────────────────────────────────────────────

    /// MCP tool: export_memories -- export memories as a JSON array.
    pub(crate) fn tool_export_memories(&self, args: &Value) -> ToolResult {
        let namespace_filter: Option<&str> = args.get("namespace").and_then(|v| v.as_str());
        let memory_type_filter: Option<MemoryType> = args
            .get("memory_type")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse().ok());
        let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(100) as usize;

        let ids = match namespace_filter {
            Some(ns) => match self.engine.storage.list_memory_ids_for_namespace(ns) {
                Ok(ids) => ids,
                Err(e) => return ToolResult::tool_error(format!("Storage error: {e}")),
            },
            None => match self.engine.storage.list_memory_ids() {
                Ok(ids) => ids,
                Err(e) => return ToolResult::tool_error(format!("Storage error: {e}")),
            },
        };

        let mut exported: Vec<Value> = Vec::new();

        for id in &ids {
            if exported.len() >= limit {
                break;
            }
            if let Ok(Some(memory)) = self.engine.storage.get_memory(id) {
                // Apply memory_type filter
                if let Some(ref filter_type) = memory_type_filter {
                    if memory.memory_type != *filter_type {
                        continue;
                    }
                }

                // Get edges for this memory
                let edges: Vec<Value> = self
                    .engine
                    .storage
                    .get_edges_for_node(id)
                    .unwrap_or_default()
                    .iter()
                    .map(|e| {
                        json!({
                            "id": e.id,
                            "src": e.src,
                            "dst": e.dst,
                            "relationship": e.relationship.to_string(),
                            "weight": e.weight,
                        })
                    })
                    .collect();

                exported.push(json!({
                    "id": memory.id,
                    "content": memory.content,
                    "memory_type": memory.memory_type.to_string(),
                    "importance": memory.importance,
                    "confidence": memory.confidence,
                    "tags": memory.tags,
                    "namespace": memory.namespace,
                    "metadata": memory.metadata,
                    "created_at": memory.created_at.to_rfc3339(),
                    "updated_at": memory.updated_at.to_rfc3339(),
                    "edges": edges,
                }));
            }
        }

        ToolResult::text(
            serde_json::to_string_pretty(&exported).expect("JSON serialization of literal"),
        )
    }

    /// MCP tool: import_memories -- import memories from a JSON array.
    pub(crate) fn tool_import_memories(&self, args: &Value) -> ToolResult {
        let memories_arr = match args.get("memories").and_then(|v| v.as_array()) {
            Some(arr) => arr,
            None => return ToolResult::tool_error("Missing 'memories' parameter (expected array)"),
        };

        let mut imported = 0usize;
        let mut skipped = 0usize;
        let mut ids: Vec<String> = Vec::new();

        for mem_val in memories_arr {
            let content = match mem_val.get("content").and_then(|v| v.as_str()) {
                Some(c) if !c.is_empty() => c,
                _ => {
                    skipped += 1;
                    continue;
                }
            };

            let memory_type: MemoryType = mem_val
                .get("memory_type")
                .and_then(|v| v.as_str())
                .and_then(|s| s.parse().ok())
                .unwrap_or(MemoryType::Context);

            let importance = mem_val
                .get("importance")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.5);

            let confidence = mem_val
                .get("confidence")
                .and_then(|v| v.as_f64())
                .unwrap_or(1.0);

            let tags = parse_string_array(mem_val, "tags");
            let namespace = parse_opt_string(mem_val, "namespace");

            let metadata: HashMap<String, serde_json::Value> = mem_val
                .get("metadata")
                .and_then(|v| serde_json::from_value(v.clone()).ok())
                .unwrap_or_default();

            let now = chrono::Utc::now();
            let id = uuid::Uuid::new_v4().to_string();
            let hash = Storage::content_hash(content);

            let memory = MemoryNode {
                id: id.clone(),
                content: content.to_string(),
                memory_type,
                importance,
                confidence,
                access_count: 0,
                content_hash: hash,
                tags,
                metadata,
                namespace,
                created_at: now,
                updated_at: now,
                last_accessed_at: now,
            };

            match self.engine.persist_memory(&memory) {
                Ok(()) => {
                    ids.push(id);
                    imported += 1;
                }
                Err(_) => {
                    skipped += 1;
                }
            }
        }

        // Persist vector index to disk
        self.save_index();

        ToolResult::text(
            serde_json::to_string_pretty(&json!({
                "imported": imported,
                "skipped": skipped,
                "ids": ids,
            }))
            .expect("JSON serialization of literal"),
        )
    }
}

#[cfg(test)]
#[path = "tests/tools_recall_tests.rs"]
mod tests;
