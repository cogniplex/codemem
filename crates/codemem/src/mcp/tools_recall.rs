//! Namespace management tools: list_namespaces (with inline stats), namespace_stats, delete_namespace.

use super::types::ToolResult;
use super::McpServer;
use serde_json::{json, Value};

impl McpServer {
    /// MCP tool: list_namespaces -- list all namespaces with memory counts and inline stats.
    pub(crate) fn tool_list_namespaces(&self) -> ToolResult {
        let namespaces = match self.engine.storage().list_namespaces() {
            Ok(ns) => ns,
            Err(e) => return ToolResult::tool_error(format!("Storage error: {e}")),
        };

        let mut ns_list: Vec<Value> = Vec::new();
        for ns in &namespaces {
            // Include inline stats for each namespace
            match self.engine.namespace_stats(ns) {
                Ok(stats) => {
                    ns_list.push(json!({
                        "name": ns,
                        "memory_count": stats.count,
                        "avg_importance": format!("{:.4}", stats.avg_importance),
                        "avg_confidence": format!("{:.4}", stats.avg_confidence),
                        "type_distribution": stats.type_distribution,
                        "oldest": stats.oldest.map(|d| d.to_rfc3339()),
                        "newest": stats.newest.map(|d| d.to_rfc3339()),
                    }));
                }
                Err(_) => {
                    let count = match self.engine.storage().list_memory_ids_for_namespace(ns) {
                        Ok(ids) => ids.len(),
                        Err(_) => 0,
                    };
                    ns_list.push(json!({
                        "name": ns,
                        "memory_count": count,
                    }));
                }
            }
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
}

#[cfg(test)]
#[path = "tests/tools_recall_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "tests/tools_namespace_tests.rs"]
mod namespace_tests;
