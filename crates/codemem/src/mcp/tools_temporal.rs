//! Temporal query tools: what_changed, graph_at_time, find_stale_files,
//! detect_drift, symbol_history.

use super::types::ToolResult;
use super::McpServer;
use chrono::DateTime;
use serde_json::{json, Value};

impl McpServer {
    pub(crate) fn tool_what_changed(&self, args: &Value) -> ToolResult {
        let from_str = match args.get("from").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => return ToolResult::tool_error("Missing 'from' parameter"),
        };
        let to_str = match args.get("to").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => return ToolResult::tool_error("Missing 'to' parameter"),
        };

        let from = match DateTime::parse_from_rfc3339(from_str) {
            Ok(dt) => dt.to_utc(),
            Err(e) => return ToolResult::tool_error(format!("Invalid 'from' date: {e}")),
        };
        let to = match DateTime::parse_from_rfc3339(to_str) {
            Ok(dt) => dt.to_utc(),
            Err(e) => return ToolResult::tool_error(format!("Invalid 'to' date: {e}")),
        };

        let namespace = args.get("namespace").and_then(|v| v.as_str());

        match self.engine.what_changed(from, to, namespace) {
            Ok(entries) => {
                let output: Vec<Value> = entries
                    .iter()
                    .map(|e| {
                        json!({
                            "commit_id": e.commit_id,
                            "hash": e.hash,
                            "author": e.author,
                            "date": e.date,
                            "subject": e.subject,
                            "changed_files": e.changed_files,
                            "changed_symbols": e.changed_symbols,
                        })
                    })
                    .collect();
                ToolResult::text(
                    serde_json::to_string_pretty(&json!({
                        "commits": output.len(),
                        "entries": output,
                    }))
                    .unwrap_or_default(),
                )
            }
            Err(e) => ToolResult::tool_error(format!("what_changed failed: {e}")),
        }
    }

    pub(crate) fn tool_graph_at_time(&self, args: &Value) -> ToolResult {
        let at_str = match args.get("at").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => return ToolResult::tool_error("Missing 'at' parameter"),
        };

        let at = match DateTime::parse_from_rfc3339(at_str) {
            Ok(dt) => dt.to_utc(),
            Err(e) => return ToolResult::tool_error(format!("Invalid 'at' date: {e}")),
        };

        match self.engine.graph_at_time(at) {
            Ok(snapshot) => {
                ToolResult::text(serde_json::to_string_pretty(&snapshot).unwrap_or_default())
            }
            Err(e) => ToolResult::tool_error(format!("graph_at_time failed: {e}")),
        }
    }

    pub(crate) fn tool_find_stale_files(&self, args: &Value) -> ToolResult {
        let namespace = args.get("namespace").and_then(|v| v.as_str());
        let stale_days = args
            .get("stale_days")
            .and_then(|v| v.as_u64())
            .unwrap_or(90);
        let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(20) as usize;

        match self.engine.find_stale_files(namespace, stale_days) {
            Ok(mut files) => {
                files.truncate(limit);
                ToolResult::text(
                    serde_json::to_string_pretty(&json!({
                        "stale_files": files.len(),
                        "stale_days": stale_days,
                        "files": files,
                    }))
                    .unwrap_or_default(),
                )
            }
            Err(e) => ToolResult::tool_error(format!("find_stale_files failed: {e}")),
        }
    }

    pub(crate) fn tool_detect_drift(&self, args: &Value) -> ToolResult {
        let from_str = match args.get("from").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => return ToolResult::tool_error("Missing 'from' parameter"),
        };
        let to_str = match args.get("to").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => return ToolResult::tool_error("Missing 'to' parameter"),
        };

        let from = match DateTime::parse_from_rfc3339(from_str) {
            Ok(dt) => dt.to_utc(),
            Err(e) => return ToolResult::tool_error(format!("Invalid 'from' date: {e}")),
        };
        let to = match DateTime::parse_from_rfc3339(to_str) {
            Ok(dt) => dt.to_utc(),
            Err(e) => return ToolResult::tool_error(format!("Invalid 'to' date: {e}")),
        };

        let namespace = args.get("namespace").and_then(|v| v.as_str());

        match self.engine.detect_drift(from, to, namespace) {
            Ok(report) => {
                ToolResult::text(serde_json::to_string_pretty(&report).unwrap_or_default())
            }
            Err(e) => ToolResult::tool_error(format!("detect_drift failed: {e}")),
        }
    }

    pub(crate) fn tool_symbol_history(&self, args: &Value) -> ToolResult {
        let node_id = match args.get("node_id").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => return ToolResult::tool_error("Missing 'node_id' parameter"),
        };

        match self.engine.symbol_history(node_id) {
            Ok(entries) => {
                let output: Vec<Value> = entries
                    .iter()
                    .map(|e| {
                        json!({
                            "hash": e.hash,
                            "author": e.author,
                            "date": e.date,
                            "subject": e.subject,
                            "changed_files": e.changed_files,
                            "changed_symbols": e.changed_symbols,
                        })
                    })
                    .collect();
                ToolResult::text(
                    serde_json::to_string_pretty(&json!({
                        "node_id": node_id,
                        "commits": output.len(),
                        "history": output,
                    }))
                    .unwrap_or_default(),
                )
            }
            Err(e) => ToolResult::tool_error(format!("symbol_history failed: {e}")),
        }
    }
}
