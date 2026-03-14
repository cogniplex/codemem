//! codemem-mcp: MCP server for Codemem (JSON-RPC 2.0 over stdio).
//!
//! Implements 26 tools:
//! store_memory, recall, delete_memory, associate_memories, refine_memory,
//! split_memory, merge_memories, graph_traverse, summary_tree,
//! codemem_status, search_code, get_symbol_info,
//! get_symbol_graph, find_important_nodes, find_related_groups,
//! get_node_memories, node_coverage,
//! get_cross_repo, consolidate, detect_patterns, get_decision_chain,
//! list_namespaces, namespace_stats, delete_namespace,
//! session_checkpoint, session_context.
//!
//! Transport: Newline-delimited JSON-RPC messages over stdio.
//! All logging goes to stderr; stdout is reserved for JSON-RPC only.

use codemem_core::{CodememError, StorageBackend};
use codemem_engine::CodememEngine;
use serde_json::{json, Value};
use std::io::{self, BufRead};
use std::path::Path;

pub(crate) mod args;
mod definitions;
pub mod http;
pub mod scoring;
pub mod tools_consolidation;
pub mod tools_graph;
pub mod tools_memory;
pub mod tools_recall;
pub mod types;

#[cfg(test)]
pub(crate) mod test_helpers;

// Re-export public types for downstream consumers.
pub use types::{JsonRpcError, JsonRpcRequest, JsonRpcResponse, ToolContent, ToolResult};

use codemem_engine::metrics;

use scoring::write_response;

// ── MCP Server ──────────────────────────────────────────────────────────────

/// MCP server that reads JSON-RPC from stdin, writes responses to stdout.
/// Wraps a `CodememEngine` for domain logic, adding only transport concerns.
pub struct McpServer {
    pub name: String,
    pub version: String,
    /// Core domain engine holding all backends and business logic.
    pub engine: CodememEngine,
}

impl McpServer {
    /// Create a server with storage, vector, graph, and optional embeddings backends.
    pub fn new(
        storage: Box<dyn StorageBackend>,
        vector: Box<dyn codemem_core::VectorBackend>,
        graph: Box<dyn codemem_core::GraphBackend>,
        embeddings: Option<Box<dyn codemem_engine::EmbeddingProvider>>,
    ) -> Self {
        Self {
            name: "codemem".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            engine: CodememEngine::new(storage, vector, graph, embeddings),
        }
    }

    /// Create a server from a database path, loading all backends.
    pub fn from_db_path(db_path: &Path) -> Result<Self, CodememError> {
        let engine = CodememEngine::from_db_path(db_path)?;
        Ok(Self {
            name: "codemem".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            engine,
        })
    }

    /// Create a minimal server for testing (no backends wired).
    pub fn for_testing() -> Self {
        Self {
            name: "codemem".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            engine: CodememEngine::for_testing(),
        }
    }

    // ── Delegate Accessors ─────────────────────────────────────────────────

    /// Core recall logic: delegates to `CodememEngine::recall()`.
    /// Used by the REST API layer and consolidation tools.
    pub fn recall(
        &self,
        q: &codemem_engine::RecallQuery<'_>,
    ) -> Result<Vec<codemem_core::SearchResult>, CodememError> {
        self.engine.recall(q)
    }

    pub fn storage(&self) -> &dyn StorageBackend {
        self.engine.storage()
    }

    pub fn reload_graph(&self) -> Result<(), CodememError> {
        self.engine.reload_graph()
    }

    pub fn db_path(&self) -> Option<&Path> {
        self.engine.db_path()
    }

    pub fn config(&self) -> &codemem_core::CodememConfig {
        self.engine.config()
    }

    pub fn metrics_collector(&self) -> &std::sync::Arc<metrics::InMemoryMetrics> {
        self.engine.metrics()
    }

    pub fn save_index(&self) {
        self.engine.save_index()
    }

    /// Run the MCP server over stdio. Convenience method that creates a
    /// `StdioTransport` and runs it. Blocks until stdin is closed.
    pub fn run(&self) -> io::Result<()> {
        let transport = StdioTransport::new(self);
        transport.run()
    }

    pub fn handle_notification(&self, method: &str) {
        match method {
            "notifications/initialized" => {
                tracing::info!("Client initialized, codemem MCP server ready");
            }
            "notifications/cancelled" => {
                tracing::debug!("Request cancelled by client");
            }
            _ => {
                tracing::debug!("Unknown notification: {method}");
            }
        }
    }

    pub fn handle_request(
        &self,
        method: &str,
        params: Option<&Value>,
        id: Value,
    ) -> JsonRpcResponse {
        match method {
            "initialize" => self.handle_initialize(id),
            "tools/list" => self.handle_tools_list(id),
            "tools/call" => self.handle_tools_call(id, params),
            "ping" => JsonRpcResponse::success(id, json!({})),
            _ => JsonRpcResponse::error(id, -32601, format!("Method not found: {method}")),
        }
    }

    fn handle_initialize(&self, id: Value) -> JsonRpcResponse {
        JsonRpcResponse::success(
            id,
            json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": { "listChanged": false }
                },
                "serverInfo": {
                    "name": self.name,
                    "version": self.version
                }
            }),
        )
    }

    fn handle_tools_list(&self, id: Value) -> JsonRpcResponse {
        JsonRpcResponse::success(
            id,
            json!({
                "tools": definitions::tool_definitions()
            }),
        )
    }

    fn handle_tools_call(&self, id: Value, params: Option<&Value>) -> JsonRpcResponse {
        let params = match params {
            Some(p) => p,
            None => return JsonRpcResponse::error(id, -32602, "Missing params"),
        };

        let tool_name = params.get("name").and_then(|n| n.as_str()).unwrap_or("");
        let arguments = params.get("arguments").cloned().unwrap_or(json!({}));

        let result = self.dispatch_tool(tool_name, &arguments);

        match serde_json::to_value(result) {
            Ok(v) => JsonRpcResponse::success(id, v),
            Err(e) => JsonRpcResponse::error(id, -32603, format!("Serialization error: {e}")),
        }
    }

    // ── Tool Dispatch ───────────────────────────────────────────────────────

    fn dispatch_tool(&self, name: &str, args: &Value) -> ToolResult {
        let start = std::time::Instant::now();
        let result = self.dispatch_tool_inner(name, args);
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        codemem_core::Metrics::record_latency(self.engine.metrics().as_ref(), name, elapsed);
        codemem_core::Metrics::increment_counter(
            self.engine.metrics().as_ref(),
            "tool_calls_total",
            1,
        );

        // Record session activity against the most recent active session (if any).
        // This tracks MCP tool usage counts for cross-session pattern detection.
        if let Ok(sessions) = self.engine.storage().list_sessions(None, 10) {
            if let Some(active) = sessions.iter().find(|s| s.ended_at.is_none()) {
                let _ = self
                    .engine
                    .storage()
                    .record_session_activity(&active.id, name, None, None, None);
            }
        }

        result
    }

    fn dispatch_tool_inner(&self, name: &str, args: &Value) -> ToolResult {
        match name {
            // ── Memory CRUD ─────────────────────────────────────────────
            "store_memory" => self.tool_store_memory(args),
            "recall" => self.tool_recall(args),
            "delete_memory" => self.tool_delete_memory(args),
            "associate_memories" => self.tool_associate_memories(args),
            "refine_memory" => self.tool_refine_memory(args),
            "split_memory" => self.tool_split_memory(args),
            "merge_memories" => self.tool_merge_memories(args),

            // ── Graph & Structure ───────────────────────────────────────
            "graph_traverse" => self.tool_graph_traverse(args),
            "summary_tree" => self.tool_summary_tree(args),
            "codemem_status" => self.tool_codemem_status(args),
            "search_code" => self.tool_search_code(args),
            "get_symbol_info" => self.tool_get_symbol_info(args),
            "get_symbol_graph" => self.tool_get_symbol_graph(args),
            "find_important_nodes" => self.tool_find_important_nodes(args),
            "find_related_groups" => self.tool_find_related_groups(args),
            "get_cross_repo" => self.tool_get_cross_repo(args),
            "get_node_memories" => self.tool_get_node_memories(args),
            "node_coverage" => self.tool_node_coverage(args),

            // ── Consolidation & Patterns ────────────────────────────────
            "consolidate" => self.tool_consolidate(args),
            "detect_patterns" => self.tool_detect_patterns(args),
            "get_decision_chain" => self.tool_get_decision_chain(args),

            // ── Namespace Management ────────────────────────────────────
            "list_namespaces" => self.tool_list_namespaces(),
            "namespace_stats" => self.tool_namespace_stats(args),
            "delete_namespace" => self.tool_delete_namespace(args),

            // ── Session & Context ───────────────────────────────────────
            "session_checkpoint" => self.tool_session_checkpoint(args),
            "session_context" => self.tool_session_context(args),

            _ => ToolResult::tool_error(format!("Unknown tool: {name}")),
        }
    }
}

// ── Stdio Transport ────────────────────────────────────────────────────────

/// Stdio transport for the MCP server.
/// Reads newline-delimited JSON-RPC from stdin, writes responses to stdout.
pub struct StdioTransport<'a> {
    server: &'a McpServer,
}

impl<'a> StdioTransport<'a> {
    /// Create a new stdio transport wrapping the given server.
    pub fn new(server: &'a McpServer) -> Self {
        Self { server }
    }

    /// Run the transport loop. Blocks until stdin is closed.
    pub fn run(&self) -> io::Result<()> {
        let stdin = io::stdin();
        let stdout = io::stdout();
        let mut stdout = stdout.lock();

        for line in stdin.lock().lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }

            let request: JsonRpcRequest = match serde_json::from_str(&line) {
                Ok(req) => req,
                Err(e) => {
                    let resp =
                        JsonRpcResponse::error(Value::Null, -32700, format!("Parse error: {e}"));
                    write_response(&mut stdout, &resp)?;
                    continue;
                }
            };

            // Notifications (no id) don't get a response
            if request.id.is_none() {
                self.server.handle_notification(&request.method);
                continue;
            }

            let id = request.id.unwrap();
            let response = self
                .server
                .handle_request(&request.method, request.params.as_ref(), id);
            write_response(&mut stdout, &response)?;
        }

        Ok(())
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests/lib_tests.rs"]
mod tests;
