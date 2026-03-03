//! codemem-mcp: MCP server for Codemem (JSON-RPC 2.0 over stdio).
//!
//! Implements 28 tools:
//! store_memory, recall, delete_memory, associate_memories, refine_memory,
//! split_memory, merge_memories, graph_traverse, summary_tree,
//! codemem_status, index_codebase, search_code, get_symbol_info,
//! get_symbol_graph, find_important_nodes, find_related_groups,
//! get_cross_repo, consolidate, detect_patterns, get_decision_chain,
//! list_namespaces, namespace_stats, delete_namespace,
//! session_checkpoint, session_context,
//! enrich_codebase, analyze_codebase,
//! enrich_git_history.
//!
//! Transport: Newline-delimited JSON-RPC messages over stdio.
//! All logging goes to stderr; stdout is reserved for JSON-RPC only.

use codemem_core::{CodememError, MemoryType, StorageBackend};
use codemem_engine::CodememEngine;
use codemem_storage::graph::GraphEngine;
use codemem_storage::HnswIndex;
use serde_json::{json, Value};
use std::io::{self, BufRead};
use std::path::Path;
use std::sync::Mutex;

pub(crate) mod args;
pub mod http;
pub mod scoring;
pub mod tools_consolidation;
pub mod tools_enrich;
pub mod tools_graph;
pub mod tools_memory;
pub mod tools_recall;
pub mod types;

#[cfg(test)]
pub(crate) mod test_helpers;

// Re-export public types for downstream consumers.
pub use types::{JsonRpcError, JsonRpcRequest, JsonRpcResponse, ToolContent, ToolResult};

use codemem_engine::bm25;
use codemem_engine::metrics;
use codemem_engine::IndexCache;

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
        vector: HnswIndex,
        graph: GraphEngine,
        embeddings: Option<Box<dyn codemem_embeddings::EmbeddingProvider>>,
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

    // ── Delegate Lock Helpers ────────────────────────────────────────────────
    // Thin wrappers so tool modules can still call self.lock_graph() etc.

    pub(crate) fn lock_vector(&self) -> Result<std::sync::MutexGuard<'_, HnswIndex>, CodememError> {
        self.engine.lock_vector()
    }

    pub fn lock_graph(&self) -> Result<std::sync::MutexGuard<'_, GraphEngine>, CodememError> {
        self.engine.lock_graph()
    }

    pub(crate) fn lock_embeddings(
        &self,
    ) -> Result<
        Option<std::sync::MutexGuard<'_, Box<dyn codemem_embeddings::EmbeddingProvider>>>,
        CodememError,
    > {
        self.engine.lock_embeddings()
    }

    pub(crate) fn lock_index_cache(
        &self,
    ) -> Result<std::sync::MutexGuard<'_, Option<IndexCache>>, CodememError> {
        self.engine.lock_index_cache()
    }

    /// Core recall logic: delegates to `CodememEngine::recall()`.
    /// Used by the REST API layer and consolidation tools.
    #[allow(clippy::too_many_arguments)]
    pub fn recall(
        &self,
        query: &str,
        k: usize,
        memory_type_filter: Option<MemoryType>,
        namespace_filter: Option<&str>,
        exclude_tags: &[String],
        min_importance: Option<f64>,
        min_confidence: Option<f64>,
    ) -> Result<Vec<codemem_core::SearchResult>, CodememError> {
        self.engine.recall(
            query,
            k,
            memory_type_filter,
            namespace_filter,
            exclude_tags,
            min_importance,
            min_confidence,
        )
    }

    // ── Public Accessors (for REST API layer) ─────────────────────────────

    pub fn storage(&self) -> &dyn StorageBackend {
        &*self.engine.storage
    }

    pub fn graph(&self) -> &Mutex<GraphEngine> {
        &self.engine.graph
    }

    pub fn vector(&self) -> &Mutex<HnswIndex> {
        &self.engine.vector
    }

    pub fn embeddings(&self) -> Option<&Mutex<Box<dyn codemem_embeddings::EmbeddingProvider>>> {
        self.engine.embeddings.as_ref()
    }

    pub fn bm25(&self) -> &Mutex<bm25::Bm25Index> {
        &self.engine.bm25_index
    }

    pub fn reload_graph(&self) -> Result<(), CodememError> {
        self.engine.reload_graph()
    }

    pub fn db_path(&self) -> Option<&Path> {
        self.engine.db_path.as_deref()
    }

    pub fn config(&self) -> &codemem_core::CodememConfig {
        &self.engine.config
    }

    pub fn metrics_collector(&self) -> &std::sync::Arc<metrics::InMemoryMetrics> {
        &self.engine.metrics
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
                "tools": tool_definitions()
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
        codemem_core::Metrics::record_latency(&*self.engine.metrics, name, elapsed);
        codemem_core::Metrics::increment_counter(&*self.engine.metrics, "tool_calls_total", 1);
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
            "index_codebase" => self.tool_index_codebase(args),
            "search_code" => self.tool_search_code(args),
            "get_symbol_info" => self.tool_get_symbol_info(args),
            "get_symbol_graph" => self.tool_get_symbol_graph(args),
            "find_important_nodes" => self.tool_find_important_nodes(args),
            "find_related_groups" => self.tool_find_related_groups(args),
            "get_cross_repo" => self.tool_get_cross_repo(args),

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

            // ── Enrichment ──────────────────────────────────────────────
            "enrich_codebase" => self.tool_enrich_codebase(args),
            "analyze_codebase" => self.tool_analyze_codebase(args),
            "enrich_git_history" => self.tool_enrich_git_history(args),

            // ── Legacy aliases (backwards compatibility) ─────────────────
            "recall_memory" => self.tool_recall(args),
            "recall_with_expansion" => {
                // Translate to unified recall with expand=true
                let mut patched = args.clone();
                patched["expand"] = json!(true);
                self.tool_recall(&patched)
            }
            "recall_with_impact" => {
                // Translate to unified recall with include_impact=true
                let mut patched = args.clone();
                patched["include_impact"] = json!(true);
                self.tool_recall(&patched)
            }
            "update_memory" => {
                // Translate to refine_memory with destructive=true
                let mut patched = args.clone();
                patched["destructive"] = json!(true);
                self.tool_refine_memory(&patched)
            }
            "codemem_stats" => self.tool_codemem_status(&json!({"include": ["stats"]})),
            "codemem_health" => self.tool_codemem_status(&json!({"include": ["health"]})),
            "codemem_metrics" => self.tool_codemem_status(&json!({"include": ["metrics"]})),
            "consolidate_decay" => {
                let mut patched = args.clone();
                patched["mode"] = json!("decay");
                self.tool_consolidate(&patched)
            }
            "consolidate_creative" => {
                let mut patched = args.clone();
                patched["mode"] = json!("creative");
                self.tool_consolidate(&patched)
            }
            "consolidate_cluster" => {
                let mut patched = args.clone();
                patched["mode"] = json!("cluster");
                self.tool_consolidate(&patched)
            }
            "consolidate_forget" => {
                let mut patched = args.clone();
                patched["mode"] = json!("forget");
                self.tool_consolidate(&patched)
            }
            "consolidate_summarize" => {
                let mut patched = args.clone();
                patched["mode"] = json!("summarize");
                self.tool_consolidate(&patched)
            }
            "consolidation_status" => self.tool_consolidate(&json!({"mode": "auto"})),
            "pattern_insights" => {
                let mut patched = args.clone();
                patched["format"] = json!("markdown");
                self.tool_detect_patterns(&patched)
            }
            "search_symbols" => {
                let mut patched = args.clone();
                patched["mode"] = json!("text");
                self.tool_search_code(&patched)
            }
            "get_dependencies" => self.tool_get_symbol_graph(args),
            "get_impact" => self.tool_get_symbol_graph(args),
            "get_clusters" => self.tool_find_related_groups(args),
            "get_pagerank" => self.tool_find_important_nodes(args),
            "set_scoring_weights" | "export_memories" | "import_memories" => {
                ToolResult::tool_error(format!(
                    "Tool '{name}' has been removed from MCP. Use CLI or config instead."
                ))
            }
            "enrich_security" => self.tool_enrich_security(args),
            "enrich_performance" => self.tool_enrich_performance(args),

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

// ── Tool Definitions ────────────────────────────────────────────────────────

fn tool_definitions() -> Vec<Value> {
    vec![
        // ── Memory CRUD (7 tools) ──────────────────────────────────────────
        json!({
            "name": "store_memory",
            "description": "Store a new memory with auto-embedding, type classification, and graph linking. Automatically links to code nodes mentioned in content.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "content": { "type": "string", "description": "The memory content to store" },
                    "memory_type": {
                        "type": "string",
                        "enum": ["decision", "pattern", "preference", "style", "habit", "insight", "context"],
                        "description": "Type of memory (default: context)"
                    },
                    "importance": { "type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.5 },
                    "tags": { "type": "array", "items": { "type": "string" } },
                    "namespace": { "type": "string", "description": "Namespace to scope the memory (e.g. project path)" },
                    "links": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "List of graph node IDs to link this memory to"
                    },
                    "auto_link": { "type": "boolean", "default": true, "description": "Auto-link to code nodes mentioned in content (default: true)" }
                },
                "required": ["content"]
            }
        }),
        json!({
            "name": "recall",
            "description": "Unified memory search: 9-component hybrid scoring with optional graph expansion and impact analysis. Use expand=true for graph-expanded recall, include_impact=true for PageRank-enriched results.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "Natural language search query" },
                    "k": { "type": "integer", "default": 10, "description": "Number of results" },
                    "memory_type": { "type": "string", "description": "Filter by memory type" },
                    "namespace": { "type": "string", "description": "Filter results to a specific namespace" },
                    "exclude_tags": { "type": "array", "items": { "type": "string" }, "description": "Exclude memories with any of these tags" },
                    "min_importance": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
                    "min_confidence": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
                    "expand": { "type": "boolean", "default": false, "description": "Enable graph expansion to discover related memories" },
                    "expansion_depth": { "type": "integer", "default": 1, "description": "Max graph hops for expansion (when expand=true)" },
                    "include_impact": { "type": "boolean", "default": false, "description": "Include PageRank, centrality, connected decisions, dependent files" }
                },
                "required": ["query"]
            }
        }),
        json!({
            "name": "delete_memory",
            "description": "Delete a memory by ID, removing from vector index, graph, and storage",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "id": { "type": "string" }
                },
                "required": ["id"]
            }
        }),
        json!({
            "name": "associate_memories",
            "description": "Create a typed relationship between two nodes in the knowledge graph",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "source_id": { "type": "string" },
                    "target_id": { "type": "string" },
                    "relationship": {
                        "type": "string",
                        "enum": ["RELATES_TO","LEADS_TO","PART_OF","REINFORCES","CONTRADICTS",
                                 "EVOLVED_INTO","DERIVED_FROM","INVALIDATED_BY","DEPENDS_ON",
                                 "IMPORTS","EXTENDS","CALLS","CONTAINS","SUPERSEDES","BLOCKS",
                                 "IMPLEMENTS","INHERITS","SIMILAR_TO","PRECEDED_BY",
                                 "EXEMPLIFIES","EXPLAINS","SHARES_THEME","SUMMARIZES","CO_CHANGED"]
                    },
                    "weight": { "type": "number", "default": 1.0 }
                },
                "required": ["source_id", "target_id", "relationship"]
            }
        }),
        json!({
            "name": "refine_memory",
            "description": "Refine an existing memory. Default: creates a new version linked via EVOLVED_INTO. With destructive=true: updates in-place.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "id": { "type": "string", "description": "ID of the memory to refine" },
                    "content": { "type": "string", "description": "Updated content (optional unless destructive=true)" },
                    "importance": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
                    "tags": { "type": "array", "items": { "type": "string" } },
                    "destructive": { "type": "boolean", "default": false, "description": "When true, update in-place instead of creating a new version" }
                },
                "required": ["id"]
            }
        }),
        json!({
            "name": "split_memory",
            "description": "Split a memory into multiple parts, each linked to the original via PART_OF edges",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "id": { "type": "string", "description": "ID of the memory to split" },
                    "parts": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": { "type": "string" },
                                "tags": { "type": "array", "items": { "type": "string" } },
                                "importance": { "type": "number", "minimum": 0.0, "maximum": 1.0 }
                            },
                            "required": ["content"]
                        }
                    }
                },
                "required": ["id", "parts"]
            }
        }),
        json!({
            "name": "merge_memories",
            "description": "Merge multiple memories into a single summary memory linked via SUMMARIZES edges",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "source_ids": { "type": "array", "items": { "type": "string" }, "minItems": 2 },
                    "content": { "type": "string", "description": "Content for the merged summary memory" },
                    "memory_type": { "type": "string", "enum": ["decision", "pattern", "preference", "style", "habit", "insight", "context"] },
                    "importance": { "type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.7 },
                    "tags": { "type": "array", "items": { "type": "string" } }
                },
                "required": ["source_ids", "content"]
            }
        }),
        // ── Graph & Structure (7 tools) ────────────────────────────────────
        json!({
            "name": "graph_traverse",
            "description": "Multi-hop graph traversal from a start node with optional filtering by node kind and relationship type",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "start_id": { "type": "string" },
                    "max_depth": { "type": "integer", "default": 2 },
                    "algorithm": { "type": "string", "enum": ["bfs", "dfs"], "default": "bfs" },
                    "exclude_kinds": { "type": "array", "items": { "type": "string" } },
                    "include_relationships": { "type": "array", "items": { "type": "string" } }
                },
                "required": ["start_id"]
            }
        }),
        json!({
            "name": "summary_tree",
            "description": "Hierarchical summary tree (packages -> files -> symbols)",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "start_id": { "type": "string", "description": "Node ID to start from (e.g. 'pkg:src/')" },
                    "max_depth": { "type": "integer", "default": 3 },
                    "include_chunks": { "type": "boolean", "default": false }
                },
                "required": ["start_id"]
            }
        }),
        json!({
            "name": "codemem_status",
            "description": "Unified status: database stats, health check, and operational metrics. Use include=[\"stats\",\"health\",\"metrics\"] to select sections.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "include": {
                        "type": "array",
                        "items": { "type": "string", "enum": ["stats", "health", "metrics"] },
                        "description": "Sections to include (default: all)"
                    }
                }
            }
        }),
        json!({
            "name": "index_codebase",
            "description": "Index a codebase directory to extract symbols and references using tree-sitter",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Absolute path to the codebase directory" }
                },
                "required": ["path"]
            }
        }),
        json!({
            "name": "search_code",
            "description": "Search code by meaning or name. mode=semantic (vector search, default), mode=text (symbol name substring), mode=hybrid (both merged).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "Search query (natural language for semantic, substring for text)" },
                    "k": { "type": "integer", "default": 10, "description": "Number of results" },
                    "mode": { "type": "string", "enum": ["semantic", "text", "hybrid"], "default": "semantic" },
                    "kind": { "type": "string", "enum": ["function", "method", "class", "struct", "enum", "interface", "type", "constant", "module", "test"], "description": "Filter by symbol kind (text/hybrid modes)" }
                },
                "required": ["query"]
            }
        }),
        json!({
            "name": "get_symbol_info",
            "description": "Get full details of a symbol by qualified name. Optionally include graph dependencies.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "qualified_name": { "type": "string", "description": "Fully qualified name (e.g. 'module::Struct::method')" },
                    "include_dependencies": { "type": "boolean", "default": false, "description": "Include graph edges (calls, imports, etc.)" }
                },
                "required": ["qualified_name"]
            }
        }),
        json!({
            "name": "get_symbol_graph",
            "description": "Get symbol dependency graph. depth=1: direct edges (calls, imports). depth>1: full impact analysis (BFS reachability).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "qualified_name": { "type": "string" },
                    "depth": { "type": "integer", "default": 1, "description": "1=direct deps, >1=impact analysis" },
                    "direction": { "type": "string", "enum": ["incoming", "outgoing", "both"], "default": "both" }
                },
                "required": ["qualified_name"]
            }
        }),
        json!({
            "name": "find_important_nodes",
            "description": "Run PageRank to find the most important/central nodes in the knowledge graph",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "top_k": { "type": "integer", "default": 20 },
                    "damping": { "type": "number", "default": 0.85 }
                }
            }
        }),
        json!({
            "name": "find_related_groups",
            "description": "Run Louvain community detection to find clusters of related symbols",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "resolution": { "type": "number", "default": 1.0, "description": "Higher = more clusters" }
                }
            }
        }),
        json!({
            "name": "get_cross_repo",
            "description": "Scan workspace manifests and report cross-package dependencies",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Path to scan" }
                }
            }
        }),
        // ── Consolidation & Patterns (3 tools) ─────────────────────────────
        json!({
            "name": "consolidate",
            "description": "Run memory consolidation. mode=auto runs all cycles. Individual modes: decay, creative, cluster, forget, summarize.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["auto", "decay", "creative", "cluster", "forget", "summarize"],
                        "default": "auto"
                    },
                    "threshold_days": { "type": "integer", "description": "For decay mode (default: 30)" },
                    "similarity_threshold": { "type": "number", "description": "For cluster mode (default: 0.92)" },
                    "importance_threshold": { "type": "number", "description": "For forget mode (default: 0.1)" },
                    "target_tags": { "type": "array", "items": { "type": "string" }, "description": "For forget mode" },
                    "max_access_count": { "type": "integer", "description": "For forget mode" },
                    "cluster_size": { "type": "integer", "description": "For summarize mode (default: 5)" }
                }
            }
        }),
        json!({
            "name": "detect_patterns",
            "description": "Detect cross-session patterns. format=json (default), format=markdown (human-readable), format=both.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "min_frequency": { "type": "integer", "minimum": 1, "default": 3 },
                    "namespace": { "type": "string" },
                    "format": { "type": "string", "enum": ["json", "markdown", "both"], "default": "json" }
                }
            }
        }),
        json!({
            "name": "get_decision_chain",
            "description": "Follow decision evolution through the knowledge graph via EVOLVED_INTO/LEADS_TO/DERIVED_FROM edges",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "file_path": { "type": "string" },
                    "topic": { "type": "string" }
                }
            }
        }),
        // ── Namespace Management (3 tools) ──────────────────────────────────
        json!({
            "name": "list_namespaces",
            "description": "List all namespaces with inline stats (counts, avg importance, type distribution, date range)",
            "inputSchema": { "type": "object", "properties": {} }
        }),
        json!({
            "name": "namespace_stats",
            "description": "Detailed statistics for a specific namespace",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "namespace": { "type": "string" }
                },
                "required": ["namespace"]
            }
        }),
        json!({
            "name": "delete_namespace",
            "description": "Delete all memories in a namespace (requires confirm=true)",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "namespace": { "type": "string" },
                    "confirm": { "type": "boolean" }
                },
                "required": ["namespace", "confirm"]
            }
        }),
        // ── Session & Context (2 tools) ─────────────────────────────────────
        json!({
            "name": "session_checkpoint",
            "description": "Mid-session progress report with activity summary, pattern detection, and focus areas",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "session_id": { "type": "string" },
                    "namespace": { "type": "string" }
                },
                "required": ["session_id"]
            }
        }),
        json!({
            "name": "session_context",
            "description": "Get session context: recent memories, pending analyses, active patterns, and focus areas",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "namespace": { "type": "string" },
                    "k": { "type": "integer", "default": 10, "description": "Number of recent memories" }
                }
            }
        }),
        // ── Enrichment (3 tools) ────────────────────────────────────────────
        json!({
            "name": "enrich_codebase",
            "description": "Composite enrichment: runs git history, security, and performance analysis in one call",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Absolute path to the git repository root" },
                    "days": { "type": "integer", "default": 90 },
                    "namespace": { "type": "string" },
                    "analyses": {
                        "type": "array",
                        "items": { "type": "string", "enum": ["git", "security", "performance"] },
                        "description": "Which analyses to run (default: all)"
                    }
                },
                "required": ["path"]
            }
        }),
        json!({
            "name": "analyze_codebase",
            "description": "Full pipeline: index -> enrich (git+security+performance) -> pagerank -> clusters -> summary",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Absolute path to the codebase" },
                    "namespace": { "type": "string" },
                    "days": { "type": "integer", "default": 90 }
                },
                "required": ["path"]
            }
        }),
        json!({
            "name": "enrich_git_history",
            "description": "Enrich knowledge graph with git history: commit counts, churn rate, CoChanged edges, activity Insights",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Git repository root" },
                    "days": { "type": "integer", "default": 90 },
                    "namespace": { "type": "string" }
                },
                "required": ["path"]
            }
        }),
    ]
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests/lib_tests.rs"]
mod tests;
