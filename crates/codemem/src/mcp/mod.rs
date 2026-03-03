//! codemem-mcp: MCP server for Codemem (JSON-RPC 2.0 over stdio).
//!
//! Implements 43 tools: store_memory, recall_memory, update_memory,
//! delete_memory, associate_memories, graph_traverse, summary_tree,
//! codemem_stats, codemem_health,
//! index_codebase, search_symbols, get_symbol_info, get_dependencies, get_impact,
//! get_clusters, get_cross_repo, get_pagerank, search_code, set_scoring_weights,
//! export_memories, import_memories, recall_with_expansion, list_namespaces,
//! namespace_stats, delete_namespace, consolidate_decay, consolidate_creative,
//! consolidate_cluster, consolidate_forget, consolidation_status,
//! recall_with_impact, get_decision_chain, detect_patterns, pattern_insights,
//! refine_memory, split_memory, merge_memories, consolidate_summarize,
//! codemem_metrics, session_checkpoint,
//! enrich_git_history, enrich_security, enrich_performance.
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

    pub(crate) fn lock_bm25(
        &self,
    ) -> Result<std::sync::MutexGuard<'_, bm25::Bm25Index>, CodememError> {
        self.engine.lock_bm25()
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

    pub(crate) fn scoring_weights(
        &self,
    ) -> Result<std::sync::RwLockReadGuard<'_, codemem_core::ScoringWeights>, CodememError> {
        self.engine.scoring_weights()
    }

    pub(crate) fn scoring_weights_mut(
        &self,
    ) -> Result<std::sync::RwLockWriteGuard<'_, codemem_core::ScoringWeights>, CodememError> {
        self.engine.scoring_weights_mut()
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
            "store_memory" => self.tool_store_memory(args),
            "recall_memory" => self.tool_recall_memory(args),
            "update_memory" => self.tool_update_memory(args),
            "delete_memory" => self.tool_delete_memory(args),
            "associate_memories" => self.tool_associate_memories(args),
            "graph_traverse" => self.tool_graph_traverse(args),
            "summary_tree" => self.tool_summary_tree(args),
            "codemem_stats" => self.tool_stats(),
            "codemem_health" => self.tool_health(),
            "index_codebase" => self.tool_index_codebase(args),
            "search_symbols" => self.tool_search_symbols(args),
            "get_symbol_info" => self.tool_get_symbol_info(args),
            "get_dependencies" => self.tool_get_dependencies(args),
            "get_impact" => self.tool_get_impact(args),
            "get_clusters" => self.tool_get_clusters(args),
            "get_cross_repo" => self.tool_get_cross_repo(args),
            "get_pagerank" => self.tool_get_pagerank(args),
            "search_code" => self.tool_search_code(args),
            "set_scoring_weights" => self.tool_set_scoring_weights(args),
            "consolidate_decay" => self.tool_consolidate_decay(args),
            "consolidate_creative" => self.tool_consolidate_creative(args),
            "consolidate_cluster" => self.tool_consolidate_cluster(args),
            "consolidate_forget" => self.tool_consolidate_forget(args),
            "consolidation_status" => self.tool_consolidation_status(),
            "recall_with_expansion" => self.tool_recall_with_expansion(args),
            "recall_with_impact" => self.tool_recall_with_impact(args),
            "get_decision_chain" => self.tool_get_decision_chain(args),
            "list_namespaces" => self.tool_list_namespaces(),
            "namespace_stats" => self.tool_namespace_stats(args),
            "delete_namespace" => self.tool_delete_namespace(args),
            "export_memories" => self.tool_export_memories(args),
            "import_memories" => self.tool_import_memories(args),
            "detect_patterns" => self.tool_detect_patterns(args),
            "pattern_insights" => self.tool_pattern_insights(args),
            "refine_memory" => self.tool_refine_memory(args),
            "split_memory" => self.tool_split_memory(args),
            "merge_memories" => self.tool_merge_memories(args),
            "consolidate_summarize" => self.tool_consolidate_summarize(args),
            "codemem_metrics" => self.tool_metrics(),
            "enrich_git_history" => self.tool_enrich_git_history(args),
            "enrich_security" => self.tool_enrich_security(args),
            "enrich_performance" => self.tool_enrich_performance(args),
            "session_checkpoint" => self.tool_session_checkpoint(args),
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
        json!({
            "name": "store_memory",
            "description": "Store a new memory with auto-embedding, type classification, and graph linking",
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
                        "description": "List of graph node IDs to link this memory to (e.g., structural symbol IDs)"
                    }
                },
                "required": ["content"]
            }
        }),
        json!({
            "name": "recall_memory",
            "description": "Semantic search using 9-component hybrid scoring with graph expansion and bridge discovery",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "Natural language search query" },
                    "k": { "type": "integer", "default": 10, "description": "Number of results" },
                    "memory_type": { "type": "string", "description": "Filter by memory type" },
                    "namespace": { "type": "string", "description": "Filter results to a specific namespace" },
                    "exclude_tags": { "type": "array", "items": { "type": "string" }, "description": "Exclude memories with any of these tags (e.g. [\"static-analysis\"])" },
                    "min_importance": { "type": "number", "minimum": 0.0, "maximum": 1.0, "description": "Only return memories with importance >= this value" },
                    "min_confidence": { "type": "number", "minimum": 0.0, "maximum": 1.0, "description": "Only return memories with confidence >= this value" }
                },
                "required": ["query"]
            }
        }),
        json!({
            "name": "update_memory",
            "description": "Update an existing memory's content and re-embed",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "id": { "type": "string" },
                    "content": { "type": "string" },
                    "importance": { "type": "number", "minimum": 0.0, "maximum": 1.0 }
                },
                "required": ["id", "content"]
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
            "description": "Create a typed relationship between two memories in the knowledge graph",
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
            "name": "graph_traverse",
            "description": "Multi-hop graph traversal from a start node with optional filtering by node kind and relationship type",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "start_id": { "type": "string" },
                    "max_depth": { "type": "integer", "default": 2 },
                    "algorithm": { "type": "string", "enum": ["bfs", "dfs"], "default": "bfs" },
                    "exclude_kinds": {
                        "type": "array",
                        "items": { "type": "string", "enum": ["file","package","function","class","module","memory","method","interface","type","constant","endpoint","test","chunk"] },
                        "description": "Node kinds to exclude from results and traversal (e.g. [\"chunk\"] to skip chunks)"
                    },
                    "include_relationships": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Only follow edges of these relationship types (e.g. [\"CALLS\",\"IMPORTS\"]). If omitted, all relationships are followed."
                    }
                },
                "required": ["start_id"]
            }
        }),
        json!({
            "name": "summary_tree",
            "description": "Return a hierarchical summary tree (packages → files → symbols). Start from a pkg: node to see the directory structure.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "start_id": { "type": "string", "description": "Node ID to start from (e.g. 'pkg:src/')" },
                    "max_depth": { "type": "integer", "default": 3, "description": "Maximum tree depth" },
                    "include_chunks": { "type": "boolean", "default": false, "description": "Include chunk nodes in the tree" }
                },
                "required": ["start_id"]
            }
        }),
        json!({
            "name": "codemem_stats",
            "description": "Get database and index statistics",
            "inputSchema": { "type": "object", "properties": {} }
        }),
        json!({
            "name": "codemem_health",
            "description": "Health check across all Codemem subsystems (storage, vector, graph, embeddings)",
            "inputSchema": { "type": "object", "properties": {} }
        }),
        // ── Structural Index Tools ──────────────────────────────────────────
        json!({
            "name": "index_codebase",
            "description": "Index a codebase directory to extract symbols and references using tree-sitter, populating the structural knowledge graph",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Absolute path to the codebase directory to index" }
                },
                "required": ["path"]
            }
        }),
        json!({
            "name": "search_symbols",
            "description": "Search indexed code symbols by name substring, optionally filtering by kind (function, method, struct, etc.)",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "Substring to search for in symbol names" },
                    "kind": {
                        "type": "string",
                        "enum": ["function", "method", "class", "struct", "enum", "interface", "type", "constant", "module", "test"],
                        "description": "Filter by symbol kind"
                    },
                    "limit": { "type": "integer", "default": 20, "description": "Maximum number of results" }
                },
                "required": ["query"]
            }
        }),
        json!({
            "name": "get_symbol_info",
            "description": "Get full details of a symbol by qualified name, including signature, file path, doc comment, and parent",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "qualified_name": { "type": "string", "description": "Fully qualified name of the symbol (e.g. 'module::Struct::method')" }
                },
                "required": ["qualified_name"]
            }
        }),
        json!({
            "name": "get_dependencies",
            "description": "Get graph edges (calls, imports, extends, etc.) connected to a symbol",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "qualified_name": { "type": "string", "description": "Fully qualified name of the symbol" },
                    "direction": {
                        "type": "string",
                        "enum": ["incoming", "outgoing", "both"],
                        "default": "both",
                        "description": "Direction of dependencies to return"
                    }
                },
                "required": ["qualified_name"]
            }
        }),
        json!({
            "name": "get_impact",
            "description": "Impact analysis: find all graph nodes reachable from a symbol within N hops (what breaks if this changes?)",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "qualified_name": { "type": "string", "description": "Fully qualified name of the symbol to analyze" },
                    "depth": { "type": "integer", "default": 2, "description": "Maximum BFS depth for reachability" }
                },
                "required": ["qualified_name"]
            }
        }),
        json!({
            "name": "get_clusters",
            "description": "Run Louvain community detection on the knowledge graph to find clusters of related symbols",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "resolution": { "type": "number", "default": 1.0, "description": "Louvain resolution parameter (higher = more clusters)" }
                }
            }
        }),
        json!({
            "name": "get_cross_repo",
            "description": "Scan for workspace manifests (Cargo.toml, package.json) and report workspace structure and cross-package dependencies",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Path to scan (defaults to the last indexed codebase root)" }
                }
            }
        }),
        json!({
            "name": "get_pagerank",
            "description": "Run PageRank on the full knowledge graph to find the most important/central nodes",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "top_k": { "type": "integer", "default": 20, "description": "Number of top-ranked nodes to return" },
                    "damping": { "type": "number", "default": 0.85, "description": "PageRank damping factor" }
                }
            }
        }),
        json!({
            "name": "search_code",
            "description": "Semantic search over indexed code symbols using signature embeddings. Finds functions, types, and methods by meaning rather than exact name match.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "Natural language description of the code you're looking for (e.g. 'parse JSON config', 'HTTP request handler')" },
                    "k": { "type": "integer", "default": 10, "description": "Number of results to return" }
                },
                "required": ["query"]
            }
        }),
        json!({
            "name": "set_scoring_weights",
            "description": "Update the 9-component hybrid scoring weights at runtime. Weights are normalized to sum to 1.0. Omitted weights use their default values.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "vector_similarity": { "type": "number", "minimum": 0.0, "description": "Weight for vector cosine similarity (default: 0.25)" },
                    "graph_strength": { "type": "number", "minimum": 0.0, "description": "Weight for graph relationship strength (default: 0.25)" },
                    "token_overlap": { "type": "number", "minimum": 0.0, "description": "Weight for content token overlap (default: 0.15)" },
                    "temporal": { "type": "number", "minimum": 0.0, "description": "Weight for temporal alignment (default: 0.10)" },
                    "tag_matching": { "type": "number", "minimum": 0.0, "description": "Weight for tag matching (default: 0.10)" },
                    "importance": { "type": "number", "minimum": 0.0, "description": "Weight for importance score (default: 0.05)" },
                    "confidence": { "type": "number", "minimum": 0.0, "description": "Weight for memory confidence (default: 0.05)" },
                    "recency": { "type": "number", "minimum": 0.0, "description": "Weight for recency boost (default: 0.05)" }
                }
            }
        }),
        // ── Export/Import Tools ──────────────────────────────────────────────
        json!({
            "name": "export_memories",
            "description": "Export memories as a JSON array with optional namespace and memory_type filters. Returns memory objects with their graph edges.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "namespace": { "type": "string", "description": "Filter by namespace" },
                    "memory_type": {
                        "type": "string",
                        "enum": ["decision", "pattern", "preference", "style", "habit", "insight", "context"],
                        "description": "Filter by memory type"
                    },
                    "limit": { "type": "integer", "default": 100, "description": "Maximum number of memories to export" }
                }
            }
        }),
        json!({
            "name": "import_memories",
            "description": "Import memories from a JSON array. Each object must have at least a 'content' field. Auto-deduplicates by content hash.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "memories": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": { "type": "string", "description": "The memory content (required)" },
                                "memory_type": {
                                    "type": "string",
                                    "enum": ["decision", "pattern", "preference", "style", "habit", "insight", "context"],
                                    "description": "Type of memory (default: context)"
                                },
                                "importance": { "type": "number", "minimum": 0.0, "maximum": 1.0, "description": "Importance score (default: 0.5)" },
                                "confidence": { "type": "number", "minimum": 0.0, "maximum": 1.0, "description": "Confidence score (default: 1.0)" },
                                "tags": { "type": "array", "items": { "type": "string" } },
                                "namespace": { "type": "string", "description": "Namespace to scope the memory" },
                                "metadata": { "type": "object", "description": "Arbitrary metadata key-value pairs" }
                            },
                            "required": ["content"]
                        },
                        "description": "Array of memory objects to import"
                    }
                },
                "required": ["memories"]
            }
        }),
        // ── Graph-Expanded Recall & Namespace Management ────────────────────
        json!({
            "name": "recall_with_expansion",
            "description": "Semantic search with graph expansion: finds memories via vector similarity then expands through the knowledge graph to discover related memories up to N hops away",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "Natural language search query" },
                    "k": { "type": "integer", "default": 5, "description": "Number of results to return" },
                    "expansion_depth": { "type": "integer", "default": 1, "description": "Maximum graph hops for expansion (0 = no expansion)" },
                    "namespace": { "type": "string", "description": "Filter results to a specific namespace" }
                },
                "required": ["query"]
            }
        }),
        json!({
            "name": "list_namespaces",
            "description": "List all namespaces with their memory counts",
            "inputSchema": { "type": "object", "properties": {} }
        }),
        json!({
            "name": "namespace_stats",
            "description": "Get detailed statistics for a specific namespace: count, avg importance/confidence, type distribution, tag frequency, date range",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "namespace": { "type": "string", "description": "Namespace to get stats for" }
                },
                "required": ["namespace"]
            }
        }),
        json!({
            "name": "delete_namespace",
            "description": "Delete all memories in a namespace (destructive, requires confirmation)",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "namespace": { "type": "string", "description": "Namespace to delete" },
                    "confirm": { "type": "boolean", "description": "Must be true to confirm deletion" }
                },
                "required": ["namespace", "confirm"]
            }
        }),
        // ── Impact-Aware Recall & Decision Chain Tools ──────────────────────
        json!({
            "name": "recall_with_impact",
            "description": "Semantic search with PageRank-enriched impact data. Returns memories with pagerank, centrality, connected decisions, dependent files, and modification counts.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "Natural language search query" },
                    "k": { "type": "integer", "default": 10, "description": "Number of results" },
                    "namespace": { "type": "string", "description": "Filter results to a specific namespace" }
                },
                "required": ["query"]
            }
        }),
        json!({
            "name": "get_decision_chain",
            "description": "Follow the evolution of decisions through the knowledge graph. Traces EVOLVED_INTO, LEADS_TO, and DERIVED_FROM edges to build a chronologically ordered decision chain.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "file_path": { "type": "string", "description": "File path to find decisions about (e.g. 'src/auth.rs')" },
                    "topic": { "type": "string", "description": "Topic to find decisions about (e.g. 'authentication')" }
                }
            }
        }),
        // ── Consolidation Tools ─────────────────────────────────────────────
        json!({
            "name": "consolidate_decay",
            "description": "Run decay consolidation: reduce importance by 10% for memories not accessed within threshold_days",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "threshold_days": { "type": "integer", "default": 30, "description": "Memories not accessed in this many days will decay (default: 30)" }
                }
            }
        }),
        json!({
            "name": "consolidate_creative",
            "description": "Run creative consolidation: find pairs of memories with overlapping tags but different types, create RELATES_TO edges between them",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        }),
        json!({
            "name": "consolidate_cluster",
            "description": "Run cluster consolidation: group memories by content_hash prefix, keep highest-importance per group, delete duplicates",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "similarity_threshold": { "type": "number", "minimum": 0.5, "maximum": 1.0, "default": 0.92, "description": "Cosine similarity threshold for semantic deduplication (default: 0.92)" }
                }
            }
        }),
        json!({
            "name": "consolidate_forget",
            "description": "Run forget consolidation: delete memories with importance below threshold. Optionally target specific tags for cleanup.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "importance_threshold": { "type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.1, "description": "Delete memories with importance below this value (default: 0.1)" },
                    "target_tags": { "type": "array", "items": { "type": "string" }, "description": "Only forget memories with any of these tags (e.g. [\"static-analysis\"])" },
                    "max_access_count": { "type": "integer", "default": 0, "description": "Only forget memories accessed at most this many times (default: 0)" }
                }
            }
        }),
        json!({
            "name": "consolidation_status",
            "description": "Show the last run timestamp and affected count for each consolidation cycle type",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        }),
        json!({
            "name": "detect_patterns",
            "description": "Detect cross-session patterns in stored memories. Analyzes repeated searches, file hotspots, decision chains, and tool usage preferences across sessions.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "min_frequency": {
                        "type": "integer",
                        "minimum": 1,
                        "default": 3,
                        "description": "Minimum number of occurrences before a pattern is flagged (default: 3)"
                    },
                    "namespace": {
                        "type": "string",
                        "description": "Optional namespace to scope the pattern detection"
                    }
                }
            }
        }),
        json!({
            "name": "pattern_insights",
            "description": "Generate human-readable markdown insights from cross-session patterns. Summarizes file hotspots, repeated searches, decision chains, and tool preferences.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "min_frequency": {
                        "type": "integer",
                        "minimum": 1,
                        "default": 2,
                        "description": "Minimum number of occurrences before a pattern is included (default: 2)"
                    },
                    "namespace": {
                        "type": "string",
                        "description": "Optional namespace to scope the pattern insights"
                    }
                }
            }
        }),
        // ── Memory Refinement & Merge Tools ──────────────────────────────────
        json!({
            "name": "refine_memory",
            "description": "Refine an existing memory: creates a new version linked via EVOLVED_INTO edge, preserving the original for provenance tracking",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "id": { "type": "string", "description": "ID of the memory to refine" },
                    "content": { "type": "string", "description": "Updated content (optional, inherits from original)" },
                    "importance": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
                    "tags": { "type": "array", "items": { "type": "string" } }
                },
                "required": ["id"]
            }
        }),
        json!({
            "name": "split_memory",
            "description": "Split a memory into multiple parts, each linked to the original via PART_OF edges for provenance tracking",
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
                        },
                        "description": "Array of parts to create from the source memory"
                    }
                },
                "required": ["id", "parts"]
            }
        }),
        json!({
            "name": "merge_memories",
            "description": "Merge multiple memories into a single summary memory linked via SUMMARIZES edges for provenance tracking",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "source_ids": {
                        "type": "array",
                        "items": { "type": "string" },
                        "minItems": 2,
                        "description": "IDs of memories to merge (minimum 2)"
                    },
                    "content": { "type": "string", "description": "Content for the merged summary memory" },
                    "memory_type": {
                        "type": "string",
                        "enum": ["decision", "pattern", "preference", "style", "habit", "insight", "context"],
                        "description": "Type for the merged memory (default: insight)"
                    },
                    "importance": { "type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.7 },
                    "tags": { "type": "array", "items": { "type": "string" } }
                },
                "required": ["source_ids", "content"]
            }
        }),
        json!({
            "name": "consolidate_summarize",
            "description": "LLM-powered consolidation: find connected components, summarize large clusters into Insight memories linked via SUMMARIZES edges. Requires CODEMEM_COMPRESS_PROVIDER env var.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "cluster_size": { "type": "integer", "minimum": 2, "default": 5, "description": "Minimum cluster size to summarize (default: 5)" }
                }
            }
        }),
        json!({
            "name": "codemem_metrics",
            "description": "Return operational metrics: per-tool latency percentiles (p50/p95/p99), call counters, and gauge values. No parameters required.",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        }),
        // ── Enrichment Tools ──────────────────────────────────────────────────
        json!({
            "name": "enrich_git_history",
            "description": "Enrich the knowledge graph with git history: annotate file nodes with commit counts, authors, and churn rate; create CoChanged edges between files that change together; store activity Insights.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Absolute path to the git repository root" },
                    "days": { "type": "integer", "default": 90, "description": "Number of days of history to analyze (default: 90)" },
                    "namespace": { "type": "string", "description": "Namespace for stored insights" }
                },
                "required": ["path"]
            }
        }),
        json!({
            "name": "enrich_security",
            "description": "Scan the knowledge graph for security-sensitive files, endpoints, and functions. Annotates nodes with security flags and stores security Insights.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "namespace": { "type": "string", "description": "Namespace filter for insights" }
                }
            }
        }),
        json!({
            "name": "enrich_performance",
            "description": "Analyze graph coupling, dependency depth, critical path (PageRank), and file complexity. Annotates nodes and stores performance Insights.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "namespace": { "type": "string", "description": "Namespace filter for insights" },
                    "top": { "type": "integer", "default": 10, "description": "Number of top items to report (default: 10)" }
                }
            }
        }),
        // ── Session Checkpoint Tool ─────────────────────────────────────────────
        json!({
            "name": "session_checkpoint",
            "description": "Mid-session checkpoint: summarize activity so far, detect session-scoped and cross-session patterns, identify focus areas, and store new pattern insights. Returns a markdown progress report.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "session_id": { "type": "string", "description": "The current session ID" },
                    "namespace": { "type": "string", "description": "Optional namespace to scope pattern detection" }
                },
                "required": ["session_id"]
            }
        }),
    ]
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests/lib_tests.rs"]
mod tests;
