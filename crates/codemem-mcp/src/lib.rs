//! codemem-mcp: MCP server for Codemem (JSON-RPC 2.0 over stdio).
//!
//! Implements 33 tools: store_memory, recall_memory, update_memory,
//! delete_memory, associate_memories, graph_traverse, codemem_stats, codemem_health,
//! index_codebase, search_symbols, get_symbol_info, get_dependencies, get_impact,
//! get_clusters, get_cross_repo, get_pagerank, search_code, set_scoring_weights,
//! export_memories, import_memories, recall_with_expansion, list_namespaces,
//! namespace_stats, delete_namespace, consolidate_decay, consolidate_creative,
//! consolidate_cluster, consolidate_forget, consolidation_status,
//! recall_with_impact, get_decision_chain, detect_patterns, pattern_insights.
//!
//! Transport: Newline-delimited JSON-RPC messages over stdio.
//! All logging goes to stderr; stdout is reserved for JSON-RPC only.

use codemem_core::{
    CodememError, GraphBackend, MemoryType, ScoringWeights, StorageBackend, VectorBackend,
};
use codemem_graph::GraphEngine;
use codemem_storage::Storage;
use codemem_vector::HnswIndex;
use serde_json::{json, Value};
use std::io::{self, BufRead};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, RwLock};

pub mod bm25;
pub mod patterns;
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

use scoring::write_response;
use types::IndexCache;

// ── MCP Server ──────────────────────────────────────────────────────────────

/// MCP server that reads JSON-RPC from stdin, writes responses to stdout.
/// Holds Storage, HnswIndex, and GraphEngine for real tool dispatch.
pub struct McpServer {
    pub name: String,
    pub version: String,
    pub(crate) storage: Box<dyn StorageBackend>,
    pub(crate) vector: Mutex<HnswIndex>,
    pub(crate) graph: Mutex<GraphEngine>,
    /// Optional embedding provider (None if not configured).
    pub(crate) embeddings: Option<Mutex<Box<dyn codemem_embeddings::EmbeddingProvider>>>,
    /// Path to the database file, used to derive the index save path.
    pub(crate) db_path: Option<PathBuf>,
    /// Cached index results for structural queries.
    pub(crate) index_cache: Mutex<Option<IndexCache>>,
    /// Configurable scoring weights for the 9-component hybrid scoring system.
    pub(crate) scoring_weights: RwLock<ScoringWeights>,
    /// BM25 index for code-aware token overlap scoring.
    /// Updated incrementally on store/update/delete operations.
    pub(crate) bm25_index: Mutex<bm25::Bm25Index>,
    /// Loaded configuration (used by scoring_weights initialization and future features).
    #[allow(dead_code)]
    pub(crate) config: codemem_core::CodememConfig,
}

impl McpServer {
    /// Create a server with storage, vector, graph, and optional embeddings backends.
    pub fn new(
        storage: Box<dyn StorageBackend>,
        vector: HnswIndex,
        graph: GraphEngine,
        embeddings: Option<Box<dyn codemem_embeddings::EmbeddingProvider>>,
    ) -> Self {
        let config = codemem_core::CodememConfig::load_or_default();
        Self {
            name: "codemem".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            storage,
            vector: Mutex::new(vector),
            graph: Mutex::new(graph),
            embeddings: embeddings.map(Mutex::new),
            db_path: None,
            index_cache: Mutex::new(None),
            scoring_weights: RwLock::new(config.scoring.clone()),
            bm25_index: Mutex::new(bm25::Bm25Index::new()),
            config,
        }
    }

    /// Create a server from a database path, loading all backends.
    pub fn from_db_path(db_path: &Path) -> Result<Self, CodememError> {
        let storage = Storage::open(db_path)?;
        let mut vector = HnswIndex::with_defaults()?;

        // Load existing vector index if it exists
        let index_path = db_path.with_extension("idx");
        if index_path.exists() {
            vector.load(&index_path)?;
        }

        // Load graph from storage
        let graph = GraphEngine::from_storage(&storage)?;

        // Try loading embeddings (optional - selects provider from env vars)
        let embeddings = codemem_embeddings::from_env().ok();

        let mut server = Self::new(Box::new(storage), vector, graph, embeddings);
        server.db_path = Some(db_path.to_path_buf());

        // Recompute centrality metrics (PageRank + betweenness) on startup
        server.lock_graph()?.recompute_centrality();

        // Populate BM25 index from all existing memories
        if let Ok(ids) = server.storage.list_memory_ids() {
            let mut bm25 = server.lock_bm25()?;
            for id in &ids {
                if let Ok(Some(memory)) = server.storage.get_memory(id) {
                    bm25.add_document(id, &memory.content);
                }
            }
        }

        Ok(server)
    }

    /// Create a minimal server for testing (no backends wired).
    pub fn for_testing() -> Self {
        let storage = Storage::open_in_memory().unwrap();
        let vector = HnswIndex::with_defaults().unwrap();
        let graph = GraphEngine::new();
        Self::new(Box::new(storage), vector, graph, None)
    }

    // ── Lock Helpers ─────────────────────────────────────────────────────────

    pub(crate) fn lock_vector(&self) -> Result<std::sync::MutexGuard<'_, HnswIndex>, CodememError> {
        self.vector
            .lock()
            .map_err(|e| CodememError::LockPoisoned(format!("vector: {e}")))
    }

    pub(crate) fn lock_graph(
        &self,
    ) -> Result<std::sync::MutexGuard<'_, GraphEngine>, CodememError> {
        self.graph
            .lock()
            .map_err(|e| CodememError::LockPoisoned(format!("graph: {e}")))
    }

    pub(crate) fn lock_bm25(
        &self,
    ) -> Result<std::sync::MutexGuard<'_, bm25::Bm25Index>, CodememError> {
        self.bm25_index
            .lock()
            .map_err(|e| CodememError::LockPoisoned(format!("bm25: {e}")))
    }

    pub(crate) fn lock_embeddings(
        &self,
    ) -> Result<
        Option<std::sync::MutexGuard<'_, Box<dyn codemem_embeddings::EmbeddingProvider>>>,
        CodememError,
    > {
        match &self.embeddings {
            Some(m) => Ok(Some(m.lock().map_err(|e| {
                CodememError::LockPoisoned(format!("embeddings: {e}"))
            })?)),
            None => Ok(None),
        }
    }

    pub(crate) fn lock_index_cache(
        &self,
    ) -> Result<std::sync::MutexGuard<'_, Option<types::IndexCache>>, CodememError> {
        self.index_cache
            .lock()
            .map_err(|e| CodememError::LockPoisoned(format!("index_cache: {e}")))
    }

    pub(crate) fn scoring_weights(
        &self,
    ) -> Result<std::sync::RwLockReadGuard<'_, codemem_core::ScoringWeights>, CodememError> {
        self.scoring_weights
            .read()
            .map_err(|e| CodememError::LockPoisoned(format!("scoring_weights read: {e}")))
    }

    pub(crate) fn scoring_weights_mut(
        &self,
    ) -> Result<std::sync::RwLockWriteGuard<'_, codemem_core::ScoringWeights>, CodememError> {
        self.scoring_weights
            .write()
            .map_err(|e| CodememError::LockPoisoned(format!("scoring_weights write: {e}")))
    }

    // ── Contextual Enrichment ────────────────────────────────────────────────
    // Prepend metadata context to text before embedding, following the
    // "contextual embeddings" methodology: enriched vectors capture semantic
    // relationships that raw content alone would miss.

    /// Build contextual text for a memory node.
    /// Prepends memory type, tags, namespace, and graph relationships so the
    /// embedding captures the memory's role, not just its content.
    pub(crate) fn enrich_memory_text(
        &self,
        content: &str,
        memory_type: MemoryType,
        tags: &[String],
        namespace: Option<&str>,
        node_id: Option<&str>,
    ) -> String {
        let mut ctx = String::new();

        // Memory type
        ctx.push_str(&format!("[{}]", memory_type));

        // Namespace
        if let Some(ns) = namespace {
            ctx.push_str(&format!(" [namespace:{}]", ns));
        }

        // Tags
        if !tags.is_empty() {
            ctx.push_str(&format!(" [tags:{}]", tags.join(",")));
        }

        // Graph relationships — pull connected edges for this node
        if let Some(nid) = node_id {
            let graph = match self.lock_graph() {
                Ok(g) => g,
                Err(_) => return format!("{ctx}\n{content}"),
            };
            if let Ok(edges) = graph.get_edges(nid) {
                let mut rels: Vec<String> = Vec::new();
                for edge in edges.iter().take(8) {
                    let other = if edge.src == nid {
                        &edge.dst
                    } else {
                        &edge.src
                    };
                    // Resolve the other node's label for readable context
                    let label = graph
                        .get_node(other)
                        .ok()
                        .flatten()
                        .map(|n| n.label.clone())
                        .unwrap_or_else(|| other.to_string());
                    let dir = if edge.src == nid { "->" } else { "<-" };
                    rels.push(format!("{dir} {} ({})", label, edge.relationship));
                }
                if !rels.is_empty() {
                    ctx.push_str(&format!("\nRelated: {}", rels.join("; ")));
                }
            }
        }

        format!("{ctx}\n{content}")
    }

    /// Build contextual text for a code symbol.
    /// Prepends symbol kind, file path, parent, visibility, and resolved
    /// edges so the embedding captures the symbol's structural context.
    pub(crate) fn enrich_symbol_text(
        &self,
        sym: &codemem_index::Symbol,
        edges: &[codemem_index::ResolvedEdge],
    ) -> String {
        let mut ctx = String::new();

        // Symbol kind + visibility
        ctx.push_str(&format!("[{} {}]", sym.visibility, sym.kind));

        // File path
        ctx.push_str(&format!(" File: {}", sym.file_path));

        // Parent (e.g., struct for a method)
        if let Some(ref parent) = sym.parent {
            ctx.push_str(&format!(" Parent: {}", parent));
        }

        // Resolved edges — calls, imports, inherits, etc.
        let related: Vec<String> = edges
            .iter()
            .filter(|e| {
                e.source_qualified_name == sym.qualified_name
                    || e.target_qualified_name == sym.qualified_name
            })
            .take(8)
            .map(|e| {
                if e.source_qualified_name == sym.qualified_name {
                    format!("-> {} ({})", e.target_qualified_name, e.relationship)
                } else {
                    format!("<- {} ({})", e.source_qualified_name, e.relationship)
                }
            })
            .collect();
        if !related.is_empty() {
            ctx.push_str(&format!("\nRelated: {}", related.join("; ")));
        }

        // Signature + doc comment
        let mut body = format!("{}: {}", sym.qualified_name, sym.signature);
        if let Some(ref doc) = sym.doc_comment {
            body.push('\n');
            body.push_str(doc);
        }

        format!("{ctx}\n{body}")
    }

    /// Save the HNSW index to disk. The index file path is derived from
    /// the database path with an `.idx` extension. No-op if db_path is None
    /// (e.g., in-memory / testing mode).
    pub fn save_index(&self) {
        if let Some(ref db_path) = self.db_path {
            let index_path = db_path.with_extension("idx");
            match self.lock_vector() {
                Ok(vec) => {
                    if let Err(e) = vec.save(&index_path) {
                        tracing::warn!("Failed to save vector index: {e}");
                    }
                }
                Err(e) => {
                    tracing::warn!("Failed to acquire vector lock for save: {e}");
                }
            }
        }
    }

    /// Run the MCP server. Reads newline-delimited JSON-RPC from stdin,
    /// writes responses to stdout. Blocks until stdin is closed.
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
                self.handle_notification(&request.method);
                continue;
            }

            let id = request.id.unwrap();
            let response = self.handle_request(&request.method, request.params.as_ref(), id);
            write_response(&mut stdout, &response)?;
        }

        Ok(())
    }

    fn handle_notification(&self, method: &str) {
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
        match name {
            "store_memory" => self.tool_store_memory(args),
            "recall_memory" => self.tool_recall_memory(args),
            "update_memory" => self.tool_update_memory(args),
            "delete_memory" => self.tool_delete_memory(args),
            "associate_memories" => self.tool_associate_memories(args),
            "graph_traverse" => self.tool_graph_traverse(args),
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
            _ => ToolResult::tool_error(format!("Unknown tool: {name}")),
        }
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
                    "namespace": { "type": "string", "description": "Filter results to a specific namespace" }
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
                                 "EXEMPLIFIES","EXPLAINS","SHARES_THEME","SUMMARIZES"]
                    },
                    "weight": { "type": "number", "default": 1.0 }
                },
                "required": ["source_id", "target_id", "relationship"]
            }
        }),
        json!({
            "name": "graph_traverse",
            "description": "Multi-hop graph traversal from a start node for reasoning and bridge discovery",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "start_id": { "type": "string" },
                    "max_depth": { "type": "integer", "default": 2 },
                    "algorithm": { "type": "string", "enum": ["bfs", "dfs"], "default": "bfs" }
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
                "properties": {}
            }
        }),
        json!({
            "name": "consolidate_forget",
            "description": "Run forget consolidation: delete memories with importance below threshold and zero access count",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "importance_threshold": { "type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.1, "description": "Delete memories with importance below this value (default: 0.1)" }
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
    ]
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use test_helpers::*;

    #[test]
    fn handle_initialize() {
        let server = test_server();
        let resp = server.handle_request("initialize", None, json!(1));
        assert!(resp.result.is_some());
        assert!(resp.error.is_none());

        let result = resp.result.unwrap();
        assert_eq!(result["protocolVersion"], "2024-11-05");
        assert_eq!(result["serverInfo"]["name"], "codemem");
    }

    #[test]
    fn handle_tools_list_returns_33_tools() {
        let server = test_server();
        let resp = server.handle_request("tools/list", None, json!(2));
        let result = resp.result.unwrap();
        let tools = result["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 33);

        let names: Vec<&str> = tools.iter().filter_map(|t| t["name"].as_str()).collect();
        assert!(names.contains(&"store_memory"));
        assert!(names.contains(&"recall_memory"));
        assert!(names.contains(&"graph_traverse"));
        assert!(names.contains(&"codemem_health"));
        assert!(names.contains(&"index_codebase"));
        assert!(names.contains(&"search_symbols"));
        assert!(names.contains(&"get_symbol_info"));
        assert!(names.contains(&"get_dependencies"));
        assert!(names.contains(&"get_impact"));
        assert!(names.contains(&"get_clusters"));
        assert!(names.contains(&"get_cross_repo"));
        assert!(names.contains(&"get_pagerank"));
        assert!(names.contains(&"search_code"));
        assert!(names.contains(&"set_scoring_weights"));
        assert!(names.contains(&"export_memories"));
        assert!(names.contains(&"import_memories"));
        assert!(names.contains(&"recall_with_expansion"));
        assert!(names.contains(&"list_namespaces"));
        assert!(names.contains(&"namespace_stats"));
        assert!(names.contains(&"delete_namespace"));
        assert!(names.contains(&"consolidate_decay"));
        assert!(names.contains(&"consolidate_creative"));
        assert!(names.contains(&"consolidate_cluster"));
        assert!(names.contains(&"consolidate_forget"));
        assert!(names.contains(&"consolidation_status"));
        assert!(names.contains(&"recall_with_impact"));
        assert!(names.contains(&"get_decision_chain"));
        assert!(names.contains(&"detect_patterns"));
        assert!(names.contains(&"pattern_insights"));
    }

    #[test]
    fn handle_unknown_method() {
        let server = test_server();
        let resp = server.handle_request("some/unknown", None, json!(5));
        assert!(resp.error.is_some());
        assert_eq!(resp.error.unwrap().code, -32601);
    }

    #[test]
    fn handle_ping() {
        let server = test_server();
        let resp = server.handle_request("ping", None, json!(6));
        assert!(resp.result.is_some());
    }
}
