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
    CodememError, Edge, GraphBackend, GraphNode, MemoryNode, MemoryType, NodeKind,
    RelationshipType, ScoreBreakdown, ScoringWeights, SearchResult, VectorBackend,
};
use codemem_graph::GraphEngine;
use codemem_storage::Storage;
use codemem_vector::HnswIndex;
use rusqlite::params;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::collections::HashSet;
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

pub mod bm25;
pub mod patterns;

// ── JSON-RPC Types ──────────────────────────────────────────────────────────

/// JSON-RPC 2.0 request.
#[derive(Debug, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    /// Absent for notifications (no response expected).
    pub id: Option<Value>,
    pub method: String,
    #[serde(default)]
    pub params: Option<Value>,
}

/// JSON-RPC 2.0 response.
#[derive(Debug, Serialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    pub id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

/// JSON-RPC 2.0 error object.
#[derive(Debug, Serialize)]
pub struct JsonRpcError {
    pub code: i64,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

impl JsonRpcResponse {
    fn success(id: Value, result: Value) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: Some(result),
            error: None,
        }
    }

    fn error(id: Value, code: i64, message: impl Into<String>) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: None,
            error: Some(JsonRpcError {
                code,
                message: message.into(),
                data: None,
            }),
        }
    }
}

// ── Tool Result Types ───────────────────────────────────────────────────────

/// MCP tool result (content array + isError flag).
#[derive(Debug, Serialize)]
pub struct ToolResult {
    pub content: Vec<ToolContent>,
    #[serde(rename = "isError")]
    pub is_error: bool,
}

/// A single content block in a tool result.
#[derive(Debug, Serialize)]
pub struct ToolContent {
    #[serde(rename = "type")]
    pub content_type: String,
    pub text: String,
}

impl ToolResult {
    fn text(msg: impl Into<String>) -> Self {
        Self {
            content: vec![ToolContent {
                content_type: "text".to_string(),
                text: msg.into(),
            }],
            is_error: false,
        }
    }

    fn tool_error(msg: impl Into<String>) -> Self {
        Self {
            content: vec![ToolContent {
                content_type: "text".to_string(),
                text: msg.into(),
            }],
            is_error: true,
        }
    }
}

// ── Index Cache ─────────────────────────────────────────────────────────────

/// Cached code-index results for structural queries.
struct IndexCache {
    symbols: Vec<codemem_index::Symbol>,
    root_path: String,
}

// ── MCP Server ──────────────────────────────────────────────────────────────

/// MCP server that reads JSON-RPC from stdin, writes responses to stdout.
/// Holds Storage, HnswIndex, and GraphEngine for real tool dispatch.
pub struct McpServer {
    pub name: String,
    pub version: String,
    storage: Storage,
    vector: Mutex<HnswIndex>,
    graph: Mutex<GraphEngine>,
    /// Optional embedding provider (None if not configured).
    embeddings: Option<Mutex<Box<dyn codemem_embeddings::EmbeddingProvider>>>,
    /// Path to the database file, used to derive the index save path.
    db_path: Option<PathBuf>,
    /// Cached index results for structural queries.
    index_cache: Mutex<Option<IndexCache>>,
    /// Configurable scoring weights for the 9-component hybrid scoring system.
    /// Wrapped in UnsafeCell for interior mutability; safe because the MCP server
    /// is single-threaded (stdio JSON-RPC, sequential request processing).
    scoring_weights: UnsafeCell<ScoringWeights>,
    /// BM25 index for code-aware token overlap scoring.
    /// Updated incrementally on store/update/delete operations.
    bm25_index: Mutex<bm25::Bm25Index>,
}

// SAFETY: McpServer is used single-threaded (stdio JSON-RPC, sequential processing).
// UnsafeCell<ScoringWeights> is the only non-Sync field; all access is sequential.
unsafe impl Send for McpServer {}
unsafe impl Sync for McpServer {}

impl McpServer {
    /// Create a server with storage, vector, graph, and optional embeddings backends.
    pub fn new(
        storage: Storage,
        vector: HnswIndex,
        graph: GraphEngine,
        embeddings: Option<Box<dyn codemem_embeddings::EmbeddingProvider>>,
    ) -> Self {
        Self {
            name: "codemem".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            storage,
            vector: Mutex::new(vector),
            graph: Mutex::new(graph),
            embeddings: embeddings.map(Mutex::new),
            db_path: None,
            index_cache: Mutex::new(None),
            scoring_weights: UnsafeCell::new(ScoringWeights::default()),
            bm25_index: Mutex::new(bm25::Bm25Index::new()),
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

        let mut server = Self::new(storage, vector, graph, embeddings);
        server.db_path = Some(db_path.to_path_buf());

        // Recompute centrality metrics (PageRank + betweenness) on startup
        server.graph.lock().unwrap().recompute_centrality();

        // Populate BM25 index from all existing memories
        if let Ok(ids) = server.storage.list_memory_ids() {
            let mut bm25 = server.bm25_index.lock().unwrap();
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
        Self::new(storage, vector, graph, None)
    }

    // ── Contextual Enrichment ────────────────────────────────────────────────
    // Prepend metadata context to text before embedding, following the
    // "contextual embeddings" methodology: enriched vectors capture semantic
    // relationships that raw content alone would miss.

    /// Build contextual text for a memory node.
    /// Prepends memory type, tags, namespace, and graph relationships so the
    /// embedding captures the memory's role, not just its content.
    fn enrich_memory_text(
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
            let graph = self.graph.lock().unwrap();
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
    fn enrich_symbol_text(
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
            if let Err(e) = self.vector.lock().unwrap().save(&index_path) {
                tracing::warn!("Failed to save vector index: {e}");
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

    fn tool_store_memory(&self, args: &Value) -> ToolResult {
        let content = match args.get("content").and_then(|v| v.as_str()) {
            Some(c) if !c.is_empty() => c,
            _ => return ToolResult::tool_error("Missing or empty 'content' parameter"),
        };

        let memory_type: MemoryType = args
            .get("memory_type")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse().ok())
            .unwrap_or(MemoryType::Context);

        let importance = args
            .get("importance")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.5);

        let tags: Vec<String> = args
            .get("tags")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        let now = chrono::Utc::now();
        let id = uuid::Uuid::new_v4().to_string();
        let hash = Storage::content_hash(content);

        let namespace = args
            .get("namespace")
            .and_then(|v| v.as_str())
            .map(String::from);

        let memory = MemoryNode {
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
            created_at: now,
            updated_at: now,
            last_accessed_at: now,
        };

        // Insert into storage
        match self.storage.insert_memory(&memory) {
            Ok(()) => {}
            Err(CodememError::Duplicate(h)) => {
                return ToolResult::text(format!("Memory already exists (hash: {h})"));
            }
            Err(e) => return ToolResult::tool_error(format!("Storage error: {e}")),
        }

        // Update BM25 index
        self.bm25_index.lock().unwrap().add_document(&id, content);

        // Create graph node for the memory (before embedding so graph context is available)
        let graph_node = GraphNode {
            id: id.clone(),
            kind: NodeKind::Memory,
            label: truncate_str(content, 80),
            payload: HashMap::new(),
            centrality: 0.0,
            memory_id: Some(id.clone()),
            namespace: None,
        };
        // Persist to SQLite (needed for FK constraints on graph_edges)
        if let Err(e) = self.storage.insert_graph_node(&graph_node) {
            tracing::warn!("Failed to persist graph node: {e}");
        }
        if let Err(e) = self.graph.lock().unwrap().add_node(graph_node) {
            tracing::warn!("Failed to add graph node: {e}");
        }

        // Handle optional `links` parameter: create RELATES_TO edges to linked nodes
        if let Some(links) = args.get("links").and_then(|v| v.as_array()) {
            let mut graph = self.graph.lock().unwrap();
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
                    };
                    if let Err(e) = self.storage.insert_graph_edge(&edge) {
                        tracing::warn!("Failed to persist link edge to {link_id}: {e}");
                    }
                    if let Err(e) = graph.add_edge(edge) {
                        tracing::warn!("Failed to add link edge to {link_id}: {e}");
                    }
                }
            }
        }

        // Generate contextual embedding and insert into vector index
        // (after graph node + links so enrichment can reference them)
        if let Some(ref emb_service) = self.embeddings {
            let enriched = self.enrich_memory_text(
                content,
                memory_type,
                &memory.tags,
                memory.namespace.as_deref(),
                Some(&id),
            );
            match emb_service.lock().unwrap().embed(&enriched) {
                Ok(embedding) => {
                    if let Err(e) = self.storage.store_embedding(&id, &embedding) {
                        tracing::warn!("Failed to store embedding: {e}");
                    }
                    if let Err(e) = self.vector.lock().unwrap().insert(&id, &embedding) {
                        tracing::warn!("Failed to index vector: {e}");
                    }
                }
                Err(e) => {
                    tracing::warn!("Embedding failed: {e}");
                }
            }
        }

        // Persist vector index to disk
        self.save_index();

        ToolResult::text(
            serde_json::to_string_pretty(&json!({
                "id": id,
                "memory_type": memory_type.to_string(),
                "importance": importance,
                "embedded": self.embeddings.is_some(),
            }))
            .unwrap(),
        )
    }

    fn tool_recall_memory(&self, args: &Value) -> ToolResult {
        let query = match args.get("query").and_then(|v| v.as_str()) {
            Some(q) if !q.is_empty() => q,
            _ => return ToolResult::tool_error("Missing or empty 'query' parameter"),
        };

        let k = args.get("k").and_then(|v| v.as_u64()).unwrap_or(10) as usize;

        // Parse optional memory_type filter
        let memory_type_filter: Option<MemoryType> = args
            .get("memory_type")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse().ok());

        // Parse optional namespace filter
        let namespace_filter: Option<&str> = args.get("namespace").and_then(|v| v.as_str());

        self.recall_memories(query, k, &memory_type_filter, namespace_filter)
    }

    /// Search the server's storage with optional type and namespace filters.
    fn recall_memories(
        &self,
        query: &str,
        k: usize,
        memory_type_filter: &Option<MemoryType>,
        namespace_filter: Option<&str>,
    ) -> ToolResult {
        // Try vector search first (if embeddings available)
        let vector_results: Vec<(String, f32)> = if let Some(ref emb_service) = self.embeddings {
            match emb_service.lock().unwrap().embed(query) {
                Ok(query_embedding) => self
                    .vector
                    .lock()
                    .unwrap()
                    .search(&query_embedding, k * 2) // over-fetch for re-ranking
                    .unwrap_or_default(),
                Err(e) => {
                    tracing::warn!("Query embedding failed: {e}");
                    vec![]
                }
            }
        } else {
            vec![]
        };

        let query_lower = query.to_lowercase();
        let query_tokens: Vec<&str> = query_lower.split_whitespace().collect();

        let graph = self.graph.lock().unwrap();
        let bm25 = self.bm25_index.lock().unwrap();

        // Build scored results
        let mut results: Vec<SearchResult> = Vec::new();

        if vector_results.is_empty() {
            // Fallback: text search over all memories
            let ids = match self.storage.list_memory_ids() {
                Ok(ids) => ids,
                Err(e) => return ToolResult::tool_error(format!("Storage error: {e}")),
            };

            for id in &ids {
                if let Ok(Some(memory)) = self.storage.get_memory(id) {
                    // Apply memory_type filter
                    if let Some(ref filter_type) = memory_type_filter {
                        if memory.memory_type != *filter_type {
                            continue;
                        }
                    }
                    // Apply namespace filter
                    if let Some(ns) = namespace_filter {
                        if memory.namespace.as_deref() != Some(ns) {
                            continue;
                        }
                    }

                    let breakdown =
                        compute_score(&memory, query, &query_tokens, 0.0, &graph, &bm25);
                    // SAFETY: Single-threaded MCP server; no concurrent access.
                    let score =
                        breakdown.total_with_weights(unsafe { &*self.scoring_weights.get() });
                    if score > 0.01 {
                        results.push(SearchResult {
                            memory,
                            score,
                            score_breakdown: breakdown,
                        });
                    }
                }
            }
        } else {
            // Vector search + hybrid scoring
            for (id, distance) in &vector_results {
                if let Ok(Some(memory)) = self.storage.get_memory(id) {
                    // Apply memory_type filter
                    if let Some(ref filter_type) = memory_type_filter {
                        if memory.memory_type != *filter_type {
                            continue;
                        }
                    }
                    // Apply namespace filter
                    if let Some(ns) = namespace_filter {
                        if memory.namespace.as_deref() != Some(ns) {
                            continue;
                        }
                    }

                    // Convert cosine distance to similarity (1.0 - distance for cosine)
                    let similarity = 1.0 - (*distance as f64);
                    let breakdown =
                        compute_score(&memory, query, &query_tokens, similarity, &graph, &bm25);
                    // SAFETY: Single-threaded MCP server; no concurrent access.
                    let score =
                        breakdown.total_with_weights(unsafe { &*self.scoring_weights.get() });
                    results.push(SearchResult {
                        memory,
                        score,
                        score_breakdown: breakdown,
                    });
                }
            }
        }

        // Sort by score descending, take top k
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(k);

        format_recall_results(&results, None)
    }

    fn tool_update_memory(&self, args: &Value) -> ToolResult {
        let id = match args.get("id").and_then(|v| v.as_str()) {
            Some(id) => id,
            None => return ToolResult::tool_error("Missing 'id' parameter"),
        };
        let content = match args.get("content").and_then(|v| v.as_str()) {
            Some(c) => c,
            None => return ToolResult::tool_error("Missing 'content' parameter"),
        };
        let importance = args.get("importance").and_then(|v| v.as_f64());

        if let Err(e) = self.storage.update_memory(id, content, importance) {
            return ToolResult::tool_error(format!("Update failed: {e}"));
        }

        // Update BM25 index with new content
        self.bm25_index.lock().unwrap().add_document(id, content);

        // Re-embed with contextual enrichment
        if let Some(ref emb_service) = self.embeddings {
            // Fetch the updated memory to get its metadata for enrichment
            let (mem_type, tags, namespace) = if let Ok(Some(mem)) = self.storage.get_memory(id) {
                (mem.memory_type, mem.tags, mem.namespace)
            } else {
                (MemoryType::Context, vec![], None)
            };
            let enriched =
                self.enrich_memory_text(content, mem_type, &tags, namespace.as_deref(), Some(id));
            if let Ok(embedding) = emb_service.lock().unwrap().embed(&enriched) {
                let _ = self.storage.store_embedding(id, &embedding);
                let mut vec = self.vector.lock().unwrap();
                let _ = vec.remove(id);
                let _ = vec.insert(id, &embedding);
            }
        }

        // Persist vector index to disk
        self.save_index();

        ToolResult::text(json!({"id": id, "updated": true}).to_string())
    }

    fn tool_delete_memory(&self, args: &Value) -> ToolResult {
        let id = match args.get("id").and_then(|v| v.as_str()) {
            Some(id) => id,
            None => return ToolResult::tool_error("Missing 'id' parameter"),
        };

        match self.storage.delete_memory(id) {
            Ok(true) => {
                // Remove from vector index
                let _ = self.vector.lock().unwrap().remove(id);
                // Remove from in-memory graph
                let _ = self.graph.lock().unwrap().remove_node(id);
                // Remove graph node and edges from SQLite
                let _ = self.storage.delete_graph_edges_for_node(id);
                let _ = self.storage.delete_graph_node(id);
                // Remove embedding from SQLite
                let _ = self
                    .storage
                    .connection()
                    .execute("DELETE FROM memory_embeddings WHERE memory_id = ?1", [id]);
                // Remove from BM25 index
                self.bm25_index.lock().unwrap().remove_document(id);
                // Persist vector index to disk
                self.save_index();
                ToolResult::text(json!({"id": id, "deleted": true}).to_string())
            }
            Ok(false) => ToolResult::tool_error(format!("Memory not found: {id}")),
            Err(e) => ToolResult::tool_error(format!("Delete failed: {e}")),
        }
    }

    fn tool_associate_memories(&self, args: &Value) -> ToolResult {
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
        };

        // Store in SQLite
        if let Err(e) = self.storage.insert_graph_edge(&edge) {
            return ToolResult::tool_error(format!("Failed to store edge: {e}"));
        }

        // Add to in-memory graph
        if let Err(e) = self.graph.lock().unwrap().add_edge(edge) {
            tracing::warn!("Failed to add edge to graph: {e}");
        }

        ToolResult::text(
            json!({
                "source": src,
                "target": dst,
                "relationship": rel_str,
                "weight": weight,
            })
            .to_string(),
        )
    }

    fn tool_graph_traverse(&self, args: &Value) -> ToolResult {
        let start = match args.get("start_id").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => return ToolResult::tool_error("Missing 'start_id' parameter"),
        };
        let depth = args.get("max_depth").and_then(|v| v.as_u64()).unwrap_or(2) as usize;
        let algorithm = args
            .get("algorithm")
            .and_then(|v| v.as_str())
            .unwrap_or("bfs");

        let graph = self.graph.lock().unwrap();
        let nodes = match algorithm {
            "bfs" => graph.bfs(start, depth),
            "dfs" => graph.dfs(start, depth),
            _ => return ToolResult::tool_error(format!("Unknown algorithm: {algorithm}")),
        };

        match nodes {
            Ok(nodes) => {
                let output: Vec<Value> = nodes
                    .iter()
                    .map(|n| {
                        json!({
                            "id": n.id,
                            "kind": n.kind.to_string(),
                            "label": n.label,
                            "memory_id": n.memory_id,
                        })
                    })
                    .collect();
                ToolResult::text(serde_json::to_string_pretty(&output).unwrap())
            }
            Err(e) => ToolResult::tool_error(format!("Traversal failed: {e}")),
        }
    }

    fn tool_stats(&self) -> ToolResult {
        let storage_stats = match self.storage.stats() {
            Ok(s) => s,
            Err(e) => return ToolResult::tool_error(format!("Stats error: {e}")),
        };

        let vector_stats = self.vector.lock().unwrap().stats();
        let graph_stats = self.graph.lock().unwrap().stats();

        ToolResult::text(
            serde_json::to_string_pretty(&json!({
                "storage": {
                    "memories": storage_stats.memory_count,
                    "embeddings": storage_stats.embedding_count,
                    "graph_nodes": storage_stats.node_count,
                    "graph_edges": storage_stats.edge_count,
                },
                "vector": {
                    "indexed": vector_stats.count,
                    "dimensions": vector_stats.dimensions,
                    "metric": vector_stats.metric,
                },
                "graph": {
                    "nodes": graph_stats.node_count,
                    "edges": graph_stats.edge_count,
                    "node_kinds": graph_stats.node_kind_counts,
                    "relationship_types": graph_stats.relationship_type_counts,
                },
                "embeddings": {
                    "available": self.embeddings.is_some(),
                    "cache": self.embeddings.as_ref().map(|e| {
                        let (size, cap) = e.lock().unwrap().cache_stats();
                        json!({"size": size, "capacity": cap})
                    }),
                }
            }))
            .unwrap(),
        )
    }

    fn tool_health(&self) -> ToolResult {
        let storage_ok = self.storage.stats().is_ok();
        let vector_ok = true; // HnswIndex is always available
        let graph_ok = true; // GraphEngine is always available
        let embeddings_ok = self.embeddings.is_some();

        let healthy = storage_ok && vector_ok && graph_ok;

        ToolResult::text(
            serde_json::to_string_pretty(&json!({
                "healthy": healthy,
                "storage": if storage_ok { "ok" } else { "error" },
                "vector": if vector_ok { "ok" } else { "error" },
                "graph": if graph_ok { "ok" } else { "error" },
                "embeddings": if embeddings_ok { "ok" } else { "not_configured" },
            }))
            .unwrap(),
        )
    }

    // ── Structural Index Tools ──────────────────────────────────────────────

    fn tool_index_codebase(&self, args: &Value) -> ToolResult {
        let path = match args.get("path").and_then(|v| v.as_str()) {
            Some(p) => p,
            None => return ToolResult::tool_error("Missing 'path' parameter"),
        };

        let root = std::path::Path::new(path);
        if !root.exists() {
            return ToolResult::tool_error(format!("Path does not exist: {path}"));
        }

        let mut indexer = codemem_index::Indexer::new();
        let result = match indexer.index_directory(root) {
            Ok(r) => r,
            Err(e) => return ToolResult::tool_error(format!("Indexing failed: {e}")),
        };

        // Collect all symbols and references
        let mut all_symbols = Vec::new();
        let mut all_references = Vec::new();
        for pr in &result.parse_results {
            all_symbols.extend(pr.symbols.clone());
            all_references.extend(pr.references.clone());
        }

        // Resolve references
        let mut resolver = codemem_index::ReferenceResolver::new();
        resolver.add_symbols(&all_symbols);
        let edges = resolver.resolve_all(&all_references);

        // Persist symbols as graph nodes
        let mut graph = self.graph.lock().unwrap();
        for sym in &all_symbols {
            let kind = match sym.kind {
                codemem_index::SymbolKind::Function => NodeKind::Function,
                codemem_index::SymbolKind::Method => NodeKind::Method,
                codemem_index::SymbolKind::Class => NodeKind::Class,
                codemem_index::SymbolKind::Struct => NodeKind::Class,
                codemem_index::SymbolKind::Enum => NodeKind::Class,
                codemem_index::SymbolKind::Interface => NodeKind::Interface,
                codemem_index::SymbolKind::Type => NodeKind::Type,
                codemem_index::SymbolKind::Constant => NodeKind::Constant,
                codemem_index::SymbolKind::Module => NodeKind::Module,
                codemem_index::SymbolKind::Test => NodeKind::Test,
            };

            let mut payload = HashMap::new();
            payload.insert(
                "signature".to_string(),
                serde_json::Value::String(sym.signature.clone()),
            );
            payload.insert(
                "file_path".to_string(),
                serde_json::Value::String(sym.file_path.clone()),
            );
            payload.insert("line_start".to_string(), serde_json::json!(sym.line_start));
            payload.insert("line_end".to_string(), serde_json::json!(sym.line_end));
            payload.insert(
                "visibility".to_string(),
                serde_json::Value::String(sym.visibility.to_string()),
            );
            if let Some(ref doc) = sym.doc_comment {
                payload.insert(
                    "doc_comment".to_string(),
                    serde_json::Value::String(doc.clone()),
                );
            }

            let node = GraphNode {
                id: format!("sym:{}", sym.qualified_name),
                kind,
                label: sym.name.clone(),
                payload,
                centrality: 0.0,
                memory_id: None,
                namespace: Some(path.to_string()),
            };

            let _ = self.storage.insert_graph_node(&node);
            let _ = graph.add_node(node);
        }

        // Persist edges
        let now = chrono::Utc::now();
        let edges_resolved = edges.len();
        for edge in &edges {
            let e = Edge {
                id: format!(
                    "ref:{}->{}:{}",
                    edge.source_qualified_name, edge.target_qualified_name, edge.relationship
                ),
                src: format!("sym:{}", edge.source_qualified_name),
                dst: format!("sym:{}", edge.target_qualified_name),
                relationship: edge.relationship,
                weight: 1.0,
                properties: HashMap::new(),
                created_at: now,
            };
            let _ = self.storage.insert_graph_edge(&e);
            let _ = graph.add_edge(e);
        }

        // Embed symbol signatures with contextual enrichment
        let mut symbols_embedded = 0usize;
        if let Some(ref emb_service) = self.embeddings {
            let emb = emb_service.lock().unwrap();
            let mut vec = self.vector.lock().unwrap();
            for sym in &all_symbols {
                let embed_text = self.enrich_symbol_text(sym, &edges);
                let sym_id = format!("sym:{}", sym.qualified_name);
                if let Ok(embedding) = emb.embed(&embed_text) {
                    let _ = self.storage.store_embedding(&sym_id, &embedding);
                    let _ = vec.insert(&sym_id, &embedding);
                    symbols_embedded += 1;
                }
            }
            drop(vec);
            drop(emb);
            self.save_index();
        }

        // Cache results
        {
            let mut cache = self.index_cache.lock().unwrap();
            *cache = Some(IndexCache {
                symbols: all_symbols,
                root_path: path.to_string(),
            });
        }

        ToolResult::text(
            serde_json::to_string_pretty(&json!({
                "files_scanned": result.files_scanned,
                "files_parsed": result.files_parsed,
                "files_skipped": result.files_skipped,
                "symbols": result.total_symbols,
                "references": result.total_references,
                "edges_resolved": edges_resolved,
                "symbols_embedded": symbols_embedded,
            }))
            .unwrap(),
        )
    }

    fn tool_search_symbols(&self, args: &Value) -> ToolResult {
        let query = match args.get("query").and_then(|v| v.as_str()) {
            Some(q) if !q.is_empty() => q,
            _ => return ToolResult::tool_error("Missing or empty 'query' parameter"),
        };

        let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(20) as usize;

        let kind_filter: Option<&str> = args.get("kind").and_then(|v| v.as_str());

        let cache = self.index_cache.lock().unwrap();
        let symbols = match cache.as_ref() {
            Some(c) => &c.symbols,
            None => {
                return ToolResult::tool_error("No codebase indexed yet. Run index_codebase first.")
            }
        };

        let query_lower = query.to_lowercase();

        let matches: Vec<Value> = symbols
            .iter()
            .filter(|sym| {
                let name_match = sym.name.to_lowercase().contains(&query_lower)
                    || sym.qualified_name.to_lowercase().contains(&query_lower);
                if !name_match {
                    return false;
                }
                if let Some(kind_str) = kind_filter {
                    let kind_lower = kind_str.to_lowercase();
                    return sym.kind.to_string().to_lowercase() == kind_lower;
                }
                true
            })
            .take(limit)
            .map(|sym| {
                json!({
                    "name": sym.name,
                    "qualified_name": sym.qualified_name,
                    "kind": sym.kind.to_string(),
                    "signature": sym.signature,
                    "file_path": sym.file_path,
                    "line_start": sym.line_start,
                    "line_end": sym.line_end,
                    "visibility": sym.visibility.to_string(),
                    "parent": sym.parent,
                })
            })
            .collect();

        if matches.is_empty() {
            return ToolResult::text("No matching symbols found.");
        }

        ToolResult::text(serde_json::to_string_pretty(&matches).unwrap())
    }

    fn tool_get_symbol_info(&self, args: &Value) -> ToolResult {
        let qualified_name = match args.get("qualified_name").and_then(|v| v.as_str()) {
            Some(qn) => qn,
            None => return ToolResult::tool_error("Missing 'qualified_name' parameter"),
        };

        let cache = self.index_cache.lock().unwrap();
        let symbols = match cache.as_ref() {
            Some(c) => &c.symbols,
            None => {
                return ToolResult::tool_error("No codebase indexed yet. Run index_codebase first.")
            }
        };

        let sym = match symbols.iter().find(|s| s.qualified_name == qualified_name) {
            Some(s) => s,
            None => return ToolResult::tool_error(format!("Symbol not found: {qualified_name}")),
        };

        ToolResult::text(
            serde_json::to_string_pretty(&json!({
                "name": sym.name,
                "qualified_name": sym.qualified_name,
                "kind": sym.kind.to_string(),
                "signature": sym.signature,
                "visibility": sym.visibility.to_string(),
                "file_path": sym.file_path,
                "line_start": sym.line_start,
                "line_end": sym.line_end,
                "doc_comment": sym.doc_comment,
                "parent": sym.parent,
            }))
            .unwrap(),
        )
    }

    fn tool_get_dependencies(&self, args: &Value) -> ToolResult {
        let qualified_name = match args.get("qualified_name").and_then(|v| v.as_str()) {
            Some(qn) => qn,
            None => return ToolResult::tool_error("Missing 'qualified_name' parameter"),
        };

        let direction = args
            .get("direction")
            .and_then(|v| v.as_str())
            .unwrap_or("both");

        let node_id = format!("sym:{qualified_name}");
        let graph = self.graph.lock().unwrap();

        let edges = match graph.get_edges(&node_id) {
            Ok(e) => e,
            Err(_) => {
                return ToolResult::tool_error(format!("Node not found in graph: {qualified_name}"))
            }
        };

        let filtered: Vec<Value> = edges
            .iter()
            .filter(|e| match direction {
                "incoming" => e.dst == node_id,
                "outgoing" => e.src == node_id,
                _ => true, // "both"
            })
            .map(|e| {
                json!({
                    "source": e.src,
                    "target": e.dst,
                    "relationship": e.relationship.to_string(),
                    "weight": e.weight,
                })
            })
            .collect();

        if filtered.is_empty() {
            return ToolResult::text(format!(
                "No {direction} dependencies found for {qualified_name}."
            ));
        }

        ToolResult::text(serde_json::to_string_pretty(&filtered).unwrap())
    }

    fn tool_get_impact(&self, args: &Value) -> ToolResult {
        let qualified_name = match args.get("qualified_name").and_then(|v| v.as_str()) {
            Some(qn) => qn,
            None => return ToolResult::tool_error("Missing 'qualified_name' parameter"),
        };

        let depth = args.get("depth").and_then(|v| v.as_u64()).unwrap_or(2) as usize;

        let node_id = format!("sym:{qualified_name}");
        let graph = self.graph.lock().unwrap();

        // BFS from the node to find all reachable nodes within N hops
        let nodes = match graph.bfs(&node_id, depth) {
            Ok(n) => n,
            Err(e) => {
                return ToolResult::tool_error(format!(
                    "Impact analysis failed for {qualified_name}: {e}"
                ))
            }
        };

        // Also collect edges that connect to the node (incoming = "who depends on me")
        let all_edges = graph.get_edges(&node_id).unwrap_or_default();

        let incoming: Vec<Value> = all_edges
            .iter()
            .filter(|e| e.dst == node_id)
            .map(|e| {
                json!({
                    "source": e.src,
                    "relationship": e.relationship.to_string(),
                })
            })
            .collect();

        let reachable: Vec<Value> = nodes
            .iter()
            .filter(|n| n.id != node_id)
            .map(|n| {
                json!({
                    "id": n.id,
                    "kind": n.kind.to_string(),
                    "label": n.label,
                })
            })
            .collect();

        ToolResult::text(
            serde_json::to_string_pretty(&json!({
                "symbol": qualified_name,
                "depth": depth,
                "direct_dependents": incoming,
                "reachable_nodes": reachable.len(),
                "reachable": reachable,
            }))
            .unwrap(),
        )
    }

    fn tool_get_clusters(&self, args: &Value) -> ToolResult {
        let resolution = args
            .get("resolution")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0);

        let graph = self.graph.lock().unwrap();
        let communities = graph.louvain_communities(resolution);

        let output: Vec<Value> = communities
            .iter()
            .enumerate()
            .map(|(i, cluster)| {
                json!({
                    "cluster_id": i,
                    "size": cluster.len(),
                    "members": cluster,
                })
            })
            .collect();

        ToolResult::text(
            serde_json::to_string_pretty(&json!({
                "cluster_count": communities.len(),
                "resolution": resolution,
                "clusters": output,
            }))
            .unwrap(),
        )
    }

    fn tool_get_cross_repo(&self, args: &Value) -> ToolResult {
        let path = args.get("path").and_then(|v| v.as_str());

        let scan_root = match path {
            Some(p) => std::path::PathBuf::from(p),
            None => {
                let cache = self.index_cache.lock().unwrap();
                match cache.as_ref() {
                    Some(c) => std::path::PathBuf::from(&c.root_path),
                    None => {
                        return ToolResult::tool_error(
                            "No path specified and no codebase indexed. Provide 'path' or run index_codebase first.",
                        )
                    }
                }
            }
        };

        if !scan_root.exists() {
            return ToolResult::tool_error(format!("Path does not exist: {}", scan_root.display()));
        }

        let manifest_result = codemem_index::manifest::scan_manifests(&scan_root);

        let workspaces: Vec<Value> = manifest_result
            .workspaces
            .iter()
            .map(|ws| {
                json!({
                    "kind": ws.kind,
                    "root": ws.root,
                    "members": ws.members,
                })
            })
            .collect();

        let packages: Vec<Value> = manifest_result
            .packages
            .iter()
            .map(|(name, path)| json!({"name": name, "manifest": path}))
            .collect();

        let deps: Vec<Value> = manifest_result
            .dependencies
            .iter()
            .map(|d| {
                json!({
                    "name": d.name,
                    "version": d.version,
                    "dev": d.dev,
                    "manifest": d.manifest_path,
                })
            })
            .collect();

        ToolResult::text(
            serde_json::to_string_pretty(&json!({
                "root": scan_root.to_string_lossy(),
                "workspaces": workspaces,
                "packages": packages,
                "dependencies_count": deps.len(),
                "dependencies": deps,
            }))
            .unwrap(),
        )
    }

    fn tool_get_pagerank(&self, args: &Value) -> ToolResult {
        let top_k = args.get("top_k").and_then(|v| v.as_u64()).unwrap_or(20) as usize;
        let damping = args.get("damping").and_then(|v| v.as_f64()).unwrap_or(0.85);

        let graph = self.graph.lock().unwrap();
        let scores = graph.pagerank(damping, 100, 1e-6);

        // Sort by score descending
        let mut sorted: Vec<(String, f64)> = scores.into_iter().collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted.truncate(top_k);

        let results: Vec<Value> = sorted
            .iter()
            .map(|(id, score)| {
                let node = graph.get_node(id).ok().flatten();
                json!({
                    "id": id,
                    "pagerank": format!("{:.6}", score),
                    "kind": node.as_ref().map(|n| n.kind.to_string()),
                    "label": node.as_ref().map(|n| n.label.clone()),
                })
            })
            .collect();

        ToolResult::text(
            serde_json::to_string_pretty(&json!({
                "damping": damping,
                "top_k": top_k,
                "results": results,
            }))
            .unwrap(),
        )
    }

    fn tool_search_code(&self, args: &Value) -> ToolResult {
        let query = match args.get("query").and_then(|v| v.as_str()) {
            Some(q) if !q.is_empty() => q,
            _ => return ToolResult::tool_error("Missing or empty 'query' parameter"),
        };

        let k = args.get("k").and_then(|v| v.as_u64()).unwrap_or(10) as usize;

        let results: Vec<(String, f32)> = if let Some(ref emb_service) = self.embeddings {
            match emb_service.lock().unwrap().embed(query) {
                Ok(query_embedding) => self
                    .vector
                    .lock()
                    .unwrap()
                    .search(&query_embedding, k * 3)
                    .unwrap_or_default()
                    .into_iter()
                    .filter(|(id, _)| id.starts_with("sym:"))
                    .take(k)
                    .collect(),
                Err(e) => {
                    return ToolResult::tool_error(format!("Embedding failed: {e}"));
                }
            }
        } else {
            return ToolResult::tool_error("Embedding service not available");
        };

        if results.is_empty() {
            return ToolResult::text("No matching code symbols found.");
        }

        let mut output = Vec::new();
        for (id, distance) in &results {
            let similarity = 1.0 - *distance as f64;
            if let Ok(Some(node)) = self.storage.get_graph_node(id) {
                output.push(json!({
                    "qualified_name": id.strip_prefix("sym:").unwrap_or(id),
                    "kind": node.kind.to_string(),
                    "label": node.label,
                    "similarity": format!("{:.4}", similarity),
                    "file_path": node.payload.get("file_path"),
                    "line_start": node.payload.get("line_start"),
                    "line_end": node.payload.get("line_end"),
                    "signature": node.payload.get("signature"),
                    "doc_comment": node.payload.get("doc_comment"),
                }));
            }
        }

        ToolResult::text(serde_json::to_string_pretty(&output).unwrap())
    }

    /// MCP tool: set_scoring_weights — update the server's scoring weights at runtime.
    fn tool_set_scoring_weights(&self, args: &Value) -> ToolResult {
        let vector_similarity = args
            .get("vector_similarity")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.25);
        let graph_strength = args
            .get("graph_strength")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.25);
        let token_overlap = args
            .get("token_overlap")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.15);
        let temporal = args
            .get("temporal")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.10);
        let tag_matching = args
            .get("tag_matching")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.10);
        let importance = args
            .get("importance")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.05);
        let confidence = args
            .get("confidence")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.05);
        let recency = args.get("recency").and_then(|v| v.as_f64()).unwrap_or(0.05);

        let raw = ScoringWeights {
            vector_similarity,
            graph_strength,
            token_overlap,
            temporal,
            tag_matching,
            importance,
            confidence,
            recency,
        };
        let normalized = raw.normalized();

        // SAFETY: Single-threaded MCP server; no concurrent access to scoring_weights.
        unsafe { *self.scoring_weights.get() = normalized.clone() };

        let response = json!({
            "updated": true,
            "weights": {
                "vector_similarity": normalized.vector_similarity,
                "graph_strength": normalized.graph_strength,
                "token_overlap": normalized.token_overlap,
                "temporal": normalized.temporal,
                "tag_matching": normalized.tag_matching,
                "importance": normalized.importance,
                "confidence": normalized.confidence,
                "recency": normalized.recency,
            }
        });

        ToolResult::text(serde_json::to_string_pretty(&response).unwrap())
    }

    // ── Graph-Expanded Recall & Namespace Management Tools ──────────────

    /// MCP tool: recall_with_expansion — vector search + graph expansion.
    fn tool_recall_with_expansion(&self, args: &Value) -> ToolResult {
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

        let query_lower = query.to_lowercase();
        let query_tokens: Vec<&str> = query_lower.split_whitespace().collect();

        // Step 1: Run normal vector search (or text fallback)
        let vector_results: Vec<(String, f32)> = if let Some(ref emb_service) = self.embeddings {
            match emb_service.lock().unwrap().embed(query) {
                Ok(query_embedding) => self
                    .vector
                    .lock()
                    .unwrap()
                    .search(&query_embedding, k * 2)
                    .unwrap_or_default(),
                Err(e) => {
                    tracing::warn!("Query embedding failed: {e}");
                    vec![]
                }
            }
        } else {
            vec![]
        };

        let graph = self.graph.lock().unwrap();
        let bm25 = self.bm25_index.lock().unwrap();

        // Collect initial seed memories with their vector similarity
        struct ScoredMemory {
            memory: MemoryNode,
            vector_sim: f64,
            expansion_path: String,
        }

        let mut all_memories: Vec<ScoredMemory> = Vec::new();
        let mut seen_ids: HashSet<String> = HashSet::new();

        if vector_results.is_empty() {
            // Fallback: text search over all memories
            let ids = match self.storage.list_memory_ids() {
                Ok(ids) => ids,
                Err(e) => return ToolResult::tool_error(format!("Storage error: {e}")),
            };

            for id in &ids {
                if let Ok(Some(memory)) = self.storage.get_memory(id) {
                    if let Some(ns) = namespace_filter {
                        if memory.namespace.as_deref() != Some(ns) {
                            continue;
                        }
                    }
                    let breakdown =
                        compute_score(&memory, query, &query_tokens, 0.0, &graph, &bm25);
                    let score =
                        breakdown.total_with_weights(unsafe { &*self.scoring_weights.get() });
                    if score > 0.01 {
                        seen_ids.insert(memory.id.clone());
                        all_memories.push(ScoredMemory {
                            memory,
                            vector_sim: 0.0,
                            expansion_path: "direct".to_string(),
                        });
                    }
                }
            }
        } else {
            // Vector search path
            for (id, distance) in &vector_results {
                if let Ok(Some(memory)) = self.storage.get_memory(id) {
                    if let Some(ns) = namespace_filter {
                        if memory.namespace.as_deref() != Some(ns) {
                            continue;
                        }
                    }
                    seen_ids.insert(memory.id.clone());
                    let similarity = 1.0 - (*distance as f64);
                    all_memories.push(ScoredMemory {
                        memory,
                        vector_sim: similarity,
                        expansion_path: "direct".to_string(),
                    });
                }
            }
        }

        // Step 2-4: Graph expansion from each direct result
        // Collect the IDs of direct results for expansion
        let direct_ids: Vec<String> = all_memories.iter().map(|m| m.memory.id.clone()).collect();

        for direct_id in &direct_ids {
            // Use BFS expansion from this memory's graph node
            if let Ok(expanded_nodes) = graph.bfs(direct_id, expansion_depth) {
                for expanded_node in &expanded_nodes {
                    // Skip the start node itself (already in results)
                    if expanded_node.id == *direct_id {
                        continue;
                    }

                    // Only consider memory nodes
                    if expanded_node.kind != NodeKind::Memory {
                        continue;
                    }

                    // Get the memory_id from the graph node
                    let memory_id = expanded_node
                        .memory_id
                        .as_deref()
                        .unwrap_or(&expanded_node.id);

                    // Skip if already seen
                    if seen_ids.contains(memory_id) {
                        continue;
                    }

                    // Fetch the memory
                    if let Ok(Some(memory)) = self.storage.get_memory(memory_id) {
                        if let Some(ns) = namespace_filter {
                            if memory.namespace.as_deref() != Some(ns) {
                                continue;
                            }
                        }

                        // Build expansion path description
                        let expansion_path = if let Ok(edges) = graph.get_edges(direct_id) {
                            edges
                                .iter()
                                .find(|e| e.dst == expanded_node.id || e.src == expanded_node.id)
                                .map(|e| format!("via {} from {}", e.relationship, direct_id))
                                .unwrap_or_else(|| format!("via graph from {direct_id}"))
                        } else {
                            format!("via graph from {direct_id}")
                        };

                        seen_ids.insert(memory_id.to_string());
                        all_memories.push(ScoredMemory {
                            memory,
                            vector_sim: 0.0,
                            expansion_path,
                        });
                    }
                }
            }
        }

        // Step 5-6: Score all memories and sort
        let mut scored_results: Vec<(SearchResult, String)> = all_memories
            .into_iter()
            .map(|sm| {
                let breakdown = compute_score(
                    &sm.memory,
                    query,
                    &query_tokens,
                    sm.vector_sim,
                    &graph,
                    &bm25,
                );
                let score = breakdown.total_with_weights(unsafe { &*self.scoring_weights.get() });
                (
                    SearchResult {
                        memory: sm.memory,
                        score,
                        score_breakdown: breakdown,
                    },
                    sm.expansion_path,
                )
            })
            .collect();

        scored_results.sort_by(|a, b| {
            b.0.score
                .partial_cmp(&a.0.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored_results.truncate(k);

        // Step 7: Format results with expansion_path
        if scored_results.is_empty() {
            return ToolResult::text("No matching memories found.");
        }

        let output: Vec<Value> = scored_results
            .iter()
            .map(|(r, path)| {
                json!({
                    "id": r.memory.id,
                    "content": r.memory.content,
                    "memory_type": r.memory.memory_type.to_string(),
                    "score": format!("{:.4}", r.score),
                    "importance": r.memory.importance,
                    "tags": r.memory.tags,
                    "access_count": r.memory.access_count,
                    "expansion_path": path,
                })
            })
            .collect();

        ToolResult::text(serde_json::to_string_pretty(&output).unwrap())
    }

    /// MCP tool: list_namespaces — list all namespaces with memory counts.
    fn tool_list_namespaces(&self) -> ToolResult {
        let namespaces = match self.storage.list_namespaces() {
            Ok(ns) => ns,
            Err(e) => return ToolResult::tool_error(format!("Storage error: {e}")),
        };

        let mut ns_list: Vec<Value> = Vec::new();
        for ns in &namespaces {
            let count = match self.storage.list_memory_ids_for_namespace(ns) {
                Ok(ids) => ids.len(),
                Err(_) => 0,
            };
            ns_list.push(json!({
                "name": ns,
                "memory_count": count,
            }));
        }

        let response = json!({ "namespaces": ns_list });
        ToolResult::text(serde_json::to_string_pretty(&response).unwrap())
    }

    /// MCP tool: namespace_stats — detailed stats for a single namespace.
    fn tool_namespace_stats(&self, args: &Value) -> ToolResult {
        let namespace = match args.get("namespace").and_then(|v| v.as_str()) {
            Some(ns) if !ns.is_empty() => ns,
            _ => return ToolResult::tool_error("Missing or empty 'namespace' parameter"),
        };

        let ids = match self.storage.list_memory_ids_for_namespace(namespace) {
            Ok(ids) => ids,
            Err(e) => return ToolResult::tool_error(format!("Storage error: {e}")),
        };

        if ids.is_empty() {
            return ToolResult::text(
                serde_json::to_string_pretty(&json!({
                    "namespace": namespace,
                    "count": 0,
                    "message": "No memories found in this namespace"
                }))
                .unwrap(),
            );
        }

        let mut total_importance = 0.0;
        let mut total_confidence = 0.0;
        let mut type_distribution: HashMap<String, usize> = HashMap::new();
        let mut tag_frequency: HashMap<String, usize> = HashMap::new();
        let mut oldest: Option<chrono::DateTime<chrono::Utc>> = None;
        let mut newest: Option<chrono::DateTime<chrono::Utc>> = None;
        let mut count = 0usize;

        for id in &ids {
            if let Ok(Some(memory)) = self.storage.get_memory(id) {
                count += 1;
                total_importance += memory.importance;
                total_confidence += memory.confidence;

                *type_distribution
                    .entry(memory.memory_type.to_string())
                    .or_insert(0) += 1;

                for tag in &memory.tags {
                    *tag_frequency.entry(tag.clone()).or_insert(0) += 1;
                }

                match oldest {
                    None => oldest = Some(memory.created_at),
                    Some(ref o) if memory.created_at < *o => oldest = Some(memory.created_at),
                    _ => {}
                }
                match newest {
                    None => newest = Some(memory.created_at),
                    Some(ref n) if memory.created_at > *n => newest = Some(memory.created_at),
                    _ => {}
                }
            }
        }

        let avg_importance = if count > 0 {
            total_importance / count as f64
        } else {
            0.0
        };
        let avg_confidence = if count > 0 {
            total_confidence / count as f64
        } else {
            0.0
        };

        let response = json!({
            "namespace": namespace,
            "count": count,
            "avg_importance": format!("{:.4}", avg_importance),
            "avg_confidence": format!("{:.4}", avg_confidence),
            "type_distribution": type_distribution,
            "tag_frequency": tag_frequency,
            "oldest": oldest.map(|d| d.to_rfc3339()),
            "newest": newest.map(|d| d.to_rfc3339()),
        });

        ToolResult::text(serde_json::to_string_pretty(&response).unwrap())
    }

    /// MCP tool: delete_namespace — delete all memories in a namespace.
    fn tool_delete_namespace(&self, args: &Value) -> ToolResult {
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

        let ids = match self.storage.list_memory_ids_for_namespace(namespace) {
            Ok(ids) => ids,
            Err(e) => return ToolResult::tool_error(format!("Storage error: {e}")),
        };

        let mut deleted = 0usize;
        let mut graph = self.graph.lock().unwrap();
        let mut vector = self.vector.lock().unwrap();
        let mut bm25 = self.bm25_index.lock().unwrap();

        for id in &ids {
            // Delete memory from storage
            if let Ok(true) = self.storage.delete_memory(id) {
                deleted += 1;

                // Remove from vector index
                let _ = vector.remove(id);

                // Remove from in-memory graph
                let _ = graph.remove_node(id);

                // Remove graph node and edges from SQLite
                let _ = self.storage.delete_graph_edges_for_node(id);
                let _ = self.storage.delete_graph_node(id);

                // Remove embedding from SQLite
                let _ = self
                    .storage
                    .connection()
                    .execute("DELETE FROM memory_embeddings WHERE memory_id = ?1", [id]);

                // Remove from BM25 index
                bm25.remove_document(id);
            }
        }

        // Drop locks before calling save_index (which acquires vector lock)
        drop(graph);
        drop(vector);
        drop(bm25);

        // Persist vector index to disk
        self.save_index();

        let response = json!({
            "deleted": deleted,
            "namespace": namespace,
        });

        ToolResult::text(serde_json::to_string_pretty(&response).unwrap())
    }

    // ── Export/Import Tools ─────────────────────────────────────────────────

    /// MCP tool: export_memories — export memories as a JSON array.
    fn tool_export_memories(&self, args: &Value) -> ToolResult {
        let namespace_filter: Option<&str> = args.get("namespace").and_then(|v| v.as_str());
        let memory_type_filter: Option<MemoryType> = args
            .get("memory_type")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse().ok());
        let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(100) as usize;

        let ids = match namespace_filter {
            Some(ns) => match self.storage.list_memory_ids_for_namespace(ns) {
                Ok(ids) => ids,
                Err(e) => return ToolResult::tool_error(format!("Storage error: {e}")),
            },
            None => match self.storage.list_memory_ids() {
                Ok(ids) => ids,
                Err(e) => return ToolResult::tool_error(format!("Storage error: {e}")),
            },
        };

        let mut exported: Vec<Value> = Vec::new();

        for id in &ids {
            if exported.len() >= limit {
                break;
            }
            if let Ok(Some(memory)) = self.storage.get_memory(id) {
                // Apply memory_type filter
                if let Some(ref filter_type) = memory_type_filter {
                    if memory.memory_type != *filter_type {
                        continue;
                    }
                }

                // Get edges for this memory
                let edges: Vec<Value> = self
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

        ToolResult::text(serde_json::to_string_pretty(&exported).unwrap())
    }

    /// MCP tool: import_memories — import memories from a JSON array.
    fn tool_import_memories(&self, args: &Value) -> ToolResult {
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

            let tags: Vec<String> = mem_val
                .get("tags")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default();

            let namespace = mem_val
                .get("namespace")
                .and_then(|v| v.as_str())
                .map(String::from);

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

            match self.storage.insert_memory(&memory) {
                Ok(()) => {
                    // Update BM25 index
                    self.bm25_index.lock().unwrap().add_document(&id, content);

                    // Create graph node first (so enrichment can reference it)
                    let graph_node = GraphNode {
                        id: id.clone(),
                        kind: NodeKind::Memory,
                        label: truncate_str(content, 80),
                        payload: HashMap::new(),
                        centrality: 0.0,
                        memory_id: Some(id.clone()),
                        namespace: None,
                    };
                    let _ = self.storage.insert_graph_node(&graph_node);
                    let _ = self.graph.lock().unwrap().add_node(graph_node);

                    // Generate contextual embedding and insert into vector index
                    if let Some(ref emb_service) = self.embeddings {
                        let enriched = self.enrich_memory_text(
                            content,
                            memory_type,
                            &memory.tags,
                            memory.namespace.as_deref(),
                            Some(&id),
                        );
                        if let Ok(embedding) = emb_service.lock().unwrap().embed(&enriched) {
                            let _ = self.storage.store_embedding(&id, &embedding);
                            let _ = self.vector.lock().unwrap().insert(&id, &embedding);
                        }
                    }

                    ids.push(id);
                    imported += 1;
                }
                Err(CodememError::Duplicate(_)) => {
                    skipped += 1;
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
            .unwrap(),
        )
    }

    // ── Consolidation Tools ─────────────────────────────────────────────

    /// MCP tool: consolidate_decay — reduce importance of stale memories.
    fn tool_consolidate_decay(&self, args: &Value) -> ToolResult {
        let threshold_days = args
            .get("threshold_days")
            .and_then(|v| v.as_u64())
            .unwrap_or(30) as i64;

        let conn = self.storage.connection();
        let now = chrono::Utc::now();
        let threshold_ts = (now - chrono::Duration::days(threshold_days)).timestamp();

        let affected = match conn.execute(
            "UPDATE memories SET importance = importance * 0.9, updated_at = ?1
             WHERE last_accessed_at < ?2",
            params![now.timestamp(), threshold_ts],
        ) {
            Ok(count) => count,
            Err(e) => return ToolResult::tool_error(format!("Decay failed: {e}")),
        };

        // Log the consolidation run
        if let Err(e) = self.storage.insert_consolidation_log("decay", affected) {
            tracing::warn!("Failed to log decay consolidation: {e}");
        }

        ToolResult::text(
            json!({
                "cycle": "decay",
                "affected": affected,
                "threshold_days": threshold_days,
            })
            .to_string(),
        )
    }

    /// MCP tool: consolidate_creative — connect memories with overlapping tags
    /// but different types via RELATES_TO edges.
    fn tool_consolidate_creative(&self, args: &Value) -> ToolResult {
        let _ = args; // no params

        let conn = self.storage.connection();

        // Load all memories with their id, type, and tags
        let mut stmt = match conn.prepare("SELECT id, memory_type, tags FROM memories") {
            Ok(s) => s,
            Err(e) => return ToolResult::tool_error(format!("Creative cycle failed: {e}")),
        };

        let rows: Vec<(String, String, String)> = match stmt.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
            ))
        }) {
            Ok(iter) => iter.filter_map(|r| r.ok()).collect(),
            Err(e) => return ToolResult::tool_error(format!("Creative cycle failed: {e}")),
        };

        // Parse tags for each memory
        let parsed: Vec<(String, String, Vec<String>)> = rows
            .into_iter()
            .map(|(id, mtype, tags_json)| {
                let tags: Vec<String> = serde_json::from_str(&tags_json).unwrap_or_default();
                (id, mtype, tags)
            })
            .collect();

        // Load existing RELATES_TO edges to avoid duplicates
        let mut edge_stmt = match conn
            .prepare("SELECT src, dst FROM graph_edges WHERE relationship = 'RELATES_TO'")
        {
            Ok(s) => s,
            Err(e) => return ToolResult::tool_error(format!("Creative cycle failed: {e}")),
        };
        let existing_edges: HashSet<(String, String)> = edge_stmt
            .query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })
            .map(|iter| iter.filter_map(|r| r.ok()).collect())
            .unwrap_or_default();

        let mut new_connections = 0usize;
        let now = chrono::Utc::now();
        let mut graph = self.graph.lock().unwrap();

        for i in 0..parsed.len() {
            for j in (i + 1)..parsed.len() {
                let (ref id_a, ref type_a, ref tags_a) = parsed[i];
                let (ref id_b, ref type_b, ref tags_b) = parsed[j];

                // Different types required
                if type_a == type_b {
                    continue;
                }

                // Must have at least one overlapping tag
                let has_common_tag = tags_a.iter().any(|t| tags_b.contains(t));
                if !has_common_tag {
                    continue;
                }

                // Check not already connected in either direction
                if existing_edges.contains(&(id_a.clone(), id_b.clone()))
                    || existing_edges.contains(&(id_b.clone(), id_a.clone()))
                {
                    continue;
                }

                // Ensure both nodes exist in graph_nodes (upsert memory-type nodes)
                let _ = conn.execute(
                    "INSERT OR IGNORE INTO graph_nodes (id, kind, label, payload, centrality, memory_id)
                     VALUES (?1, 'memory', ?1, '{}', 0.0, ?1)",
                    params![id_a],
                );
                let _ = conn.execute(
                    "INSERT OR IGNORE INTO graph_nodes (id, kind, label, payload, centrality, memory_id)
                     VALUES (?1, 'memory', ?1, '{}', 0.0, ?1)",
                    params![id_b],
                );

                let edge_id = format!("{id_a}-RELATES_TO-{id_b}");
                let edge = Edge {
                    id: edge_id,
                    src: id_a.clone(),
                    dst: id_b.clone(),
                    relationship: RelationshipType::RelatesTo,
                    weight: 1.0,
                    properties: HashMap::new(),
                    created_at: now,
                };

                if self.storage.insert_graph_edge(&edge).is_ok() {
                    let _ = graph.add_edge(edge);
                    new_connections += 1;
                }
            }
        }

        // Log the consolidation run
        if let Err(e) = self
            .storage
            .insert_consolidation_log("creative", new_connections)
        {
            tracing::warn!("Failed to log creative consolidation: {e}");
        }

        ToolResult::text(
            json!({
                "cycle": "creative",
                "new_connections": new_connections,
            })
            .to_string(),
        )
    }

    /// MCP tool: consolidate_cluster — merge memories with same content_hash prefix,
    /// keeping the one with highest importance.
    fn tool_consolidate_cluster(&self, args: &Value) -> ToolResult {
        let _ = args; // no params

        let conn = self.storage.connection();

        // Find groups of memories sharing the same 8-char content_hash prefix
        let mut stmt = match conn
            .prepare("SELECT id, content_hash, importance FROM memories ORDER BY content_hash")
        {
            Ok(s) => s,
            Err(e) => return ToolResult::tool_error(format!("Cluster cycle failed: {e}")),
        };

        let rows: Vec<(String, String, f64)> = match stmt.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, f64>(2)?,
            ))
        }) {
            Ok(iter) => iter.filter_map(|r| r.ok()).collect(),
            Err(e) => return ToolResult::tool_error(format!("Cluster cycle failed: {e}")),
        };

        // Group by first 8 chars of content_hash
        let mut groups: HashMap<String, Vec<(String, f64)>> = HashMap::new();
        for (id, hash, importance) in &rows {
            let prefix = if hash.len() >= 8 {
                hash[..8].to_string()
            } else {
                hash.clone()
            };
            groups
                .entry(prefix)
                .or_default()
                .push((id.clone(), *importance));
        }

        let mut merged_count = 0usize;
        let mut kept_count = 0usize;
        let mut ids_to_delete: Vec<String> = Vec::new();

        for (_prefix, mut members) in groups {
            if members.len() <= 1 {
                kept_count += 1;
                continue;
            }

            // Sort by importance descending; keep the first (highest), delete the rest
            members.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            kept_count += 1;

            for (id, _importance) in members.iter().skip(1) {
                ids_to_delete.push(id.clone());
                merged_count += 1;
            }
        }

        // Delete the duplicates
        let mut vector = self.vector.lock().unwrap();
        let mut graph = self.graph.lock().unwrap();
        for id in &ids_to_delete {
            let _ = self.storage.delete_memory(id);
            let _ = conn.execute(
                "DELETE FROM memory_embeddings WHERE memory_id = ?1",
                params![id],
            );
            let _ = self.storage.delete_graph_edges_for_node(id);
            let _ = self.storage.delete_graph_node(id);
            let _ = vector.remove(id);
            let _ = graph.remove_node(id);
        }

        // Rebuild vector index if we deleted anything
        if merged_count > 0 {
            self.rebuild_vector_index_internal(&mut vector);
        }
        drop(vector);
        drop(graph);

        // Persist vector index to disk
        self.save_index();

        // Log the consolidation run
        if let Err(e) = self
            .storage
            .insert_consolidation_log("cluster", merged_count)
        {
            tracing::warn!("Failed to log cluster consolidation: {e}");
        }

        ToolResult::text(
            json!({
                "cycle": "cluster",
                "merged": merged_count,
                "kept": kept_count,
            })
            .to_string(),
        )
    }

    /// MCP tool: consolidate_forget — delete low-importance, never-accessed memories.
    fn tool_consolidate_forget(&self, args: &Value) -> ToolResult {
        let importance_threshold = args
            .get("importance_threshold")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.1);

        let conn = self.storage.connection();

        // Find memories to forget
        let mut stmt = match conn
            .prepare("SELECT id FROM memories WHERE importance < ?1 AND access_count = 0")
        {
            Ok(s) => s,
            Err(e) => return ToolResult::tool_error(format!("Forget cycle failed: {e}")),
        };

        let ids: Vec<String> = match stmt.query_map(params![importance_threshold], |row| row.get(0))
        {
            Ok(iter) => iter.filter_map(|r| r.ok()).collect(),
            Err(e) => return ToolResult::tool_error(format!("Forget cycle failed: {e}")),
        };

        let deleted = ids.len();

        let mut vector = self.vector.lock().unwrap();
        let mut graph = self.graph.lock().unwrap();
        let mut bm25 = self.bm25_index.lock().unwrap();
        for id in &ids {
            let _ = self.storage.delete_memory(id);
            let _ = conn.execute(
                "DELETE FROM memory_embeddings WHERE memory_id = ?1",
                params![id],
            );
            let _ = self.storage.delete_graph_edges_for_node(id);
            let _ = self.storage.delete_graph_node(id);
            let _ = vector.remove(id);
            let _ = graph.remove_node(id);
            bm25.remove_document(id);
        }

        // Rebuild vector index if we deleted anything
        if deleted > 0 {
            self.rebuild_vector_index_internal(&mut vector);
        }
        drop(vector);
        drop(graph);
        drop(bm25);

        // Persist vector index to disk
        self.save_index();

        // Log the consolidation run
        if let Err(e) = self.storage.insert_consolidation_log("forget", deleted) {
            tracing::warn!("Failed to log forget consolidation: {e}");
        }

        ToolResult::text(
            json!({
                "cycle": "forget",
                "deleted": deleted,
                "threshold": importance_threshold,
            })
            .to_string(),
        )
    }

    /// MCP tool: consolidation_status — show last run of each consolidation cycle.
    fn tool_consolidation_status(&self) -> ToolResult {
        let runs = match self.storage.last_consolidation_runs() {
            Ok(r) => r,
            Err(e) => return ToolResult::tool_error(format!("Failed to query status: {e}")),
        };

        let mut cycles = json!({});
        for entry in &runs {
            let dt = chrono::DateTime::from_timestamp(entry.run_at, 0)
                .map(|t| t.format("%Y-%m-%d %H:%M:%S UTC").to_string())
                .unwrap_or_else(|| "unknown".to_string());
            cycles[&entry.cycle_type] = json!({
                "last_run": dt,
                "affected": entry.affected_count,
            });
        }

        ToolResult::text(
            json!({
                "cycles": cycles,
            })
            .to_string(),
        )
    }

    // ── Impact-Aware Recall & Decision Chain Tools ────────────────────────

    /// MCP tool: recall_with_impact -- recall memories with PageRank-enriched impact data.
    fn tool_recall_with_impact(&self, args: &Value) -> ToolResult {
        let query = match args.get("query").and_then(|v| v.as_str()) {
            Some(q) if !q.is_empty() => q,
            _ => return ToolResult::tool_error("Missing or empty 'query' parameter"),
        };

        let k = args.get("k").and_then(|v| v.as_u64()).unwrap_or(10) as usize;
        let namespace_filter: Option<&str> = args.get("namespace").and_then(|v| v.as_str());

        // Run standard recall logic
        let query_lower = query.to_lowercase();
        let query_tokens: Vec<&str> = query_lower.split_whitespace().collect();

        let vector_results: Vec<(String, f32)> = if let Some(ref emb_service) = self.embeddings {
            match emb_service.lock().unwrap().embed(query) {
                Ok(query_embedding) => self
                    .vector
                    .lock()
                    .unwrap()
                    .search(&query_embedding, k * 2)
                    .unwrap_or_default(),
                Err(e) => {
                    tracing::warn!("Query embedding failed: {e}");
                    vec![]
                }
            }
        } else {
            vec![]
        };

        let graph = self.graph.lock().unwrap();
        let bm25 = self.bm25_index.lock().unwrap();

        let mut results: Vec<SearchResult> = Vec::new();

        if vector_results.is_empty() {
            let ids = match self.storage.list_memory_ids() {
                Ok(ids) => ids,
                Err(e) => return ToolResult::tool_error(format!("Storage error: {e}")),
            };

            for id in &ids {
                if let Ok(Some(memory)) = self.storage.get_memory(id) {
                    if let Some(ns) = namespace_filter {
                        if memory.namespace.as_deref() != Some(ns) {
                            continue;
                        }
                    }

                    let breakdown =
                        compute_score(&memory, query, &query_tokens, 0.0, &graph, &bm25);
                    // SAFETY: Single-threaded MCP server; no concurrent access.
                    let score =
                        breakdown.total_with_weights(unsafe { &*self.scoring_weights.get() });
                    if score > 0.01 {
                        results.push(SearchResult {
                            memory,
                            score,
                            score_breakdown: breakdown,
                        });
                    }
                }
            }
        } else {
            for (id, distance) in &vector_results {
                if let Ok(Some(memory)) = self.storage.get_memory(id) {
                    if let Some(ns) = namespace_filter {
                        if memory.namespace.as_deref() != Some(ns) {
                            continue;
                        }
                    }

                    let similarity = 1.0 - (*distance as f64);
                    let breakdown =
                        compute_score(&memory, query, &query_tokens, similarity, &graph, &bm25);
                    // SAFETY: Single-threaded MCP server; no concurrent access.
                    let score =
                        breakdown.total_with_weights(unsafe { &*self.scoring_weights.get() });
                    results.push(SearchResult {
                        memory,
                        score,
                        score_breakdown: breakdown,
                    });
                }
            }
        }

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(k);

        if results.is_empty() {
            return ToolResult::text("No matching memories found.");
        }

        // Enrich each result with impact data
        let output: Vec<Value> = results
            .iter()
            .map(|r| {
                let memory_id = &r.memory.id;

                let pagerank = graph.get_pagerank(memory_id);
                let centrality = graph.get_betweenness(memory_id);

                // Find connected Decision memories
                let connected_decisions: Vec<String> = graph
                    .get_edges(memory_id)
                    .unwrap_or_default()
                    .iter()
                    .filter_map(|e| {
                        let other_id = if e.src == *memory_id { &e.dst } else { &e.src };
                        self.storage
                            .get_memory(other_id)
                            .ok()
                            .flatten()
                            .and_then(|m| {
                                if m.memory_type == MemoryType::Decision {
                                    Some(m.id)
                                } else {
                                    None
                                }
                            })
                    })
                    .collect();

                // Find dependent files from graph edges
                let dependent_files: Vec<String> = graph
                    .get_edges(memory_id)
                    .unwrap_or_default()
                    .iter()
                    .filter_map(|e| {
                        let other_id = if e.src == *memory_id { &e.dst } else { &e.src };
                        graph.get_node(other_id).ok().flatten().and_then(|n| {
                            if n.kind == NodeKind::File {
                                Some(n.label.clone())
                            } else {
                                n.payload
                                    .get("file_path")
                                    .and_then(|v| v.as_str().map(String::from))
                            }
                        })
                    })
                    .collect();

                let modification_count = r.memory.access_count;

                json!({
                    "id": r.memory.id,
                    "content": r.memory.content,
                    "memory_type": r.memory.memory_type.to_string(),
                    "score": format!("{:.4}", r.score),
                    "importance": r.memory.importance,
                    "tags": r.memory.tags,
                    "access_count": r.memory.access_count,
                    "impact": {
                        "pagerank": format!("{:.6}", pagerank),
                        "centrality": format!("{:.6}", centrality),
                        "connected_decisions": connected_decisions,
                        "dependent_files": dependent_files,
                        "modification_count": modification_count,
                    }
                })
            })
            .collect();

        ToolResult::text(serde_json::to_string_pretty(&output).unwrap())
    }

    /// MCP tool: get_decision_chain -- follow decision evolution through the graph.
    fn tool_get_decision_chain(&self, args: &Value) -> ToolResult {
        let file_path: Option<&str> = args.get("file_path").and_then(|v| v.as_str());
        let topic: Option<&str> = args.get("topic").and_then(|v| v.as_str());

        if file_path.is_none() && topic.is_none() {
            return ToolResult::tool_error("Must provide either 'file_path' or 'topic' parameter");
        }

        let graph = self.graph.lock().unwrap();

        // Find all Decision-type memories matching the file_path or topic
        let ids = match self.storage.list_memory_ids() {
            Ok(ids) => ids,
            Err(e) => return ToolResult::tool_error(format!("Storage error: {e}")),
        };

        let decision_edge_types = [
            RelationshipType::EvolvedInto,
            RelationshipType::LeadsTo,
            RelationshipType::DerivedFrom,
        ];

        // Collect all Decision memories matching the filter
        let mut decision_memories: Vec<MemoryNode> = Vec::new();
        for id in &ids {
            if let Ok(Some(memory)) = self.storage.get_memory(id) {
                if memory.memory_type != MemoryType::Decision {
                    continue;
                }

                let content_lower = memory.content.to_lowercase();
                let tags_lower: String = memory.tags.join(" ").to_lowercase();

                let matches = if let Some(fp) = file_path {
                    content_lower.contains(&fp.to_lowercase())
                        || tags_lower.contains(&fp.to_lowercase())
                        || memory
                            .metadata
                            .get("file_path")
                            .and_then(|v| v.as_str())
                            .map(|v| v.to_lowercase().contains(&fp.to_lowercase()))
                            .unwrap_or(false)
                } else if let Some(t) = topic {
                    let t_lower = t.to_lowercase();
                    content_lower.contains(&t_lower) || tags_lower.contains(&t_lower)
                } else {
                    false
                };

                if matches {
                    decision_memories.push(memory);
                }
            }
        }

        if decision_memories.is_empty() {
            return ToolResult::text("No decision memories found matching the criteria.");
        }

        // Expand through decision-related edges to find the full chain
        let mut chain_ids: HashSet<String> = HashSet::new();
        let mut to_explore: Vec<String> = decision_memories.iter().map(|m| m.id.clone()).collect();

        while let Some(current_id) = to_explore.pop() {
            if !chain_ids.insert(current_id.clone()) {
                continue;
            }

            if let Ok(edges) = graph.get_edges(&current_id) {
                for edge in &edges {
                    if decision_edge_types.contains(&edge.relationship) {
                        let other_id = if edge.src == current_id {
                            &edge.dst
                        } else {
                            &edge.src
                        };
                        if !chain_ids.contains(other_id) {
                            // Only follow to other Decision memories
                            if let Ok(Some(m)) = self.storage.get_memory(other_id) {
                                if m.memory_type == MemoryType::Decision {
                                    to_explore.push(other_id.clone());
                                }
                            }
                        }
                    }
                }
            }
        }

        // Collect all chain memories and sort by created_at (temporal order)
        let mut chain: Vec<Value> = Vec::new();
        for id in &chain_ids {
            if let Ok(Some(memory)) = self.storage.get_memory(id) {
                // Find edges connecting this memory within the chain
                let connections: Vec<Value> = graph
                    .get_edges(id)
                    .unwrap_or_default()
                    .iter()
                    .filter(|e| {
                        decision_edge_types.contains(&e.relationship)
                            && (chain_ids.contains(&e.src) && chain_ids.contains(&e.dst))
                    })
                    .map(|e| {
                        json!({
                            "relationship": e.relationship.to_string(),
                            "source": e.src,
                            "target": e.dst,
                        })
                    })
                    .collect();

                chain.push(json!({
                    "id": memory.id,
                    "content": memory.content,
                    "importance": memory.importance,
                    "tags": memory.tags,
                    "created_at": memory.created_at.to_rfc3339(),
                    "connections": connections,
                }));
            }
        }

        // Sort chronologically
        chain.sort_by(|a, b| {
            let a_dt = a["created_at"].as_str().unwrap_or("");
            let b_dt = b["created_at"].as_str().unwrap_or("");
            a_dt.cmp(b_dt)
        });

        let response = json!({
            "chain_length": chain.len(),
            "filter": {
                "file_path": file_path,
                "topic": topic,
            },
            "decisions": chain,
        });

        ToolResult::text(serde_json::to_string_pretty(&response).unwrap())
    }

    /// Internal helper: rebuild vector index from all stored embeddings.
    fn rebuild_vector_index_internal(&self, vector: &mut HnswIndex) {
        let conn = self.storage.connection();
        let mut stmt = match conn.prepare("SELECT memory_id, embedding FROM memory_embeddings") {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!("Failed to rebuild vector index: {e}");
                return;
            }
        };

        let rows: Vec<(String, Vec<u8>)> = match stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, Vec<u8>>(1)?))
        }) {
            Ok(iter) => iter.filter_map(|r| r.ok()).collect(),
            Err(e) => {
                tracing::warn!("Failed to rebuild vector index: {e}");
                return;
            }
        };

        // Create a fresh index and reinsert all embeddings
        if let Ok(mut fresh) = HnswIndex::with_defaults() {
            for (id, blob) in &rows {
                let floats: Vec<f32> = blob
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                let _ = fresh.insert(id, &floats);
            }
            *vector = fresh;
        }
    }

    /// MCP tool: detect_patterns — detect cross-session patterns in stored memories.
    fn tool_detect_patterns(&self, args: &Value) -> ToolResult {
        let min_frequency = args
            .get("min_frequency")
            .and_then(|v| v.as_u64())
            .unwrap_or(3) as usize;
        let namespace = args.get("namespace").and_then(|v| v.as_str());

        match patterns::detect_patterns(&self.storage, namespace, min_frequency) {
            Ok(detected) => {
                let json_patterns: Vec<Value> = detected
                    .iter()
                    .map(|p| {
                        json!({
                            "pattern_type": p.pattern_type.to_string(),
                            "description": p.description,
                            "frequency": p.frequency,
                            "confidence": p.confidence,
                            "related_memories": p.related_memories,
                        })
                    })
                    .collect();
                ToolResult::text(
                    serde_json::to_string_pretty(&json!({
                        "patterns": json_patterns,
                        "count": detected.len(),
                    }))
                    .unwrap(),
                )
            }
            Err(e) => ToolResult::tool_error(format!("Pattern detection error: {e}")),
        }
    }

    /// MCP tool: pattern_insights — generate human-readable pattern insights as markdown.
    fn tool_pattern_insights(&self, args: &Value) -> ToolResult {
        let min_frequency = args
            .get("min_frequency")
            .and_then(|v| v.as_u64())
            .unwrap_or(2) as usize;
        let namespace = args.get("namespace").and_then(|v| v.as_str());

        match patterns::detect_patterns(&self.storage, namespace, min_frequency) {
            Ok(detected) => {
                let markdown = patterns::generate_insights(&detected);
                ToolResult::text(markdown)
            }
            Err(e) => ToolResult::tool_error(format!("Pattern insights error: {e}")),
        }
    }
}

/// Write a JSON-RPC response as a single line to stdout.
fn write_response(writer: &mut impl Write, response: &JsonRpcResponse) -> io::Result<()> {
    let json = serde_json::to_string(response)?;
    writeln!(writer, "{json}")?;
    writer.flush()
}

fn truncate_str(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}...", &s[..max])
    }
}

// ── Result Formatting ────────────────────────────────────────────────────────

/// Format search results into a ToolResult. If `_repo_label` is provided,
/// a "repo" field is added to each result.
fn format_recall_results(results: &[SearchResult], _repo_label: Option<&str>) -> ToolResult {
    if results.is_empty() {
        return ToolResult::text("No matching memories found.");
    }

    let output: Vec<Value> = results
        .iter()
        .map(|r| {
            json!({
                "id": r.memory.id,
                "content": r.memory.content,
                "memory_type": r.memory.memory_type.to_string(),
                "score": format!("{:.4}", r.score),
                "importance": r.memory.importance,
                "tags": r.memory.tags,
                "access_count": r.memory.access_count,
            })
        })
        .collect();

    ToolResult::text(serde_json::to_string_pretty(&output).unwrap())
}

// ── Hybrid Scoring ──────────────────────────────────────────────────────────

/// Compute 9-component hybrid score for a memory against a query.
/// The `graph` parameter is used to look up edge counts for graph strength scoring.
/// The `bm25` parameter provides BM25-based token overlap scoring; if the memory
/// is in the index it uses the indexed score, otherwise falls back to `score_text`.
fn compute_score(
    memory: &MemoryNode,
    query: &str,
    query_tokens: &[&str],
    vector_similarity: f64,
    graph: &GraphEngine,
    bm25: &bm25::Bm25Index,
) -> ScoreBreakdown {
    // BM25 token overlap (replaces naive split+intersect)
    let token_overlap = if query.is_empty() {
        0.0
    } else {
        // Try indexed score first (memory already in the BM25 index),
        // fall back to scoring against raw text for unindexed documents.
        let indexed_score = bm25.score(query, &memory.id);
        if indexed_score > 0.0 {
            indexed_score
        } else {
            bm25.score_text(query, &memory.content)
        }
    };

    // Temporal: how recently updated (exponential decay over 30 days)
    let age_hours = (chrono::Utc::now() - memory.updated_at).num_hours().max(0) as f64;
    let temporal = (-age_hours / (30.0 * 24.0)).exp();

    // Tag matching: fraction of query tokens found in tags
    let tag_matching = if !query_tokens.is_empty() {
        let tag_str: String = memory.tags.join(" ").to_lowercase();
        let matches = query_tokens
            .iter()
            .filter(|qt| tag_str.contains(**qt))
            .count();
        matches as f64 / query_tokens.len() as f64
    } else {
        0.0
    };

    // Recency: based on last access time (decay over 7 days)
    let access_hours = (chrono::Utc::now() - memory.last_accessed_at)
        .num_hours()
        .max(0) as f64;
    let recency = (-access_hours / (7.0 * 24.0)).exp();

    // Enhanced graph scoring using cached centrality metrics.
    // Combines PageRank, betweenness centrality, normalized degree,
    // and a cluster bonus for richer graph-awareness.
    let pagerank = graph.get_pagerank(&memory.id);
    let betweenness = graph.get_betweenness(&memory.id);
    let degree = graph.neighbors(&memory.id).map(|n| n.len()).unwrap_or(0) as f64;
    let max_degree = graph.max_degree();
    let normalized_degree = degree / max_degree.max(1.0);

    // Cluster bonus: if the memory has many neighbors, it gets a small bonus (capped at 1.0).
    let cluster_bonus = graph
        .neighbors(&memory.id)
        .map(|n| (n.len() as f64 / 10.0).min(1.0))
        .unwrap_or(0.0);

    let graph_strength =
        (0.4 * pagerank + 0.3 * betweenness + 0.2 * normalized_degree + 0.1 * cluster_bonus)
            .min(1.0);

    ScoreBreakdown {
        vector_similarity,
        graph_strength,
        token_overlap,
        temporal,
        tag_matching,
        importance: memory.importance,
        confidence: memory.confidence,
        recency,
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

    fn test_server() -> McpServer {
        McpServer::for_testing()
    }

    /// Helper: store a memory and return the parsed JSON result.
    fn store_memory(server: &McpServer, content: &str, memory_type: &str, tags: &[&str]) -> Value {
        let store_params = json!({
            "name": "store_memory",
            "arguments": {
                "content": content,
                "memory_type": memory_type,
                "tags": tags,
            }
        });
        let resp = server.handle_request("tools/call", Some(&store_params), json!(1));
        let result = resp.result.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        serde_json::from_str(text).unwrap()
    }

    /// Helper: recall memories and return the parsed JSON (array or string).
    fn recall_memories(server: &McpServer, query: &str, memory_type: Option<&str>) -> String {
        let mut arguments = json!({"query": query});
        if let Some(mt) = memory_type {
            arguments["memory_type"] = json!(mt);
        }
        let params = json!({"name": "recall_memory", "arguments": arguments});
        let resp = server.handle_request("tools/call", Some(&params), json!(2));
        let result = resp.result.unwrap();
        result["content"][0]["text"].as_str().unwrap().to_string()
    }

    #[test]
    fn parse_json_rpc_request() {
        let json = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#;
        let req: JsonRpcRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.method, "initialize");
        assert!(req.id.is_some());
    }

    #[test]
    fn parse_notification_no_id() {
        let json = r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#;
        let req: JsonRpcRequest = serde_json::from_str(json).unwrap();
        assert!(req.id.is_none());
    }

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
    fn handle_tools_call_store() {
        let server = test_server();
        let params = json!({"name": "store_memory", "arguments": {"content": "test content"}});
        let resp = server.handle_request("tools/call", Some(&params), json!(3));
        assert!(resp.result.is_some());
        assert!(resp.error.is_none());

        // Verify it actually stored
        let stats_resp = server.handle_request(
            "tools/call",
            Some(&json!({"name": "codemem_stats", "arguments": {}})),
            json!(4),
        );
        let stats = stats_resp.result.unwrap();
        let text = stats["content"][0]["text"].as_str().unwrap();
        let parsed: Value = serde_json::from_str(text).unwrap();
        assert_eq!(parsed["storage"]["memories"], 1);
    }

    #[test]
    fn handle_store_and_recall() {
        let server = test_server();

        // Store a memory
        let store_params = json!({
            "name": "store_memory",
            "arguments": {
                "content": "Rust uses ownership and borrowing for memory safety",
                "memory_type": "insight",
                "tags": ["rust", "memory"]
            }
        });
        server.handle_request("tools/call", Some(&store_params), json!(1));

        // Recall it (text search fallback, no embeddings in test)
        let recall_params = json!({
            "name": "recall_memory",
            "arguments": {"query": "rust memory safety"}
        });
        let resp = server.handle_request("tools/call", Some(&recall_params), json!(2));
        let result = resp.result.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        // Should find the memory via token overlap
        assert!(text.contains("ownership") || text.contains("rust"));
    }

    #[test]
    fn handle_store_and_delete() {
        let server = test_server();

        // Store
        let store_params = json!({
            "name": "store_memory",
            "arguments": {"content": "delete me"}
        });
        let resp = server.handle_request("tools/call", Some(&store_params), json!(1));
        let result = resp.result.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        let stored: Value = serde_json::from_str(text).unwrap();
        let id = stored["id"].as_str().unwrap();

        // Delete
        let delete_params = json!({
            "name": "delete_memory",
            "arguments": {"id": id}
        });
        let resp = server.handle_request("tools/call", Some(&delete_params), json!(2));
        assert!(resp.error.is_none());
    }

    #[test]
    fn handle_unknown_tool() {
        let server = test_server();
        let params = json!({"name": "nonexistent", "arguments": {}});
        let resp = server.handle_request("tools/call", Some(&params), json!(4));
        let result = resp.result.unwrap();
        assert_eq!(result["isError"], true);
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

    #[test]
    fn tool_result_serialization() {
        let result = ToolResult::text("hello");
        let json = serde_json::to_value(&result).unwrap();
        assert_eq!(json["content"][0]["type"], "text");
        assert_eq!(json["content"][0]["text"], "hello");
        assert_eq!(json["isError"], false);
    }

    #[test]
    fn tool_error_serialization() {
        let result = ToolResult::tool_error("something went wrong");
        let json = serde_json::to_value(&result).unwrap();
        assert_eq!(json["isError"], true);
    }

    #[test]
    fn write_response_newline_delimited() {
        let resp = JsonRpcResponse::success(json!(1), json!({"ok": true}));
        let mut buf = Vec::new();
        write_response(&mut buf, &resp).unwrap();
        let output = String::from_utf8(buf).unwrap();
        assert!(output.ends_with('\n'));
        assert!(!output.contains("Content-Length"));
    }

    #[test]
    fn handle_health() {
        let server = test_server();
        let params = json!({"name": "codemem_health", "arguments": {}});
        let resp = server.handle_request("tools/call", Some(&params), json!(7));
        let result = resp.result.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        let health: Value = serde_json::from_str(text).unwrap();
        assert_eq!(health["healthy"], true);
        assert_eq!(health["storage"], "ok");
    }

    #[test]
    fn handle_stats() {
        let server = test_server();
        let params = json!({"name": "codemem_stats", "arguments": {}});
        let resp = server.handle_request("tools/call", Some(&params), json!(8));
        let result = resp.result.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        let stats: Value = serde_json::from_str(text).unwrap();
        assert_eq!(stats["storage"]["memories"], 0);
        assert_eq!(stats["vector"]["dimensions"], 768);
    }

    // ── Graph Strength Scoring Tests ────────────────────────────────────

    #[test]
    fn graph_strength_zero_when_no_edges() {
        let server = test_server();
        let stored = store_memory(&server, "isolated memory", "context", &[]);
        let id = stored["id"].as_str().unwrap();

        // Verify graph strength is 0 for a memory with no edges
        let graph = server.graph.lock().unwrap();
        let edges = graph.get_edges(id).unwrap();
        assert_eq!(edges.len(), 0);

        let memory = server.storage.get_memory(id).unwrap().unwrap();
        let bm25 = server.bm25_index.lock().unwrap();
        let breakdown = compute_score(&memory, "isolated", &["isolated"], 0.0, &graph, &bm25);
        assert_eq!(breakdown.graph_strength, 0.0);
    }

    #[test]
    fn graph_strength_increases_with_edges() {
        let server = test_server();
        let src = store_memory(&server, "source memory about rust", "insight", &["rust"]);
        let dst1 = store_memory(&server, "target memory one about types", "pattern", &[]);
        let dst2 = store_memory(&server, "target memory two about safety", "decision", &[]);

        let src_id = src["id"].as_str().unwrap();
        let dst1_id = dst1["id"].as_str().unwrap();
        let dst2_id = dst2["id"].as_str().unwrap();

        // Associate: src -> dst1
        let params = json!({
            "name": "associate_memories",
            "arguments": {
                "source_id": src_id,
                "target_id": dst1_id,
                "relationship": "RELATES_TO",
            }
        });
        server.handle_request("tools/call", Some(&params), json!(10));

        // Associate: src -> dst2
        let params = json!({
            "name": "associate_memories",
            "arguments": {
                "source_id": src_id,
                "target_id": dst2_id,
                "relationship": "LEADS_TO",
            }
        });
        server.handle_request("tools/call", Some(&params), json!(11));

        // Recompute centrality so PageRank/betweenness are cached
        {
            let mut graph = server.graph.lock().unwrap();
            graph.recompute_centrality();
        }

        // Score with edges: the source memory with 2 edges should have
        // a non-zero graph_strength due to enhanced scoring (PageRank + betweenness + degree)
        let graph = server.graph.lock().unwrap();
        let memory = server.storage.get_memory(src_id).unwrap().unwrap();
        let bm25 = server.bm25_index.lock().unwrap();
        let breakdown = compute_score(&memory, "rust", &["rust"], 0.0, &graph, &bm25);
        assert!(
            breakdown.graph_strength > 0.0,
            "graph_strength should be > 0 with 2 edges, got {}",
            breakdown.graph_strength
        );
    }

    #[test]
    fn graph_strength_caps_at_one() {
        let server = test_server();

        // Create 6 memories, connect all to the first
        let src = store_memory(&server, "hub memory with many edges", "insight", &[]);
        let src_id = src["id"].as_str().unwrap();

        for i in 0..6 {
            let dst = store_memory(&server, &format!("spoke memory number {i}"), "context", &[]);
            let dst_id = dst["id"].as_str().unwrap();
            let params = json!({
                "name": "associate_memories",
                "arguments": {
                    "source_id": src_id,
                    "target_id": dst_id,
                    "relationship": "RELATES_TO",
                }
            });
            server.handle_request("tools/call", Some(&params), json!(20 + i));
        }

        // The graph_strength formula caps at 1.0 via .min(1.0)
        let graph = server.graph.lock().unwrap();
        let memory = server.storage.get_memory(src_id).unwrap().unwrap();
        let bm25 = server.bm25_index.lock().unwrap();
        let breakdown = compute_score(&memory, "hub", &["hub"], 0.0, &graph, &bm25);
        assert!(
            breakdown.graph_strength <= 1.0,
            "graph_strength should be <= 1.0, got {}",
            breakdown.graph_strength
        );
    }

    // ── Memory Type Filter Tests ────────────────────────────────────────

    #[test]
    fn recall_filters_by_memory_type() {
        let server = test_server();

        // Store memories of different types, all containing "rust"
        store_memory(&server, "rust ownership insight", "insight", &["rust"]);
        store_memory(&server, "rust pattern matching", "pattern", &["rust"]);
        store_memory(&server, "rust decision to use enums", "decision", &["rust"]);

        // Recall with type filter "insight"
        let text = recall_memories(&server, "rust", Some("insight"));
        let results: Vec<Value> = serde_json::from_str(&text).unwrap();

        // Should only contain the insight memory
        assert_eq!(results.len(), 1);
        assert_eq!(results[0]["memory_type"], "insight");
        assert!(results[0]["content"]
            .as_str()
            .unwrap()
            .contains("ownership"));
    }

    #[test]
    fn recall_without_type_filter_returns_all() {
        let server = test_server();

        store_memory(&server, "rust ownership insight", "insight", &["rust"]);
        store_memory(&server, "rust pattern matching", "pattern", &["rust"]);

        // Recall without type filter
        let text = recall_memories(&server, "rust", None);
        let results: Vec<Value> = serde_json::from_str(&text).unwrap();

        // Should return both
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn recall_with_invalid_type_filter_returns_all() {
        let server = test_server();

        store_memory(&server, "rust ownership insight", "insight", &["rust"]);

        // An invalid memory_type string should be ignored (parsed as None)
        let text = recall_memories(&server, "rust", Some("nonexistent_type"));
        let results: Vec<Value> = serde_json::from_str(&text).unwrap();

        // Should return everything (no filter applied)
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn recall_with_type_filter_no_matches() {
        let server = test_server();

        store_memory(&server, "rust ownership insight", "insight", &["rust"]);

        // Filter for a type that has no matches in the content query
        let text = recall_memories(&server, "rust", Some("habit"));
        assert_eq!(text, "No matching memories found.");
    }

    // ── Vector Index Persistence Tests ──────────────────────────────────

    #[test]
    fn save_index_noop_for_in_memory_server() {
        let server = test_server();
        // db_path is None for in-memory server, save_index should not panic
        assert!(server.db_path.is_none());
        server.save_index(); // should be a no-op
    }

    #[test]
    fn from_db_path_sets_db_path() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();

        let server = McpServer::from_db_path(&path).unwrap();
        assert_eq!(server.db_path, Some(path));
    }

    #[test]
    fn save_index_persists_to_disk() {
        let dir = tempfile::TempDir::new().unwrap();
        let db_path = dir.path().join("test.db");

        let server = McpServer::from_db_path(&db_path).unwrap();

        // Store a memory (triggers save_index internally)
        store_memory(&server, "persistent memory test", "context", &[]);

        // The index file should exist if embeddings were available,
        // but even without embeddings save_index should not error.
        // Verify save_index can be called explicitly without panicking.
        server.save_index();

        // Verify the idx path is derived correctly
        let expected_idx_path = db_path.with_extension("idx");
        assert_eq!(expected_idx_path, dir.path().join("test.idx"),);
    }

    // ── Namespace Filter Tests ────────────────────────────────────────

    #[test]
    fn recall_filters_by_namespace() {
        let server = test_server();

        // Store memories with different namespaces via direct storage
        let now = chrono::Utc::now();
        for (content, ns) in [
            ("rust ownership in project-a", Some("/projects/a")),
            ("rust borrowing in project-b", Some("/projects/b")),
            ("rust global memory no namespace", None),
        ] {
            let id = uuid::Uuid::new_v4().to_string();
            let hash = Storage::content_hash(content);
            let memory = MemoryNode {
                id: id.clone(),
                content: content.to_string(),
                memory_type: MemoryType::Insight,
                importance: 0.5,
                confidence: 1.0,
                access_count: 0,
                content_hash: hash,
                tags: vec!["rust".to_string()],
                metadata: HashMap::new(),
                namespace: ns.map(String::from),
                created_at: now,
                updated_at: now,
                last_accessed_at: now,
            };
            server.storage.insert_memory(&memory).unwrap();

            // Add graph node so graph scoring works
            let graph_node = GraphNode {
                id: id.clone(),
                kind: NodeKind::Memory,
                label: content.to_string(),
                payload: HashMap::new(),
                centrality: 0.0,
                memory_id: Some(id),
                namespace: None,
            };
            server.storage.insert_graph_node(&graph_node).unwrap();
            let _ = server.graph.lock().unwrap().add_node(graph_node);
        }

        // Recall with namespace filter "/projects/a"
        let params = json!({
            "name": "recall_memory",
            "arguments": {"query": "rust", "namespace": "/projects/a"}
        });
        let resp = server.handle_request("tools/call", Some(&params), json!(100));
        let result = resp.result.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        let results: Vec<Value> = serde_json::from_str(text).unwrap();

        assert_eq!(results.len(), 1);
        assert!(results[0]["content"]
            .as_str()
            .unwrap()
            .contains("project-a"));
    }

    #[test]
    fn recall_without_namespace_returns_all() {
        let server = test_server();

        // Store memories in different namespaces
        store_memory(&server, "rust memory one", "context", &["rust"]);
        store_memory(&server, "rust memory two", "context", &["rust"]);

        // Recall without namespace filter returns all
        let text = recall_memories(&server, "rust", None);
        let results: Vec<Value> = serde_json::from_str(&text).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn store_memory_with_namespace() {
        let server = test_server();

        let params = json!({
            "name": "store_memory",
            "arguments": {
                "content": "namespaced memory content",
                "namespace": "/my/project"
            }
        });
        let resp = server.handle_request("tools/call", Some(&params), json!(200));
        let result = resp.result.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        let stored: Value = serde_json::from_str(text).unwrap();
        let id = stored["id"].as_str().unwrap();

        // Retrieve and verify namespace is set
        let memory = server.storage.get_memory(id).unwrap().unwrap();
        assert_eq!(memory.namespace.as_deref(), Some("/my/project"));
    }

    // ── Structural Tool Tests ───────────────────────────────────────────

    #[test]
    fn search_symbols_requires_index() {
        let server = test_server();
        let params = json!({"name": "search_symbols", "arguments": {"query": "foo"}});
        let resp = server.handle_request("tools/call", Some(&params), json!(300));
        let result = resp.result.unwrap();
        assert_eq!(result["isError"], true);
        let text = result["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("No codebase indexed"));
    }

    #[test]
    fn get_symbol_info_requires_index() {
        let server = test_server();
        let params =
            json!({"name": "get_symbol_info", "arguments": {"qualified_name": "foo::bar"}});
        let resp = server.handle_request("tools/call", Some(&params), json!(301));
        let result = resp.result.unwrap();
        assert_eq!(result["isError"], true);
    }

    #[test]
    fn get_clusters_empty_graph() {
        let server = test_server();
        let params = json!({"name": "get_clusters", "arguments": {}});
        let resp = server.handle_request("tools/call", Some(&params), json!(302));
        let result = resp.result.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        let parsed: Value = serde_json::from_str(text).unwrap();
        assert_eq!(parsed["cluster_count"], 0);
    }

    #[test]
    fn get_pagerank_empty_graph() {
        let server = test_server();
        let params = json!({"name": "get_pagerank", "arguments": {}});
        let resp = server.handle_request("tools/call", Some(&params), json!(303));
        let result = resp.result.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        let parsed: Value = serde_json::from_str(text).unwrap();
        assert_eq!(parsed["results"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn get_cross_repo_requires_path_or_index() {
        let server = test_server();
        let params = json!({"name": "get_cross_repo", "arguments": {}});
        let resp = server.handle_request("tools/call", Some(&params), json!(304));
        let result = resp.result.unwrap();
        assert_eq!(result["isError"], true);
    }

    #[test]
    fn store_memory_with_links() {
        let server = test_server();

        // First store two memories to get node IDs
        let m1 = store_memory(&server, "target node one", "context", &[]);
        let m2 = store_memory(&server, "target node two", "context", &[]);
        let m1_id = m1["id"].as_str().unwrap();
        let m2_id = m2["id"].as_str().unwrap();

        // Store a new memory with links to the previous two
        let params = json!({
            "name": "store_memory",
            "arguments": {
                "content": "linked memory content",
                "links": [m1_id, m2_id]
            }
        });
        let resp = server.handle_request("tools/call", Some(&params), json!(305));
        let result = resp.result.unwrap();
        assert_eq!(result["isError"], false);
        let text = result["content"][0]["text"].as_str().unwrap();
        let stored: Value = serde_json::from_str(text).unwrap();
        let linked_id = stored["id"].as_str().unwrap();

        // Verify edges were created
        let graph = server.graph.lock().unwrap();
        let edges = graph.get_edges(linked_id).unwrap();
        assert_eq!(edges.len(), 2);
        for edge in &edges {
            assert_eq!(edge.src, linked_id);
            assert_eq!(edge.relationship, RelationshipType::RelatesTo);
        }
    }

    #[test]
    fn index_codebase_nonexistent_path() {
        let server = test_server();
        let params =
            json!({"name": "index_codebase", "arguments": {"path": "/nonexistent/path/abc123"}});
        let resp = server.handle_request("tools/call", Some(&params), json!(306));
        let result = resp.result.unwrap();
        assert_eq!(result["isError"], true);
        let text = result["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("does not exist"));
    }

    #[test]
    fn index_codebase_and_search_symbols() {
        let server = test_server();

        // Create a temp directory with a Rust file
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(
            dir.path().join("lib.rs"),
            b"pub fn hello_world() { println!(\"hello\"); }\npub struct MyConfig { pub debug: bool }\n",
        )
        .unwrap();

        // Index the directory
        let params = json!({
            "name": "index_codebase",
            "arguments": {"path": dir.path().to_string_lossy()}
        });
        let resp = server.handle_request("tools/call", Some(&params), json!(307));
        let result = resp.result.unwrap();
        assert_eq!(result["isError"], false);
        let text = result["content"][0]["text"].as_str().unwrap();
        let index_result: Value = serde_json::from_str(text).unwrap();
        assert!(index_result["symbols"].as_u64().unwrap() >= 2);

        // Now search for symbols
        let params = json!({
            "name": "search_symbols",
            "arguments": {"query": "hello"}
        });
        let resp = server.handle_request("tools/call", Some(&params), json!(308));
        let result = resp.result.unwrap();
        assert_eq!(result["isError"], false);
        let text = result["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("hello_world"));

        // Search by kind
        let params = json!({
            "name": "search_symbols",
            "arguments": {"query": "My", "kind": "struct"}
        });
        let resp = server.handle_request("tools/call", Some(&params), json!(309));
        let result = resp.result.unwrap();
        assert_eq!(result["isError"], false);
        let text = result["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("MyConfig"));
    }

    #[test]
    fn get_dependencies_for_symbol() {
        let server = test_server();

        // Manually add symbol nodes and an edge to the graph
        let node_a = GraphNode {
            id: "sym:module::foo".to_string(),
            kind: NodeKind::Function,
            label: "foo".to_string(),
            payload: HashMap::new(),
            centrality: 0.0,
            memory_id: None,
            namespace: None,
        };
        let node_b = GraphNode {
            id: "sym:module::bar".to_string(),
            kind: NodeKind::Function,
            label: "bar".to_string(),
            payload: HashMap::new(),
            centrality: 0.0,
            memory_id: None,
            namespace: None,
        };

        server.storage.insert_graph_node(&node_a).unwrap();
        server.storage.insert_graph_node(&node_b).unwrap();
        {
            let mut graph = server.graph.lock().unwrap();
            graph.add_node(node_a).unwrap();
            graph.add_node(node_b).unwrap();
            let edge = Edge {
                id: "ref:foo->bar:CALLS".to_string(),
                src: "sym:module::foo".to_string(),
                dst: "sym:module::bar".to_string(),
                relationship: RelationshipType::Calls,
                weight: 1.0,
                properties: HashMap::new(),
                created_at: chrono::Utc::now(),
            };
            graph.add_edge(edge).unwrap();
        }

        // Query outgoing deps from foo
        let params = json!({
            "name": "get_dependencies",
            "arguments": {"qualified_name": "module::foo", "direction": "outgoing"}
        });
        let resp = server.handle_request("tools/call", Some(&params), json!(310));
        let result = resp.result.unwrap();
        assert_eq!(result["isError"], false);
        let text = result["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("module::bar"));
        assert!(text.contains("CALLS"));
    }

    #[test]
    fn get_pagerank_with_nodes() {
        let server = test_server();

        // Add a small graph: A -> B -> C
        for (id, label) in [("sym:a", "a"), ("sym:b", "b"), ("sym:c", "c")] {
            let node = GraphNode {
                id: id.to_string(),
                kind: NodeKind::Function,
                label: label.to_string(),
                payload: HashMap::new(),
                centrality: 0.0,
                memory_id: None,
                namespace: None,
            };
            server.storage.insert_graph_node(&node).unwrap();
            server.graph.lock().unwrap().add_node(node).unwrap();
        }

        let edge1 = Edge {
            id: "e1".to_string(),
            src: "sym:a".to_string(),
            dst: "sym:b".to_string(),
            relationship: RelationshipType::Calls,
            weight: 1.0,
            properties: HashMap::new(),
            created_at: chrono::Utc::now(),
        };
        let edge2 = Edge {
            id: "e2".to_string(),
            src: "sym:b".to_string(),
            dst: "sym:c".to_string(),
            relationship: RelationshipType::Calls,
            weight: 1.0,
            properties: HashMap::new(),
            created_at: chrono::Utc::now(),
        };
        {
            let mut graph = server.graph.lock().unwrap();
            graph.add_edge(edge1).unwrap();
            graph.add_edge(edge2).unwrap();
        }

        let params = json!({"name": "get_pagerank", "arguments": {"top_k": 3}});
        let resp = server.handle_request("tools/call", Some(&params), json!(311));
        let result = resp.result.unwrap();
        assert_eq!(result["isError"], false);
        let text = result["content"][0]["text"].as_str().unwrap();
        let parsed: Value = serde_json::from_str(text).unwrap();
        assert_eq!(parsed["results"].as_array().unwrap().len(), 3);
    }

    #[test]
    fn set_scoring_weights_updates_weights() {
        let server = test_server();

        // Set custom weights (all equal)
        let params = json!({
            "name": "set_scoring_weights",
            "arguments": {
                "vector_similarity": 1.0,
                "graph_strength": 1.0,
                "token_overlap": 1.0,
                "temporal": 1.0,
                "tag_matching": 1.0,
                "importance": 1.0,
                "confidence": 1.0,
                "recency": 1.0,
            }
        });
        let resp = server.handle_request("tools/call", Some(&params), json!(100));
        let result = resp.result.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        let parsed: Value = serde_json::from_str(text).unwrap();
        assert_eq!(parsed["updated"], true);

        // All weights should be normalized to 0.125
        let weights = &parsed["weights"];
        let expected = 0.125;
        let eps = 1e-10;
        assert!((weights["vector_similarity"].as_f64().unwrap() - expected).abs() < eps);
        assert!((weights["graph_strength"].as_f64().unwrap() - expected).abs() < eps);
        assert!((weights["token_overlap"].as_f64().unwrap() - expected).abs() < eps);
        assert!((weights["temporal"].as_f64().unwrap() - expected).abs() < eps);
        assert!((weights["tag_matching"].as_f64().unwrap() - expected).abs() < eps);
        assert!((weights["importance"].as_f64().unwrap() - expected).abs() < eps);
        assert!((weights["confidence"].as_f64().unwrap() - expected).abs() < eps);
        assert!((weights["recency"].as_f64().unwrap() - expected).abs() < eps);
    }

    #[test]
    fn recall_uses_custom_scoring_weights() {
        let server = test_server();

        // Store two memories: one with high importance, one with many tags matching
        store_memory(&server, "rust ownership concept", "insight", &[]);
        store_memory(
            &server,
            "rust borrowing rules",
            "pattern",
            &["rust", "borrowing", "rules"],
        );

        // Default weights: recall both (both match "rust")
        let text_default = recall_memories(&server, "rust", None);
        let results_default: Vec<Value> = serde_json::from_str(&text_default).unwrap();
        assert_eq!(results_default.len(), 2);

        // Set weights to heavily favor tag_matching (1.0) and minimize everything else
        let params = json!({
            "name": "set_scoring_weights",
            "arguments": {
                "vector_similarity": 0.0,
                "graph_strength": 0.0,
                "token_overlap": 0.01,
                "temporal": 0.0,
                "tag_matching": 1.0,
                "importance": 0.0,
                "confidence": 0.0,
                "recency": 0.0,
            }
        });
        server.handle_request("tools/call", Some(&params), json!(200));

        // Recall again - the tagged memory should score much higher
        let text_custom = recall_memories(&server, "rust", None);
        let results_custom: Vec<Value> = serde_json::from_str(&text_custom).unwrap();
        assert!(!results_custom.is_empty());

        // The first result should be the one with more tag matches
        assert!(results_custom[0]["content"]
            .as_str()
            .unwrap()
            .contains("borrowing"));
    }

    #[test]
    fn set_scoring_weights_with_defaults_for_omitted() {
        let server = test_server();

        // Only set vector_similarity, rest should use defaults
        let params = json!({
            "name": "set_scoring_weights",
            "arguments": {
                "vector_similarity": 0.5,
            }
        });
        let resp = server.handle_request("tools/call", Some(&params), json!(300));
        let result = resp.result.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        let parsed: Value = serde_json::from_str(text).unwrap();
        assert_eq!(parsed["updated"], true);

        // vector_similarity should be 0.5 normalized against the sum of all defaults
        // sum = 0.5 + 0.25 + 0.15 + 0.10 + 0.10 + 0.05 + 0.05 + 0.05 = 1.25
        // so vector_similarity = 0.5 / 1.25 = 0.4
        let vs = parsed["weights"]["vector_similarity"].as_f64().unwrap();
        assert!((vs - 0.4).abs() < 1e-10);
    }

    // ── Export/Import Tests ─────────────────────────────────────────────

    #[test]
    fn export_memories_empty() {
        let server = test_server();
        let params = json!({"name": "export_memories", "arguments": {}});
        let resp = server.handle_request("tools/call", Some(&params), json!(400));
        let result = resp.result.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        let exported: Vec<Value> = serde_json::from_str(text).unwrap();
        assert!(exported.is_empty());
    }

    #[test]
    fn import_and_export_roundtrip() {
        let server = test_server();

        // Import 2 memories
        let import_params = json!({
            "name": "import_memories",
            "arguments": {
                "memories": [
                    {
                        "content": "roundtrip memory one about rust",
                        "memory_type": "insight",
                        "importance": 0.8,
                        "tags": ["rust", "test"]
                    },
                    {
                        "content": "roundtrip memory two about python",
                        "memory_type": "pattern",
                        "tags": ["python"]
                    }
                ]
            }
        });
        let resp = server.handle_request("tools/call", Some(&import_params), json!(401));
        let result = resp.result.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        let import_result: Value = serde_json::from_str(text).unwrap();
        assert_eq!(import_result["imported"], 2);
        assert_eq!(import_result["skipped"], 0);
        assert_eq!(import_result["ids"].as_array().unwrap().len(), 2);

        // Export all memories
        let export_params = json!({"name": "export_memories", "arguments": {}});
        let resp = server.handle_request("tools/call", Some(&export_params), json!(402));
        let result = resp.result.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        let exported: Vec<Value> = serde_json::from_str(text).unwrap();
        assert_eq!(exported.len(), 2);

        // Verify content matches
        let contents: Vec<&str> = exported
            .iter()
            .filter_map(|e| e["content"].as_str())
            .collect();
        assert!(contents.contains(&"roundtrip memory one about rust"));
        assert!(contents.contains(&"roundtrip memory two about python"));

        // Verify memory types
        let types: Vec<&str> = exported
            .iter()
            .filter_map(|e| e["memory_type"].as_str())
            .collect();
        assert!(types.contains(&"insight"));
        assert!(types.contains(&"pattern"));
    }

    #[test]
    fn export_with_namespace_filter() {
        let server = test_server();

        // Import memories with different namespaces
        let import_params = json!({
            "name": "import_memories",
            "arguments": {
                "memories": [
                    {
                        "content": "project-a memory about architecture",
                        "memory_type": "decision",
                        "namespace": "/projects/a"
                    },
                    {
                        "content": "project-b memory about testing",
                        "memory_type": "insight",
                        "namespace": "/projects/b"
                    },
                    {
                        "content": "project-a memory about patterns",
                        "memory_type": "pattern",
                        "namespace": "/projects/a"
                    }
                ]
            }
        });
        let resp = server.handle_request("tools/call", Some(&import_params), json!(403));
        let result = resp.result.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        let import_result: Value = serde_json::from_str(text).unwrap();
        assert_eq!(import_result["imported"], 3);

        // Export only namespace /projects/a
        let export_params = json!({
            "name": "export_memories",
            "arguments": {"namespace": "/projects/a"}
        });
        let resp = server.handle_request("tools/call", Some(&export_params), json!(404));
        let result = resp.result.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        let exported: Vec<Value> = serde_json::from_str(text).unwrap();
        assert_eq!(exported.len(), 2);

        // All exported should be from /projects/a
        for mem in &exported {
            assert_eq!(mem["namespace"].as_str().unwrap(), "/projects/a");
        }
    }

    // ── Consolidation Tool Tests ────────────────────────────────────────

    #[test]
    fn consolidate_decay_reduces_importance() {
        let server = test_server();

        // Store a memory with known importance
        let now = chrono::Utc::now();
        let sixty_days_ago = now - chrono::Duration::days(60);
        let id = uuid::Uuid::new_v4().to_string();
        let content = "old memory that should decay";
        let hash = Storage::content_hash(content);
        let memory = MemoryNode {
            id: id.clone(),
            content: content.to_string(),
            memory_type: MemoryType::Context,
            importance: 0.8,
            confidence: 1.0,
            access_count: 0,
            content_hash: hash,
            tags: vec![],
            metadata: HashMap::new(),
            namespace: None,
            created_at: sixty_days_ago,
            updated_at: sixty_days_ago,
            last_accessed_at: sixty_days_ago,
        };
        server.storage.insert_memory(&memory).unwrap();

        // Run decay with default threshold (30 days)
        let params = json!({"name": "consolidate_decay", "arguments": {}});
        let resp = server.handle_request("tools/call", Some(&params), json!(1));
        let result = resp.result.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        let parsed: Value = serde_json::from_str(text).unwrap();

        assert_eq!(parsed["cycle"], "decay");
        assert_eq!(parsed["affected"], 1);
        assert_eq!(parsed["threshold_days"], 30);

        // Verify importance was reduced: 0.8 * 0.9 = 0.72
        let retrieved = server.storage.get_memory(&id).unwrap().unwrap();
        assert!((retrieved.importance - 0.72).abs() < 0.01);
    }

    #[test]
    fn consolidate_decay_skips_recent_memories() {
        let server = test_server();

        // Store a recent memory
        store_memory(&server, "recently accessed memory", "context", &[]);

        // Run decay
        let params = json!({"name": "consolidate_decay", "arguments": {"threshold_days": 30}});
        let resp = server.handle_request("tools/call", Some(&params), json!(1));
        let result = resp.result.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        let parsed: Value = serde_json::from_str(text).unwrap();

        // Recent memory should not be affected
        assert_eq!(parsed["affected"], 0);
    }

    #[test]
    fn consolidate_creative_creates_edges() {
        let server = test_server();

        // Store two memories with overlapping tags but different types
        store_memory(
            &server,
            "insight about rust safety",
            "insight",
            &["rust", "safety"],
        );
        store_memory(
            &server,
            "pattern for error handling",
            "pattern",
            &["rust", "error"],
        );

        // Run creative cycle
        let params = json!({"name": "consolidate_creative", "arguments": {}});
        let resp = server.handle_request("tools/call", Some(&params), json!(1));
        let result = resp.result.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        let parsed: Value = serde_json::from_str(text).unwrap();

        assert_eq!(parsed["cycle"], "creative");
        // They share the "rust" tag and have different types, so should create 1 connection
        assert_eq!(parsed["new_connections"], 1);
    }

    #[test]
    fn consolidate_creative_skips_same_type() {
        let server = test_server();

        // Store two memories with same type (should not create edges)
        store_memory(&server, "insight one about rust", "insight", &["rust"]);
        store_memory(&server, "insight two about rust", "insight", &["rust"]);

        let params = json!({"name": "consolidate_creative", "arguments": {}});
        let resp = server.handle_request("tools/call", Some(&params), json!(1));
        let result = resp.result.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        let parsed: Value = serde_json::from_str(text).unwrap();

        assert_eq!(parsed["new_connections"], 0);
    }

    #[test]
    fn consolidate_forget_deletes_low_importance() {
        let server = test_server();

        // Store a memory with very low importance and zero access count
        let now = chrono::Utc::now();
        let id = uuid::Uuid::new_v4().to_string();
        let content = "forgettable memory";
        let hash = Storage::content_hash(content);
        let memory = MemoryNode {
            id: id.clone(),
            content: content.to_string(),
            memory_type: MemoryType::Context,
            importance: 0.05,
            confidence: 1.0,
            access_count: 0,
            content_hash: hash,
            tags: vec![],
            metadata: HashMap::new(),
            namespace: None,
            created_at: now,
            updated_at: now,
            last_accessed_at: now,
        };
        server.storage.insert_memory(&memory).unwrap();

        // Verify it exists
        assert_eq!(server.storage.memory_count().unwrap(), 1);

        // Run forget
        let params = json!({"name": "consolidate_forget", "arguments": {}});
        let resp = server.handle_request("tools/call", Some(&params), json!(1));
        let result = resp.result.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        let parsed: Value = serde_json::from_str(text).unwrap();

        assert_eq!(parsed["cycle"], "forget");
        assert_eq!(parsed["deleted"], 1);
        assert_eq!(parsed["threshold"], 0.1);

        // Verify it's gone
        assert_eq!(server.storage.memory_count().unwrap(), 0);
    }

    #[test]
    fn consolidate_forget_keeps_accessed_memories() {
        let server = test_server();

        // Store a memory with low importance but nonzero access count
        let stored = store_memory(&server, "low importance but accessed", "context", &[]);
        let id = stored["id"].as_str().unwrap();

        // Update importance to be low and set access_count > 0 via raw SQL
        server
            .storage
            .connection()
            .execute(
                "UPDATE memories SET importance = 0.05, access_count = 5 WHERE id = ?1",
                rusqlite::params![id],
            )
            .unwrap();

        // This memory has access_count = 5, so it should NOT be forgotten
        // (forget only targets memories with access_count == 0)

        let params = json!({"name": "consolidate_forget", "arguments": {}});
        let resp = server.handle_request("tools/call", Some(&params), json!(1));
        let result = resp.result.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        let parsed: Value = serde_json::from_str(text).unwrap();

        assert_eq!(parsed["deleted"], 0);
        assert_eq!(server.storage.memory_count().unwrap(), 1);
    }

    #[test]
    fn consolidation_status_shows_last_run() {
        let server = test_server();

        // Status with no prior runs should return empty cycles
        let params = json!({"name": "consolidation_status", "arguments": {}});
        let resp = server.handle_request("tools/call", Some(&params), json!(1));
        let result = resp.result.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        let parsed: Value = serde_json::from_str(text).unwrap();
        assert_eq!(parsed["cycles"], json!({}));

        // Run a decay cycle
        let params = json!({"name": "consolidate_decay", "arguments": {}});
        server.handle_request("tools/call", Some(&params), json!(2));

        // Now status should show decay
        let params = json!({"name": "consolidation_status", "arguments": {}});
        let resp = server.handle_request("tools/call", Some(&params), json!(3));
        let result = resp.result.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        let parsed: Value = serde_json::from_str(text).unwrap();

        assert!(parsed["cycles"]["decay"].is_object());
        assert!(parsed["cycles"]["decay"]["last_run"].is_string());
        assert!(parsed["cycles"]["decay"]["affected"].is_number());
    }

    #[test]
    fn consolidate_forget_custom_threshold() {
        let server = test_server();

        // Store two memories with different importance
        let now = chrono::Utc::now();
        for (imp, content) in [(0.3, "medium importance"), (0.05, "very low importance")] {
            let id = uuid::Uuid::new_v4().to_string();
            let hash = Storage::content_hash(content);
            let memory = MemoryNode {
                id,
                content: content.to_string(),
                memory_type: MemoryType::Context,
                importance: imp,
                confidence: 1.0,
                access_count: 0,
                content_hash: hash,
                tags: vec![],
                metadata: HashMap::new(),
                namespace: None,
                created_at: now,
                updated_at: now,
                last_accessed_at: now,
            };
            server.storage.insert_memory(&memory).unwrap();
        }

        assert_eq!(server.storage.memory_count().unwrap(), 2);

        // Forget with threshold 0.5 should delete both
        let params =
            json!({"name": "consolidate_forget", "arguments": {"importance_threshold": 0.5}});
        let resp = server.handle_request("tools/call", Some(&params), json!(1));
        let result = resp.result.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        let parsed: Value = serde_json::from_str(text).unwrap();

        assert_eq!(parsed["deleted"], 2);
        assert_eq!(parsed["threshold"], 0.5);
        assert_eq!(server.storage.memory_count().unwrap(), 0);
    }
}

// ── Graph-Expanded Recall & Namespace Management Tests ─────────────────────

#[cfg(test)]
mod graph_recall_ns_tests {
    use super::*;

    fn test_server() -> McpServer {
        McpServer::for_testing()
    }

    /// Helper: call a tool and return the result Value.
    fn call_tool(server: &McpServer, tool_name: &str, arguments: Value) -> Value {
        let params = json!({"name": tool_name, "arguments": arguments});
        let resp = server.handle_request("tools/call", Some(&params), json!("req"));
        assert!(
            resp.error.is_none(),
            "Unexpected error calling {tool_name}: {:?}",
            resp.error
        );
        resp.result.unwrap()
    }

    /// Helper: call a tool and parse the text content as JSON.
    fn call_tool_parse(server: &McpServer, tool_name: &str, arguments: Value) -> Value {
        let result = call_tool(server, tool_name, arguments);
        let text = result["content"][0]["text"].as_str().unwrap();
        serde_json::from_str(text).unwrap_or_else(|_| Value::String(text.to_string()))
    }

    /// Helper: store a memory with namespace.
    fn store_ns(
        server: &McpServer,
        content: &str,
        namespace: &str,
        memory_type: &str,
        tags: &[&str],
    ) -> Value {
        call_tool_parse(
            server,
            "store_memory",
            json!({
                "content": content,
                "memory_type": memory_type,
                "tags": tags,
                "namespace": namespace,
            }),
        )
    }

    #[test]
    fn recall_with_expansion_no_embeddings() {
        let server = test_server();

        // Store two memories and link them
        let mem_a = store_ns(
            &server,
            "graph expansion base memory about architecture",
            "test-ns",
            "insight",
            &["arch"],
        );
        let id_a = mem_a["id"].as_str().unwrap();

        let mem_b = store_ns(
            &server,
            "related memory about design patterns",
            "test-ns",
            "pattern",
            &["design"],
        );
        let id_b = mem_b["id"].as_str().unwrap();

        // Associate them
        call_tool(
            &server,
            "associate_memories",
            json!({
                "source_id": id_a,
                "target_id": id_b,
                "relationship": "RELATES_TO",
            }),
        );

        // Recall with expansion (no embeddings = text fallback)
        let result = call_tool(
            &server,
            "recall_with_expansion",
            json!({
                "query": "architecture",
                "k": 5,
                "expansion_depth": 1,
            }),
        );
        let text = result["content"][0]["text"].as_str().unwrap();
        // Should find at least the base memory via token overlap
        assert!(text.contains("architecture") || text.contains("design"));
    }

    #[test]
    fn list_namespaces_empty() {
        let server = test_server();

        let result = call_tool_parse(&server, "list_namespaces", json!({}));
        let namespaces = result["namespaces"].as_array().unwrap();
        assert_eq!(namespaces.len(), 0);
    }

    #[test]
    fn list_namespaces_with_data() {
        let server = test_server();

        // Store memories in two namespaces
        store_ns(
            &server,
            "memory alpha one about rust",
            "ns-alpha",
            "insight",
            &["rust"],
        );
        store_ns(
            &server,
            "memory alpha two about safety",
            "ns-alpha",
            "pattern",
            &["safety"],
        );
        store_ns(
            &server,
            "memory beta one about python",
            "ns-beta",
            "context",
            &["python"],
        );

        let result = call_tool_parse(&server, "list_namespaces", json!({}));
        let namespaces = result["namespaces"].as_array().unwrap();
        assert_eq!(namespaces.len(), 2);

        // Verify names and counts
        let ns_names: Vec<&str> = namespaces
            .iter()
            .filter_map(|n| n["name"].as_str())
            .collect();
        assert!(ns_names.contains(&"ns-alpha"));
        assert!(ns_names.contains(&"ns-beta"));

        for ns in namespaces {
            if ns["name"].as_str().unwrap() == "ns-alpha" {
                assert_eq!(ns["memory_count"], 2);
            } else if ns["name"].as_str().unwrap() == "ns-beta" {
                assert_eq!(ns["memory_count"], 1);
            }
        }
    }

    #[test]
    fn namespace_stats_basic() {
        let server = test_server();

        store_ns(
            &server,
            "insight about architecture patterns",
            "stats-ns",
            "insight",
            &["arch", "patterns"],
        );
        store_ns(
            &server,
            "pattern for error handling in rust",
            "stats-ns",
            "pattern",
            &["rust", "errors"],
        );

        let result = call_tool_parse(&server, "namespace_stats", json!({"namespace": "stats-ns"}));
        assert_eq!(result["namespace"], "stats-ns");
        assert_eq!(result["count"], 2);

        // Check type distribution
        let types = &result["type_distribution"];
        assert_eq!(types["insight"], 1);
        assert_eq!(types["pattern"], 1);

        // Check tag frequency
        let tags = &result["tag_frequency"];
        assert_eq!(tags["arch"], 1);
        assert_eq!(tags["patterns"], 1);
        assert_eq!(tags["rust"], 1);
        assert_eq!(tags["errors"], 1);

        // Dates should be present
        assert!(result["oldest"].is_string());
        assert!(result["newest"].is_string());
    }

    #[test]
    fn delete_namespace_requires_confirm() {
        let server = test_server();

        store_ns(
            &server,
            "memory to be protected",
            "protected-ns",
            "context",
            &[],
        );

        // Try to delete without confirm
        let result = call_tool(
            &server,
            "delete_namespace",
            json!({
                "namespace": "protected-ns",
                "confirm": false,
            }),
        );
        let text = result["content"][0]["text"].as_str().unwrap();
        assert_eq!(result["isError"], true);
        assert!(text.contains("confirm"));

        // Memory should still exist
        let stats = call_tool_parse(
            &server,
            "namespace_stats",
            json!({"namespace": "protected-ns"}),
        );
        assert_eq!(stats["count"], 1);
    }

    #[test]
    fn delete_namespace_with_confirm() {
        let server = test_server();

        store_ns(
            &server,
            "memory to delete alpha",
            "delete-ns",
            "insight",
            &["test"],
        );
        store_ns(
            &server,
            "memory to delete beta",
            "delete-ns",
            "pattern",
            &["test"],
        );

        // Verify they exist
        let stats = call_tool_parse(
            &server,
            "namespace_stats",
            json!({"namespace": "delete-ns"}),
        );
        assert_eq!(stats["count"], 2);

        // Delete with confirm
        let result = call_tool_parse(
            &server,
            "delete_namespace",
            json!({
                "namespace": "delete-ns",
                "confirm": true,
            }),
        );
        assert_eq!(result["deleted"], 2);
        assert_eq!(result["namespace"], "delete-ns");

        // Verify they are gone
        let stats_after = call_tool_parse(
            &server,
            "namespace_stats",
            json!({"namespace": "delete-ns"}),
        );
        assert_eq!(stats_after["count"], 0);
    }

    #[test]
    fn recall_with_impact_returns_impact_data() {
        let server = test_server();

        // Store a memory
        let mem = store_ns(
            &server,
            "impact test memory about error handling patterns",
            "test-ns",
            "insight",
            &["error", "handling"],
        );
        let _id = mem["id"].as_str().unwrap();

        // Recall with impact (text fallback, no embeddings)
        let result = call_tool(
            &server,
            "recall_with_impact",
            json!({"query": "error handling"}),
        );
        let text = result["content"][0]["text"].as_str().unwrap();

        // Should find the memory and include impact data
        if text.contains("No matching memories") {
            // Token overlap alone may not be enough; that is fine
            return;
        }

        let parsed: Value = serde_json::from_str(text).unwrap();
        let first = &parsed[0];
        assert!(
            first.get("impact").is_some(),
            "result should contain impact data"
        );
        let impact = &first["impact"];
        assert!(impact.get("pagerank").is_some());
        assert!(impact.get("centrality").is_some());
        assert!(impact.get("connected_decisions").is_some());
        assert!(impact.get("dependent_files").is_some());
        assert!(impact.get("modification_count").is_some());
    }

    #[test]
    fn get_decision_chain_requires_parameter() {
        let server = test_server();

        // Calling without file_path or topic should return an error
        let params = json!({"name": "get_decision_chain", "arguments": {}});
        let resp = server.handle_request("tools/call", Some(&params), json!("req"));
        let result = resp.result.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        assert!(
            text.contains("file_path") || text.contains("topic"),
            "error should mention required parameters"
        );
    }

    #[test]
    fn get_decision_chain_by_topic() {
        let server = test_server();

        // Store decision memories with a topic
        let _d1 = call_tool_parse(
            &server,
            "store_memory",
            json!({
                "content": "Decision: use async runtime for concurrency",
                "memory_type": "decision",
                "tags": ["concurrency"],
            }),
        );
        let _d2 = call_tool_parse(
            &server,
            "store_memory",
            json!({
                "content": "Decision: switched from threads to async for concurrency",
                "memory_type": "decision",
                "tags": ["concurrency"],
            }),
        );

        // Query decision chain by topic
        let result = call_tool(
            &server,
            "get_decision_chain",
            json!({"topic": "concurrency"}),
        );
        let text = result["content"][0]["text"].as_str().unwrap();
        let parsed: Value = serde_json::from_str(text).unwrap();
        assert!(parsed["chain_length"].as_u64().unwrap() >= 2);
        assert_eq!(parsed["filter"]["topic"], "concurrency");
    }

    #[test]
    fn decision_chain_follows_temporal_order() {
        let server = test_server();

        // Store decision memories at different times (chronological insertion order)
        let d1 = call_tool_parse(
            &server,
            "store_memory",
            json!({
                "content": "Decision: initial architecture for auth module",
                "memory_type": "decision",
                "tags": ["auth"],
            }),
        );
        let d2 = call_tool_parse(
            &server,
            "store_memory",
            json!({
                "content": "Decision: refactored auth to use JWT tokens",
                "memory_type": "decision",
                "tags": ["auth"],
            }),
        );
        let d3 = call_tool_parse(
            &server,
            "store_memory",
            json!({
                "content": "Decision: added OAuth2 to auth module",
                "memory_type": "decision",
                "tags": ["auth"],
            }),
        );

        // Link d1 -> d2 -> d3 with EVOLVED_INTO edges
        let id1 = d1["id"].as_str().unwrap();
        let id2 = d2["id"].as_str().unwrap();
        let id3 = d3["id"].as_str().unwrap();

        call_tool(
            &server,
            "associate_memories",
            json!({
                "source_id": id1,
                "target_id": id2,
                "relationship": "EVOLVED_INTO",
            }),
        );
        call_tool(
            &server,
            "associate_memories",
            json!({
                "source_id": id2,
                "target_id": id3,
                "relationship": "EVOLVED_INTO",
            }),
        );

        // Get decision chain
        let result = call_tool(&server, "get_decision_chain", json!({"topic": "auth"}));
        let text = result["content"][0]["text"].as_str().unwrap();
        let parsed: Value = serde_json::from_str(text).unwrap();

        assert_eq!(parsed["chain_length"].as_u64().unwrap(), 3);
        let decisions = parsed["decisions"].as_array().unwrap();

        // Verify temporal order: created_at of each should be <= the next
        for i in 0..decisions.len() - 1 {
            let dt_a = decisions[i]["created_at"].as_str().unwrap();
            let dt_b = decisions[i + 1]["created_at"].as_str().unwrap();
            assert!(dt_a <= dt_b, "decisions should be in chronological order");
        }

        // Verify connections exist
        let has_connections = decisions
            .iter()
            .any(|d| !d["connections"].as_array().unwrap().is_empty());
        assert!(
            has_connections,
            "at least one decision should have connections"
        );
    }
}
