//! Graph & analysis tools: traverse, stats, health, index, symbols, deps, impact,
//! clusters, cross-repo, pagerank, search-code, scoring weights, metrics.

use crate::types::{IndexCache, ToolResult};
use crate::McpServer;
use codemem_core::{
    Edge, GraphBackend, GraphConfig, GraphNode, NodeKind, RelationshipType, ScoringWeights,
    VectorBackend,
};
use serde_json::{json, Value};
use std::collections::HashMap;

/// Return the edge weight for a given relationship type, using config overrides
/// for the three most common types (Contains, Calls, Imports).
fn edge_weight_for(rel: &RelationshipType, config: &GraphConfig) -> f64 {
    match rel {
        RelationshipType::Calls => config.calls_edge_weight,
        RelationshipType::Imports => config.imports_edge_weight,
        RelationshipType::Contains => config.contains_edge_weight,
        RelationshipType::Implements | RelationshipType::Inherits => 0.8,
        RelationshipType::DependsOn => 0.7,
        RelationshipType::CoChanged => 0.6,
        RelationshipType::EvolvedInto | RelationshipType::Summarizes => 0.7,
        RelationshipType::PartOf => 0.4,
        RelationshipType::RelatesTo | RelationshipType::SharesTheme => 0.3,
        _ => 0.5,
    }
}

impl McpServer {
    pub(crate) fn tool_graph_traverse(&self, args: &Value) -> ToolResult {
        let start = match args.get("start_id").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => return ToolResult::tool_error("Missing 'start_id' parameter"),
        };
        let depth = args.get("max_depth").and_then(|v| v.as_u64()).unwrap_or(2) as usize;
        let algorithm = args
            .get("algorithm")
            .and_then(|v| v.as_str())
            .unwrap_or("bfs");

        // Parse optional exclude_kinds filter
        let exclude_kinds: Vec<NodeKind> = args
            .get("exclude_kinds")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str()?.parse::<NodeKind>().ok())
                    .collect()
            })
            .unwrap_or_default();

        // Parse optional include_relationships filter
        let include_relationships: Option<Vec<RelationshipType>> = args
            .get("include_relationships")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str()?.parse::<RelationshipType>().ok())
                    .collect()
            });

        let graph = match self.lock_graph() {
            Ok(g) => g,
            Err(e) => return ToolResult::tool_error(format!("Lock error: {e}")),
        };

        let has_filters = !exclude_kinds.is_empty() || include_relationships.is_some();

        let nodes = if has_filters {
            match algorithm {
                "bfs" => graph.bfs_filtered(
                    start,
                    depth,
                    &exclude_kinds,
                    include_relationships.as_deref(),
                ),
                "dfs" => graph.dfs_filtered(
                    start,
                    depth,
                    &exclude_kinds,
                    include_relationships.as_deref(),
                ),
                _ => return ToolResult::tool_error(format!("Unknown algorithm: {algorithm}")),
            }
        } else {
            match algorithm {
                "bfs" => graph.bfs(start, depth),
                "dfs" => graph.dfs(start, depth),
                _ => return ToolResult::tool_error(format!("Unknown algorithm: {algorithm}")),
            }
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
                ToolResult::text(
                    serde_json::to_string_pretty(&output).expect("JSON serialization of literal"),
                )
            }
            Err(e) => ToolResult::tool_error(format!("Traversal failed: {e}")),
        }
    }

    pub(crate) fn tool_stats(&self) -> ToolResult {
        let storage_stats = match self.storage.stats() {
            Ok(s) => s,
            Err(e) => return ToolResult::tool_error(format!("Stats error: {e}")),
        };

        let vector_stats = match self.lock_vector() {
            Ok(v) => v.stats(),
            Err(e) => return ToolResult::tool_error(format!("Lock error: {e}")),
        };
        let graph_stats = match self.lock_graph() {
            Ok(g) => g.stats(),
            Err(e) => return ToolResult::tool_error(format!("Lock error: {e}")),
        };

        let cache_info = match self.lock_embeddings() {
            Ok(Some(emb)) => {
                let (size, cap) = emb.cache_stats();
                Some(json!({"size": size, "capacity": cap}))
            }
            Ok(None) => None,
            Err(e) => return ToolResult::tool_error(format!("Lock error: {e}")),
        };

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
                    "cache": cache_info,
                }
            }))
            .expect("JSON serialization of literal"),
        )
    }

    pub(crate) fn tool_health(&self) -> ToolResult {
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
            .expect("JSON serialization of literal"),
        )
    }

    // ── Structural Index Tools ──────────────────────────────────────────────

    pub(crate) fn tool_index_codebase(&self, args: &Value) -> ToolResult {
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

        // Collect all symbols, references, chunks, and unique file paths
        let mut all_symbols = Vec::new();
        let mut all_references = Vec::new();
        let mut all_chunks = Vec::new();
        let mut seen_files = std::collections::HashSet::new();
        for pr in &result.parse_results {
            all_symbols.extend(pr.symbols.clone());
            all_references.extend(pr.references.clone());
            all_chunks.extend(pr.chunks.clone());
            seen_files.insert(pr.file_path.clone());
        }

        // Resolve references
        let mut resolver = codemem_index::ReferenceResolver::new();
        resolver.add_symbols(&all_symbols);
        let edges = resolver.resolve_all(&all_references);

        let now = chrono::Utc::now();

        // Persist file nodes, then symbols as graph nodes
        let mut graph = match self.lock_graph() {
            Ok(g) => g,
            Err(e) => return ToolResult::tool_error(format!("Lock error: {e}")),
        };

        // Create file:* nodes for each unique file path
        for file_path in &seen_files {
            let node_id = format!("file:{file_path}");
            let mut payload = HashMap::new();
            payload.insert(
                "file_path".to_string(),
                serde_json::Value::String(file_path.clone()),
            );
            let node = GraphNode {
                id: node_id,
                kind: NodeKind::File,
                label: file_path.clone(),
                payload,
                centrality: 0.0,
                memory_id: None,
                namespace: Some(path.to_string()),
            };
            let _ = self.storage.insert_graph_node(&node);
            let _ = graph.add_node(node);
        }

        // Create pkg:* directory nodes forming a tree above file nodes
        let mut created_dirs: std::collections::HashSet<String> = std::collections::HashSet::new();
        for file_path in &seen_files {
            let p = std::path::Path::new(file_path);
            // Walk parent directories from immediate parent up to the root
            let mut ancestors: Vec<String> = Vec::new();
            let mut current = p.parent();
            while let Some(dir) = current {
                let dir_str = dir.to_string_lossy().to_string();
                if dir_str.is_empty() || dir_str == "." {
                    break;
                }
                ancestors.push(dir_str);
                current = dir.parent();
            }
            // Create directory nodes from outermost to innermost
            ancestors.reverse();
            for (i, dir_str) in ancestors.iter().enumerate() {
                let pkg_id = format!("pkg:{dir_str}/");
                if created_dirs.insert(pkg_id.clone()) {
                    let node = GraphNode {
                        id: pkg_id.clone(),
                        kind: NodeKind::Package,
                        label: format!("{dir_str}/"),
                        payload: HashMap::new(),
                        centrality: 0.0,
                        memory_id: None,
                        namespace: Some(path.to_string()),
                    };
                    let _ = self.storage.insert_graph_node(&node);
                    let _ = graph.add_node(node);
                }
                // Create CONTAINS edge from parent dir to this dir
                if i > 0 {
                    let parent_pkg_id = format!("pkg:{}/", ancestors[i - 1]);
                    let edge_id = format!("contains:{parent_pkg_id}->{pkg_id}");
                    if !graph
                        .get_edges(&parent_pkg_id)
                        .unwrap_or_default()
                        .iter()
                        .any(|e| e.id == edge_id)
                    {
                        let edge = Edge {
                            id: edge_id,
                            src: parent_pkg_id,
                            dst: pkg_id.clone(),
                            relationship: codemem_core::RelationshipType::Contains,
                            weight: edge_weight_for(
                                &RelationshipType::Contains,
                                &self.config.graph,
                            ),
                            valid_from: None,
                            valid_to: None,
                            properties: HashMap::new(),
                            created_at: now,
                        };
                        let _ = self.storage.insert_graph_edge(&edge);
                        let _ = graph.add_edge(edge);
                    }
                }
            }
            // Create CONTAINS edge from innermost directory to this file
            if let Some(last_dir) = ancestors.last() {
                let parent_pkg_id = format!("pkg:{last_dir}/");
                let file_node_id = format!("file:{file_path}");
                let edge_id = format!("contains:{parent_pkg_id}->{file_node_id}");
                let edge = Edge {
                    id: edge_id,
                    src: parent_pkg_id,
                    dst: file_node_id,
                    relationship: codemem_core::RelationshipType::Contains,
                    weight: edge_weight_for(&RelationshipType::Contains, &self.config.graph),
                    valid_from: None,
                    valid_to: None,
                    properties: HashMap::new(),
                    created_at: now,
                };
                let _ = self.storage.insert_graph_edge(&edge);
                let _ = graph.add_edge(edge);
            }
        }

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
                label: sym.qualified_name.clone(),
                payload,
                centrality: 0.0,
                memory_id: None,
                namespace: Some(path.to_string()),
            };

            let sym_node_id = node.id.clone();
            let _ = self.storage.insert_graph_node(&node);
            let _ = graph.add_node(node);

            // Add CONTAINS edge: file:{path} → sym:{qualified_name}
            let file_node_id = format!("file:{}", sym.file_path);
            let contains_edge = Edge {
                id: format!("contains:{file_node_id}->{sym_node_id}"),
                src: file_node_id,
                dst: sym_node_id,
                relationship: codemem_core::RelationshipType::Contains,
                weight: edge_weight_for(&RelationshipType::Contains, &self.config.graph),
                valid_from: None,
                valid_to: None,
                properties: HashMap::new(),
                created_at: now,
            };
            let _ = self.storage.insert_graph_edge(&contains_edge);
            let _ = graph.add_edge(contains_edge);
        }

        // Persist edges
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
                weight: edge_weight_for(&edge.relationship, &self.config.graph),
                valid_from: None,
                valid_to: None,
                properties: HashMap::new(),
                created_at: now,
            };
            let _ = self.storage.insert_graph_edge(&e);
            let _ = graph.add_edge(e);
        }

        // Cleanup stale chunk nodes for files being re-indexed, then persist new chunks
        let mut chunk_count = 0usize;
        {
            for file_path in &seen_files {
                let prefix = format!("chunk:{file_path}:");
                let _ = self.storage.delete_graph_nodes_by_prefix(&prefix);
            }

            // Persist chunk nodes and CONTAINS edges
            for chunk in &all_chunks {
                let chunk_id = format!("chunk:{}:{}", chunk.file_path, chunk.index);

                let mut payload = HashMap::new();
                payload.insert(
                    "file_path".to_string(),
                    serde_json::Value::String(chunk.file_path.clone()),
                );
                payload.insert(
                    "line_start".to_string(),
                    serde_json::json!(chunk.line_start),
                );
                payload.insert("line_end".to_string(), serde_json::json!(chunk.line_end));
                payload.insert(
                    "node_kind".to_string(),
                    serde_json::Value::String(chunk.node_kind.clone()),
                );
                payload.insert(
                    "non_ws_chars".to_string(),
                    serde_json::json!(chunk.non_ws_chars),
                );
                if let Some(ref parent) = chunk.parent_symbol {
                    payload.insert(
                        "parent_symbol".to_string(),
                        serde_json::Value::String(parent.clone()),
                    );
                }

                let node = GraphNode {
                    id: chunk_id.clone(),
                    kind: NodeKind::Chunk,
                    label: format!(
                        "chunk:{}:{}..{}",
                        chunk.file_path, chunk.line_start, chunk.line_end
                    ),
                    payload,
                    centrality: 0.0,
                    memory_id: None,
                    namespace: Some(path.to_string()),
                };

                let _ = self.storage.insert_graph_node(&node);
                let _ = graph.add_node(node);

                // Add CONTAINS edge from file to chunk
                let file_node_id = format!("file:{}", chunk.file_path);
                let file_chunk_edge = Edge {
                    id: format!("contains:{file_node_id}->{chunk_id}"),
                    src: file_node_id,
                    dst: chunk_id.clone(),
                    relationship: codemem_core::RelationshipType::Contains,
                    weight: edge_weight_for(&RelationshipType::Contains, &self.config.graph),
                    valid_from: None,
                    valid_to: None,
                    properties: HashMap::new(),
                    created_at: now,
                };
                let _ = self.storage.insert_graph_edge(&file_chunk_edge);
                let _ = graph.add_edge(file_chunk_edge);

                // Add CONTAINS edge from parent symbol to chunk
                if let Some(ref parent_sym) = chunk.parent_symbol {
                    let parent_node_id = format!("sym:{parent_sym}");
                    let contains_edge = Edge {
                        id: format!("contains:{parent_node_id}->{chunk_id}"),
                        src: parent_node_id,
                        dst: chunk_id,
                        relationship: codemem_core::RelationshipType::Contains,
                        weight: edge_weight_for(&RelationshipType::Contains, &self.config.graph),
                        valid_from: None,
                        valid_to: None,
                        properties: HashMap::new(),
                        created_at: now,
                    };
                    let _ = self.storage.insert_graph_edge(&contains_edge);
                    let _ = graph.add_edge(contains_edge);
                }

                chunk_count += 1;
            }
        }
        drop(graph);

        // Embed symbol signatures and chunks with contextual enrichment
        let mut symbols_embedded = 0usize;
        let mut chunks_embedded = 0usize;
        if let Some(emb) = match self.lock_embeddings() {
            Ok(g) => g,
            Err(e) => return ToolResult::tool_error(format!("Lock error: {e}")),
        } {
            let mut vec = match self.lock_vector() {
                Ok(v) => v,
                Err(e) => return ToolResult::tool_error(format!("Lock error: {e}")),
            };
            for sym in &all_symbols {
                let embed_text = self.enrich_symbol_text(sym, &edges);
                let sym_id = format!("sym:{}", sym.qualified_name);
                if let Ok(embedding) = emb.embed(&embed_text) {
                    let _ = self.storage.store_embedding(&sym_id, &embedding);
                    let _ = vec.insert(&sym_id, &embedding);
                    symbols_embedded += 1;
                }
            }
            for chunk in &all_chunks {
                let embed_text = self.enrich_chunk_text(chunk);
                let chunk_id = format!("chunk:{}:{}", chunk.file_path, chunk.index);
                if let Ok(embedding) = emb.embed(&embed_text) {
                    let _ = self.storage.store_embedding(&chunk_id, &embedding);
                    let _ = vec.insert(&chunk_id, &embedding);
                    chunks_embedded += 1;
                }
            }
            drop(vec);
            drop(emb);
            self.save_index();
        }

        // Auto-compact chunk and symbol graph nodes if configured
        let (chunks_pruned, symbols_pruned) = if self.config.chunking.auto_compact {
            self.compact_graph(&seen_files)
        } else {
            (0, 0)
        };

        // Cache results
        {
            match self.lock_index_cache() {
                Ok(mut cache) => {
                    *cache = Some(IndexCache {
                        symbols: all_symbols,
                        chunks: all_chunks,
                        root_path: path.to_string(),
                    });
                }
                Err(e) => return ToolResult::tool_error(format!("Lock error: {e}")),
            }
        }

        ToolResult::text(
            serde_json::to_string_pretty(&json!({
                "files_scanned": result.files_scanned,
                "files_parsed": result.files_parsed,
                "files_skipped": result.files_skipped,
                "files_created": seen_files.len(),
                "symbols": result.total_symbols,
                "references": result.total_references,
                "edges_resolved": edges_resolved,
                "symbols_embedded": symbols_embedded,
                "chunks": chunk_count,
                "chunks_embedded": chunks_embedded,
                "chunks_pruned": chunks_pruned,
                "symbols_pruned": symbols_pruned,
                "packages_created": created_dirs.len(),
            }))
            .expect("JSON serialization of literal"),
        )
    }

    pub(crate) fn tool_search_symbols(&self, args: &Value) -> ToolResult {
        let query = match args.get("query").and_then(|v| v.as_str()) {
            Some(q) if !q.is_empty() => q,
            _ => return ToolResult::tool_error("Missing or empty 'query' parameter"),
        };

        let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(20) as usize;

        let kind_filter: Option<&str> = args.get("kind").and_then(|v| v.as_str());

        let cache = match self.lock_index_cache() {
            Ok(c) => c,
            Err(e) => return ToolResult::tool_error(format!("Lock error: {e}")),
        };
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

        ToolResult::text(
            serde_json::to_string_pretty(&matches).expect("JSON serialization of literal"),
        )
    }

    pub(crate) fn tool_get_symbol_info(&self, args: &Value) -> ToolResult {
        let qualified_name = match args.get("qualified_name").and_then(|v| v.as_str()) {
            Some(qn) => qn,
            None => return ToolResult::tool_error("Missing 'qualified_name' parameter"),
        };

        let cache = match self.lock_index_cache() {
            Ok(c) => c,
            Err(e) => return ToolResult::tool_error(format!("Lock error: {e}")),
        };
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
            .expect("JSON serialization of literal"),
        )
    }

    pub(crate) fn tool_get_dependencies(&self, args: &Value) -> ToolResult {
        let qualified_name = match args.get("qualified_name").and_then(|v| v.as_str()) {
            Some(qn) => qn,
            None => return ToolResult::tool_error("Missing 'qualified_name' parameter"),
        };

        let direction = args
            .get("direction")
            .and_then(|v| v.as_str())
            .unwrap_or("both");

        let node_id = format!("sym:{qualified_name}");
        let graph = match self.lock_graph() {
            Ok(g) => g,
            Err(e) => return ToolResult::tool_error(format!("Lock error: {e}")),
        };

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

        ToolResult::text(
            serde_json::to_string_pretty(&filtered).expect("JSON serialization of literal"),
        )
    }

    pub(crate) fn tool_get_impact(&self, args: &Value) -> ToolResult {
        let qualified_name = match args.get("qualified_name").and_then(|v| v.as_str()) {
            Some(qn) => qn,
            None => return ToolResult::tool_error("Missing 'qualified_name' parameter"),
        };

        let depth = args.get("depth").and_then(|v| v.as_u64()).unwrap_or(2) as usize;

        let node_id = format!("sym:{qualified_name}");
        let graph = match self.lock_graph() {
            Ok(g) => g,
            Err(e) => return ToolResult::tool_error(format!("Lock error: {e}")),
        };

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
            .expect("JSON serialization of literal"),
        )
    }

    pub(crate) fn tool_get_clusters(&self, args: &Value) -> ToolResult {
        let resolution = args
            .get("resolution")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0);

        let graph = match self.lock_graph() {
            Ok(g) => g,
            Err(e) => return ToolResult::tool_error(format!("Lock error: {e}")),
        };
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
            .expect("JSON serialization of literal"),
        )
    }

    pub(crate) fn tool_get_cross_repo(&self, args: &Value) -> ToolResult {
        let path = args.get("path").and_then(|v| v.as_str());

        let scan_root = match path {
            Some(p) => std::path::PathBuf::from(p),
            None => {
                let cache = match self.lock_index_cache() {
                    Ok(c) => c,
                    Err(e) => return ToolResult::tool_error(format!("Lock error: {e}")),
                };
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
            .expect("JSON serialization of literal"),
        )
    }

    pub(crate) fn tool_get_pagerank(&self, args: &Value) -> ToolResult {
        let top_k = args.get("top_k").and_then(|v| v.as_u64()).unwrap_or(20) as usize;
        let damping = args.get("damping").and_then(|v| v.as_f64()).unwrap_or(0.85);

        let graph = match self.lock_graph() {
            Ok(g) => g,
            Err(e) => return ToolResult::tool_error(format!("Lock error: {e}")),
        };
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
            .expect("JSON serialization of literal"),
        )
    }

    pub(crate) fn tool_search_code(&self, args: &Value) -> ToolResult {
        let query = match args.get("query").and_then(|v| v.as_str()) {
            Some(q) if !q.is_empty() => q,
            _ => return ToolResult::tool_error("Missing or empty 'query' parameter"),
        };

        let k = args.get("k").and_then(|v| v.as_u64()).unwrap_or(10) as usize;

        let results: Vec<(String, f32)> = if let Some(emb_guard) = match self.lock_embeddings() {
            Ok(g) => g,
            Err(e) => return ToolResult::tool_error(format!("Lock error: {e}")),
        } {
            match emb_guard.embed(query) {
                Ok(query_embedding) => {
                    drop(emb_guard);
                    let vec = match self.lock_vector() {
                        Ok(v) => v,
                        Err(e) => return ToolResult::tool_error(format!("Lock error: {e}")),
                    };
                    vec.search(&query_embedding, k * 3)
                        .unwrap_or_default()
                        .into_iter()
                        .filter(|(id, _)| id.starts_with("sym:") || id.starts_with("chunk:"))
                        .take(k)
                        .collect()
                }
                Err(e) => {
                    return ToolResult::tool_error(format!("Embedding failed: {e}"));
                }
            }
        } else {
            return ToolResult::tool_error("Embedding service not available");
        };

        if results.is_empty() {
            return ToolResult::text("No matching code found.");
        }

        let mut output = Vec::new();
        for (id, distance) in &results {
            let similarity = 1.0 - *distance as f64;
            if let Ok(Some(node)) = self.storage.get_graph_node(id) {
                if id.starts_with("chunk:") {
                    output.push(json!({
                        "id": id,
                        "kind": "chunk",
                        "label": node.label,
                        "similarity": format!("{:.4}", similarity),
                        "file_path": node.payload.get("file_path"),
                        "line_start": node.payload.get("line_start"),
                        "line_end": node.payload.get("line_end"),
                        "node_kind": node.payload.get("node_kind"),
                        "parent_symbol": node.payload.get("parent_symbol"),
                        "non_ws_chars": node.payload.get("non_ws_chars"),
                    }));
                } else {
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
        }

        ToolResult::text(
            serde_json::to_string_pretty(&output).expect("JSON serialization of literal"),
        )
    }

    /// MCP tool: set_scoring_weights -- update the server's scoring weights at runtime.
    pub(crate) fn tool_set_scoring_weights(&self, args: &Value) -> ToolResult {
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

        match self.scoring_weights_mut() {
            Ok(mut w) => *w = normalized.clone(),
            Err(e) => return ToolResult::tool_error(format!("Lock error: {e}")),
        }

        // Persist scoring weights to config file
        let mut config = codemem_core::CodememConfig::load_or_default();
        config.scoring = normalized.clone();
        if let Err(e) = config.save(&codemem_core::CodememConfig::default_path()) {
            tracing::warn!("Failed to persist scoring weights: {e}");
        }

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

        ToolResult::text(
            serde_json::to_string_pretty(&response).expect("JSON serialization of literal"),
        )
    }

    /// Return a snapshot of operational metrics (latency percentiles, counters, gauges).
    pub(crate) fn tool_metrics(&self) -> ToolResult {
        let snapshot = self.metrics.snapshot();
        match serde_json::to_string_pretty(&snapshot) {
            Ok(json) => ToolResult::text(json),
            Err(e) => ToolResult::tool_error(format!("Failed to serialize metrics: {e}")),
        }
    }

    // ── Graph Compaction ─────────────────────────────────────────────────

    /// Compact chunk graph-nodes after indexing. Keeps the top-K most important
    /// chunks per file and removes the rest from the graph (but leaves embeddings
    /// in the vector index for semantic search).
    ///
    /// Returns the number of chunks pruned.
    fn compact_graph(&self, seen_files: &std::collections::HashSet<String>) -> (usize, usize) {
        let mut graph = match self.lock_graph() {
            Ok(g) => g,
            Err(_) => return (0, 0),
        };

        // ── Pass 1: Chunk compaction (existing logic) ──────────────────────

        let max_chunks_per_file = self.config.chunking.max_retained_chunks_per_file;
        let chunk_score_threshold = self.config.chunking.min_chunk_score_threshold;

        let mut chunks_by_file: HashMap<String, Vec<(String, f64)>> = HashMap::new();

        let mut max_degree: f64 = 1.0;
        let mut max_non_ws: f64 = 1.0;

        let all_nodes = graph.get_all_nodes();
        let chunk_nodes: Vec<&GraphNode> = all_nodes
            .iter()
            .filter(|n| n.kind == NodeKind::Chunk)
            .collect();

        for node in &chunk_nodes {
            let degree = graph
                .get_edges(&node.id)
                .map(|edges| edges.len() as f64)
                .unwrap_or(0.0);
            max_degree = max_degree.max(degree);

            let non_ws = node
                .payload
                .get("non_ws_chars")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            max_non_ws = max_non_ws.max(non_ws);
        }

        for node in &chunk_nodes {
            let file_path = match node.payload.get("file_path").and_then(|v| v.as_str()) {
                Some(fp) => fp.to_string(),
                None => continue,
            };
            if !seen_files.contains(&file_path) {
                continue;
            }

            let degree = graph
                .get_edges(&node.id)
                .map(|edges| edges.len() as f64)
                .unwrap_or(0.0);
            let centrality_rank = degree / max_degree;

            let has_symbol_parent = if node.payload.contains_key("parent_symbol") {
                1.0
            } else {
                0.0
            };

            let has_memory_link = graph
                .get_edges(&node.id)
                .map(|edges| {
                    edges.iter().any(|e| {
                        let other = if e.src == node.id { &e.dst } else { &e.src };
                        !other.starts_with("sym:")
                            && !other.starts_with("file:")
                            && !other.starts_with("chunk:")
                            && !other.starts_with("pkg:")
                            && !other.starts_with("contains:")
                            && !other.starts_with("ref:")
                    })
                })
                .unwrap_or(false);
            let memory_link_score = if has_memory_link { 1.0 } else { 0.0 };

            let non_ws = node
                .payload
                .get("non_ws_chars")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let non_ws_rank = non_ws / max_non_ws;

            let chunk_score = centrality_rank * 0.3
                + has_symbol_parent * 0.2
                + memory_link_score * 0.3
                + non_ws_rank * 0.2;

            chunks_by_file
                .entry(file_path)
                .or_default()
                .push((node.id.clone(), chunk_score));
        }

        let mut symbol_count_by_file: HashMap<String, usize> = HashMap::new();
        for node in &all_nodes {
            if matches!(
                node.kind,
                NodeKind::Function
                    | NodeKind::Method
                    | NodeKind::Class
                    | NodeKind::Interface
                    | NodeKind::Type
                    | NodeKind::Constant
                    | NodeKind::Test
            ) {
                if let Some(fp) = node.payload.get("file_path").and_then(|v| v.as_str()) {
                    *symbol_count_by_file.entry(fp.to_string()).or_default() += 1;
                }
            }
        }

        let mut chunks_pruned = 0usize;
        for (file_path, mut chunks) in chunks_by_file {
            chunks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let sym_count = symbol_count_by_file.get(&file_path).copied().unwrap_or(0);
            let k = max_chunks_per_file.min(chunks.len()).max(3.max(sym_count));

            for (i, (chunk_id, score)) in chunks.iter().enumerate() {
                if i >= k || *score < chunk_score_threshold {
                    if let Ok(Some(chunk_node)) = graph.get_node(chunk_id) {
                        if let Some(parent_sym) = chunk_node
                            .payload
                            .get("parent_symbol")
                            .and_then(|v| v.as_str())
                        {
                            let parent_id = format!("sym:{parent_sym}");
                            if let Ok(Some(mut parent_node)) = graph.get_node(&parent_id) {
                                let line_start = chunk_node
                                    .payload
                                    .get("line_start")
                                    .and_then(|v| v.as_u64())
                                    .unwrap_or(0);
                                let line_end = chunk_node
                                    .payload
                                    .get("line_end")
                                    .and_then(|v| v.as_u64())
                                    .unwrap_or(0);
                                let ranges = parent_node
                                    .payload
                                    .entry("covered_ranges".to_string())
                                    .or_insert_with(|| json!([]));
                                if let Some(arr) = ranges.as_array_mut() {
                                    arr.push(json!([line_start, line_end]));
                                }
                                let count = parent_node
                                    .payload
                                    .entry("pruned_chunk_count".to_string())
                                    .or_insert_with(|| json!(0));
                                if let Some(n) = count.as_u64() {
                                    *count = json!(n + 1);
                                }
                                let _ = self.storage.insert_graph_node(&parent_node);
                                let _ = graph.add_node(parent_node);
                            }
                        }
                    }

                    let _ = self.storage.delete_graph_edges_for_node(chunk_id);
                    let _ = self.storage.delete_graph_node(chunk_id);
                    let _ = graph.remove_node(chunk_id);
                    chunks_pruned += 1;
                }
            }
        }

        // ── Pass 2: Symbol compaction ──────────────────────────────────────

        let max_syms_per_file = self.config.chunking.max_retained_symbols_per_file;
        let sym_score_threshold = self.config.chunking.min_symbol_score_threshold;

        // Re-fetch nodes after chunk pruning
        let all_nodes = graph.get_all_nodes();
        let sym_nodes: Vec<&GraphNode> = all_nodes
            .iter()
            .filter(|n| n.id.starts_with("sym:"))
            .collect();

        // Find max values for normalization
        let mut max_calls_degree: f64 = 1.0;
        let mut max_code_size: f64 = 1.0;

        for node in &sym_nodes {
            let calls_degree = graph
                .get_edges(&node.id)
                .map(|edges| {
                    edges
                        .iter()
                        .filter(|e| e.relationship == RelationshipType::Calls)
                        .count() as f64
                })
                .unwrap_or(0.0);
            max_calls_degree = max_calls_degree.max(calls_degree);

            let line_start = node
                .payload
                .get("line_start")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let line_end = node
                .payload
                .get("line_end")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let code_size = (line_end - line_start).max(0.0);
            max_code_size = max_code_size.max(code_size);
        }

        // Helper: check if a node has any edge to a Memory node
        let has_memory_link = |graph: &dyn GraphBackend, node_id: &str| -> bool {
            graph
                .get_edges(node_id)
                .map(|edges| {
                    edges.iter().any(|e| {
                        let other = if e.src == node_id { &e.dst } else { &e.src };
                        !other.starts_with("sym:")
                            && !other.starts_with("file:")
                            && !other.starts_with("chunk:")
                            && !other.starts_with("pkg:")
                            && !other.starts_with("contains:")
                            && !other.starts_with("ref:")
                    })
                })
                .unwrap_or(false)
        };

        // Score symbols and group by file
        let mut syms_by_file: HashMap<String, Vec<(String, f64, bool, bool)>> = HashMap::new();
        // Tuple: (node_id, score, is_structural_anchor, has_memory_link)

        for node in &sym_nodes {
            let file_path = match node.payload.get("file_path").and_then(|v| v.as_str()) {
                Some(fp) => fp.to_string(),
                None => continue,
            };
            if !seen_files.contains(&file_path) {
                continue;
            }

            // call_connectivity: inbound+outbound CALLS edge count, normalized
            let calls_degree = graph
                .get_edges(&node.id)
                .map(|edges| {
                    edges
                        .iter()
                        .filter(|e| e.relationship == RelationshipType::Calls)
                        .count() as f64
                })
                .unwrap_or(0.0);
            let call_connectivity = calls_degree / max_calls_degree;

            // visibility_score: public=1.0, crate=0.5, private=0.0
            let visibility_score = match node
                .payload
                .get("visibility")
                .and_then(|v| v.as_str())
                .unwrap_or("private")
            {
                "public" => 1.0,
                "crate" => 0.5,
                _ => 0.0,
            };

            // kind_score: class/interface=1.0, function/method=0.6, test=0.3, constant=0.1
            let kind_score = match node.kind {
                NodeKind::Class | NodeKind::Interface => 1.0,
                NodeKind::Module => 1.0,
                NodeKind::Function | NodeKind::Method => 0.6,
                NodeKind::Test => 0.3,
                NodeKind::Constant => 0.1,
                _ => 0.5,
            };

            // memory_link: 1.0 if linked to any memory node, 0.0 otherwise
            let mem_linked = has_memory_link(&*graph, &node.id);
            let memory_link_val = if mem_linked { 1.0 } else { 0.0 };

            // code_size_rank: (line_end - line_start) normalized by max
            let line_start = node
                .payload
                .get("line_start")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let line_end = node
                .payload
                .get("line_end")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let code_size = (line_end - line_start).max(0.0);
            let code_size_rank = code_size / max_code_size;

            let symbol_score = call_connectivity * 0.30
                + visibility_score * 0.20
                + kind_score * 0.15
                + memory_link_val * 0.20
                + code_size_rank * 0.15;

            let is_structural = matches!(
                node.kind,
                NodeKind::Class | NodeKind::Interface | NodeKind::Module
            );

            syms_by_file.entry(file_path).or_default().push((
                node.id.clone(),
                symbol_score,
                is_structural,
                mem_linked,
            ));
        }

        let mut symbols_pruned = 0usize;
        for (_file_path, mut syms) in syms_by_file {
            // Sort by score descending
            syms.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // K = max(max_retained_symbols_per_file, count of public symbols in file)
            let public_count = syms
                .iter()
                .filter(|(id, ..)| {
                    graph
                        .get_node(id)
                        .ok()
                        .flatten()
                        .and_then(|n| {
                            n.payload
                                .get("visibility")
                                .and_then(|v| v.as_str())
                                .map(|v| v == "public")
                        })
                        .unwrap_or(false)
                })
                .count();
            let k = max_syms_per_file.max(public_count);

            for (i, (sym_id, score, is_structural, mem_linked)) in syms.iter().enumerate() {
                // Always retain structural anchors and memory-linked symbols
                if *is_structural || *mem_linked {
                    continue;
                }
                if i < k && *score >= sym_score_threshold {
                    continue;
                }

                // Copy covered_ranges from pruned symbol's child chunks into parent file node
                if let Ok(Some(sym_node)) = graph.get_node(sym_id) {
                    if let Some(fp) = sym_node.payload.get("file_path").and_then(|v| v.as_str()) {
                        let file_id = format!("file:{fp}");
                        if let Ok(Some(mut file_node)) = graph.get_node(&file_id) {
                            let line_start = sym_node
                                .payload
                                .get("line_start")
                                .and_then(|v| v.as_u64())
                                .unwrap_or(0);
                            let line_end = sym_node
                                .payload
                                .get("line_end")
                                .and_then(|v| v.as_u64())
                                .unwrap_or(0);
                            if line_end > line_start {
                                let ranges = file_node
                                    .payload
                                    .entry("pruned_symbol_ranges".to_string())
                                    .or_insert_with(|| json!([]));
                                if let Some(arr) = ranges.as_array_mut() {
                                    arr.push(json!([line_start, line_end]));
                                }
                                let _ = self.storage.insert_graph_node(&file_node);
                                let _ = graph.add_node(file_node);
                            }
                        }
                    }
                }

                // Remove symbol from graph (but keep embeddings for vector search)
                let _ = self.storage.delete_graph_edges_for_node(sym_id);
                let _ = self.storage.delete_graph_node(sym_id);
                let _ = graph.remove_node(sym_id);
                symbols_pruned += 1;
            }
        }

        // Recompute centrality on the now-smaller graph
        if chunks_pruned > 0 || symbols_pruned > 0 {
            graph.compute_centrality();
            graph.recompute_centrality();
        }

        (chunks_pruned, symbols_pruned)
    }

    // ── Summary Tree Tool ────────────────────────────────────────────────

    /// Return a hierarchical summary tree starting from a given node.
    /// Shows packages → files → symbols (no chunks unless explicitly requested).
    pub(crate) fn tool_summary_tree(&self, args: &Value) -> ToolResult {
        let start_id = match args.get("start_id").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => return ToolResult::tool_error("Missing 'start_id' parameter"),
        };
        let max_depth = args.get("max_depth").and_then(|v| v.as_u64()).unwrap_or(3) as usize;
        let include_chunks = args
            .get("include_chunks")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let graph = match self.lock_graph() {
            Ok(g) => g,
            Err(e) => return ToolResult::tool_error(format!("Lock error: {e}")),
        };

        fn build_tree(
            graph: &dyn GraphBackend,
            node_id: &str,
            depth: usize,
            max_depth: usize,
            include_chunks: bool,
        ) -> Option<Value> {
            if depth > max_depth {
                return None;
            }
            let node = match graph.get_node(node_id) {
                Ok(Some(n)) => n,
                _ => return None,
            };

            let mut children: Vec<Value> = Vec::new();
            if depth < max_depth {
                // Find CONTAINS edges from this node
                if let Ok(edges) = graph.get_edges(node_id) {
                    let mut child_ids: Vec<String> = edges
                        .iter()
                        .filter(|e| {
                            e.src == node_id && e.relationship == RelationshipType::Contains
                        })
                        .map(|e| e.dst.clone())
                        .collect();
                    child_ids.sort();

                    for child_id in &child_ids {
                        // Skip chunks unless requested
                        if !include_chunks && child_id.starts_with("chunk:") {
                            continue;
                        }
                        if let Some(child) =
                            build_tree(graph, child_id, depth + 1, max_depth, include_chunks)
                        {
                            children.push(child);
                        }
                    }
                }
            }

            let mut result = json!({
                "id": node.id,
                "kind": node.kind.to_string(),
                "label": node.label,
                "centrality": node.centrality,
            });

            // Add summary metadata for files
            if node.kind == NodeKind::File {
                let symbol_count = children
                    .iter()
                    .filter(|c| {
                        let k = c.get("kind").and_then(|v| v.as_str()).unwrap_or("");
                        k != "chunk" && k != "package" && k != "file"
                    })
                    .count();
                let chunk_count = if include_chunks {
                    children
                        .iter()
                        .filter(|c| c.get("kind").and_then(|v| v.as_str()) == Some("chunk"))
                        .count()
                } else {
                    0
                };
                result["symbol_count"] = json!(symbol_count);
                result["chunk_count"] = json!(chunk_count);
            }

            if !children.is_empty() {
                result["children"] = json!(children);
            }

            Some(result)
        }

        match build_tree(&*graph, start_id, 0, max_depth, include_chunks) {
            Some(tree) => {
                ToolResult::text(serde_json::to_string_pretty(&tree).expect("JSON serialization"))
            }
            None => ToolResult::tool_error(format!("Node not found: {start_id}")),
        }
    }
}

#[cfg(test)]
#[path = "tests/tools_graph_tests.rs"]
mod tests;
