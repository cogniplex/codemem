//! Graph & analysis tools: traverse, stats, health, index, symbols, deps, impact,
//! clusters, cross-repo, pagerank, search-code, scoring weights, metrics.

use super::types::ToolResult;
use super::McpServer;
use codemem_core::{GraphBackend, NodeKind, RelationshipType, ScoringWeights, VectorBackend};
use codemem_engine::IndexCache;
use serde_json::{json, Value};

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
        let storage_stats = match self.engine.storage.stats() {
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
                    "available": self.engine.embeddings.is_some(),
                    "cache": cache_info,
                }
            }))
            .expect("JSON serialization of literal"),
        )
    }

    pub(crate) fn tool_health(&self) -> ToolResult {
        let storage_ok = self.engine.storage.stats().is_ok();
        let vector_ok = true; // HnswIndex is always available
        let graph_ok = true; // GraphEngine is always available
        let embeddings_ok = self.engine.embeddings.is_some();

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

        let mut indexer = codemem_engine::Indexer::new();
        let resolved = match indexer.index_and_resolve(root) {
            Ok(r) => r,
            Err(e) => return ToolResult::tool_error(format!("Indexing failed: {e}")),
        };

        let files_scanned = resolved.index.files_scanned;
        let files_parsed = resolved.index.files_parsed;
        let files_skipped = resolved.index.files_skipped;
        let total_symbols = resolved.index.total_symbols;
        let total_references = resolved.index.total_references;

        // Delegate all persistence to the engine
        let persist_result = match self.engine.persist_index_results(&resolved, Some(path)) {
            Ok(r) => r,
            Err(e) => return ToolResult::tool_error(format!("Persistence failed: {e}")),
        };

        // Cache results for structural queries
        {
            match self.lock_index_cache() {
                Ok(mut cache) => {
                    *cache = Some(IndexCache {
                        symbols: resolved.symbols,
                        chunks: resolved.chunks,
                        root_path: path.to_string(),
                    });
                }
                Err(e) => return ToolResult::tool_error(format!("Lock error: {e}")),
            }
        }

        ToolResult::text(
            serde_json::to_string_pretty(&json!({
                "files_scanned": files_scanned,
                "files_parsed": files_parsed,
                "files_skipped": files_skipped,
                "files_created": persist_result.files_created,
                "symbols": total_symbols,
                "references": total_references,
                "edges_resolved": persist_result.edges_resolved,
                "symbols_embedded": persist_result.symbols_embedded,
                "chunks": persist_result.chunks_stored,
                "chunks_embedded": persist_result.chunks_embedded,
                "chunks_pruned": persist_result.chunks_pruned,
                "symbols_pruned": persist_result.symbols_pruned,
                "packages_created": persist_result.packages_created,
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

        let manifest_result = codemem_engine::index::manifest::scan_manifests(&scan_root);

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
            if let Ok(Some(node)) = self.engine.storage.get_graph_node(id) {
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
        let snapshot = self.engine.metrics.snapshot();
        match serde_json::to_string_pretty(&snapshot) {
            Ok(json) => ToolResult::text(json),
            Err(e) => ToolResult::tool_error(format!("Failed to serialize metrics: {e}")),
        }
    }

    // ── Graph Compaction ─────────────────────────────────────────────────

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
