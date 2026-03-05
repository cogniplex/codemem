//! Graph & analysis tools: traverse, codemem_status (merged stats+health+metrics),
//! index, search_code (with mode), get_symbol_info (with deps), get_symbol_graph,
//! find_important_nodes, find_related_groups, cross-repo, summary_tree.

use super::types::ToolResult;
use super::McpServer;
use codemem_core::{GraphBackend, NodeKind, RelationshipType, VectorBackend};
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

    /// Unified status tool: combines stats, health, and metrics.
    /// `include` parameter controls which sections to return (default: all).
    pub(crate) fn tool_codemem_status(&self, args: &Value) -> ToolResult {
        let include: Vec<&str> = args
            .get("include")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
            .unwrap_or_else(|| vec!["stats", "health", "metrics"]);

        let mut response = json!({});

        if include.contains(&"stats") {
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

            response["stats"] = json!({
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
            });
        }

        if include.contains(&"health") {
            let storage_ok = self.engine.storage.stats().is_ok();
            let vector_ok = true;
            let graph_ok = true;
            let embeddings_ok = self.engine.embeddings.is_some();

            let healthy = storage_ok && vector_ok && graph_ok;

            response["health"] = json!({
                "healthy": healthy,
                "storage": if storage_ok { "ok" } else { "error" },
                "vector": if vector_ok { "ok" } else { "error" },
                "graph": if graph_ok { "ok" } else { "error" },
                "embeddings": if embeddings_ok { "ok" } else { "not_configured" },
            });
        }

        if include.contains(&"metrics") {
            let snapshot = self.engine.metrics.snapshot();
            response["metrics"] = serde_json::to_value(snapshot).unwrap_or(json!({}));
        }

        ToolResult::text(
            serde_json::to_string_pretty(&response).expect("JSON serialization of literal"),
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

        // Use the directory basename as namespace (not the full path) so it
        // matches the short names agents use when calling store_memory.
        let namespace = root.file_name().and_then(|f| f.to_str()).unwrap_or(path);

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
        let persist_result = match self
            .engine
            .persist_index_results(&resolved, Some(namespace))
        {
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

    /// Enhanced search_code with `mode` parameter: "semantic" (default), "text", "hybrid".
    /// Optional `kind` filter for symbol kind.
    pub(crate) fn tool_search_code(&self, args: &Value) -> ToolResult {
        let query = match args.get("query").and_then(|v| v.as_str()) {
            Some(q) if !q.is_empty() => q,
            _ => return ToolResult::tool_error("Missing or empty 'query' parameter"),
        };
        let k = args.get("k").and_then(|v| v.as_u64()).unwrap_or(10) as usize;
        let mode = args
            .get("mode")
            .and_then(|v| v.as_str())
            .unwrap_or("semantic");
        let kind_filter: Option<&str> = args.get("kind").and_then(|v| v.as_str());

        match mode {
            "text" => {
                // Text mode = search_symbols (substring match)
                match self.engine.search_symbols(query, k, kind_filter) {
                    Ok(matches) if matches.is_empty() => {
                        ToolResult::text("No matching symbols found.")
                    }
                    Ok(matches) => {
                        let output: Vec<Value> = matches
                            .iter()
                            .map(|sym| {
                                json!({
                                    "name": sym.name,
                                    "qualified_name": sym.qualified_name,
                                    "kind": sym.kind,
                                    "signature": sym.signature,
                                    "file_path": sym.file_path,
                                    "line_start": sym.line_start,
                                    "line_end": sym.line_end,
                                    "visibility": sym.visibility,
                                    "parent": sym.parent,
                                })
                            })
                            .collect();
                        ToolResult::text(
                            serde_json::to_string_pretty(&output)
                                .expect("JSON serialization of literal"),
                        )
                    }
                    Err(e) => ToolResult::tool_error(format!("{e}")),
                }
            }
            "hybrid" => {
                // Hybrid: run both text and semantic, merge-rank results
                let text_results = self
                    .engine
                    .search_symbols(query, k, kind_filter)
                    .unwrap_or_default();
                let semantic_results = self.engine.search_code(query, k).unwrap_or_default();

                let mut combined: Vec<Value> = Vec::new();
                let mut seen_ids = std::collections::HashSet::new();

                // Semantic results first (higher quality)
                for r in &semantic_results {
                    if seen_ids.insert(r.id.clone()) {
                        combined.push(Self::format_code_search_result(r));
                    }
                }
                // Then text results
                for sym in &text_results {
                    let id = format!("sym:{}", sym.qualified_name);
                    if seen_ids.insert(id) {
                        combined.push(json!({
                            "name": sym.name,
                            "qualified_name": sym.qualified_name,
                            "kind": sym.kind,
                            "signature": sym.signature,
                            "file_path": sym.file_path,
                            "line_start": sym.line_start,
                            "line_end": sym.line_end,
                            "visibility": sym.visibility,
                            "parent": sym.parent,
                            "match_mode": "text",
                        }));
                    }
                }
                combined.truncate(k);

                if combined.is_empty() {
                    ToolResult::text("No matching code found.")
                } else {
                    ToolResult::text(
                        serde_json::to_string_pretty(&combined)
                            .expect("JSON serialization of literal"),
                    )
                }
            }
            _ => {
                // Default: "semantic" (original search_code behavior)
                match self.engine.search_code(query, k) {
                    Ok(results) if results.is_empty() => {
                        ToolResult::text("No matching code found.")
                    }
                    Ok(results) => {
                        let output: Vec<Value> = results
                            .iter()
                            .map(Self::format_code_search_result)
                            .collect();
                        ToolResult::text(
                            serde_json::to_string_pretty(&output)
                                .expect("JSON serialization of literal"),
                        )
                    }
                    Err(e) => ToolResult::tool_error(format!("{e}")),
                }
            }
        }
    }

    fn format_code_search_result(r: &codemem_engine::CodeSearchResult) -> Value {
        if r.kind == "chunk" {
            json!({
                "id": r.id,
                "kind": "chunk",
                "label": r.label,
                "similarity": format!("{:.4}", r.similarity),
                "file_path": r.file_path,
                "line_start": r.line_start,
                "line_end": r.line_end,
                "node_kind": r.node_kind,
                "parent_symbol": r.parent_symbol,
                "non_ws_chars": r.non_ws_chars,
            })
        } else {
            json!({
                "qualified_name": r.qualified_name,
                "kind": r.kind,
                "label": r.label,
                "similarity": format!("{:.4}", r.similarity),
                "file_path": r.file_path,
                "line_start": r.line_start,
                "line_end": r.line_end,
                "signature": r.signature,
                "doc_comment": r.doc_comment,
            })
        }
    }

    /// Enhanced get_symbol_info with optional `include_dependencies`.
    /// Uses `engine.get_symbol()` for cache-through DB fallback.
    pub(crate) fn tool_get_symbol_info(&self, args: &Value) -> ToolResult {
        let qualified_name = match args.get("qualified_name").and_then(|v| v.as_str()) {
            Some(qn) => qn,
            None => return ToolResult::tool_error("Missing 'qualified_name' parameter"),
        };
        let include_deps = args
            .get("include_dependencies")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let sym = match self.engine.get_symbol(qualified_name) {
            Ok(Some(s)) => s,
            Ok(None) => {
                return ToolResult::tool_error(format!("Symbol not found: {qualified_name}"))
            }
            Err(e) => return ToolResult::tool_error(format!("{e}")),
        };

        let mut result = json!({
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
        });

        // Optionally include dependencies from graph
        if include_deps {
            let node_id = format!("sym:{qualified_name}");
            let graph = match self.lock_graph() {
                Ok(g) => g,
                Err(e) => return ToolResult::tool_error(format!("Lock error: {e}")),
            };

            let edges: Vec<Value> = graph
                .get_edges(&node_id)
                .unwrap_or_default()
                .iter()
                .map(|e| {
                    json!({
                        "source": e.src,
                        "target": e.dst,
                        "relationship": e.relationship.to_string(),
                        "weight": e.weight,
                    })
                })
                .collect();
            result["dependencies"] = json!(edges);
        }

        ToolResult::text(
            serde_json::to_string_pretty(&result).expect("JSON serialization of literal"),
        )
    }

    /// Merged get_dependencies + get_impact into get_symbol_graph.
    /// depth=1 is like get_dependencies, depth>1 is like get_impact.
    pub(crate) fn tool_get_symbol_graph(&self, args: &Value) -> ToolResult {
        let qualified_name = match args.get("qualified_name").and_then(|v| v.as_str()) {
            Some(qn) => qn,
            None => return ToolResult::tool_error("Missing 'qualified_name' parameter"),
        };

        let depth = args.get("depth").and_then(|v| v.as_u64()).unwrap_or(1) as usize;
        let direction = args
            .get("direction")
            .and_then(|v| v.as_str())
            .unwrap_or("both");

        let node_id = format!("sym:{qualified_name}");
        let graph = match self.lock_graph() {
            Ok(g) => g,
            Err(e) => return ToolResult::tool_error(format!("Lock error: {e}")),
        };

        // Get direct edges (filtered by direction)
        let edges = match graph.get_edges(&node_id) {
            Ok(e) => e,
            Err(_) => {
                return ToolResult::tool_error(format!("Node not found in graph: {qualified_name}"))
            }
        };

        let direct_edges: Vec<Value> = edges
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

        if depth <= 1 {
            // depth=1: just return direct edges
            if direct_edges.is_empty() {
                return ToolResult::text(format!(
                    "No {direction} dependencies found for {qualified_name}."
                ));
            }
            ToolResult::text(
                serde_json::to_string_pretty(&json!({
                    "symbol": qualified_name,
                    "depth": depth,
                    "direction": direction,
                    "edges": direct_edges,
                }))
                .expect("JSON serialization of literal"),
            )
        } else {
            // depth>1: BFS reachability (impact analysis)
            let nodes = match graph.bfs(&node_id, depth) {
                Ok(n) => n,
                Err(e) => {
                    return ToolResult::tool_error(format!(
                        "Impact analysis failed for {qualified_name}: {e}"
                    ))
                }
            };

            let incoming: Vec<Value> = edges
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
                    "direction": direction,
                    "direct_edges": direct_edges,
                    "direct_dependents": incoming,
                    "reachable_nodes": reachable.len(),
                    "reachable": reachable,
                }))
                .expect("JSON serialization of literal"),
            )
        }
    }

    /// Renamed from get_clusters: find related groups via Louvain community detection.
    pub(crate) fn tool_find_related_groups(&self, args: &Value) -> ToolResult {
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

    /// Renamed from get_pagerank: find the most important/central nodes.
    pub(crate) fn tool_find_important_nodes(&self, args: &Value) -> ToolResult {
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

    // ── Summary Tree Tool ────────────────────────────────────────────────

    /// Return a hierarchical summary tree starting from a given node.
    /// Shows packages -> files -> symbols (no chunks unless explicitly requested).
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

        match self
            .engine
            .summary_tree(start_id, max_depth, include_chunks)
        {
            Ok(tree) => {
                ToolResult::text(serde_json::to_string_pretty(&tree).expect("JSON serialization"))
            }
            Err(e) => ToolResult::tool_error(format!("{e}")),
        }
    }
    /// Retrieve all memories connected to a graph node.
    pub(crate) fn tool_get_node_memories(&self, args: &Value) -> ToolResult {
        let node_id = match args.get("node_id").and_then(|v| v.as_str()) {
            Some(id) => id,
            None => return ToolResult::tool_error("Missing 'node_id' parameter"),
        };
        let max_depth = args.get("max_depth").and_then(|v| v.as_u64()).unwrap_or(1) as usize;

        let include_relationships: Option<Vec<RelationshipType>> = args
            .get("include_relationships")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str()?.parse::<RelationshipType>().ok())
                    .collect()
            });

        match self
            .engine
            .get_node_memories(node_id, max_depth, include_relationships.as_deref())
        {
            Ok(results) if results.is_empty() => {
                ToolResult::text(format!("No memories connected to node '{node_id}'."))
            }
            Ok(results) => {
                let output: Vec<Value> = results
                    .iter()
                    .map(|r| {
                        json!({
                            "memory_id": r.memory.id,
                            "content": r.memory.content,
                            "memory_type": r.memory.memory_type.to_string(),
                            "importance": r.memory.importance,
                            "relationship": r.relationship,
                            "depth": r.depth,
                            "tags": r.memory.tags,
                        })
                    })
                    .collect();
                ToolResult::text(
                    serde_json::to_string_pretty(&output).expect("JSON serialization of literal"),
                )
            }
            Err(e) => ToolResult::tool_error(format!("{e}")),
        }
    }

    /// Batch-check which nodes have attached memories.
    pub(crate) fn tool_node_coverage(&self, args: &Value) -> ToolResult {
        let node_ids: Vec<&str> = match args.get("node_ids").and_then(|v| v.as_array()) {
            Some(arr) => arr.iter().filter_map(|v| v.as_str()).collect(),
            None => return ToolResult::tool_error("Missing 'node_ids' parameter (string array)"),
        };

        if node_ids.is_empty() {
            return ToolResult::tool_error("'node_ids' must be a non-empty array");
        }

        match self.engine.node_coverage(&node_ids) {
            Ok(entries) => {
                let output: Vec<Value> = entries
                    .iter()
                    .map(|e| {
                        json!({
                            "node_id": e.node_id,
                            "memory_count": e.memory_count,
                            "has_coverage": e.has_coverage,
                        })
                    })
                    .collect();
                ToolResult::text(
                    serde_json::to_string_pretty(&output).expect("JSON serialization of literal"),
                )
            }
            Err(e) => ToolResult::tool_error(format!("{e}")),
        }
    }
}

#[cfg(test)]
#[path = "tests/tools_graph_tests.rs"]
mod tests;
