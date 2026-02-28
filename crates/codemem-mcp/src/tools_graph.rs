//! Graph & analysis tools: traverse, stats, health, index, symbols, deps, impact,
//! clusters, cross-repo, pagerank, search-code, scoring weights.

use crate::types::{IndexCache, ToolResult};
use crate::McpServer;
use codemem_core::{Edge, GraphBackend, GraphNode, NodeKind, ScoringWeights, VectorBackend};
use serde_json::{json, Value};
use std::collections::HashMap;

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

    pub(crate) fn tool_stats(&self) -> ToolResult {
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
            .unwrap(),
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

    pub(crate) fn tool_search_symbols(&self, args: &Value) -> ToolResult {
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

    pub(crate) fn tool_get_symbol_info(&self, args: &Value) -> ToolResult {
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

    pub(crate) fn tool_get_impact(&self, args: &Value) -> ToolResult {
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

    pub(crate) fn tool_get_clusters(&self, args: &Value) -> ToolResult {
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

    pub(crate) fn tool_get_cross_repo(&self, args: &Value) -> ToolResult {
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

    pub(crate) fn tool_get_pagerank(&self, args: &Value) -> ToolResult {
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

    pub(crate) fn tool_search_code(&self, args: &Value) -> ToolResult {
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scoring::compute_score;
    use crate::test_helpers::*;
    use codemem_core::RelationshipType;

    #[test]
    fn handle_unknown_tool() {
        let server = test_server();
        let params = json!({"name": "nonexistent", "arguments": {}});
        let resp = server.handle_request("tools/call", Some(&params), json!(4));
        let result = resp.result.unwrap();
        assert_eq!(result["isError"], true);
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
}
