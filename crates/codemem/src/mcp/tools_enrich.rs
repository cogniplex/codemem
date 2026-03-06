//! Enrichment tools: git history, security, performance analysis, and composite tools.
//!
//! Each tool annotates existing graph nodes with additional metadata and stores
//! Insight-type memories tagged with `track:*` tags for the Insights UI.

use super::args::parse_string_array;
use super::types::ToolResult;
use super::McpServer;
use codemem_core::GraphBackend;
use codemem_engine::IndexCache;
use serde_json::{json, Value};

impl McpServer {
    pub(crate) fn tool_enrich_git_history(&self, args: &Value) -> ToolResult {
        let path = match args.get("path").and_then(|v| v.as_str()) {
            Some(p) if !p.is_empty() => p,
            _ => return ToolResult::tool_error("Missing required 'path' parameter (repo root)"),
        };

        let days = args.get("days").and_then(|v| v.as_u64()).unwrap_or(90);
        let namespace = args.get("namespace").and_then(|v| v.as_str());

        match self.engine.enrich_git_history(path, days, namespace) {
            Ok(result) => {
                ToolResult::text(serde_json::to_string_pretty(&result.details).unwrap_or_default())
            }
            Err(e) => ToolResult::tool_error(format!("{e}")),
        }
    }

    pub(crate) fn tool_enrich_security(&self, args: &Value) -> ToolResult {
        let namespace = args.get("namespace").and_then(|v| v.as_str());

        match self.engine.enrich_security(namespace) {
            Ok(result) => {
                ToolResult::text(serde_json::to_string_pretty(&result.details).unwrap_or_default())
            }
            Err(e) => ToolResult::tool_error(format!("{e}")),
        }
    }

    pub(crate) fn tool_enrich_performance(&self, args: &Value) -> ToolResult {
        let namespace = args.get("namespace").and_then(|v| v.as_str());
        let top = args.get("top").and_then(|v| v.as_u64()).unwrap_or(10) as usize;

        match self.engine.enrich_performance(top, namespace) {
            Ok(result) => {
                ToolResult::text(serde_json::to_string_pretty(&result.details).unwrap_or_default())
            }
            Err(e) => ToolResult::tool_error(format!("{e}")),
        }
    }

    /// Composite enrichment tool: runs selected (or all) enrichment analyses in one call.
    ///
    /// The 14 supported analyses must match the `enum` in `tool_definitions()` (mcp/mod.rs):
    /// git, security, performance, complexity, code_smells, security_scan, architecture,
    /// test_mapping, api_surface, doc_coverage, hot_complex, blame, quality, change_impact.
    pub(crate) fn tool_enrich_codebase(&self, args: &Value) -> ToolResult {
        let path = match args.get("path").and_then(|v| v.as_str()) {
            Some(p) if !p.is_empty() => p,
            _ => return ToolResult::tool_error("Missing required 'path' parameter"),
        };

        let days = args.get("days").and_then(|v| v.as_u64()).unwrap_or(90);
        let namespace = args.get("namespace").and_then(|v| v.as_str());
        let analyses = parse_string_array(args, "analyses");
        let file_path = args.get("file_path").and_then(|v| v.as_str());

        let enrichment = self
            .engine
            .run_enrichments(path, &analyses, days, namespace, file_path);

        ToolResult::text(
            serde_json::to_string_pretty(&enrichment.results)
                .expect("JSON serialization of literal"),
        )
    }

    /// Full pipeline tool: index → enrich → pagerank → clusters → summary.
    pub(crate) fn tool_analyze_codebase(&self, args: &Value) -> ToolResult {
        let path = match args.get("path").and_then(|v| v.as_str()) {
            Some(p) if !p.is_empty() => p,
            _ => return ToolResult::tool_error("Missing required 'path' parameter"),
        };

        let days = args.get("days").and_then(|v| v.as_u64()).unwrap_or(90);

        let mut summary = json!({});

        // Step 1: Index
        let root = std::path::Path::new(path);
        if !root.exists() {
            return ToolResult::tool_error(format!("Path does not exist: {path}"));
        }

        // Namespace: use explicit param, or derive basename from path
        let namespace = args
            .get("namespace")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| {
                root.file_name()
                    .and_then(|f| f.to_str())
                    .unwrap_or(path)
                    .to_string()
            });

        let mut indexer = codemem_engine::Indexer::new();
        let resolved = match indexer.index_and_resolve(root) {
            Ok(r) => r,
            Err(e) => return ToolResult::tool_error(format!("Indexing failed: {e}")),
        };

        let persist_result = match self
            .engine
            .persist_index_results(&resolved, Some(&namespace))
        {
            Ok(r) => r,
            Err(e) => return ToolResult::tool_error(format!("Persistence failed: {e}")),
        };

        // Cache results
        {
            if let Ok(mut cache) = self.lock_index_cache() {
                *cache = Some(IndexCache {
                    symbols: resolved.symbols,
                    chunks: resolved.chunks,
                    root_path: path.to_string(),
                });
            }
        }

        summary["index"] = json!({
            "files_parsed": resolved.index.files_parsed,
            "symbols": resolved.index.total_symbols,
            "edges_resolved": persist_result.edges_resolved,
            "chunks": persist_result.chunks_stored,
        });

        // Step 2: Enrich (all 14 analyses)
        let ns_ref = Some(namespace.as_str());
        let enrichment = self.engine.run_enrichments(path, &[], days, ns_ref, None);
        summary["enrichment"] = enrichment.results;

        // Step 3: PageRank (top 10)
        if let Ok(graph) = self.lock_graph() {
            let scores = graph.pagerank(0.85, 100, 1e-6);
            let mut sorted: Vec<(String, f64)> = scores.into_iter().collect();
            sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            sorted.truncate(10);
            let top_nodes: Vec<Value> = sorted
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
            summary["important_nodes"] = json!(top_nodes);

            // Step 4: Clusters
            let communities = graph.louvain_communities(1.0);
            summary["cluster_count"] = json!(communities.len());
        }

        ToolResult::text(
            serde_json::to_string_pretty(&summary).expect("JSON serialization of literal"),
        )
    }
}
