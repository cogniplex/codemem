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

    /// Composite enrichment tool: runs git + security + performance enrichment in one call.
    pub(crate) fn tool_enrich_codebase(&self, args: &Value) -> ToolResult {
        let path = match args.get("path").and_then(|v| v.as_str()) {
            Some(p) if !p.is_empty() => p,
            _ => return ToolResult::tool_error("Missing required 'path' parameter"),
        };

        let days = args.get("days").and_then(|v| v.as_u64()).unwrap_or(90);
        let namespace = args.get("namespace").and_then(|v| v.as_str());
        let analyses = parse_string_array(args, "analyses");
        let run_all = analyses.is_empty();

        let mut results = json!({});

        if run_all || analyses.iter().any(|a| a == "git") {
            match self.engine.enrich_git_history(path, days, namespace) {
                Ok(r) => {
                    results["git"] = r.details;
                }
                Err(e) => {
                    results["git"] = json!({"error": format!("{e}")});
                }
            }
        }

        if run_all || analyses.iter().any(|a| a == "security") {
            match self.engine.enrich_security(namespace) {
                Ok(r) => {
                    results["security"] = r.details;
                }
                Err(e) => {
                    results["security"] = json!({"error": format!("{e}")});
                }
            }
        }

        if run_all || analyses.iter().any(|a| a == "performance") {
            match self.engine.enrich_performance(10, namespace) {
                Ok(r) => {
                    results["performance"] = r.details;
                }
                Err(e) => {
                    results["performance"] = json!({"error": format!("{e}")});
                }
            }
        }

        let root = std::path::Path::new(path);
        let project_root = Some(root);

        if run_all || analyses.iter().any(|a| a == "complexity") {
            match self.engine.enrich_complexity(namespace, project_root) {
                Ok(r) => {
                    results["complexity"] = r.details;
                }
                Err(e) => {
                    results["complexity"] = json!({"error": format!("{e}")});
                }
            }
        }

        if run_all || analyses.iter().any(|a| a == "code_smells") {
            match self.engine.enrich_code_smells(namespace, project_root) {
                Ok(r) => {
                    results["code_smells"] = r.details;
                }
                Err(e) => {
                    results["code_smells"] = json!({"error": format!("{e}")});
                }
            }
        }

        if run_all || analyses.iter().any(|a| a == "security_scan") {
            match self
                .engine
                .enrich_security_scan(namespace, project_root)
            {
                Ok(r) => {
                    results["security_scan"] = r.details;
                }
                Err(e) => {
                    results["security_scan"] = json!({"error": format!("{e}")});
                }
            }
        }

        if run_all || analyses.iter().any(|a| a == "architecture") {
            match self.engine.enrich_architecture(namespace) {
                Ok(r) => results["architecture"] = r.details,
                Err(e) => results["architecture"] = json!({"error": format!("{e}")}),
            }
        }

        if run_all || analyses.iter().any(|a| a == "test_mapping") {
            match self.engine.enrich_test_mapping(namespace) {
                Ok(r) => results["test_mapping"] = r.details,
                Err(e) => results["test_mapping"] = json!({"error": format!("{e}")}),
            }
        }

        if run_all || analyses.iter().any(|a| a == "api_surface") {
            match self.engine.enrich_api_surface(namespace) {
                Ok(r) => results["api_surface"] = r.details,
                Err(e) => results["api_surface"] = json!({"error": format!("{e}")}),
            }
        }

        if run_all || analyses.iter().any(|a| a == "doc_coverage") {
            match self.engine.enrich_doc_coverage(namespace) {
                Ok(r) => results["doc_coverage"] = r.details,
                Err(e) => results["doc_coverage"] = json!({"error": format!("{e}")}),
            }
        }

        if run_all || analyses.iter().any(|a| a == "hot_complex") {
            match self.engine.enrich_hot_complex(namespace) {
                Ok(r) => results["hot_complex"] = r.details,
                Err(e) => results["hot_complex"] = json!({"error": format!("{e}")}),
            }
        }

        if run_all || analyses.iter().any(|a| a == "blame") {
            match self.engine.enrich_blame(path, namespace) {
                Ok(r) => results["blame"] = r.details,
                Err(e) => results["blame"] = json!({"error": format!("{e}")}),
            }
        }

        if run_all || analyses.iter().any(|a| a == "quality") {
            match self.engine.enrich_quality_stratification(namespace) {
                Ok(r) => results["quality"] = r.details,
                Err(e) => results["quality"] = json!({"error": format!("{e}")}),
            }
        }

        ToolResult::text(
            serde_json::to_string_pretty(&results).expect("JSON serialization of literal"),
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

        // Step 2: Enrich
        let ns_ref = Some(namespace.as_str());
        if let Ok(r) = self.engine.enrich_git_history(path, days, ns_ref) {
            summary["enrichment_git"] = r.details;
        }
        if let Ok(r) = self.engine.enrich_security(ns_ref) {
            summary["enrichment_security"] = r.details;
        }
        if let Ok(r) = self.engine.enrich_performance(10, ns_ref) {
            summary["enrichment_performance"] = r.details;
        }

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
