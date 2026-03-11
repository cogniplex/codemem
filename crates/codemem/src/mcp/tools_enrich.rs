//! Enrichment tools: git history, security, performance analysis, and composite tools.
//!
//! Each tool annotates existing graph nodes with additional metadata and stores
//! Insight-type memories tagged with `track:*` tags for the Insights UI.

use super::args::parse_string_array;
use super::types::ToolResult;
use super::McpServer;
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

        let root = std::path::Path::new(path);
        if !root.exists() {
            return ToolResult::tool_error(format!("Path does not exist: {path}"));
        }

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

        let options = codemem_engine::AnalyzeOptions {
            path: root,
            namespace: &namespace,
            git_days: days,
            change_detector: None,
            progress: None,
            no_embed: false,
        };

        match self.engine.analyze(options) {
            Ok(result) => {
                let top_nodes: Vec<Value> = result
                    .top_nodes
                    .iter()
                    .map(|r| {
                        json!({
                            "id": r.id,
                            "pagerank": format!("{:.6}", r.score),
                            "kind": r.kind,
                            "label": r.label,
                        })
                    })
                    .collect();

                let summary = json!({
                    "index": {
                        "files_parsed": result.files_parsed,
                        "symbols": result.symbols_found,
                        "edges_resolved": result.edges_resolved,
                        "chunks": result.chunks_stored,
                    },
                    "enrichment": result.enrichment_results,
                    "important_nodes": top_nodes,
                    "cluster_count": result.community_count,
                });

                ToolResult::text(
                    serde_json::to_string_pretty(&summary).expect("JSON serialization of literal"),
                )
            }
            Err(e) => ToolResult::tool_error(format!("Analysis failed: {e}")),
        }
    }
}
