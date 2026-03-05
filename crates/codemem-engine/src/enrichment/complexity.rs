//! Cyclomatic and cognitive complexity metrics for functions/methods.

use super::{resolve_path, EnrichResult};
use crate::CodememEngine;
use codemem_core::{CodememError, GraphBackend, NodeKind};
use serde_json::json;
use std::collections::HashMap;
use std::path::Path;

impl CodememEngine {
    /// Enrich the graph with cyclomatic and cognitive complexity metrics for functions/methods.
    ///
    /// For each Function/Method node, reads the source file, counts decision points
    /// (if/else/match/for/while/loop/&&/||) as cyclomatic complexity and measures
    /// max nesting depth as a cognitive complexity proxy. High-complexity functions
    /// (cyclomatic > 10) produce Insight memories.
    pub fn enrich_complexity(
        &self,
        namespace: Option<&str>,
        project_root: Option<&Path>,
    ) -> Result<EnrichResult, CodememError> {
        let all_nodes = {
            let graph = self.lock_graph()?;
            graph.get_all_nodes()
        };

        // Collect function/method nodes with file info
        struct SymbolInfo {
            node_id: String,
            label: String,
            file_path: String,
            line_start: usize,
            line_end: usize,
        }

        let mut symbols: Vec<SymbolInfo> = Vec::new();
        for node in &all_nodes {
            if !matches!(node.kind, NodeKind::Function | NodeKind::Method) {
                continue;
            }
            let file_path = match node.payload.get("file_path").and_then(|v| v.as_str()) {
                Some(fp) => fp.to_string(),
                None => continue,
            };
            let line_start = node
                .payload
                .get("line_start")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as usize;
            let line_end = node
                .payload
                .get("line_end")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as usize;
            if line_end <= line_start {
                continue;
            }
            symbols.push(SymbolInfo {
                node_id: node.id.clone(),
                label: node.label.clone(),
                file_path,
                line_start,
                line_end,
            });
        }

        // Cache file contents to avoid re-reading
        let mut file_cache: HashMap<String, Vec<String>> = HashMap::new();
        let mut annotated = 0usize;
        let mut insights_stored = 0usize;

        // Nodes to annotate (collected first, then applied in a single lock scope)
        struct ComplexityData {
            node_id: String,
            cyclomatic: usize,
            cognitive: usize,
        }
        let mut complexity_data: Vec<ComplexityData> = Vec::new();

        // Insights to store (collected first, then stored outside the lock)
        struct ComplexityInsight {
            content: String,
            importance: f64,
            node_id: String,
        }
        let mut pending_insights: Vec<ComplexityInsight> = Vec::new();

        for sym in &symbols {
            let lines = file_cache.entry(sym.file_path.clone()).or_insert_with(|| {
                std::fs::read_to_string(resolve_path(&sym.file_path, project_root))
                    .unwrap_or_default()
                    .lines()
                    .map(String::from)
                    .collect()
            });

            // Extract the function's lines (1-indexed to 0-indexed)
            let start = sym.line_start.saturating_sub(1);
            let end = sym.line_end.min(lines.len());
            if start >= end {
                continue;
            }
            let fn_lines = &lines[start..end];

            // Count cyclomatic complexity: decision points
            let mut cyclomatic: usize = 1; // base path
            let mut max_depth: usize = 0;
            let mut current_depth: usize = 0;

            for line in fn_lines {
                let trimmed = line.trim();

                // Count decision points
                for keyword in &[
                    "if ", "if(", "else if", "match ", "for ", "for(", "while ", "while(", "loop ",
                    "loop{",
                ] {
                    if trimmed.starts_with(keyword) || trimmed.contains(&format!(" {keyword}")) {
                        cyclomatic += 1;
                        break;
                    }
                }
                // Count logical operators as additional branches
                cyclomatic += trimmed.matches("&&").count();
                cyclomatic += trimmed.matches("||").count();

                // Track nesting depth via braces
                for ch in trimmed.chars() {
                    match ch {
                        '{' => {
                            current_depth += 1;
                            max_depth = max_depth.max(current_depth);
                        }
                        '}' => {
                            current_depth = current_depth.saturating_sub(1);
                        }
                        _ => {}
                    }
                }
            }

            complexity_data.push(ComplexityData {
                node_id: sym.node_id.clone(),
                cyclomatic,
                cognitive: max_depth,
            });
            annotated += 1;

            // High complexity threshold
            if cyclomatic > 10 {
                let importance = if cyclomatic > 20 { 0.9 } else { 0.7 };
                pending_insights.push(ComplexityInsight {
                    content: format!(
                        "High complexity: {} — cyclomatic={}, max_nesting={} in {}",
                        sym.label, cyclomatic, max_depth, sym.file_path
                    ),
                    importance,
                    node_id: sym.node_id.clone(),
                });
            }
        }

        // Annotate graph nodes
        {
            let mut graph = self.lock_graph()?;
            for data in &complexity_data {
                if let Ok(Some(mut node)) = graph.get_node(&data.node_id) {
                    node.payload
                        .insert("cyclomatic_complexity".into(), json!(data.cyclomatic));
                    node.payload
                        .insert("cognitive_complexity".into(), json!(data.cognitive));
                    let _ = graph.add_node(node);
                }
            }
        }

        // Store insights (outside graph lock)
        for insight in &pending_insights {
            if self
                .store_insight(
                    &insight.content,
                    "complexity",
                    &[],
                    insight.importance,
                    namespace,
                    std::slice::from_ref(&insight.node_id),
                )
                .is_some()
            {
                insights_stored += 1;
            }
        }

        self.save_index();

        Ok(EnrichResult {
            insights_stored,
            details: json!({
                "symbols_analyzed": annotated,
                "high_complexity_count": pending_insights.len(),
                "insights_stored": insights_stored,
            }),
        })
    }
}
