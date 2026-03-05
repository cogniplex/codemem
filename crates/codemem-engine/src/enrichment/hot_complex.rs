//! Hot+complex correlation: cross-reference git churn with complexity.

use super::EnrichResult;
use crate::CodememEngine;
use codemem_core::{CodememError, NodeKind};
use serde_json::json;
use std::collections::HashMap;

impl CodememEngine {
    /// Cross-reference git churn with complexity to find high-risk files.
    ///
    /// Files that are BOTH high-churn AND high-complexity represent the highest
    /// maintenance risk. Requires E1 (complexity) and git enrichment to have run first.
    pub fn enrich_hot_complex(
        &self,
        namespace: Option<&str>,
    ) -> Result<EnrichResult, CodememError> {
        let all_nodes = {
            let graph = self.lock_graph()?;
            graph.get_all_nodes()
        };

        // Find files with git churn data
        let mut file_churn: HashMap<String, f64> = HashMap::new();
        for node in &all_nodes {
            if node.kind != NodeKind::File {
                continue;
            }
            if let Some(churn) = node.payload.get("git_churn_rate").and_then(|v| v.as_f64()) {
                if churn > 0.0 {
                    file_churn.insert(node.label.clone(), churn);
                }
            }
        }

        // Find functions with high complexity and aggregate per file
        let mut file_max_complexity: HashMap<String, (usize, String)> = HashMap::new();
        for node in &all_nodes {
            if !matches!(node.kind, NodeKind::Function | NodeKind::Method) {
                continue;
            }
            let cyclomatic = node
                .payload
                .get("cyclomatic_complexity")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as usize;
            if cyclomatic <= 5 {
                continue;
            }
            let file_path = match node.payload.get("file_path").and_then(|v| v.as_str()) {
                Some(fp) => fp.to_string(),
                None => continue,
            };
            let entry = file_max_complexity
                .entry(file_path)
                .or_insert((0, String::new()));
            if cyclomatic > entry.0 {
                *entry = (cyclomatic, node.label.clone());
            }
        }

        let mut insights_stored = 0;
        let mut hot_complex_files: Vec<serde_json::Value> = Vec::new();

        for (file_path, churn) in &file_churn {
            if let Some((complexity, fn_name)) = file_max_complexity.get(file_path) {
                // Both high churn and high complexity
                hot_complex_files.push(json!({
                    "file": file_path,
                    "churn_rate": churn,
                    "max_complexity": complexity,
                    "complex_function": fn_name,
                }));

                let content = format!(
                    "High-risk file: {} — churn rate {:.1} + max cyclomatic complexity {} (in {}). \
                     Prioritize refactoring",
                    file_path, churn, complexity, fn_name
                );
                if self
                    .store_insight(
                        &content,
                        "risk",
                        &["hot-complex"],
                        0.9,
                        namespace,
                        &[format!("file:{file_path}")],
                    )
                    .is_some()
                {
                    insights_stored += 1;
                }
            }
        }

        self.save_index();

        Ok(EnrichResult {
            insights_stored,
            details: json!({
                "hot_complex_files": hot_complex_files.len(),
                "files_with_churn": file_churn.len(),
                "files_with_complexity": file_max_complexity.len(),
                "insights_stored": insights_stored,
            }),
        })
    }
}
