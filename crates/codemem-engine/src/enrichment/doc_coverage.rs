//! Documentation coverage analysis for public symbols.

use super::EnrichResult;
use crate::CodememEngine;
use codemem_core::{CodememError, NodeKind};
use serde_json::json;
use std::collections::HashMap;

impl CodememEngine {
    /// Analyze documentation coverage for public symbols.
    ///
    /// Checks if each public symbol has a non-empty `doc_comment` in its payload.
    /// Stores Insight memories for files with low documentation coverage.
    pub fn enrich_doc_coverage(
        &self,
        namespace: Option<&str>,
    ) -> Result<EnrichResult, CodememError> {
        let all_nodes = {
            let graph = self.lock_graph()?;
            graph.get_all_nodes()
        };

        struct DocStats {
            documented: usize,
            undocumented: Vec<String>,
        }
        let mut file_docs: HashMap<String, DocStats> = HashMap::new();

        for node in &all_nodes {
            if !matches!(
                node.kind,
                NodeKind::Function
                    | NodeKind::Method
                    | NodeKind::Class
                    | NodeKind::Interface
                    | NodeKind::Type
            ) {
                continue;
            }
            let visibility = node
                .payload
                .get("visibility")
                .and_then(|v| v.as_str())
                .unwrap_or("private");
            if visibility != "public" {
                continue;
            }
            let file_path = match node.payload.get("file_path").and_then(|v| v.as_str()) {
                Some(fp) => fp.to_string(),
                None => continue,
            };
            let has_doc = node
                .payload
                .get("doc_comment")
                .and_then(|v| v.as_str())
                .map(|s| !s.trim().is_empty())
                .unwrap_or(false);

            let stats = file_docs.entry(file_path).or_insert(DocStats {
                documented: 0,
                undocumented: Vec::new(),
            });
            if has_doc {
                stats.documented += 1;
            } else {
                stats.undocumented.push(node.label.clone());
            }
        }

        let mut insights_stored = 0;
        let mut total_documented = 0usize;
        let mut total_undocumented = 0usize;

        for (file_path, stats) in &file_docs {
            total_documented += stats.documented;
            total_undocumented += stats.undocumented.len();

            let total = stats.documented + stats.undocumented.len();
            if total == 0 {
                continue;
            }
            let coverage = stats.documented as f64 / total as f64;
            if coverage < 0.5 && !stats.undocumented.is_empty() {
                let names: Vec<&str> = stats
                    .undocumented
                    .iter()
                    .take(10)
                    .map(|s| s.as_str())
                    .collect();
                let suffix = if stats.undocumented.len() > 10 {
                    format!(" (and {} more)", stats.undocumented.len() - 10)
                } else {
                    String::new()
                };
                let content = format!(
                    "Undocumented public API: {} — {:.0}% coverage ({}/{} documented). Missing: {}{}",
                    file_path,
                    coverage * 100.0,
                    stats.documented,
                    total,
                    names.join(", "),
                    suffix
                );
                let importance = if coverage < 0.2 { 0.7 } else { 0.5 };
                if self
                    .store_insight(
                        &content,
                        "documentation",
                        &[],
                        importance,
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

        let total = total_documented + total_undocumented;
        let overall_coverage = if total > 0 {
            total_documented as f64 / total as f64
        } else {
            1.0
        };

        Ok(EnrichResult {
            insights_stored,
            details: json!({
                "files_analyzed": file_docs.len(),
                "total_public_documented": total_documented,
                "total_public_undocumented": total_undocumented,
                "overall_coverage": format!("{:.1}%", overall_coverage * 100.0),
                "insights_stored": insights_stored,
            }),
        })
    }
}
