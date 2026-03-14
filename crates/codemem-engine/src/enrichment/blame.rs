//! Blame/ownership enrichment: primary owner and contributors from git blame.

use super::EnrichResult;
use crate::CodememEngine;
use codemem_core::{CodememError, NodeKind};
use serde_json::json;
use std::collections::HashMap;

impl CodememEngine {
    /// Enrich file nodes with primary owner and contributors from git blame.
    pub fn enrich_blame(
        &self,
        path: &str,
        namespace: Option<&str>,
    ) -> Result<EnrichResult, CodememError> {
        let file_nodes: Vec<String> = {
            let graph = self.lock_graph()?;
            graph
                .get_all_nodes()
                .into_iter()
                .filter(|n| n.kind == NodeKind::File)
                .map(|n| n.label.clone())
                .collect()
        };

        let mut files_annotated = 0;
        let mut insights_stored = 0;

        for file_path in &file_nodes {
            let output = std::process::Command::new("git")
                .args(["-C", path, "log", "--format=%an", "--", file_path])
                .output();

            let output = match output {
                Ok(o) if o.status.success() => o,
                _ => continue,
            };

            let stdout = String::from_utf8_lossy(&output.stdout);
            let mut author_counts: HashMap<String, usize> = HashMap::new();
            for line in stdout.lines() {
                let author = line.trim();
                if !author.is_empty() {
                    *author_counts.entry(author.to_string()).or_default() += 1;
                }
            }

            if author_counts.is_empty() {
                continue;
            }

            let mut sorted_authors: Vec<_> = author_counts.into_iter().collect();
            sorted_authors.sort_by(|a, b| b.1.cmp(&a.1));

            let primary_owner = sorted_authors[0].0.clone();
            let contributors: Vec<String> = sorted_authors.iter().map(|(a, _)| a.clone()).collect();

            // Annotate graph node
            let node_id = format!("file:{file_path}");
            {
                let mut graph = self.lock_graph()?;
                if let Ok(Some(mut node)) = graph.get_node(&node_id) {
                    node.payload
                        .insert("primary_owner".into(), json!(primary_owner));
                    node.payload
                        .insert("contributors".into(), json!(contributors));
                    let _ = graph.add_node(node);
                    files_annotated += 1;
                }
            }
        }

        // Collect shared-ownership insights while holding the lock, then store outside
        let pending_ownership: Vec<(String, String)> = {
            let graph = self.lock_graph()?;
            graph
                .get_all_nodes()
                .iter()
                .filter(|n| n.kind == NodeKind::File)
                .filter_map(|node| {
                    let contribs = node.payload.get("contributors")?.as_array()?;
                    if contribs.len() > 5 {
                        let primary = node
                            .payload
                            .get("primary_owner")
                            .and_then(|v| v.as_str())
                            .unwrap_or("unknown");
                        let content = format!(
                            "Shared ownership: {} has {} contributors (primary: {}) — may need clear ownership",
                            node.label, contribs.len(), primary
                        );
                        Some((content, node.id.clone()))
                    } else {
                        None
                    }
                })
                .collect()
        };

        for (content, node_id) in &pending_ownership {
            if self
                .store_insight(
                    content,
                    "ownership",
                    &[],
                    0.5,
                    namespace,
                    std::slice::from_ref(node_id),
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
                "files_annotated": files_annotated,
                "insights_stored": insights_stored,
            }),
        })
    }
}
