//! Change impact prediction: co-change, call graph, and test associations.

use super::EnrichResult;
use crate::CodememEngine;
use codemem_core::{CodememError, NodeKind, RelationshipType};
use serde_json::json;

impl CodememEngine {
    /// Predict the impact of changes to a given file by combining co-change edges,
    /// call graph edges, and test file associations.
    pub fn enrich_change_impact(
        &self,
        file_path: &str,
        namespace: Option<&str>,
    ) -> Result<EnrichResult, CodememError> {
        let graph = self.lock_graph()?;

        let file_id = format!("file:{file_path}");
        if graph.get_node(&file_id).ok().flatten().is_none() {
            return Err(CodememError::NotFound(format!(
                "File node not found: {file_path}"
            )));
        }

        let mut co_changed: Vec<String> = Vec::new();
        let mut callers: Vec<String> = Vec::new();
        let mut callees: Vec<String> = Vec::new();
        let mut test_files: Vec<String> = Vec::new();

        // Get edges for the file node
        if let Ok(edges) = graph.get_edges(&file_id) {
            for edge in &edges {
                match edge.relationship {
                    RelationshipType::CoChanged => {
                        let other = if edge.src == file_id {
                            &edge.dst
                        } else {
                            &edge.src
                        };
                        if let Some(path) = other.strip_prefix("file:") {
                            co_changed.push(path.to_string());
                        }
                    }
                    RelationshipType::Calls => {
                        let other = if edge.src == file_id {
                            callees.push(edge.dst.clone());
                            &edge.dst
                        } else {
                            callers.push(edge.src.clone());
                            &edge.src
                        };
                        let _ = other;
                    }
                    RelationshipType::RelatesTo => {
                        // Check if this is a test mapping edge
                        if edge.properties.contains_key("test_mapping") {
                            let other = if edge.src == file_id {
                                &edge.dst
                            } else {
                                &edge.src
                            };
                            if let Ok(Some(node)) = graph.get_node(other) {
                                if node.kind == NodeKind::Test {
                                    if let Some(fp) =
                                        node.payload.get("file_path").and_then(|v| v.as_str())
                                    {
                                        test_files.push(fp.to_string());
                                    }
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        // Also check symbols contained in this file for their callers
        let all_nodes = graph.get_all_nodes();
        for node in &all_nodes {
            if !matches!(node.kind, NodeKind::Function | NodeKind::Method) {
                continue;
            }
            let sym_file = node
                .payload
                .get("file_path")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if sym_file != file_path {
                continue;
            }
            if let Ok(edges) = graph.get_edges(&node.id) {
                for edge in &edges {
                    if edge.relationship == RelationshipType::Calls && edge.dst == node.id {
                        // Something calls this symbol
                        if let Ok(Some(caller_node)) = graph.get_node(&edge.src) {
                            if let Some(fp) = caller_node
                                .payload
                                .get("file_path")
                                .and_then(|v| v.as_str())
                            {
                                if fp != file_path {
                                    callers.push(fp.to_string());
                                }
                            }
                        }
                    }
                }
            }
        }

        drop(graph);

        // Dedup
        co_changed.sort();
        co_changed.dedup();
        callers.sort();
        callers.dedup();
        callees.sort();
        callees.dedup();
        test_files.sort();
        test_files.dedup();

        let impact_score = co_changed.len() + callers.len() + callees.len();

        let mut insights_stored = 0;

        if impact_score > 0 {
            let mut parts: Vec<String> = Vec::new();
            if !callers.is_empty() {
                parts.push(format!(
                    "{} caller files ({})",
                    callers.len(),
                    callers
                        .iter()
                        .take(5)
                        .cloned()
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
            }
            if !co_changed.is_empty() {
                parts.push(format!(
                    "{} co-changed files ({})",
                    co_changed.len(),
                    co_changed
                        .iter()
                        .take(5)
                        .cloned()
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
            }
            if !test_files.is_empty() {
                parts.push(format!(
                    "{} test files ({})",
                    test_files.len(),
                    test_files.join(", ")
                ));
            }
            let content = format!("Change impact for {}: {}", file_path, parts.join("; "));
            let importance = (impact_score as f64 / 20.0).clamp(0.4, 0.9);
            if self
                .store_insight(&content, "impact", &[], importance, namespace, &[file_id])
                .is_some()
            {
                insights_stored += 1;
            }
        }

        self.save_index();

        Ok(EnrichResult {
            insights_stored,
            details: json!({
                "file": file_path,
                "callers": callers.len(),
                "callees": callees.len(),
                "co_changed": co_changed.len(),
                "test_files": test_files.len(),
                "impact_score": impact_score,
                "insights_stored": insights_stored,
            }),
        })
    }
}
