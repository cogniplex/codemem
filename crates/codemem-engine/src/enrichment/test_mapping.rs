//! Test-to-code mapping: link tests to tested symbols, identify untested public functions.

use super::EnrichResult;
use crate::CodememEngine;
use codemem_core::{CodememError, Edge, GraphBackend, GraphNode, NodeKind, RelationshipType};
use serde_json::json;
use std::collections::{HashMap, HashSet};

impl CodememEngine {
    /// Map test functions to the code they test and identify untested public functions.
    ///
    /// For Test-kind nodes, infers tested symbols by naming convention (`test_foo` -> `foo`)
    /// and by CALLS edges. Creates RELATES_TO edges between test and tested symbols.
    /// Produces Insight memories for files with untested public functions.
    pub fn enrich_test_mapping(
        &self,
        namespace: Option<&str>,
    ) -> Result<EnrichResult, CodememError> {
        let all_nodes;
        let mut test_edges_info: Vec<(String, String)> = Vec::new();

        {
            let graph = self.lock_graph()?;
            all_nodes = graph.get_all_nodes();

            // Collect test nodes and non-test function/method nodes
            let test_nodes: Vec<&GraphNode> = all_nodes
                .iter()
                .filter(|n| n.kind == NodeKind::Test)
                .collect();
            // Index by simple name (last segment of qualified name)
            let mut fn_by_simple_name: HashMap<String, Vec<&GraphNode>> = HashMap::new();
            for node in all_nodes
                .iter()
                .filter(|n| matches!(n.kind, NodeKind::Function | NodeKind::Method))
            {
                let simple = node
                    .label
                    .rsplit("::")
                    .next()
                    .unwrap_or(&node.label)
                    .to_string();
                fn_by_simple_name.entry(simple).or_default().push(node);
            }

            for test_node in &test_nodes {
                // Extract what this test might be testing from its name
                let test_name = test_node
                    .label
                    .rsplit("::")
                    .next()
                    .unwrap_or(&test_node.label);

                // Convention: test_foo tests foo, test_foo_bar tests foo_bar
                let tested_name = test_name
                    .strip_prefix("test_")
                    .or_else(|| test_name.strip_prefix("test"))
                    .unwrap_or("");

                if !tested_name.is_empty() {
                    // Check by simple name
                    if let Some(targets) = fn_by_simple_name.get(tested_name) {
                        for target in targets {
                            test_edges_info.push((test_node.id.clone(), target.id.clone()));
                        }
                    }
                }

                // Also check CALLS edges from the test to find tested symbols
                if let Ok(edges) = graph.get_edges(&test_node.id) {
                    for edge in &edges {
                        if edge.relationship == RelationshipType::Calls && edge.src == test_node.id
                        {
                            // Only link to function/method nodes
                            if let Ok(Some(dst_node)) = graph.get_node(&edge.dst) {
                                if matches!(dst_node.kind, NodeKind::Function | NodeKind::Method) {
                                    test_edges_info
                                        .push((test_node.id.clone(), dst_node.id.clone()));
                                }
                            }
                        }
                    }
                }
            }
        }

        // Dedup edges
        let unique_edges: HashSet<(String, String)> = test_edges_info.into_iter().collect();

        // Create RELATES_TO edges for test mappings
        let mut edges_created = 0;
        {
            let mut graph = self.lock_graph()?;
            let now = chrono::Utc::now();
            for (test_id, target_id) in &unique_edges {
                let edge_id = format!("test-map:{test_id}->{target_id}");
                // Skip if edge already exists
                if graph.get_node(test_id).ok().flatten().is_none()
                    || graph.get_node(target_id).ok().flatten().is_none()
                {
                    continue;
                }
                let edge = Edge {
                    id: edge_id,
                    src: test_id.clone(),
                    dst: target_id.clone(),
                    relationship: RelationshipType::RelatesTo,
                    weight: 0.8,
                    properties: HashMap::from([("test_mapping".into(), json!(true))]),
                    created_at: now,
                    valid_from: None,
                    valid_to: None,
                };
                let _ = self.storage.insert_graph_edge(&edge);
                if graph.add_edge(edge).is_ok() {
                    edges_created += 1;
                }
            }
        }

        // Identify untested public functions per file
        let tested_ids: HashSet<String> = unique_edges.iter().map(|(_, t)| t.clone()).collect();
        let mut untested_by_file: HashMap<String, Vec<String>> = HashMap::new();

        for node in &all_nodes {
            if !matches!(node.kind, NodeKind::Function | NodeKind::Method) {
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
            if tested_ids.contains(&node.id) {
                continue;
            }
            let file_path = node
                .payload
                .get("file_path")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string();
            untested_by_file
                .entry(file_path)
                .or_default()
                .push(node.label.clone());
        }

        let mut insights_stored = 0;
        for (file_path, untested) in &untested_by_file {
            if untested.is_empty() {
                continue;
            }
            let names: Vec<&str> = untested.iter().take(10).map(|s| s.as_str()).collect();
            let suffix = if untested.len() > 10 {
                format!(" (and {} more)", untested.len() - 10)
            } else {
                String::new()
            };
            let content = format!(
                "Untested public functions in {}: {}{}",
                file_path,
                names.join(", "),
                suffix
            );
            if self
                .store_insight(
                    &content,
                    "testing",
                    &[],
                    0.6,
                    namespace,
                    &[format!("file:{file_path}")],
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
                "test_edges_created": edges_created,
                "files_with_untested": untested_by_file.len(),
                "insights_stored": insights_stored,
            }),
        })
    }
}
