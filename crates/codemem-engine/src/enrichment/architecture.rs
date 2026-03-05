//! Architecture inference: layering, patterns, circular dependencies.

use super::EnrichResult;
use crate::CodememEngine;
use codemem_core::{CodememError, GraphBackend, NodeKind, RelationshipType};
use serde_json::json;
use std::collections::{HashMap, HashSet};

impl CodememEngine {
    /// Infer architectural layers and patterns from the module dependency graph.
    ///
    /// Analyzes IMPORTS/CALLS/DEPENDS_ON edges between modules to detect layering
    /// (e.g., api -> service -> storage) and recognizes common directory patterns
    /// (controllers/, models/, views/, handlers/).
    pub fn enrich_architecture(
        &self,
        namespace: Option<&str>,
    ) -> Result<EnrichResult, CodememError> {
        let all_nodes;
        let mut module_deps: HashMap<String, HashSet<String>> = HashMap::new();

        {
            let graph = self.lock_graph()?;
            all_nodes = graph.get_all_nodes();

            // Build module dependency graph from IMPORTS/CALLS edges
            for node in &all_nodes {
                if !matches!(
                    node.kind,
                    NodeKind::File | NodeKind::Module | NodeKind::Package
                ) {
                    continue;
                }
                if let Ok(edges) = graph.get_edges(&node.id) {
                    for edge in &edges {
                        if !matches!(
                            edge.relationship,
                            RelationshipType::Imports
                                | RelationshipType::Calls
                                | RelationshipType::DependsOn
                        ) {
                            continue;
                        }
                        if edge.src == node.id {
                            module_deps
                                .entry(node.id.clone())
                                .or_default()
                                .insert(edge.dst.clone());
                        }
                    }
                }
            }
        }

        let mut insights_stored = 0;

        // Detect architectural layers by analyzing dependency direction
        // Extract top-level directory from node IDs
        fn top_dir(node_id: &str) -> Option<String> {
            let path = node_id
                .strip_prefix("file:")
                .or_else(|| node_id.strip_prefix("pkg:"))
                .unwrap_or(node_id);
            let parts: Vec<&str> = path.split('/').collect();
            if parts.len() >= 2 {
                Some(parts[0].to_string())
            } else {
                None
            }
        }

        // Build directory-level dependency counts
        let mut dir_deps: HashMap<String, HashSet<String>> = HashMap::new();
        for (src, dsts) in &module_deps {
            if let Some(src_dir) = top_dir(src) {
                for dst in dsts {
                    if let Some(dst_dir) = top_dir(dst) {
                        if src_dir != dst_dir {
                            dir_deps.entry(src_dir.clone()).or_default().insert(dst_dir);
                        }
                    }
                }
            }
        }

        // Detect layers: directories with no incoming deps are "top" layers
        let all_dirs: HashSet<String> = dir_deps
            .keys()
            .chain(dir_deps.values().flat_map(|v| v.iter()))
            .cloned()
            .collect();
        let dirs_with_incoming: HashSet<String> =
            dir_deps.values().flat_map(|v| v.iter()).cloned().collect();
        let top_layers: Vec<&String> = all_dirs
            .iter()
            .filter(|d| !dirs_with_incoming.contains(*d))
            .collect();
        let bottom_layers: Vec<&String> = all_dirs
            .iter()
            .filter(|d| !dir_deps.contains_key(*d))
            .collect();

        if !dir_deps.is_empty() {
            let mut layer_desc = String::new();
            if !top_layers.is_empty() {
                let mut sorted_top: Vec<&&String> = top_layers.iter().collect();
                sorted_top.sort();
                layer_desc.push_str(&format!(
                    "Top-level (entry points): {}",
                    sorted_top
                        .iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
            }
            if !bottom_layers.is_empty() {
                if !layer_desc.is_empty() {
                    layer_desc.push_str("; ");
                }
                let mut sorted_bottom: Vec<&&String> = bottom_layers.iter().collect();
                sorted_bottom.sort();
                layer_desc.push_str(&format!(
                    "Foundation (no outbound deps): {}",
                    sorted_bottom
                        .iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
            }
            let content = format!(
                "Architecture: {} module groups with layered dependencies. {}",
                all_dirs.len(),
                layer_desc
            );
            if self
                .store_insight(&content, "architecture", &[], 0.9, namespace, &[])
                .is_some()
            {
                insights_stored += 1;
            }
        }

        // Detect common architectural patterns from directory names
        let known_patterns = [
            ("controllers", "MVC Controller layer"),
            ("handlers", "Handler/Controller layer"),
            ("models", "Data model layer"),
            ("views", "View/Template layer"),
            ("services", "Service/Business logic layer"),
            ("api", "API layer"),
            ("routes", "Routing layer"),
            ("middleware", "Middleware layer"),
            ("utils", "Utility/Helper layer"),
            ("lib", "Library/Core layer"),
        ];

        let detected: Vec<&str> = known_patterns
            .iter()
            .filter(|(name, _)| {
                all_nodes
                    .iter()
                    .any(|n| n.kind == NodeKind::Package && n.label.contains(name))
            })
            .map(|(_, desc)| *desc)
            .collect();

        if !detected.is_empty() {
            let content = format!("Architecture patterns detected: {}", detected.join(", "));
            if self
                .store_insight(&content, "architecture", &[], 0.7, namespace, &[])
                .is_some()
            {
                insights_stored += 1;
            }
        }

        // Detect circular dependencies between directories
        for (dir, deps) in &dir_deps {
            for dep in deps {
                if let Some(back_deps) = dir_deps.get(dep) {
                    if back_deps.contains(dir) && dir < dep {
                        let content = format!(
                            "Circular dependency: {} and {} depend on each other — consider refactoring",
                            dir, dep
                        );
                        if self
                            .store_insight(
                                &content,
                                "architecture",
                                &["circular-dep"],
                                0.8,
                                namespace,
                                &[],
                            )
                            .is_some()
                        {
                            insights_stored += 1;
                        }
                    }
                }
            }
        }

        self.save_index();

        Ok(EnrichResult {
            insights_stored,
            details: json!({
                "module_count": all_dirs.len(),
                "dependency_edges": module_deps.values().map(|v| v.len()).sum::<usize>(),
                "top_layers": top_layers.len(),
                "bottom_layers": bottom_layers.len(),
                "patterns_detected": detected.len(),
                "insights_stored": insights_stored,
            }),
        })
    }
}
