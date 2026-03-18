//! Dead code detection: identifies symbols with zero inbound CALLS/IMPORTS edges,
//! applying framework-aware exemptions (decorators, constructors, tests, etc.).

use super::EnrichResult;
use crate::CodememEngine;
use codemem_core::config::DeadCodeConfig;
use codemem_core::{CodememError, Edge, GraphNode, RelationshipType};
use serde_json::json;
use std::collections::HashSet;

/// A symbol detected as potentially dead (unreferenced) code.
#[derive(Debug, Clone)]
pub struct DeadCodeEntry {
    /// Graph node ID of the unreferenced symbol.
    pub node_id: String,
    /// Human-readable symbol name (from `node.label`).
    pub label: String,
    /// The symbol kind string from `node.payload["kind"]`.
    pub kind: String,
    /// Optional file path from `node.payload["file_path"]`.
    pub file_path: Option<String>,
}

/// Keyword fragments in decorator/attribute values that signal framework entry points.
const FRAMEWORK_KEYWORDS: &[&str] = &["route", "endpoint", "export", "api"];

/// Analyze the graph for unreferenced symbols (dead code candidates).
///
/// Symbols are considered dead when they have no inbound `Calls`, `Imports`,
/// `Inherits`, or `Implements` edges and do not match any exemption rule.
pub fn find_dead_code(
    nodes: &[GraphNode],
    edges: &[Edge],
    config: &DeadCodeConfig,
) -> Vec<DeadCodeEntry> {
    // Step 1: filter to symbol nodes (those with a "kind" key in payload).
    let symbol_nodes: Vec<&GraphNode> = nodes
        .iter()
        .filter(|n| n.payload.contains_key("kind"))
        .collect();

    // Step 2: min_symbols threshold — avoid false positives on tiny graphs.
    if symbol_nodes.len() < config.min_symbols {
        return Vec::new();
    }

    // Step 3: build set of node IDs that have inbound referencing edges.
    let referenced: HashSet<&str> = edges
        .iter()
        .filter(|e| {
            matches!(
                e.relationship,
                RelationshipType::Calls
                    | RelationshipType::Imports
                    | RelationshipType::Inherits
                    | RelationshipType::Implements
            )
        })
        .map(|e| e.dst.as_str())
        .collect();

    // Step 4: collect unreferenced, non-exempt symbols.
    let mut dead: Vec<DeadCodeEntry> = Vec::new();

    for node in &symbol_nodes {
        if referenced.contains(node.id.as_str()) {
            continue;
        }

        if is_exempt(node, config) {
            continue;
        }

        let kind = node
            .payload
            .get("kind")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let file_path = node
            .payload
            .get("file_path")
            .and_then(|v| v.as_str())
            .map(String::from);

        dead.push(DeadCodeEntry {
            node_id: node.id.clone(),
            label: node.label.clone(),
            kind,
            file_path,
        });
    }

    dead
}

/// Check whether a symbol node should be exempt from dead code detection.
fn is_exempt(node: &GraphNode, config: &DeadCodeConfig) -> bool {
    // Kind exemption: payload["kind"] in config.exempt_kinds
    if let Some(kind_val) = node.payload.get("kind").and_then(|v| v.as_str()) {
        let kind_lower = kind_val.to_lowercase();
        if config
            .exempt_kinds
            .iter()
            .any(|k| k.to_lowercase() == kind_lower)
        {
            return true;
        }
    }

    // Decorator/attribute exemption
    if let Some(attrs) = node.payload.get("attributes").and_then(|v| v.as_array()) {
        for attr in attrs {
            if let Some(attr_str) = attr.as_str() {
                let attr_lower = attr_str.to_lowercase();
                // Exact match against configured exempt decorators
                if config
                    .exempt_decorators
                    .iter()
                    .any(|d| attr_lower.contains(&d.to_lowercase()))
                {
                    return true;
                }
                // Framework keyword match
                if FRAMEWORK_KEYWORDS.iter().any(|kw| attr_lower.contains(kw)) {
                    return true;
                }
            }
        }
    }

    // Main entry point
    if node.label == "main" || node.label == "Main" {
        return true;
    }

    // Dunder methods (Python __init__, __str__, etc.)
    if node.label.starts_with("__") && node.label.ends_with("__") {
        return true;
    }

    // Public visibility — public symbols may be part of library API
    if let Some(vis) = node.payload.get("visibility").and_then(|v| v.as_str()) {
        if vis == "public" {
            return true;
        }
    }

    false
}

impl CodememEngine {
    /// Run dead code detection on the current graph and store insights for
    /// unreferenced symbols.
    pub fn enrich_dead_code(&self, namespace: Option<&str>) -> Result<EnrichResult, CodememError> {
        let config = &self.config.enrichment.dead_code;
        if !config.enabled {
            return Ok(EnrichResult {
                insights_stored: 0,
                details: json!({"skipped": true, "reason": "dead_code disabled"}),
            });
        }

        // Collect all nodes from the in-memory graph and all edges from storage.
        // Using storage.all_graph_edges() avoids N+1 per-node get_edges calls.
        let all_nodes = self.lock_graph()?.get_all_nodes();
        let all_edges = self.storage.all_graph_edges()?;

        let dead_entries = find_dead_code(&all_nodes, &all_edges, config);

        let mut insights_stored = 0;
        for entry in &dead_entries {
            let file_info = entry
                .file_path
                .as_deref()
                .map(|fp| format!(" in {fp}"))
                .unwrap_or_default();
            let content = format!(
                "Dead code candidate: `{}` ({}) has no callers or importers{}",
                entry.label, entry.kind, file_info,
            );
            let links = vec![entry.node_id.clone()];
            if self
                .store_insight(
                    &content,
                    "dead-code",
                    &["dead-code"],
                    0.6,
                    namespace,
                    &links,
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
                "dead_code_candidates": dead_entries.len(),
                "insights_stored": insights_stored,
            }),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use codemem_core::{GraphNode, NodeKind};
    use std::collections::HashMap;

    /// Helper: create a GraphNode with the given label, kind payload, and optional attributes.
    fn make_graph_node(name: &str, kind_str: &str, attrs: Option<Vec<&str>>) -> GraphNode {
        let mut payload: HashMap<String, serde_json::Value> = HashMap::new();
        payload.insert("kind".into(), json!(kind_str));
        if let Some(attr_list) = attrs {
            payload.insert("attributes".into(), json!(attr_list));
        }
        GraphNode {
            id: format!("sym:{name}"),
            kind: NodeKind::Function,
            label: name.to_string(),
            payload,
            centrality: 0.0,
            memory_id: None,
            namespace: None,
            valid_from: None,
            valid_to: None,
        }
    }

    /// Helper: create an Edge with the given src, dst, and relationship.
    fn make_edge(src: &str, dst: &str, rel: RelationshipType) -> Edge {
        Edge {
            id: format!("{src}-{:?}-{dst}", rel),
            src: src.to_string(),
            dst: dst.to_string(),
            relationship: rel,
            weight: 1.0,
            properties: HashMap::new(),
            created_at: chrono::Utc::now(),
            valid_from: None,
            valid_to: None,
        }
    }

    fn test_config() -> DeadCodeConfig {
        DeadCodeConfig {
            min_symbols: 2,
            ..DeadCodeConfig::default()
        }
    }

    #[test]
    fn unreachable_function_detected() {
        let nodes = vec![
            make_graph_node("main", "function", None),
            make_graph_node("helper", "function", None),
            make_graph_node("unused_fn", "function", None),
        ];
        let edges = vec![make_edge("sym:main", "sym:helper", RelationshipType::Calls)];
        let config = test_config();

        let dead = find_dead_code(&nodes, &edges, &config);

        // unused_fn has no callers and is not main → should be detected.
        // main is exempt (entry point). helper is called by main → not dead.
        let dead_labels: Vec<&str> = dead.iter().map(|d| d.label.as_str()).collect();
        assert!(
            dead_labels.contains(&"unused_fn"),
            "unused_fn should be detected as dead code"
        );
        assert!(
            !dead_labels.contains(&"helper"),
            "helper is called by main, should not be dead"
        );
        assert!(
            !dead_labels.contains(&"main"),
            "main is exempt as entry point"
        );
    }

    #[test]
    fn decorated_symbols_exempt() {
        let nodes = vec![
            make_graph_node("index", "function", Some(vec!["app.route"])),
            make_graph_node("unused", "function", None),
            make_graph_node("api_handler", "function", Some(vec!["get_endpoint"])),
        ];
        let edges = vec![];
        let config = test_config();

        let dead = find_dead_code(&nodes, &edges, &config);

        let dead_labels: Vec<&str> = dead.iter().map(|d| d.label.as_str()).collect();
        assert!(
            !dead_labels.contains(&"index"),
            "app.route decorated should be exempt"
        );
        assert!(
            !dead_labels.contains(&"api_handler"),
            "endpoint keyword in attribute should be exempt"
        );
        assert!(
            dead_labels.contains(&"unused"),
            "unused with no decorators should be detected"
        );
    }

    #[test]
    fn constructors_and_tests_exempt() {
        let nodes = vec![
            make_graph_node("__init__", "constructor", None),
            make_graph_node("test_foo", "test", None),
            make_graph_node("orphan", "function", None),
        ];
        let edges = vec![];
        let config = test_config();

        let dead = find_dead_code(&nodes, &edges, &config);

        let dead_labels: Vec<&str> = dead.iter().map(|d| d.label.as_str()).collect();
        assert!(
            !dead_labels.contains(&"__init__"),
            "constructor kind should be exempt"
        );
        assert!(
            !dead_labels.contains(&"test_foo"),
            "test kind should be exempt"
        );
        assert!(
            dead_labels.contains(&"orphan"),
            "orphan function should be detected"
        );
    }

    #[test]
    fn min_symbols_threshold_respected() {
        let nodes = vec![make_graph_node("lonely", "function", None)];
        let edges = vec![];
        let config = DeadCodeConfig {
            min_symbols: 10,
            ..DeadCodeConfig::default()
        };

        let dead = find_dead_code(&nodes, &edges, &config);
        assert!(
            dead.is_empty(),
            "Should return empty when symbol count < min_symbols"
        );
    }

    #[test]
    fn public_symbols_exempt() {
        let mut node = make_graph_node("pub_fn", "function", None);
        node.payload.insert("visibility".into(), json!("public"));
        let nodes = vec![node, make_graph_node("priv_fn", "function", None)];
        let edges = vec![];
        let config = test_config();

        let dead = find_dead_code(&nodes, &edges, &config);
        let dead_labels: Vec<&str> = dead.iter().map(|d| d.label.as_str()).collect();
        assert!(
            !dead_labels.contains(&"pub_fn"),
            "public symbol should be exempt"
        );
        assert!(
            dead_labels.contains(&"priv_fn"),
            "private symbol with no callers should be detected"
        );
    }

    #[test]
    fn dunder_methods_exempt() {
        let nodes = vec![
            make_graph_node("__str__", "method", None),
            make_graph_node("orphan_method", "method", None),
        ];
        let edges = vec![];
        let config = test_config();

        let dead = find_dead_code(&nodes, &edges, &config);
        let dead_labels: Vec<&str> = dead.iter().map(|d| d.label.as_str()).collect();
        assert!(
            !dead_labels.contains(&"__str__"),
            "dunder method should be exempt"
        );
        assert!(
            dead_labels.contains(&"orphan_method"),
            "non-dunder method should be detected"
        );
    }

    #[test]
    fn inherits_and_implements_count_as_references() {
        let nodes = vec![
            make_graph_node("BaseClass", "class", None),
            make_graph_node("MyTrait", "trait", None),
            make_graph_node("orphan_class", "class", None),
        ];
        let edges = vec![
            make_edge(
                "sym:orphan_class",
                "sym:BaseClass",
                RelationshipType::Inherits,
            ),
            make_edge(
                "sym:orphan_class",
                "sym:MyTrait",
                RelationshipType::Implements,
            ),
        ];
        let config = test_config();

        let dead = find_dead_code(&nodes, &edges, &config);
        let dead_labels: Vec<&str> = dead.iter().map(|d| d.label.as_str()).collect();
        assert!(
            !dead_labels.contains(&"BaseClass"),
            "inherited class should not be dead"
        );
        assert!(
            !dead_labels.contains(&"MyTrait"),
            "implemented trait should not be dead"
        );
    }
}
