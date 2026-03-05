//! Performance analysis: coupling, dependency depth, PageRank, complexity.

use super::EnrichResult;
use crate::CodememEngine;
use codemem_core::{CodememError, GraphBackend, NodeKind};
use serde_json::json;
use std::collections::HashMap;

impl CodememEngine {
    /// Enrich the graph with performance analysis: coupling, dependency depth, PageRank, complexity.
    pub fn enrich_performance(
        &self,
        top: usize,
        namespace: Option<&str>,
    ) -> Result<EnrichResult, CodememError> {
        // Collect data from graph into local variables, then drop lock
        let all_nodes;
        let mut coupling_data: Vec<(String, String, usize)> = Vec::new();
        let mut high_coupling_count = 0;
        let layers: Vec<Vec<String>>;
        let mut file_pagerank: Vec<(String, String, f64)> = Vec::new();
        {
            let graph = self.lock_graph()?;
            all_nodes = graph.get_all_nodes();

            // 1. Compute coupling (in-degree + out-degree) for each node
            for node in &all_nodes {
                let degree = graph.get_edges(&node.id).map(|e| e.len()).unwrap_or(0);
                coupling_data.push((node.id.clone(), node.label.clone(), degree));
                if degree > self.config.enrichment.perf_min_coupling_degree {
                    high_coupling_count += 1;
                }
            }

            // 2. Compute dependency depth via topological layers
            layers = graph.topological_layers();

            // 3. PageRank for critical path (File nodes only)
            for node in &all_nodes {
                if node.kind == NodeKind::File {
                    let pr = graph.get_pagerank(&node.id);
                    if pr > 0.0 {
                        file_pagerank.push((node.id.clone(), node.label.clone(), pr));
                    }
                }
            }
        }
        // Graph lock released here

        let max_depth = layers.len();
        file_pagerank.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        // 4. File complexity from symbol counts (computed from local all_nodes)
        let mut file_symbol_counts: HashMap<String, usize> = HashMap::new();
        for node in &all_nodes {
            match node.kind {
                NodeKind::Function
                | NodeKind::Method
                | NodeKind::Class
                | NodeKind::Interface
                | NodeKind::Type => {
                    if let Some(file_path) = node.payload.get("file_path").and_then(|v| v.as_str())
                    {
                        *file_symbol_counts.entry(file_path.to_string()).or_default() += 1;
                    }
                }
                _ => {}
            }
        }

        // Annotate graph nodes (short lock scope for writes only)
        {
            let mut graph = self.lock_graph()?;

            for (node_id, _label, degree) in &coupling_data {
                if let Ok(Some(mut node)) = graph.get_node(node_id) {
                    node.payload.insert("coupling_score".into(), json!(degree));
                    let _ = graph.add_node(node);
                }
            }

            for (layer_idx, layer) in layers.iter().enumerate() {
                for node_id in layer {
                    if let Ok(Some(mut node)) = graph.get_node(node_id) {
                        node.payload
                            .insert("dependency_layer".into(), json!(layer_idx));
                        let _ = graph.add_node(node);
                    }
                }
            }

            for (node_id, _label, rank) in file_pagerank.iter().take(top) {
                if let Ok(Some(mut node)) = graph.get_node(node_id) {
                    node.payload
                        .insert("critical_path_rank".into(), json!(rank));
                    let _ = graph.add_node(node);
                }
            }

            for (file_path, sym_count) in &file_symbol_counts {
                let node_id = format!("file:{file_path}");
                if let Ok(Some(mut node)) = graph.get_node(&node_id) {
                    node.payload.insert("symbol_count".into(), json!(sym_count));
                    let _ = graph.add_node(node);
                }
            }
        }

        // Store insights
        let mut insights_stored = 0;

        // High-coupling nodes
        coupling_data.sort_by(|a, b| b.2.cmp(&a.2));
        for (node_id, label, degree) in coupling_data.iter().take(top) {
            if *degree > self.config.enrichment.perf_min_coupling_degree {
                let content = format!(
                    "High coupling: {} has {} dependencies — refactoring risk",
                    label, degree
                );
                if self
                    .store_insight(
                        &content,
                        "performance",
                        &["coupling"],
                        0.7,
                        namespace,
                        std::slice::from_ref(node_id),
                    )
                    .is_some()
                {
                    insights_stored += 1;
                }
            }
        }

        // Deep dependency chain
        if max_depth > 5 {
            let content = format!(
                "Deep dependency chain: {} layers — impacts build and test times",
                max_depth
            );
            if self
                .store_insight(
                    &content,
                    "performance",
                    &["dependency-depth"],
                    0.6,
                    namespace,
                    &[],
                )
                .is_some()
            {
                insights_stored += 1;
            }
        }

        // Critical bottleneck (top PageRank file)
        if let Some((node_id, label, _)) = file_pagerank.first() {
            let content = format!(
                "Critical bottleneck: {} — highest centrality file, changes cascade widely",
                label
            );
            if self
                .store_insight(
                    &content,
                    "performance",
                    &["critical-path"],
                    0.8,
                    namespace,
                    std::slice::from_ref(node_id),
                )
                .is_some()
            {
                insights_stored += 1;
            }
        }

        // Complex files (high symbol count)
        let mut complex_files: Vec<_> = file_symbol_counts.iter().collect();
        complex_files.sort_by(|a, b| b.1.cmp(a.1));
        for (file_path, sym_count) in complex_files.iter().take(top) {
            if **sym_count > self.config.enrichment.perf_min_symbol_count {
                let content = format!("Complex file: {} — {} symbols", file_path, sym_count);
                if self
                    .store_insight(
                        &content,
                        "performance",
                        &["complexity"],
                        0.5,
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

        let critical_files: Vec<_> = file_pagerank
            .iter()
            .take(top)
            .map(|(_, label, score)| json!({"file": label, "pagerank": score}))
            .collect();

        Ok(EnrichResult {
            insights_stored,
            details: json!({
                "high_coupling_count": high_coupling_count,
                "max_depth": max_depth,
                "critical_files": critical_files,
                "insights_stored": insights_stored,
            }),
        })
    }
}
