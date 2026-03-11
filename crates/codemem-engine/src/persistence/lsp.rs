//! Apply LSP enrichment results to the graph: edge upgrades, ext: node creation,
//! and type annotation enrichment on existing sym: nodes.

use crate::index::lsp::{
    self, EnrichmentResult, EnrichmentTarget, LspEnrichStats, LspResolvedRef, RefToResolve,
};
use codemem_core::{
    CodememError, Edge, GraphBackend, GraphNode, NodeKind, PendingUnresolvedRef, RelationshipType,
};
use std::collections::HashMap;
use std::path::Path;

impl super::super::CodememEngine {
    /// Run LSP enrichment: discover available enrichers, collect targets from
    /// unresolved refs + low-confidence edges, run enrichers, and apply results.
    ///
    /// Returns stats about what was changed. This is Phase 3 of the pipeline.
    pub fn lsp_enrich(
        &self,
        project_root: &Path,
        namespace: &str,
    ) -> Result<LspEnrichStats, CodememError> {
        let enrichers = lsp::available_enrichers();
        if enrichers.is_empty() {
            tracing::info!("No LSP enrichers available on PATH, skipping Phase 3");
            return Ok(LspEnrichStats::default());
        }

        tracing::info!(
            "LSP enrichment: {} enricher(s) available: {}",
            enrichers.len(),
            enrichers
                .iter()
                .map(|e| e.name())
                .collect::<Vec<_>>()
                .join(", ")
        );

        // 1. Collect targets: unresolved refs + low-confidence edges
        let targets = self.collect_lsp_targets(namespace)?;
        if targets.is_empty() {
            tracing::info!("No LSP enrichment targets found");
            return Ok(LspEnrichStats::default());
        }

        tracing::info!(
            "Collected {} enrichment target files with refs to resolve",
            targets.len()
        );

        // 2. Build source_node lookup: (file, line) → source_node ID
        let mut source_lookup: HashMap<(String, usize), String> = HashMap::new();
        for target in &targets {
            for r in &target.refs {
                source_lookup.insert((r.file_path.clone(), r.line), r.source_node.clone());
            }
        }

        // 3. Run all available enrichers
        let results = lsp::run_enrichment(&targets, &enrichers, project_root);

        // 4. Apply results to the graph
        let mut stats = LspEnrichStats::default();
        for result in &results {
            self.apply_lsp_result(result, project_root, namespace, &source_lookup, &mut stats)?;
        }

        tracing::info!(
            "LSP enrichment complete: {} edges upgraded, {} ext: nodes created, {} type annotations",
            stats.edges_upgraded,
            stats.ext_nodes_created,
            stats.type_annotations_applied
        );

        Ok(stats)
    }

    /// Collect enrichment targets from unresolved refs and low-confidence edges.
    fn collect_lsp_targets(&self, namespace: &str) -> Result<Vec<EnrichmentTarget>, CodememError> {
        let mut by_file: HashMap<String, Vec<RefToResolve>> = HashMap::new();

        // Source 1: Unresolved refs from storage
        let pending: Vec<PendingUnresolvedRef> = self
            .storage
            .list_pending_unresolved_refs()
            .unwrap_or_default()
            .into_iter()
            .filter(|r| r.namespace == namespace)
            .collect();

        for r in &pending {
            by_file
                .entry(r.file_path.clone())
                .or_default()
                .push(RefToResolve {
                    source_node: r.source_node.clone(),
                    target_name: r.target_name.clone(),
                    file_path: r.file_path.clone(),
                    line: r.line,
                    current_confidence: None,
                });
        }

        // Source 2: Low-confidence edges (weight < 0.8) from the graph
        let edges = self
            .storage
            .graph_edges_for_namespace_with_cross(namespace, false)?;

        let graph = self.lock_graph()?;
        for edge in &edges {
            if edge.weight >= 0.8 {
                continue;
            }
            // Only consider ref edges (not structural Contains edges)
            if !edge.id.starts_with("ref:") {
                continue;
            }
            // Extract source file info from the source node
            let src_node = graph.get_node(&edge.src)?;
            if let Some(src) = src_node {
                let file_path = src
                    .payload
                    .get("file_path")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string();
                let line = src
                    .payload
                    .get("line_start")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as usize;

                let target_name = edge
                    .dst
                    .strip_prefix("sym:")
                    .unwrap_or(&edge.dst)
                    .to_string();

                if !file_path.is_empty() {
                    by_file
                        .entry(file_path.clone())
                        .or_default()
                        .push(RefToResolve {
                            source_node: edge.src.clone(),
                            target_name,
                            file_path,
                            line,
                            current_confidence: Some(edge.weight),
                        });
                }
            }
        }
        drop(graph);

        Ok(by_file
            .into_iter()
            .map(|(file_path, refs)| EnrichmentTarget { file_path, refs })
            .collect())
    }

    /// Apply a single enrichment result to the graph.
    fn apply_lsp_result(
        &self,
        result: &EnrichmentResult,
        project_root: &Path,
        namespace: &str,
        source_lookup: &HashMap<(String, usize), String>,
        stats: &mut LspEnrichStats,
    ) -> Result<(), CodememError> {
        let now = chrono::Utc::now();

        // Collect errors
        for err in &result.errors {
            stats.errors.push(err.clone());
        }

        // 1. Process resolved references
        for resolved in &result.resolved_refs {
            // Look up the actual source node ID from (file, line)
            let source_node_id = source_lookup
                .get(&(resolved.source_file.clone(), resolved.source_line))
                .cloned()
                .unwrap_or_else(|| {
                    // Fallback: use file-based ID (should rarely happen)
                    format!("sym:{}", resolved.source_file.replace('/', "."))
                });

            if resolved.is_external {
                self.create_ext_node(resolved, &source_node_id, namespace, stats)?;
            } else {
                self.upgrade_edge(resolved, &source_node_id, namespace, now, stats)?;
            }
        }

        // 2. Apply type annotations to existing sym: nodes
        for annotation in &result.type_annotations {
            self.apply_type_annotation(annotation, project_root, namespace, stats)?;
        }

        Ok(())
    }

    /// Create an ext: node for an external dependency reference.
    fn create_ext_node(
        &self,
        resolved: &LspResolvedRef,
        source_node_id: &str,
        namespace: &str,
        stats: &mut LspEnrichStats,
    ) -> Result<(), CodememError> {
        let pkg = resolved.package_name.as_deref().unwrap_or("unknown");
        let ext_id = format!("ext:{}.{}", pkg, resolved.target_symbol);

        let mut payload = HashMap::new();
        payload.insert(
            "package".to_string(),
            serde_json::Value::String(pkg.to_string()),
        );
        payload.insert(
            "target_file".to_string(),
            serde_json::Value::String(resolved.target_file.clone()),
        );
        payload.insert(
            "source".to_string(),
            serde_json::Value::String("lsp".to_string()),
        );

        let node = GraphNode {
            id: ext_id.clone(),
            kind: NodeKind::Function, // Default; LSP doesn't always distinguish
            label: resolved.target_symbol.clone(),
            payload,
            centrality: 0.0,
            memory_id: None,
            namespace: Some(namespace.to_string()),
        };

        self.storage.insert_graph_node(&node)?;
        {
            let mut graph = self.lock_graph()?;
            let _ = graph.add_node(node);
        }

        // Create edge from source to ext: node
        let edge = Edge {
            id: format!("lsp:{}->{}", source_node_id, ext_id),
            src: source_node_id.to_string(),
            dst: ext_id,
            relationship: RelationshipType::Calls,
            weight: 1.0,
            properties: {
                let mut props = HashMap::new();
                props.insert(
                    "source".to_string(),
                    serde_json::Value::String("lsp".to_string()),
                );
                if let Some(ref pkg_name) = resolved.package_name {
                    props.insert(
                        "package".to_string(),
                        serde_json::Value::String(pkg_name.clone()),
                    );
                }
                props
            },
            created_at: chrono::Utc::now(),
            valid_from: Some(chrono::Utc::now()),
            valid_to: None,
        };

        self.storage.insert_graph_edge(&edge)?;
        {
            let mut graph = self.lock_graph()?;
            let _ = graph.add_edge(edge);
        }

        stats.ext_nodes_created += 1;
        Ok(())
    }

    /// Upgrade an existing edge's confidence when LSP confirms it.
    fn upgrade_edge(
        &self,
        resolved: &LspResolvedRef,
        source_node_id: &str,
        _namespace: &str,
        now: chrono::DateTime<chrono::Utc>,
        stats: &mut LspEnrichStats,
    ) -> Result<(), CodememError> {
        // Find matching edges by source node
        let edges = self.storage.get_edges_for_node(source_node_id)?;

        // Look for an edge to the resolved target
        let target_suffix = &resolved.target_symbol;
        for existing in &edges {
            let dst_label = existing.dst.strip_prefix("sym:").unwrap_or(&existing.dst);

            // Match by target symbol suffix
            if dst_label == target_suffix
                || (dst_label.ends_with(target_suffix)
                    && dst_label[..dst_label.len() - target_suffix.len()].ends_with(['.', ':']))
            {
                // Upgrade: set confidence to 1.0, add "source": "lsp-confirmed"
                let mut upgraded = existing.clone();
                upgraded.weight = 1.0;
                upgraded.properties.insert(
                    "source".to_string(),
                    serde_json::Value::String("lsp-confirmed".to_string()),
                );
                upgraded.valid_from = Some(now);

                // INSERT OR REPLACE with same edge ID → upgrades in place
                self.storage.insert_graph_edge(&upgraded)?;
                {
                    let mut graph = self.lock_graph()?;
                    let _ = graph.add_edge(upgraded);
                }
                stats.edges_upgraded += 1;
                return Ok(());
            }
        }

        // No existing edge found — create a new LSP-sourced edge
        let target_node_id = format!("sym:{}", resolved.target_symbol);
        // Verify target exists in graph before creating edge
        let graph = self.lock_graph()?;
        if graph.get_node(&target_node_id)?.is_some() {
            drop(graph);

            let edge = Edge {
                id: format!("lsp:{source_node_id}->{target_node_id}"),
                src: source_node_id.to_string(),
                dst: target_node_id,
                relationship: RelationshipType::Calls,
                weight: 1.0,
                properties: {
                    let mut props = HashMap::new();
                    props.insert(
                        "source".to_string(),
                        serde_json::Value::String("lsp".to_string()),
                    );
                    props
                },
                created_at: now,
                valid_from: Some(now),
                valid_to: None,
            };

            self.storage.insert_graph_edge(&edge)?;
            {
                let mut graph = self.lock_graph()?;
                let _ = graph.add_edge(edge);
            }
            stats.edges_upgraded += 1;
        }

        Ok(())
    }

    /// Apply a type annotation to an existing sym: node's payload.
    fn apply_type_annotation(
        &self,
        annotation: &lsp::TypeAnnotation,
        _project_root: &Path,
        namespace: &str,
        stats: &mut LspEnrichStats,
    ) -> Result<(), CodememError> {
        // Find matching sym: node by file + line + name
        let graph = self.lock_graph()?;
        let mut target_id: Option<String> = None;

        for node in graph.get_all_nodes() {
            if !node.id.starts_with("sym:") {
                continue;
            }
            if node.namespace.as_deref() != Some(namespace) {
                continue;
            }

            let file_match = node.payload.get("file_path").and_then(|v| v.as_str())
                == Some(&annotation.file_path);

            let name_match = node.label.ends_with(&annotation.symbol_name)
                || node.label == annotation.symbol_name;

            if file_match && name_match {
                // Additional line proximity check
                let node_line = node
                    .payload
                    .get("line_start")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as usize;
                let line_close = annotation.line.abs_diff(node_line) <= 5;

                if line_close {
                    target_id = Some(node.id.clone());
                    break;
                }
            }
        }
        drop(graph);

        let Some(node_id) = target_id else {
            return Ok(());
        };

        // Update the node's payload with type information
        if let Ok(Some(mut node)) = self.storage.get_graph_node(&node_id) {
            let mut changed = false;

            if !annotation.resolved_type.is_empty() {
                node.payload.insert(
                    "resolved_type".to_string(),
                    serde_json::Value::String(annotation.resolved_type.clone()),
                );
                changed = true;
            }
            if let Some(ref rt) = annotation.return_type {
                node.payload.insert(
                    "return_type".to_string(),
                    serde_json::Value::String(rt.clone()),
                );
                changed = true;
            }
            if !annotation.generic_params.is_empty() {
                node.payload.insert(
                    "generic_params".to_string(),
                    serde_json::to_value(&annotation.generic_params).unwrap_or_default(),
                );
                changed = true;
            }

            if changed {
                // INSERT OR REPLACE to update
                self.storage.insert_graph_node(&node)?;
                {
                    let mut graph = self.lock_graph()?;
                    let _ = graph.add_node(node);
                }
                stats.type_annotations_applied += 1;
            }
        }

        Ok(())
    }
}
