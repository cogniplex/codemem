//! Graph persistence: persist indexing results (file/package/symbol/chunk nodes,
//! edges, embeddings, compaction) into the storage and graph backends.

mod compaction;
pub mod cross_repo;

use crate::index::{CodeChunk, ResolvedEdge, Symbol};
use crate::IndexAndResolveResult;
use codemem_core::{CodememError, Edge, GraphConfig, GraphNode, NodeKind, RelationshipType};
use std::collections::{HashMap, HashSet};

/// Counts of what was persisted by `persist_index_results`.
#[derive(Debug, Default)]
pub struct IndexPersistResult {
    pub files_created: usize,
    pub packages_created: usize,
    pub symbols_stored: usize,
    pub chunks_stored: usize,
    pub edges_resolved: usize,
    pub symbols_embedded: usize,
    pub chunks_embedded: usize,
    pub chunks_pruned: usize,
    pub symbols_pruned: usize,
}

/// Counts of what was persisted by `persist_cross_repo_data`.
#[derive(Debug, Default)]
pub struct CrossRepoPersistResult {
    pub packages_registered: usize,
    pub unresolved_refs_stored: usize,
    pub forward_edges_created: usize,
    pub backward_edges_created: usize,
    pub endpoints_detected: usize,
    pub client_calls_detected: usize,
}

/// Return the edge weight for a given relationship type, using config overrides
/// for the three most common types (Contains, Calls, Imports).
pub fn edge_weight_for(rel: &RelationshipType, config: &GraphConfig) -> f64 {
    match rel {
        RelationshipType::Calls => config.calls_edge_weight,
        RelationshipType::Imports => config.imports_edge_weight,
        RelationshipType::Contains => config.contains_edge_weight,
        RelationshipType::TypeDefinition => config.type_definition_edge_weight,
        RelationshipType::Reads => config.reads_edge_weight,
        RelationshipType::Writes => config.writes_edge_weight,
        RelationshipType::Overrides => config.overrides_edge_weight,
        RelationshipType::Implements | RelationshipType::Inherits => 0.8,
        RelationshipType::DependsOn => 0.7,
        RelationshipType::CoChanged => 0.6,
        RelationshipType::EvolvedInto | RelationshipType::Summarizes => 0.7,
        RelationshipType::PartOf => 0.4,
        RelationshipType::RelatesTo | RelationshipType::SharesTheme => 0.3,
        _ => 0.5,
    }
}

/// Intermediate counts from graph node persistence (before embedding).
struct GraphPersistCounts {
    packages_created: usize,
    chunks_stored: usize,
}

impl super::CodememEngine {
    /// Persist all indexing results (file nodes, package tree, symbol nodes, chunk nodes,
    /// edges, embeddings, compaction) into storage and the in-memory graph.
    ///
    /// This is the full persistence pipeline called after `Indexer::index_and_resolve()`.
    pub fn persist_index_results(
        &self,
        results: &IndexAndResolveResult,
        namespace: Option<&str>,
    ) -> Result<IndexPersistResult, CodememError> {
        self.persist_index_results_with_progress(results, namespace, |_, _| {})
    }

    /// Like `persist_index_results`, but skips the embedding phase entirely.
    /// Stores graph nodes, edges, and chunks without vectorizing them.
    /// Also skips cross-repo linking — this is a fast graph-only mode intended
    /// for rapid iteration (e.g., `--skip-embed`). Run a full `analyze` to
    /// populate cross-repo data.
    pub fn persist_graph_only(
        &self,
        results: &IndexAndResolveResult,
        namespace: Option<&str>,
    ) -> Result<IndexPersistResult, CodememError> {
        let seen_files = &results.file_paths;
        let graph_counts = self.persist_graph_nodes(results, namespace)?;

        let (chunks_pruned, symbols_pruned) = if self.config.chunking.auto_compact {
            self.compact_graph(seen_files)
        } else {
            (0, 0)
        };

        Ok(IndexPersistResult {
            files_created: seen_files.len(),
            packages_created: graph_counts.packages_created,
            symbols_stored: results.symbols.len(),
            chunks_stored: graph_counts.chunks_stored,
            edges_resolved: results.edges.len(),
            symbols_embedded: 0,
            chunks_embedded: 0,
            chunks_pruned,
            symbols_pruned,
        })
    }

    /// Like `persist_index_results`, but calls `on_progress(done, total)` during
    /// the embedding phase so callers can display progress.
    pub fn persist_index_results_with_progress(
        &self,
        results: &IndexAndResolveResult,
        namespace: Option<&str>,
        on_progress: impl Fn(usize, usize),
    ) -> Result<IndexPersistResult, CodememError> {
        let seen_files = &results.file_paths;

        // 1. Persist all graph nodes and edges
        let graph_counts = self.persist_graph_nodes(results, namespace)?;

        // 2. Embed symbols and chunks
        let (symbols_embedded, chunks_embedded) = self.embed_and_persist(
            &results.symbols,
            &results.chunks,
            &results.edges,
            on_progress,
        )?;

        // 3. Auto-compact
        let (chunks_pruned, symbols_pruned) = if self.config.chunking.auto_compact {
            self.compact_graph(seen_files)
        } else {
            (0, 0)
        };

        Ok(IndexPersistResult {
            files_created: seen_files.len(),
            packages_created: graph_counts.packages_created,
            symbols_stored: results.symbols.len(),
            chunks_stored: graph_counts.chunks_stored,
            edges_resolved: results.edges.len(),
            symbols_embedded,
            chunks_embedded,
            chunks_pruned,
            symbols_pruned,
        })
    }

    // ── Graph Node Persistence ───────────────────────────────────────────

    /// Persist file, package, symbol, chunk nodes and all edges into storage
    /// and the in-memory graph. Returns counts for the result struct.
    fn persist_graph_nodes(
        &self,
        results: &IndexAndResolveResult,
        namespace: Option<&str>,
    ) -> Result<GraphPersistCounts, CodememError> {
        let all_symbols = &results.symbols;
        let all_chunks = &results.chunks;
        let seen_files = &results.file_paths;
        let edges = &results.edges;

        let now = chrono::Utc::now();
        let ns_string = namespace.map(|s| s.to_string());
        let contains_weight = edge_weight_for(&RelationshipType::Contains, &self.config.graph);

        let mut graph = self.lock_graph()?;

        // ── File nodes
        let file_nodes: Vec<GraphNode> = seen_files
            .iter()
            .map(|file_path| {
                let mut payload = HashMap::new();
                payload.insert(
                    "file_path".to_string(),
                    serde_json::Value::String(file_path.clone()),
                );
                GraphNode {
                    id: format!("file:{file_path}"),
                    kind: NodeKind::File,
                    label: file_path.clone(),
                    payload,
                    centrality: 0.0,
                    memory_id: None,
                    namespace: ns_string.clone(),
                }
            })
            .collect();
        self.persist_nodes_to_storage_and_graph(&file_nodes, &mut **graph);

        // ── Package (directory) nodes
        let (dir_nodes, dir_edges, created_dirs) =
            self.build_package_tree(seen_files, &ns_string, contains_weight, now, &**graph);
        self.persist_nodes_to_storage_and_graph(&dir_nodes, &mut **graph);
        self.persist_edges_to_storage_and_graph(&dir_edges, &mut **graph);

        // ── Symbol nodes + file→symbol edges
        let (sym_nodes, sym_edges) =
            Self::build_symbol_nodes(all_symbols, &ns_string, contains_weight, now);

        // Clean up stale symbols: single pass over in-memory graph to collect
        // existing symbols grouped by file, then diff against new parse results.
        //
        // Lock protocol: We collect old symbols while holding the graph lock,
        // then drop it so `cleanup_stale_symbols` can acquire graph + vector
        // locks internally. The re-acquire below is safe: cleanup only removes
        // stale nodes that won't conflict with the inserts that follow.
        let mut old_syms_by_file: HashMap<String, HashSet<String>> = HashMap::new();
        for node in graph.get_all_nodes() {
            if !node.id.starts_with("sym:") {
                continue;
            }
            // Skip SCIP-sourced symbols (explicit and synthetic containment nodes)
            // — they're managed by the SCIP pipeline, not ast-grep. Without this
            // guard, re-indexing deletes all SCIP sym: nodes because their IDs
            // don't match ast-grep's qualified names.
            if matches!(
                node.payload.get("source").and_then(|v| v.as_str()),
                Some("scip" | "scip-synthetic")
            ) {
                continue;
            }
            let Some(fp) = node.payload.get("file_path").and_then(|v| v.as_str()) else {
                continue;
            };
            if !seen_files.contains(fp) {
                continue;
            }
            old_syms_by_file
                .entry(fp.to_string())
                .or_default()
                .insert(node.id);
        }
        drop(graph);
        for file_path in seen_files {
            let new_sym_ids: HashSet<String> = sym_nodes
                .iter()
                .filter(|n| {
                    n.payload.get("file_path").and_then(|v| v.as_str()) == Some(file_path.as_str())
                })
                .map(|n| n.id.clone())
                .collect();
            let empty = HashSet::new();
            let old_sym_ids = old_syms_by_file.get(file_path).unwrap_or(&empty);
            if let Err(e) = self.cleanup_stale_symbols(file_path, old_sym_ids, &new_sym_ids) {
                tracing::warn!("Failed to cleanup stale symbols for {file_path}: {e}");
            }
        }
        let mut graph = self.lock_graph()?; // Re-acquire lock

        self.persist_nodes_to_storage_and_graph(&sym_nodes, &mut **graph);
        self.persist_edges_to_storage_and_graph(&sym_edges, &mut **graph);

        // ── Resolved reference edges
        let ref_edges = Self::build_reference_edges(edges, &self.config.graph, now);
        self.persist_edges_to_storage_and_graph(&ref_edges, &mut **graph);

        // ── SCIP nodes + edges (compiler-grade)
        if let Some(ref scip_build) = results.scip_build {
            // Clean up stale SCIP nodes: collect existing SCIP-sourced sym: nodes
            // for files covered by this SCIP run, then remove any not in the new set.
            let new_scip_ids: HashSet<&str> =
                scip_build.nodes.iter().map(|n| n.id.as_str()).collect();
            let mut stale_scip_ids = Vec::new();
            for node in graph.get_all_nodes() {
                if !node.id.starts_with("sym:") {
                    continue;
                }
                if !matches!(
                    node.payload.get("source").and_then(|v| v.as_str()),
                    Some("scip" | "scip-synthetic")
                ) {
                    continue;
                }
                if !new_scip_ids.contains(node.id.as_str()) {
                    // Only clean up nodes in files that SCIP covered this run.
                    if let Some(fp) = node.payload.get("file_path").and_then(|v| v.as_str()) {
                        if seen_files.contains(fp) {
                            stale_scip_ids.push(node.id.clone());
                        }
                    }
                }
            }
            for stale_id in &stale_scip_ids {
                let _ = graph.remove_node(stale_id);
                let _ = self.storage.delete_graph_nodes_by_prefix(stale_id);
                // Clean up orphan doc memories for removed symbols.
                if let Some(qname) = stale_id.strip_prefix("sym:") {
                    let doc_id = format!("scip-doc:{qname}");
                    let _ = self.storage.delete_memory(&doc_id);
                }
            }
            if !stale_scip_ids.is_empty() {
                tracing::info!(
                    "Cleaned up {} stale SCIP nodes from re-index",
                    stale_scip_ids.len()
                );
            }

            self.persist_nodes_to_storage_and_graph(&scip_build.nodes, &mut **graph);

            // Multi-layer fusion: merge confidence when ast-grep and SCIP agree.
            // Superseded ast-grep edges are removed to avoid duplicates.
            let (fused_edges, superseded_ids) = Self::fuse_edges(&ref_edges, &scip_build.edges);

            // Remove the low-confidence ast-grep edges that were fused into SCIP edges.
            for edge_id in &superseded_ids {
                let _ = graph.remove_edge(edge_id);
                let _ = self.storage.delete_graph_edge(edge_id);
            }

            self.persist_edges_to_storage_and_graph(&fused_edges, &mut **graph);

            // Persist hover doc memories and their RELATES_TO edges.
            for (memory, related_node_id) in &scip_build.memories {
                let _ = self.storage.insert_memory(memory);
                let relates_edge = Edge {
                    id: format!("relates:{}->mem:{}", related_node_id, memory.id),
                    src: related_node_id.clone(),
                    dst: format!("mem:{}", memory.id),
                    relationship: RelationshipType::RelatesTo,
                    weight: 0.3,
                    properties: HashMap::new(),
                    created_at: now,
                    valid_from: Some(now),
                    valid_to: None,
                };
                let _ = graph.add_edge(relates_edge.clone());
                let _ = self.storage.insert_graph_edges_batch(&[relates_edge]);
            }
        }

        // ── Chunk nodes + file→chunk / symbol→chunk edges
        for file_path in seen_files {
            let prefix = format!("chunk:{file_path}:");
            let _ = self.storage.delete_graph_nodes_by_prefix(&prefix);
        }
        let (chunk_nodes, chunk_edges) =
            Self::build_chunk_nodes(all_chunks, &ns_string, contains_weight, now);
        let chunk_count = chunk_nodes.len();
        self.persist_nodes_to_storage_and_graph(&chunk_nodes, &mut **graph);
        self.persist_edges_to_storage_and_graph(&chunk_edges, &mut **graph);

        drop(graph);

        Ok(GraphPersistCounts {
            packages_created: created_dirs,
            chunks_stored: chunk_count,
        })
    }

    /// Batch-insert nodes into both SQLite and the in-memory graph.
    fn persist_nodes_to_storage_and_graph(
        &self,
        nodes: &[GraphNode],
        graph: &mut dyn codemem_core::GraphBackend,
    ) {
        if let Err(e) = self.storage.insert_graph_nodes_batch(nodes) {
            tracing::warn!("Failed to batch-insert {} graph nodes: {e}", nodes.len());
        }
        for node in nodes {
            let _ = graph.add_node(node.clone());
        }
    }

    /// Batch-insert edges into both SQLite and the in-memory graph.
    fn persist_edges_to_storage_and_graph(
        &self,
        edges: &[Edge],
        graph: &mut dyn codemem_core::GraphBackend,
    ) {
        if let Err(e) = self.storage.insert_graph_edges_batch(edges) {
            tracing::warn!("Failed to batch-insert {} graph edges: {e}", edges.len());
        }
        for edge in edges {
            let _ = graph.add_edge(edge.clone());
        }
    }

    /// Build directory/package nodes and CONTAINS edges from file paths.
    /// Returns (nodes, edges, number_of_dirs_created).
    fn build_package_tree(
        &self,
        seen_files: &HashSet<String>,
        ns_string: &Option<String>,
        contains_weight: f64,
        now: chrono::DateTime<chrono::Utc>,
        graph: &dyn codemem_core::GraphBackend,
    ) -> (Vec<GraphNode>, Vec<Edge>, usize) {
        let mut created_dirs: HashSet<String> = HashSet::new();
        let mut dir_nodes = Vec::new();
        let mut dir_edges = Vec::new();

        for file_path in seen_files {
            let p = std::path::Path::new(file_path);
            let mut ancestors: Vec<String> = Vec::new();
            let mut current = p.parent();
            while let Some(dir) = current {
                let dir_str = dir.to_string_lossy().to_string();
                if dir_str.is_empty() || dir_str == "." {
                    break;
                }
                ancestors.push(dir_str);
                current = dir.parent();
            }
            ancestors.reverse();
            for (i, dir_str) in ancestors.iter().enumerate() {
                let pkg_id = format!("pkg:{dir_str}/");
                if created_dirs.insert(pkg_id.clone()) {
                    dir_nodes.push(GraphNode {
                        id: pkg_id.clone(),
                        kind: NodeKind::Package,
                        label: format!("{dir_str}/"),
                        payload: HashMap::new(),
                        centrality: 0.0,
                        memory_id: None,
                        namespace: ns_string.clone(),
                    });
                }
                if i == 0 {
                    continue;
                }
                let parent_pkg_id = format!("pkg:{}/", ancestors[i - 1]);
                let edge_id = format!("contains:{parent_pkg_id}->{pkg_id}");
                if graph
                    .get_edges(&parent_pkg_id)
                    .unwrap_or_default()
                    .iter()
                    .any(|e| e.id == edge_id)
                {
                    continue;
                }
                dir_edges.push(Edge {
                    id: edge_id,
                    src: parent_pkg_id,
                    dst: pkg_id.clone(),
                    relationship: RelationshipType::Contains,
                    weight: contains_weight,
                    valid_from: Some(now),
                    valid_to: None,
                    properties: HashMap::new(),
                    created_at: now,
                });
            }
            if let Some(last_dir) = ancestors.last() {
                let parent_pkg_id = format!("pkg:{last_dir}/");
                let file_node_id = format!("file:{file_path}");
                let edge_id = format!("contains:{parent_pkg_id}->{file_node_id}");
                dir_edges.push(Edge {
                    id: edge_id,
                    src: parent_pkg_id,
                    dst: file_node_id,
                    relationship: RelationshipType::Contains,
                    weight: contains_weight,
                    valid_from: Some(now),
                    valid_to: None,
                    properties: HashMap::new(),
                    created_at: now,
                });
            }
        }

        let count = created_dirs.len();
        (dir_nodes, dir_edges, count)
    }

    /// Build symbol graph nodes and file→symbol CONTAINS edges.
    fn build_symbol_nodes(
        symbols: &[Symbol],
        ns_string: &Option<String>,
        contains_weight: f64,
        now: chrono::DateTime<chrono::Utc>,
    ) -> (Vec<GraphNode>, Vec<Edge>) {
        let mut sym_nodes = Vec::with_capacity(symbols.len());
        let mut sym_edges = Vec::with_capacity(symbols.len());

        for sym in symbols {
            let kind = NodeKind::from(sym.kind);
            let payload = Self::build_symbol_payload(sym);

            let sym_node_id = format!("sym:{}", sym.qualified_name);
            sym_nodes.push(GraphNode {
                id: sym_node_id.clone(),
                kind,
                label: sym.qualified_name.clone(),
                payload,
                centrality: 0.0,
                memory_id: None,
                namespace: ns_string.clone(),
            });

            let file_node_id = format!("file:{}", sym.file_path);
            sym_edges.push(Edge {
                id: format!("contains:{file_node_id}->{sym_node_id}"),
                src: file_node_id,
                dst: sym_node_id,
                relationship: RelationshipType::Contains,
                weight: contains_weight,
                valid_from: Some(now),
                valid_to: None,
                properties: HashMap::new(),
                created_at: now,
            });
        }

        (sym_nodes, sym_edges)
    }

    /// Build the payload HashMap for a symbol's graph node.
    fn build_symbol_payload(sym: &Symbol) -> HashMap<String, serde_json::Value> {
        let mut payload = HashMap::new();
        payload.insert(
            "symbol_kind".to_string(),
            serde_json::Value::String(sym.kind.to_string()),
        );
        payload.insert(
            "signature".to_string(),
            serde_json::Value::String(sym.signature.clone()),
        );
        payload.insert(
            "file_path".to_string(),
            serde_json::Value::String(sym.file_path.clone()),
        );
        payload.insert("line_start".to_string(), serde_json::json!(sym.line_start));
        payload.insert("line_end".to_string(), serde_json::json!(sym.line_end));
        payload.insert(
            "visibility".to_string(),
            serde_json::Value::String(sym.visibility.to_string()),
        );
        if let Some(ref doc) = sym.doc_comment {
            payload.insert(
                "doc_comment".to_string(),
                serde_json::Value::String(doc.clone()),
            );
        }
        if !sym.parameters.is_empty() {
            payload.insert(
                "parameters".to_string(),
                serde_json::to_value(&sym.parameters).unwrap_or_default(),
            );
        }
        if let Some(ref ret) = sym.return_type {
            payload.insert(
                "return_type".to_string(),
                serde_json::Value::String(ret.clone()),
            );
        }
        if sym.is_async {
            payload.insert("is_async".to_string(), serde_json::json!(true));
        }
        if !sym.attributes.is_empty() {
            payload.insert(
                "attributes".to_string(),
                serde_json::to_value(&sym.attributes).unwrap_or_default(),
            );
        }
        if !sym.throws.is_empty() {
            payload.insert(
                "throws".to_string(),
                serde_json::to_value(&sym.throws).unwrap_or_default(),
            );
        }
        if let Some(ref gp) = sym.generic_params {
            payload.insert(
                "generic_params".to_string(),
                serde_json::Value::String(gp.clone()),
            );
        }
        if sym.is_abstract {
            payload.insert("is_abstract".to_string(), serde_json::json!(true));
        }
        if let Some(ref parent) = sym.parent {
            payload.insert(
                "parent".to_string(),
                serde_json::Value::String(parent.clone()),
            );
        }
        payload
    }

    /// Build edges from resolved cross-file references.
    /// ast-grep base confidence for multi-layer fusion.
    const AST_GREP_BASE_CONFIDENCE: f64 = 0.10;

    fn build_reference_edges(
        edges: &[ResolvedEdge],
        graph_config: &GraphConfig,
        now: chrono::DateTime<chrono::Utc>,
    ) -> Vec<Edge> {
        edges
            .iter()
            .map(|edge| {
                let mut properties = HashMap::new();
                properties.insert("source".to_string(), serde_json::json!("ast-grep"));
                properties.insert(
                    "confidence".to_string(),
                    serde_json::json!(Self::AST_GREP_BASE_CONFIDENCE),
                );
                properties.insert("source_layers".to_string(), serde_json::json!(["ast-grep"]));
                Edge {
                    id: format!(
                        "ref:{}->{}:{}",
                        edge.source_qualified_name, edge.target_qualified_name, edge.relationship
                    ),
                    src: format!("sym:{}", edge.source_qualified_name),
                    dst: format!("sym:{}", edge.target_qualified_name),
                    relationship: edge.relationship,
                    weight: edge_weight_for(&edge.relationship, graph_config),
                    valid_from: Some(now),
                    valid_to: None,
                    properties,
                    created_at: now,
                }
            })
            .collect()
    }

    /// Multi-layer edge fusion: when ast-grep and SCIP produce the same edge
    /// (same src, dst, relationship), sum their confidences and merge source_layers.
    /// SCIP edges not in ast-grep pass through unchanged.
    ///
    /// Returns `(fused_scip_edges, superseded_ast_grep_edge_ids)`. The caller must
    /// remove the superseded ast-grep edges to avoid duplicates in the graph.
    fn fuse_edges(ast_grep_edges: &[Edge], scip_edges: &[Edge]) -> (Vec<Edge>, Vec<String>) {
        // Index ast-grep edges by (src, dst, relationship_str) → edge ID for O(1) lookup.
        let ast_grep_index: HashMap<(String, String, String), &str> = ast_grep_edges
            .iter()
            .map(|e| {
                (
                    (e.src.clone(), e.dst.clone(), e.relationship.to_string()),
                    e.id.as_str(),
                )
            })
            .collect();

        let mut superseded_ids = Vec::new();

        let fused = scip_edges
            .iter()
            .map(|scip_edge| {
                let key = (
                    scip_edge.src.clone(),
                    scip_edge.dst.clone(),
                    scip_edge.relationship.to_string(),
                );
                if let Some(&ast_edge_id) = ast_grep_index.get(&key) {
                    // Both layers agree — fuse confidence and mark ast-grep edge for removal.
                    superseded_ids.push(ast_edge_id.to_string());
                    let mut fused = scip_edge.clone();
                    let scip_conf = scip_edge
                        .properties
                        .get("confidence")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.15);
                    let fused_conf = scip_conf + Self::AST_GREP_BASE_CONFIDENCE;
                    fused
                        .properties
                        .insert("confidence".to_string(), serde_json::json!(fused_conf));
                    fused.properties.insert(
                        "source_layers".to_string(),
                        serde_json::json!(["ast-grep", "scip"]),
                    );
                    fused
                } else {
                    scip_edge.clone()
                }
            })
            .collect();

        (fused, superseded_ids)
    }

    /// Build chunk graph nodes and file→chunk / symbol→chunk CONTAINS edges.
    fn build_chunk_nodes(
        chunks: &[CodeChunk],
        ns_string: &Option<String>,
        contains_weight: f64,
        now: chrono::DateTime<chrono::Utc>,
    ) -> (Vec<GraphNode>, Vec<Edge>) {
        let mut chunk_nodes = Vec::with_capacity(chunks.len());
        let mut chunk_edges = Vec::with_capacity(chunks.len() * 2);

        for chunk in chunks {
            let chunk_id = format!("chunk:{}:{}", chunk.file_path, chunk.index);

            let mut payload = HashMap::new();
            payload.insert(
                "file_path".to_string(),
                serde_json::Value::String(chunk.file_path.clone()),
            );
            payload.insert(
                "line_start".to_string(),
                serde_json::json!(chunk.line_start),
            );
            payload.insert("line_end".to_string(), serde_json::json!(chunk.line_end));
            payload.insert(
                "node_kind".to_string(),
                serde_json::Value::String(chunk.node_kind.clone()),
            );
            payload.insert(
                "non_ws_chars".to_string(),
                serde_json::json!(chunk.non_ws_chars),
            );
            if let Some(ref parent) = chunk.parent_symbol {
                payload.insert(
                    "parent_symbol".to_string(),
                    serde_json::Value::String(parent.clone()),
                );
            }

            chunk_nodes.push(GraphNode {
                id: chunk_id.clone(),
                kind: NodeKind::Chunk,
                label: format!(
                    "chunk:{}:{}..{}",
                    chunk.file_path, chunk.line_start, chunk.line_end
                ),
                payload,
                centrality: 0.0,
                memory_id: None,
                namespace: ns_string.clone(),
            });

            let file_node_id = format!("file:{}", chunk.file_path);
            chunk_edges.push(Edge {
                id: format!("contains:{file_node_id}->{chunk_id}"),
                src: file_node_id,
                dst: chunk_id.clone(),
                relationship: RelationshipType::Contains,
                weight: contains_weight,
                valid_from: Some(now),
                valid_to: None,
                properties: HashMap::new(),
                created_at: now,
            });

            if let Some(ref parent_sym) = chunk.parent_symbol {
                let parent_node_id = format!("sym:{parent_sym}");
                chunk_edges.push(Edge {
                    id: format!("contains:{parent_node_id}->{chunk_id}"),
                    src: parent_node_id,
                    dst: chunk_id,
                    relationship: RelationshipType::Contains,
                    weight: contains_weight,
                    valid_from: Some(now),
                    valid_to: None,
                    properties: HashMap::new(),
                    created_at: now,
                });
            }
        }

        (chunk_nodes, chunk_edges)
    }

    // ── Embedding Persistence ────────────────────────────────────────────

    /// Embed symbols and chunks, persisting embeddings to SQLite and the
    /// vector index in batches with progress reporting.
    ///
    /// Returns (symbols_embedded, chunks_embedded).
    fn embed_and_persist(
        &self,
        symbols: &[Symbol],
        chunks: &[CodeChunk],
        edges: &[ResolvedEdge],
        on_progress: impl Fn(usize, usize),
    ) -> Result<(usize, usize), CodememError> {
        let mut symbols_embedded = 0usize;
        let mut chunks_embedded = 0usize;

        // Quick check: skip expensive text enrichment if embedding provider isn't loaded.
        // This avoids triggering lazy init during lightweight operations (hooks).
        if !self.embeddings_ready() {
            return Ok((0, 0));
        }

        // Phase 1: Collect enriched texts without holding any lock.
        let sym_texts: Vec<(String, String)> = symbols
            .iter()
            .map(|sym| {
                let id = format!("sym:{}", sym.qualified_name);
                let text = self.enrich_symbol_text(sym, edges);
                (id, text)
            })
            .collect();
        let chunk_texts: Vec<(String, String)> = chunks
            .iter()
            .map(|chunk| {
                let id = format!("chunk:{}:{}", chunk.file_path, chunk.index);
                let text = self.enrich_chunk_text(chunk);
                (id, text)
            })
            .collect();

        // Phase 2+3: Embed in batches and persist progressively.
        let embed_batch_size = self.config.embedding.batch_size;

        let all_pairs: Vec<(String, String)> = sym_texts.into_iter().chain(chunk_texts).collect();
        let total = all_pairs.len();
        let sym_count = symbols.len();
        let mut done = 0usize;

        for batch in all_pairs.chunks(embed_batch_size) {
            let texts: Vec<&str> = batch.iter().map(|(_, t)| t.as_str()).collect();

            let t0 = std::time::Instant::now();
            let embed_result = {
                let emb = self.lock_embeddings()?;
                match emb {
                    Some(emb_guard) => emb_guard.embed_batch(&texts),
                    None => break,
                }
            };

            match embed_result {
                Ok(embeddings) => {
                    let embed_ms = t0.elapsed().as_millis();

                    let t1 = std::time::Instant::now();
                    let pairs: Vec<(&str, &[f32])> = batch
                        .iter()
                        .zip(embeddings.iter())
                        .map(|((id, _), emb_vec)| (id.as_str(), emb_vec.as_slice()))
                        .collect();
                    if let Err(e) = self.storage.store_embeddings_batch(&pairs) {
                        tracing::warn!("Failed to batch-store embeddings: {e}");
                    }
                    let sqlite_ms = t1.elapsed().as_millis();

                    let t2 = std::time::Instant::now();
                    let batch_items: Vec<(String, Vec<f32>)> = batch
                        .iter()
                        .zip(embeddings.into_iter())
                        .map(|((id, _), emb_vec)| (id.clone(), emb_vec))
                        .collect();
                    let batch_len = batch_items.len();
                    {
                        let mut vec = self.lock_vector()?;
                        if let Err(e) = vec.insert_batch(&batch_items) {
                            tracing::warn!("Failed to batch-insert into vector index: {e}");
                        }
                    }
                    let vector_ms = t2.elapsed().as_millis();

                    let syms_in_batch = batch_len.min(sym_count.saturating_sub(done));
                    symbols_embedded += syms_in_batch;
                    chunks_embedded += batch_len - syms_in_batch;
                    done += batch_len;

                    tracing::debug!(
                        "Embed batch {}: embed={embed_ms}ms sqlite={sqlite_ms}ms vector={vector_ms}ms",
                        batch_len
                    );
                }
                Err(e) => {
                    tracing::warn!("embed_batch failed for chunk of {} texts: {e}", batch.len());
                }
            }
            on_progress(done, total);
        }
        self.save_index();

        Ok((symbols_embedded, chunks_embedded))
    }
}
