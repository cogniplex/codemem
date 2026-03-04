//! Graph persistence: persist indexing results (file/package/symbol/chunk nodes,
//! edges, embeddings, compaction) into the storage and graph backends.

use crate::IndexAndResolveResult;
use codemem_core::{
    CodememError, Edge, GraphBackend, GraphConfig, GraphNode, NodeKind, RelationshipType,
    VectorBackend,
};
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

/// Return the edge weight for a given relationship type, using config overrides
/// for the three most common types (Contains, Calls, Imports).
pub fn edge_weight_for(rel: &RelationshipType, config: &GraphConfig) -> f64 {
    match rel {
        RelationshipType::Calls => config.calls_edge_weight,
        RelationshipType::Imports => config.imports_edge_weight,
        RelationshipType::Contains => config.contains_edge_weight,
        RelationshipType::Implements | RelationshipType::Inherits => 0.8,
        RelationshipType::DependsOn => 0.7,
        RelationshipType::CoChanged => 0.6,
        RelationshipType::EvolvedInto | RelationshipType::Summarizes => 0.7,
        RelationshipType::PartOf => 0.4,
        RelationshipType::RelatesTo | RelationshipType::SharesTheme => 0.3,
        _ => 0.5,
    }
}

/// Check whether a graph node has any edge linking it to a memory node
/// (i.e. an edge whose other endpoint is not a code-structural ID).
fn has_memory_link_edge(graph: &dyn GraphBackend, node_id: &str) -> bool {
    graph
        .get_edges(node_id)
        .map(|edges| {
            edges.iter().any(|e| {
                let other = if e.src == node_id { &e.dst } else { &e.src };
                !other.starts_with("sym:")
                    && !other.starts_with("file:")
                    && !other.starts_with("chunk:")
                    && !other.starts_with("pkg:")
                    && !other.starts_with("contains:")
                    && !other.starts_with("ref:")
            })
        })
        .unwrap_or(false)
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
        let all_symbols = &results.symbols;
        let all_chunks = &results.chunks;
        let seen_files = &results.file_paths;
        let edges = &results.edges;

        let now = chrono::Utc::now();
        let ns_string = namespace.map(|s| s.to_string());

        let mut graph = self.lock_graph()?;

        // ── File nodes ──────────────────────────────────────────────────────
        for file_path in seen_files {
            let node_id = format!("file:{file_path}");
            let mut payload = HashMap::new();
            payload.insert(
                "file_path".to_string(),
                serde_json::Value::String(file_path.clone()),
            );
            let node = GraphNode {
                id: node_id,
                kind: NodeKind::File,
                label: file_path.clone(),
                payload,
                centrality: 0.0,
                memory_id: None,
                namespace: ns_string.clone(),
            };
            let _ = self.storage.insert_graph_node(&node);
            let _ = graph.add_node(node);
        }

        // ── Package (directory) nodes ───────────────────────────────────────
        let mut created_dirs: HashSet<String> = HashSet::new();
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
                    let node = GraphNode {
                        id: pkg_id.clone(),
                        kind: NodeKind::Package,
                        label: format!("{dir_str}/"),
                        payload: HashMap::new(),
                        centrality: 0.0,
                        memory_id: None,
                        namespace: ns_string.clone(),
                    };
                    let _ = self.storage.insert_graph_node(&node);
                    let _ = graph.add_node(node);
                }
                // CONTAINS edge from parent dir to this dir
                if i > 0 {
                    let parent_pkg_id = format!("pkg:{}/", ancestors[i - 1]);
                    let edge_id = format!("contains:{parent_pkg_id}->{pkg_id}");
                    if !graph
                        .get_edges(&parent_pkg_id)
                        .unwrap_or_default()
                        .iter()
                        .any(|e| e.id == edge_id)
                    {
                        let edge = Edge {
                            id: edge_id,
                            src: parent_pkg_id,
                            dst: pkg_id.clone(),
                            relationship: RelationshipType::Contains,
                            weight: edge_weight_for(
                                &RelationshipType::Contains,
                                &self.config.graph,
                            ),
                            valid_from: None,
                            valid_to: None,
                            properties: HashMap::new(),
                            created_at: now,
                        };
                        let _ = self.storage.insert_graph_edge(&edge);
                        let _ = graph.add_edge(edge);
                    }
                }
            }
            // CONTAINS edge from innermost directory to file
            if let Some(last_dir) = ancestors.last() {
                let parent_pkg_id = format!("pkg:{last_dir}/");
                let file_node_id = format!("file:{file_path}");
                let edge_id = format!("contains:{parent_pkg_id}->{file_node_id}");
                let edge = Edge {
                    id: edge_id,
                    src: parent_pkg_id,
                    dst: file_node_id,
                    relationship: RelationshipType::Contains,
                    weight: edge_weight_for(&RelationshipType::Contains, &self.config.graph),
                    valid_from: None,
                    valid_to: None,
                    properties: HashMap::new(),
                    created_at: now,
                };
                let _ = self.storage.insert_graph_edge(&edge);
                let _ = graph.add_edge(edge);
            }
        }

        // ── Symbol nodes ────────────────────────────────────────────────────
        for sym in all_symbols {
            let kind = NodeKind::from(sym.kind);

            let mut payload = HashMap::new();
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

            let node = GraphNode {
                id: format!("sym:{}", sym.qualified_name),
                kind,
                label: sym.qualified_name.clone(),
                payload,
                centrality: 0.0,
                memory_id: None,
                namespace: ns_string.clone(),
            };

            let sym_node_id = node.id.clone();
            let _ = self.storage.insert_graph_node(&node);
            let _ = graph.add_node(node);

            // CONTAINS edge: file → symbol
            let file_node_id = format!("file:{}", sym.file_path);
            let contains_edge = Edge {
                id: format!("contains:{file_node_id}->{sym_node_id}"),
                src: file_node_id,
                dst: sym_node_id,
                relationship: RelationshipType::Contains,
                weight: edge_weight_for(&RelationshipType::Contains, &self.config.graph),
                valid_from: None,
                valid_to: None,
                properties: HashMap::new(),
                created_at: now,
            };
            let _ = self.storage.insert_graph_edge(&contains_edge);
            let _ = graph.add_edge(contains_edge);
        }

        // ── Resolved reference edges ────────────────────────────────────────
        for edge in edges {
            let e = Edge {
                id: format!(
                    "ref:{}->{}:{}",
                    edge.source_qualified_name, edge.target_qualified_name, edge.relationship
                ),
                src: format!("sym:{}", edge.source_qualified_name),
                dst: format!("sym:{}", edge.target_qualified_name),
                relationship: edge.relationship,
                weight: edge_weight_for(&edge.relationship, &self.config.graph),
                valid_from: None,
                valid_to: None,
                properties: HashMap::new(),
                created_at: now,
            };
            let _ = self.storage.insert_graph_edge(&e);
            let _ = graph.add_edge(e);
        }

        // ── Chunk nodes ─────────────────────────────────────────────────────
        // Cleanup stale chunk nodes for files being re-indexed
        for file_path in seen_files {
            let prefix = format!("chunk:{file_path}:");
            let _ = self.storage.delete_graph_nodes_by_prefix(&prefix);
        }

        let mut chunk_count = 0usize;
        for chunk in all_chunks {
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

            let node = GraphNode {
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
            };

            let _ = self.storage.insert_graph_node(&node);
            let _ = graph.add_node(node);

            // CONTAINS edge: file → chunk
            let file_node_id = format!("file:{}", chunk.file_path);
            let file_chunk_edge = Edge {
                id: format!("contains:{file_node_id}->{chunk_id}"),
                src: file_node_id,
                dst: chunk_id.clone(),
                relationship: RelationshipType::Contains,
                weight: edge_weight_for(&RelationshipType::Contains, &self.config.graph),
                valid_from: None,
                valid_to: None,
                properties: HashMap::new(),
                created_at: now,
            };
            let _ = self.storage.insert_graph_edge(&file_chunk_edge);
            let _ = graph.add_edge(file_chunk_edge);

            // CONTAINS edge: parent symbol → chunk
            if let Some(ref parent_sym) = chunk.parent_symbol {
                let parent_node_id = format!("sym:{parent_sym}");
                let contains_edge = Edge {
                    id: format!("contains:{parent_node_id}->{chunk_id}"),
                    src: parent_node_id,
                    dst: chunk_id,
                    relationship: RelationshipType::Contains,
                    weight: edge_weight_for(&RelationshipType::Contains, &self.config.graph),
                    valid_from: None,
                    valid_to: None,
                    properties: HashMap::new(),
                    created_at: now,
                };
                let _ = self.storage.insert_graph_edge(&contains_edge);
                let _ = graph.add_edge(contains_edge);
            }

            chunk_count += 1;
        }
        drop(graph);

        // ── Embed symbols and chunks ────────────────────────────────────────
        // Phase 1: Collect enriched texts without holding the vector lock.
        // enrich_symbol_text / enrich_chunk_text only read from the passed-in
        // Symbol/CodeChunk and edges slice, so no lock is required.
        //
        // A12: Embedding bottleneck — The embedding provider is behind a Mutex,
        // so `embed_batch` runs sequentially even though CPU-bound inference
        // (Candle) could benefit from parallelism. For large codebases, this is
        // the primary bottleneck. Potential fix: wrap the provider in an Arc and
        // use `tokio::spawn_blocking` for CPU-bound Candle inference, or use a
        // channel-based work queue to decouple embedding from persistence.
        let mut symbols_embedded = 0usize;
        let mut chunks_embedded = 0usize;
        // H3: Acquire embeddings lock, embed batch, drop lock before acquiring vector lock.
        if let Some(emb) = self.lock_embeddings()? {
            let sym_texts: Vec<(String, String)> = all_symbols
                .iter()
                .map(|sym| {
                    let id = format!("sym:{}", sym.qualified_name);
                    let text = self.enrich_symbol_text(sym, edges);
                    (id, text)
                })
                .collect();
            let chunk_texts: Vec<(String, String)> = all_chunks
                .iter()
                .map(|chunk| {
                    let id = format!("chunk:{}:{}", chunk.file_path, chunk.index);
                    let text = self.enrich_chunk_text(chunk);
                    (id, text)
                })
                .collect();

            // Phase 2: Embed all texts (embedding provider lock only, no vector lock).
            let all_texts: Vec<&str> = sym_texts
                .iter()
                .map(|(_, t)| t.as_str())
                .chain(chunk_texts.iter().map(|(_, t)| t.as_str()))
                .collect();
            let all_embeddings = match emb.embed_batch(&all_texts) {
                Ok(embeddings) => embeddings,
                Err(e) => {
                    tracing::warn!(
                        "embed_batch failed for {} texts, symbols/chunks will be unembedded: {e}",
                        all_texts.len()
                    );
                    vec![]
                }
            };
            drop(emb);

            // Phase 3: Insert into vector index + storage with a single lock acquisition.
            let mut vec = self.lock_vector()?;
            let sym_count = sym_texts.len();
            for (i, (id, _)) in sym_texts.into_iter().enumerate() {
                if let Some(embedding) = all_embeddings.get(i) {
                    if let Err(e) = self.storage.store_embedding(&id, embedding) {
                        tracing::warn!("Failed to store embedding for {id}: {e}");
                    }
                    if let Err(e) = vec.insert(&id, embedding) {
                        tracing::warn!("Failed to insert {id} into vector index: {e}");
                    }
                    symbols_embedded += 1;
                }
            }
            for (i, (id, _)) in chunk_texts.into_iter().enumerate() {
                if let Some(embedding) = all_embeddings.get(sym_count + i) {
                    if let Err(e) = self.storage.store_embedding(&id, embedding) {
                        tracing::warn!("Failed to store embedding for {id}: {e}");
                    }
                    if let Err(e) = vec.insert(&id, embedding) {
                        tracing::warn!("Failed to insert {id} into vector index: {e}");
                    }
                    chunks_embedded += 1;
                }
            }
            drop(vec);
            self.save_index();
        }

        // ── Auto-compact ────────────────────────────────────────────────────
        let (chunks_pruned, symbols_pruned) = if self.config.chunking.auto_compact {
            self.compact_graph(seen_files)
        } else {
            (0, 0)
        };

        Ok(IndexPersistResult {
            files_created: seen_files.len(),
            packages_created: created_dirs.len(),
            symbols_stored: all_symbols.len(),
            chunks_stored: chunk_count,
            edges_resolved: edges.len(),
            symbols_embedded,
            chunks_embedded,
            chunks_pruned,
            symbols_pruned,
        })
    }

    // ── Graph Compaction ────────────────────────────────────────────────────

    /// Compact chunk and symbol graph-nodes after indexing.
    /// Returns (chunks_pruned, symbols_pruned).
    pub fn compact_graph(&self, seen_files: &HashSet<String>) -> (usize, usize) {
        let mut graph = match self.lock_graph() {
            Ok(g) => g,
            Err(_) => return (0, 0),
        };

        // Fetch all nodes once and share between both compaction passes.
        let all_nodes = graph.get_all_nodes();
        let chunks_pruned = self.compact_chunks(seen_files, &mut graph, &all_nodes);
        let symbols_pruned = self.compact_symbols(seen_files, &mut graph, &all_nodes);

        if chunks_pruned > 0 || symbols_pruned > 0 {
            // compute_centrality: updates node.centrality with degree centrality.
            // recompute_centrality: caches PageRank + betweenness for hybrid scoring.
            // Both are needed — they populate different data used by different scoring paths.
            graph.compute_centrality();
            graph.recompute_centrality();
        }

        (chunks_pruned, symbols_pruned)
    }

    /// Pass 1: Score and prune low-value chunks, transferring line ranges to parent symbols.
    ///
    /// Scoring weights adjust on cold start: when no memories exist yet, the
    /// `memory_link_score` weight (normally 0.3) is redistributed to the other
    /// factors so compaction still produces meaningful rankings.
    fn compact_chunks(
        &self,
        seen_files: &HashSet<String>,
        graph: &mut std::sync::MutexGuard<'_, codemem_storage::graph::GraphEngine>,
        all_nodes: &[GraphNode],
    ) -> usize {
        let max_chunks_per_file = self.config.chunking.max_retained_chunks_per_file;
        let chunk_score_threshold = self.config.chunking.min_chunk_score_threshold;

        // Cold-start detection: if no memories exist, memory_link_score is always 0
        // and its weight should be redistributed to other factors.
        let has_memories = self
            .storage
            .list_memory_ids()
            .map(|ids| !ids.is_empty())
            .unwrap_or(false);
        let (w_centrality, w_parent, w_memory, w_size) = if has_memories {
            (0.3, 0.2, 0.3, 0.2)
        } else {
            // Redistribute memory_link weight: centrality 0.4, parent 0.3, memory 0.0, size 0.3
            (0.4, 0.3, 0.0, 0.3)
        };

        let mut chunks_by_file: HashMap<String, Vec<(String, f64)>> = HashMap::new();

        let mut max_degree: f64 = 1.0;
        let mut max_non_ws: f64 = 1.0;

        let chunk_nodes: Vec<&GraphNode> = all_nodes
            .iter()
            .filter(|n| n.kind == NodeKind::Chunk)
            .collect();

        for node in &chunk_nodes {
            let degree = graph
                .get_edges(&node.id)
                .map(|edges| edges.len() as f64)
                .unwrap_or(0.0);
            max_degree = max_degree.max(degree);

            let non_ws = node
                .payload
                .get("non_ws_chars")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            max_non_ws = max_non_ws.max(non_ws);
        }

        for node in &chunk_nodes {
            let file_path = match node.payload.get("file_path").and_then(|v| v.as_str()) {
                Some(fp) => fp.to_string(),
                None => continue,
            };
            if !seen_files.contains(&file_path) {
                continue;
            }

            let degree = graph
                .get_edges(&node.id)
                .map(|edges| edges.len() as f64)
                .unwrap_or(0.0);
            let centrality_rank = degree / max_degree;

            let has_symbol_parent = if node.payload.contains_key("parent_symbol") {
                1.0
            } else {
                0.0
            };

            let memory_link_score = if has_memories && has_memory_link_edge(&**graph, &node.id) {
                1.0
            } else {
                0.0
            };

            let non_ws = node
                .payload
                .get("non_ws_chars")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let non_ws_rank = non_ws / max_non_ws;

            let chunk_score = centrality_rank * w_centrality
                + has_symbol_parent * w_parent
                + memory_link_score * w_memory
                + non_ws_rank * w_size;

            chunks_by_file
                .entry(file_path)
                .or_default()
                .push((node.id.clone(), chunk_score));
        }

        let mut symbol_count_by_file: HashMap<String, usize> = HashMap::new();
        for node in all_nodes {
            if matches!(
                node.kind,
                NodeKind::Function
                    | NodeKind::Method
                    | NodeKind::Class
                    | NodeKind::Interface
                    | NodeKind::Type
                    | NodeKind::Constant
                    | NodeKind::Test
            ) {
                if let Some(fp) = node.payload.get("file_path").and_then(|v| v.as_str()) {
                    *symbol_count_by_file.entry(fp.to_string()).or_default() += 1;
                }
            }
        }

        let mut chunks_pruned = 0usize;
        for (file_path, mut chunks) in chunks_by_file {
            chunks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let sym_count = symbol_count_by_file.get(&file_path).copied().unwrap_or(0);
            let k = max_chunks_per_file.min(chunks.len()).max(3.max(sym_count));

            for (i, (chunk_id, score)) in chunks.iter().enumerate() {
                // Prune aggressively: remove if beyond the top-k slots OR below the
                // quality threshold. Using || ensures both caps are enforced independently
                // (keep at most k chunks, and never keep any chunk below threshold).
                if i >= k || *score < chunk_score_threshold {
                    self.transfer_chunk_ranges_to_parent(graph, chunk_id);

                    if let Err(e) = self.storage.delete_graph_edges_for_node(chunk_id) {
                        tracing::warn!("Failed to delete graph edges for chunk {chunk_id}: {e}");
                    }
                    if let Err(e) = self.storage.delete_graph_node(chunk_id) {
                        tracing::warn!("Failed to delete graph node for chunk {chunk_id}: {e}");
                    }
                    if let Err(e) = graph.remove_node(chunk_id) {
                        tracing::warn!("Failed to remove chunk {chunk_id} from graph: {e}");
                    }
                    chunks_pruned += 1;
                }
            }
        }

        chunks_pruned
    }

    /// When pruning a chunk, transfer its line range to the parent symbol node.
    fn transfer_chunk_ranges_to_parent(
        &self,
        graph: &mut std::sync::MutexGuard<'_, codemem_storage::graph::GraphEngine>,
        chunk_id: &str,
    ) {
        if let Ok(Some(chunk_node)) = graph.get_node(chunk_id) {
            if let Some(parent_sym) = chunk_node
                .payload
                .get("parent_symbol")
                .and_then(|v| v.as_str())
            {
                let parent_id = format!("sym:{parent_sym}");
                if let Ok(Some(mut parent_node)) = graph.get_node(&parent_id) {
                    let line_start = chunk_node
                        .payload
                        .get("line_start")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0);
                    let line_end = chunk_node
                        .payload
                        .get("line_end")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0);
                    let ranges = parent_node
                        .payload
                        .entry("covered_ranges".to_string())
                        .or_insert_with(|| serde_json::json!([]));
                    if let Some(arr) = ranges.as_array_mut() {
                        arr.push(serde_json::json!([line_start, line_end]));
                    }
                    let count = parent_node
                        .payload
                        .entry("pruned_chunk_count".to_string())
                        .or_insert_with(|| serde_json::json!(0));
                    if let Some(n) = count.as_u64() {
                        *count = serde_json::json!(n + 1);
                    }
                    let _ = self.storage.insert_graph_node(&parent_node);
                    let _ = graph.add_node(parent_node);
                }
            }
        }
    }

    /// Pass 2: Score and prune low-value symbol nodes, transferring ranges to parent files.
    ///
    /// Like chunk compaction, scoring weights adjust on cold start: when no memories
    /// exist yet, the `memory_link_val` weight (normally 0.20) is redistributed to
    /// call connectivity and code size factors.
    fn compact_symbols(
        &self,
        seen_files: &HashSet<String>,
        graph: &mut std::sync::MutexGuard<'_, codemem_storage::graph::GraphEngine>,
        all_nodes: &[GraphNode],
    ) -> usize {
        let max_syms_per_file = self.config.chunking.max_retained_symbols_per_file;
        let sym_score_threshold = self.config.chunking.min_symbol_score_threshold;

        // Cold-start: redistribute memory_link weight when no memories exist
        let has_memories = self
            .storage
            .list_memory_ids()
            .map(|ids| !ids.is_empty())
            .unwrap_or(false);
        let (w_calls, w_vis, w_kind, w_mem, w_size) = if has_memories {
            (0.30, 0.20, 0.15, 0.20, 0.15)
        } else {
            // Redistribute memory weight to calls and code size
            (0.40, 0.20, 0.15, 0.0, 0.25)
        };

        let sym_nodes: Vec<&GraphNode> = all_nodes
            .iter()
            .filter(|n| n.id.starts_with("sym:"))
            .collect();

        let mut max_calls_degree: f64 = 1.0;
        let mut max_code_size: f64 = 1.0;

        for node in &sym_nodes {
            let calls_degree = graph
                .get_edges(&node.id)
                .map(|edges| {
                    edges
                        .iter()
                        .filter(|e| e.relationship == RelationshipType::Calls)
                        .count() as f64
                })
                .unwrap_or(0.0);
            max_calls_degree = max_calls_degree.max(calls_degree);

            let line_start = node
                .payload
                .get("line_start")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let line_end = node
                .payload
                .get("line_end")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let code_size = (line_end - line_start).max(0.0);
            max_code_size = max_code_size.max(code_size);
        }

        let mut syms_by_file: HashMap<String, Vec<(String, f64, bool, bool)>> = HashMap::new();

        for node in &sym_nodes {
            let file_path = match node.payload.get("file_path").and_then(|v| v.as_str()) {
                Some(fp) => fp.to_string(),
                None => continue,
            };
            if !seen_files.contains(&file_path) {
                continue;
            }

            let calls_degree = graph
                .get_edges(&node.id)
                .map(|edges| {
                    edges
                        .iter()
                        .filter(|e| e.relationship == RelationshipType::Calls)
                        .count() as f64
                })
                .unwrap_or(0.0);
            let call_connectivity = calls_degree / max_calls_degree;

            let visibility_score = match node
                .payload
                .get("visibility")
                .and_then(|v| v.as_str())
                .unwrap_or("private")
            {
                "public" => 1.0,
                "crate" => 0.5,
                _ => 0.0,
            };

            let kind_score = match node.kind {
                NodeKind::Class | NodeKind::Interface => 1.0,
                NodeKind::Module => 1.0,
                NodeKind::Function | NodeKind::Method => 0.6,
                NodeKind::Test => 0.3,
                NodeKind::Constant => 0.1,
                _ => 0.5,
            };

            let mem_linked = has_memories && has_memory_link_edge(&**graph, &node.id);
            let memory_link_val = if mem_linked { 1.0 } else { 0.0 };

            let line_start = node
                .payload
                .get("line_start")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let line_end = node
                .payload
                .get("line_end")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let code_size = (line_end - line_start).max(0.0);
            let code_size_rank = code_size / max_code_size;

            let symbol_score = call_connectivity * w_calls
                + visibility_score * w_vis
                + kind_score * w_kind
                + memory_link_val * w_mem
                + code_size_rank * w_size;

            let is_structural = matches!(
                node.kind,
                NodeKind::Class | NodeKind::Interface | NodeKind::Module
            );

            syms_by_file.entry(file_path).or_default().push((
                node.id.clone(),
                symbol_score,
                is_structural,
                mem_linked,
            ));
        }

        let mut symbols_pruned = 0usize;
        for (_file_path, mut syms) in syms_by_file {
            syms.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let public_count = syms
                .iter()
                .filter(|(id, ..)| {
                    graph
                        .get_node(id)
                        .ok()
                        .flatten()
                        .and_then(|n| {
                            n.payload
                                .get("visibility")
                                .and_then(|v| v.as_str())
                                .map(|v| v == "public")
                        })
                        .unwrap_or(false)
                })
                .count();
            let k = max_syms_per_file.max(public_count);

            for (i, (sym_id, score, is_structural, mem_linked)) in syms.iter().enumerate() {
                if *is_structural || *mem_linked {
                    continue;
                }
                if i < k && *score >= sym_score_threshold {
                    continue;
                }

                self.transfer_symbol_ranges_to_file(graph, sym_id);

                if let Err(e) = self.storage.delete_graph_edges_for_node(sym_id) {
                    tracing::warn!("Failed to delete graph edges for symbol {sym_id}: {e}");
                }
                if let Err(e) = self.storage.delete_graph_node(sym_id) {
                    tracing::warn!("Failed to delete graph node for symbol {sym_id}: {e}");
                }
                if let Err(e) = graph.remove_node(sym_id) {
                    tracing::warn!("Failed to remove symbol {sym_id} from graph: {e}");
                }
                symbols_pruned += 1;
            }
        }

        symbols_pruned
    }

    /// When pruning a symbol, transfer its line range to the parent file node.
    fn transfer_symbol_ranges_to_file(
        &self,
        graph: &mut std::sync::MutexGuard<'_, codemem_storage::graph::GraphEngine>,
        sym_id: &str,
    ) {
        if let Ok(Some(sym_node)) = graph.get_node(sym_id) {
            if let Some(fp) = sym_node.payload.get("file_path").and_then(|v| v.as_str()) {
                let file_id = format!("file:{fp}");
                if let Ok(Some(mut file_node)) = graph.get_node(&file_id) {
                    let line_start = sym_node
                        .payload
                        .get("line_start")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0);
                    let line_end = sym_node
                        .payload
                        .get("line_end")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0);
                    if line_end > line_start {
                        let ranges = file_node
                            .payload
                            .entry("pruned_symbol_ranges".to_string())
                            .or_insert_with(|| serde_json::json!([]));
                        if let Some(arr) = ranges.as_array_mut() {
                            arr.push(serde_json::json!([line_start, line_end]));
                        }
                        let _ = self.storage.insert_graph_node(&file_node);
                        let _ = graph.add_node(file_node);
                    }
                }
            }
        }
    }
}
