//! Graph persistence: persist indexing results (file/package/symbol/chunk nodes,
//! edges, embeddings, compaction) into the storage and graph backends.

mod compaction;

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

    /// Like `persist_index_results`, but calls `on_progress(done, total)` during
    /// the embedding phase so callers can display progress.
    pub fn persist_index_results_with_progress(
        &self,
        results: &IndexAndResolveResult,
        namespace: Option<&str>,
        on_progress: impl Fn(usize, usize),
    ) -> Result<IndexPersistResult, CodememError> {
        let all_symbols = &results.symbols;
        let all_chunks = &results.chunks;
        let seen_files = &results.file_paths;
        let edges = &results.edges;

        let now = chrono::Utc::now();
        let ns_string = namespace.map(|s| s.to_string());

        let mut graph = self.lock_graph()?;

        // ── File nodes ──────────────────────────────────────────────────────
        let mut file_nodes = Vec::with_capacity(seen_files.len());
        for file_path in seen_files {
            let node_id = format!("file:{file_path}");
            let mut payload = HashMap::new();
            payload.insert(
                "file_path".to_string(),
                serde_json::Value::String(file_path.clone()),
            );
            file_nodes.push(GraphNode {
                id: node_id,
                kind: NodeKind::File,
                label: file_path.clone(),
                payload,
                centrality: 0.0,
                memory_id: None,
                namespace: ns_string.clone(),
            });
        }
        let _ = self.storage.insert_graph_nodes_batch(&file_nodes);
        for node in file_nodes {
            let _ = graph.add_node(node);
        }

        // ── Package (directory) nodes ───────────────────────────────────────
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
                        dir_edges.push(edge);
                    }
                }
            }
            // CONTAINS edge from innermost directory to file
            if let Some(last_dir) = ancestors.last() {
                let parent_pkg_id = format!("pkg:{last_dir}/");
                let file_node_id = format!("file:{file_path}");
                let edge_id = format!("contains:{parent_pkg_id}->{file_node_id}");
                dir_edges.push(Edge {
                    id: edge_id,
                    src: parent_pkg_id,
                    dst: file_node_id,
                    relationship: RelationshipType::Contains,
                    weight: edge_weight_for(&RelationshipType::Contains, &self.config.graph),
                    valid_from: None,
                    valid_to: None,
                    properties: HashMap::new(),
                    created_at: now,
                });
            }
        }
        let _ = self.storage.insert_graph_nodes_batch(&dir_nodes);
        for node in dir_nodes {
            let _ = graph.add_node(node);
        }
        let _ = self.storage.insert_graph_edges_batch(&dir_edges);
        for edge in dir_edges {
            let _ = graph.add_edge(edge);
        }

        // ── Symbol nodes ────────────────────────────────────────────────────
        let mut sym_nodes = Vec::with_capacity(all_symbols.len());
        let mut sym_edges = Vec::with_capacity(all_symbols.len());
        for sym in all_symbols {
            let kind = NodeKind::from(sym.kind);

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

            let sym_node_id = format!("sym:{}", sym.qualified_name);
            let node = GraphNode {
                id: sym_node_id.clone(),
                kind,
                label: sym.qualified_name.clone(),
                payload,
                centrality: 0.0,
                memory_id: None,
                namespace: ns_string.clone(),
            };
            sym_nodes.push(node);

            // CONTAINS edge: file → symbol
            let file_node_id = format!("file:{}", sym.file_path);
            sym_edges.push(Edge {
                id: format!("contains:{file_node_id}->{sym_node_id}"),
                src: file_node_id,
                dst: sym_node_id,
                relationship: RelationshipType::Contains,
                weight: edge_weight_for(&RelationshipType::Contains, &self.config.graph),
                valid_from: None,
                valid_to: None,
                properties: HashMap::new(),
                created_at: now,
            });
        }
        let _ = self.storage.insert_graph_nodes_batch(&sym_nodes);
        for node in sym_nodes {
            let _ = graph.add_node(node);
        }
        let _ = self.storage.insert_graph_edges_batch(&sym_edges);
        for edge in sym_edges {
            let _ = graph.add_edge(edge);
        }

        // ── Resolved reference edges ────────────────────────────────────────
        let ref_edges: Vec<Edge> = edges
            .iter()
            .map(|edge| Edge {
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
            })
            .collect();
        let _ = self.storage.insert_graph_edges_batch(&ref_edges);
        for e in ref_edges {
            let _ = graph.add_edge(e);
        }

        // ── Chunk nodes ─────────────────────────────────────────────────────
        // Cleanup stale chunk nodes for files being re-indexed
        for file_path in seen_files {
            let prefix = format!("chunk:{file_path}:");
            let _ = self.storage.delete_graph_nodes_by_prefix(&prefix);
        }

        let mut chunk_nodes = Vec::with_capacity(all_chunks.len());
        let mut chunk_edges = Vec::with_capacity(all_chunks.len() * 2);
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

            // CONTAINS edge: file → chunk
            let file_node_id = format!("file:{}", chunk.file_path);
            chunk_edges.push(Edge {
                id: format!("contains:{file_node_id}->{chunk_id}"),
                src: file_node_id,
                dst: chunk_id.clone(),
                relationship: RelationshipType::Contains,
                weight: edge_weight_for(&RelationshipType::Contains, &self.config.graph),
                valid_from: None,
                valid_to: None,
                properties: HashMap::new(),
                created_at: now,
            });

            // CONTAINS edge: parent symbol → chunk
            if let Some(ref parent_sym) = chunk.parent_symbol {
                let parent_node_id = format!("sym:{parent_sym}");
                chunk_edges.push(Edge {
                    id: format!("contains:{parent_node_id}->{chunk_id}"),
                    src: parent_node_id,
                    dst: chunk_id,
                    relationship: RelationshipType::Contains,
                    weight: edge_weight_for(&RelationshipType::Contains, &self.config.graph),
                    valid_from: None,
                    valid_to: None,
                    properties: HashMap::new(),
                    created_at: now,
                });
            }
        }
        let chunk_count = chunk_nodes.len();
        let _ = self.storage.insert_graph_nodes_batch(&chunk_nodes);
        for node in chunk_nodes {
            let _ = graph.add_node(node);
        }
        let _ = self.storage.insert_graph_edges_batch(&chunk_edges);
        for edge in chunk_edges {
            let _ = graph.add_edge(edge);
        }
        drop(graph);

        // ── Embed symbols and chunks ────────────────────────────────────────
        // Phase 1: Collect enriched texts without holding any lock.
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

        // Check if embeddings are available before collecting texts.
        let has_embeddings = self.lock_embeddings()?.is_some();
        if has_embeddings {
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

            // Phase 2+3: Embed in chunks and persist progressively.
            // Instead of one giant embed_batch (which blocks with no progress for
            // large codebases), we process in manageable chunks, persisting each
            // batch and reporting progress.
            //
            // The embedding lock is acquired per-batch so that SQLite/vector
            // writes don't hold it, and remote providers (Ollama/OpenAI) don't
            // block other operations for the entire duration.
            // Persistence batch size: how many items to embed + flush per round.
            // Separate from the GPU batch size (configured on EmbeddingService).
            const EMBED_CHUNK_SIZE: usize = 64;

            // all_pairs is ordered: symbols first, then chunks (via chain).
            // sym_count is used to attribute embedded items to the correct counter.
            let all_pairs: Vec<(String, String)> =
                sym_texts.into_iter().chain(chunk_texts).collect();
            let total = all_pairs.len();
            let sym_count = all_symbols.len();
            let mut done = 0usize;

            for batch in all_pairs.chunks(EMBED_CHUNK_SIZE) {
                let texts: Vec<&str> = batch.iter().map(|(_, t)| t.as_str()).collect();

                // Acquire embedding lock only for the embed_batch call, then drop.
                let t0 = std::time::Instant::now();
                let embed_result = {
                    let emb = self.lock_embeddings()?;
                    match emb {
                        Some(emb_guard) => emb_guard.embed_batch(&texts),
                        None => break, // Provider disappeared between check and use
                    }
                    // emb_guard dropped here — lock released before persistence I/O
                };

                match embed_result {
                    Ok(embeddings) => {
                        let embed_ms = t0.elapsed().as_millis();

                        // Batch-store embeddings to SQLite
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

                        // Batch-insert into in-memory vector index
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

                        // Update per-type counters
                        for _ in 0..batch_len {
                            if done < sym_count {
                                symbols_embedded += 1;
                            } else {
                                chunks_embedded += 1;
                            }
                            done += 1;
                        }

                        tracing::debug!(
                            "Embed batch {}: embed={embed_ms}ms sqlite={sqlite_ms}ms vector={vector_ms}ms",
                            batch_len
                        );
                    }
                    Err(e) => {
                        tracing::warn!(
                            "embed_batch failed for chunk of {} texts: {e}",
                            batch.len()
                        );
                        // Don't advance `done` — progress should reflect actual embeddings,
                        // not failed batches. The total will no longer be reached, which
                        // correctly signals incomplete embedding to the caller.
                    }
                }
                on_progress(done, total);
            }
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
}
