use crate::index::{self, IndexAndResolveResult, Indexer};
use crate::patterns;
use crate::CodememEngine;
use codemem_core::{CodememError, DetectedPattern, GraphBackend, MemoryNode, VectorBackend};
use std::collections::HashSet;
use std::path::Path;
use std::sync::atomic::Ordering;

impl CodememEngine {
    // ── Index Persistence ────────────────────────────────────────────────

    /// Save the vector and BM25 indexes to disk if a db_path is configured.
    /// Compacts the HNSW index if ghost entries exceed 20% of live entries.
    /// Always clears the dirty flag so `flush_if_dirty()` won't double-save.
    pub fn save_index(&self) {
        if let Some(ref db_path) = self.db_path {
            let idx_path = db_path.with_extension("idx");
            if let Ok(mut vi) = self.lock_vector() {
                // Compact HNSW if ghost entries exceed threshold
                if vi.needs_compaction() {
                    let ghost = vi.ghost_count();
                    let live = vi.stats().count;
                    tracing::info!(
                        "HNSW ghost compaction: {ghost} ghosts / {live} live entries, rebuilding..."
                    );
                    if let Ok(embeddings) = self.storage.list_all_embeddings() {
                        if let Err(e) = vi.rebuild_from_entries(&embeddings) {
                            tracing::warn!("HNSW compaction failed: {e}");
                        }
                    }
                }
                if let Err(e) = vi.save(&idx_path) {
                    tracing::warn!("Failed to save vector index: {e}");
                }
            }

            // Persist BM25 index alongside the vector index
            let bm25_path = db_path.with_extension("bm25");
            if let Ok(bm25) = self.lock_bm25() {
                if bm25.needs_save() {
                    let data = bm25.serialize();
                    let tmp_path = db_path.with_extension("bm25.tmp");
                    if let Err(e) = std::fs::write(&tmp_path, &data)
                        .and_then(|_| std::fs::rename(&tmp_path, &bm25_path))
                    {
                        tracing::warn!("Failed to save BM25 index: {e}");
                    }
                }
            }
        }
        self.dirty.store(false, Ordering::Release);
    }

    /// Reload the in-memory graph from the database.
    pub fn reload_graph(&self) -> Result<(), CodememError> {
        let new_graph = codemem_storage::graph::GraphEngine::from_storage(&*self.storage)?;
        let mut graph = self.lock_graph()?;
        *graph = new_graph;
        graph.recompute_centrality();
        Ok(())
    }

    // ── A2: File Watcher Event Processing ───────────────────────────────

    /// Process a single file watcher event by re-indexing changed/created files
    /// or cleaning up deleted file nodes.
    ///
    /// Call this from a watcher event loop:
    /// ```ignore
    /// while let Ok(event) = watcher.receiver().recv() {
    ///     engine.process_watch_event(&event, namespace, Some(root));
    /// }
    /// ```
    pub fn process_watch_event(
        &self,
        event: &crate::watch::WatchEvent,
        namespace: Option<&str>,
        project_root: Option<&Path>,
    ) -> Result<(), CodememError> {
        match event {
            crate::watch::WatchEvent::FileChanged(path)
            | crate::watch::WatchEvent::FileCreated(path) => {
                self.index_single_file(path, namespace, project_root)?;
            }
            crate::watch::WatchEvent::FileDeleted(path) => {
                // Relativize the deleted path so the node ID matches what was indexed.
                let rel = if let Some(root) = project_root {
                    path.strip_prefix(root)
                        .unwrap_or(path)
                        .to_string_lossy()
                        .to_string()
                } else {
                    path.to_string_lossy().to_string()
                };
                self.cleanup_file_nodes(&rel)?;
            }
        }
        Ok(())
    }

    /// Index (or re-index) a single file: parse it, persist nodes/edges/embeddings,
    /// and update the index cache.
    ///
    /// `project_root` is used to relativize the absolute `path` so node IDs are
    /// portable. If `None`, the path is stored as-is (absolute).
    fn index_single_file(
        &self,
        path: &Path,
        namespace: Option<&str>,
        project_root: Option<&Path>,
    ) -> Result<(), CodememError> {
        let content = std::fs::read(path)?;

        let path_str = if let Some(root) = project_root {
            path.strip_prefix(root)
                .unwrap_or(path)
                .to_string_lossy()
                .to_string()
        } else {
            path.to_string_lossy().to_string()
        };
        let parser = index::CodeParser::new();

        let parse_result = match parser.parse_file(&path_str, &content) {
            Some(pr) => pr,
            None => return Ok(()), // Unsupported file type or parse failure
        };

        // Build a minimal IndexAndResolveResult for this single file
        let mut file_paths = HashSet::new();
        file_paths.insert(parse_result.file_path.clone());

        let mut resolver = index::ReferenceResolver::new();
        resolver.add_symbols(&parse_result.symbols);
        let edges = resolver.resolve_all(&parse_result.references);

        let results = IndexAndResolveResult {
            index: index::IndexResult {
                files_scanned: 1,
                files_parsed: 1,
                files_skipped: 0,
                total_symbols: parse_result.symbols.len(),
                total_references: parse_result.references.len(),
                total_chunks: parse_result.chunks.len(),
                parse_results: Vec::new(),
            },
            symbols: parse_result.symbols,
            references: parse_result.references,
            chunks: parse_result.chunks,
            file_paths,
            edges,
            root_path: project_root
                .map(|p| p.to_path_buf())
                .unwrap_or_else(|| path.to_path_buf()),
        };

        self.persist_index_results(&results, namespace)?;
        Ok(())
    }

    // ── A3: File Deletion Cleanup ───────────────────────────────────────

    /// Remove graph nodes, edges, and embeddings for a single deleted file.
    fn cleanup_file_nodes(&self, file_path: &str) -> Result<(), CodememError> {
        let file_node_id = format!("file:{file_path}");

        // Remove all chunk nodes for this file
        let chunk_prefix = format!("chunk:{file_path}:");
        if let Err(e) = self.storage.delete_graph_nodes_by_prefix(&chunk_prefix) {
            tracing::warn!("Failed to delete chunk nodes for {file_path}: {e}");
        }

        // Remove symbol nodes for this file by checking graph
        let graph = self.lock_graph()?;
        let sym_ids: Vec<String> = graph
            .get_all_nodes()
            .into_iter()
            .filter(|n| {
                n.id.starts_with("sym:")
                    && n.payload.get("file_path").and_then(|v| v.as_str()) == Some(file_path)
            })
            .map(|n| n.id.clone())
            .collect();
        drop(graph);

        for sym_id in &sym_ids {
            if let Err(e) = self.storage.delete_graph_edges_for_node(sym_id) {
                tracing::warn!("Failed to delete graph edges for {sym_id}: {e}");
            }
            if let Err(e) = self.storage.delete_graph_node(sym_id) {
                tracing::warn!("Failed to delete graph node {sym_id}: {e}");
            }
            if let Err(e) = self.storage.delete_embedding(sym_id) {
                tracing::warn!("Failed to delete embedding {sym_id}: {e}");
            }
        }

        // Remove file node itself
        if let Err(e) = self.storage.delete_graph_edges_for_node(&file_node_id) {
            tracing::warn!("Failed to delete graph edges for {file_node_id}: {e}");
        }
        if let Err(e) = self.storage.delete_graph_node(&file_node_id) {
            tracing::warn!("Failed to delete graph node {file_node_id}: {e}");
        }

        // Clean up in-memory graph
        let mut graph = self.lock_graph()?;
        for sym_id in &sym_ids {
            if let Err(e) = graph.remove_node(sym_id) {
                tracing::warn!("Failed to remove {sym_id} from in-memory graph: {e}");
            }
        }
        // Remove chunk nodes from in-memory graph
        let chunk_ids: Vec<String> = graph
            .get_all_nodes()
            .into_iter()
            .filter(|n| n.id.starts_with(&format!("chunk:{file_path}:")))
            .map(|n| n.id.clone())
            .collect();
        for chunk_id in &chunk_ids {
            if let Err(e) = graph.remove_node(chunk_id) {
                tracing::warn!("Failed to remove {chunk_id} from in-memory graph: {e}");
            }
        }
        if let Err(e) = graph.remove_node(&file_node_id) {
            tracing::warn!("Failed to remove {file_node_id} from in-memory graph: {e}");
        }
        drop(graph);

        // Remove stale embeddings from vector index
        let mut vec = self.lock_vector()?;
        for sym_id in &sym_ids {
            if let Err(e) = vec.remove(sym_id) {
                tracing::warn!("Failed to remove {sym_id} from vector index: {e}");
            }
        }
        for chunk_id in &chunk_ids {
            if let Err(e) = vec.remove(chunk_id) {
                tracing::warn!("Failed to remove {chunk_id} from vector index: {e}");
            }
        }
        drop(vec);

        self.save_index();
        Ok(())
    }

    // ── A4: Unified Analyze Pipeline ────────────────────────────────────

    /// Full analysis pipeline: index → persist → enrich → recompute centrality.
    ///
    /// This is the single entry point for all callers (CLI, MCP, API).
    /// Supports incremental indexing via `ChangeDetector`, progress callbacks,
    /// and returns comprehensive results.
    pub fn analyze(&self, options: AnalyzeOptions<'_>) -> Result<AnalyzeResult, CodememError> {
        let root = options.path;

        // 1. Index
        let mut indexer = match options.change_detector {
            Some(cd) => Indexer::with_change_detector(cd),
            None => Indexer::new(),
        };
        let resolved = indexer.index_and_resolve(root)?;

        // 2. Persist (with or without progress callback)
        let persist = if let Some(ref on_progress) = options.progress {
            self.persist_index_results_with_progress(
                &resolved,
                Some(options.namespace),
                |done, total| {
                    on_progress(AnalyzeProgress::Embedding { done, total });
                },
            )?
        } else {
            self.persist_index_results(&resolved, Some(options.namespace))?
        };

        // Cache results for structural queries
        {
            if let Ok(mut cache) = self.lock_index_cache() {
                *cache = Some(crate::IndexCache {
                    symbols: resolved.symbols,
                    chunks: resolved.chunks,
                    root_path: root.to_string_lossy().to_string(),
                });
            }
        }

        // 3. Enrich
        let path_str = root.to_str().unwrap_or("");
        let enrichment = self.run_enrichments(
            path_str,
            &[],
            options.git_days,
            Some(options.namespace),
            None,
        );

        // 4. Recompute centrality
        self.lock_graph()?.recompute_centrality();

        // 5. Compute summary stats
        let top_nodes = self.find_important_nodes(10, 0.85).unwrap_or_default();
        let community_count = self.louvain_communities(1.0).map(|c| c.len()).unwrap_or(0);

        // 6. Save indexes
        self.save_index();

        // Save incremental state
        indexer.change_detector().save_to_storage(self.storage())?;

        Ok(AnalyzeResult {
            files_parsed: resolved.index.files_parsed,
            files_skipped: resolved.index.files_skipped,
            symbols_found: resolved.index.total_symbols,
            edges_resolved: persist.edges_resolved,
            chunks_stored: persist.chunks_stored,
            symbols_embedded: persist.symbols_embedded,
            chunks_embedded: persist.chunks_embedded,
            chunks_pruned: persist.chunks_pruned,
            symbols_pruned: persist.symbols_pruned,
            enrichment_results: enrichment.results,
            total_insights: enrichment.total_insights,
            top_nodes,
            community_count,
        })
    }

    // ── A8: Session Context Synthesis ───────────────────────────────────

    /// Synthesize context for a new session: recent memories, pending analyses,
    /// active patterns, and last session summary.
    pub fn session_context(&self, namespace: Option<&str>) -> Result<SessionContext, CodememError> {
        let now = chrono::Utc::now();
        let cutoff_24h = now - chrono::Duration::hours(24);

        // 1. Recent memories (last 24h)
        let ids = match namespace {
            Some(ns) => self.storage.list_memory_ids_for_namespace(ns)?,
            None => self.storage.list_memory_ids()?,
        };

        let mut recent_memories = Vec::new();
        let mut pending_analyses = Vec::new();

        for id in ids.iter().rev().take(200) {
            if let Ok(Some(m)) = self.storage.get_memory_no_touch(id) {
                // Collect pending analyses
                if m.tags.contains(&"pending-analysis".to_string()) {
                    pending_analyses.push(m.clone());
                }
                // Collect recent memories from last 24h
                if m.created_at >= cutoff_24h {
                    recent_memories.push(m);
                }
                if recent_memories.len() >= 50 && pending_analyses.len() >= 10 {
                    break;
                }
            }
        }

        // 2. Active patterns
        let session_count = self.storage.session_count(namespace).unwrap_or(1).max(1);
        let active_patterns = patterns::detect_patterns(
            &*self.storage,
            namespace,
            2, // min_frequency
            session_count,
        )
        .unwrap_or_default();

        // 3. Last session summary
        let last_session_summary = self
            .storage
            .list_sessions(namespace, 1)?
            .into_iter()
            .next()
            .and_then(|s| s.summary);

        Ok(SessionContext {
            recent_memories,
            pending_analyses,
            active_patterns,
            last_session_summary,
        })
    }
}

// ── Result Types ────────────────────────────────────────────────────────────

/// Options for the unified `analyze()` pipeline.
pub struct AnalyzeOptions<'a> {
    pub path: &'a Path,
    pub namespace: &'a str,
    pub git_days: u64,
    pub change_detector: Option<index::incremental::ChangeDetector>,
    pub progress: Option<Box<dyn Fn(AnalyzeProgress) + Send + 'a>>,
}

/// Progress events emitted during analysis.
#[derive(Debug, Clone)]
pub enum AnalyzeProgress {
    Embedding { done: usize, total: usize },
}

/// Result of the unified `analyze()` pipeline.
#[derive(Debug)]
pub struct AnalyzeResult {
    pub files_parsed: usize,
    pub files_skipped: usize,
    pub symbols_found: usize,
    pub edges_resolved: usize,
    pub chunks_stored: usize,
    pub symbols_embedded: usize,
    pub chunks_embedded: usize,
    pub chunks_pruned: usize,
    pub symbols_pruned: usize,
    pub enrichment_results: serde_json::Value,
    pub total_insights: usize,
    pub top_nodes: Vec<crate::graph_ops::RankedNode>,
    pub community_count: usize,
}

/// Session context synthesized at session start.
#[derive(Debug)]
pub struct SessionContext {
    /// Memories created in the last 24 hours.
    pub recent_memories: Vec<MemoryNode>,
    /// Memories tagged `pending-analysis` awaiting code-mapper review.
    pub pending_analyses: Vec<MemoryNode>,
    /// Cross-session patterns detected with sufficient frequency.
    pub active_patterns: Vec<DetectedPattern>,
    /// Summary text from the most recent session (if any).
    pub last_session_summary: Option<String>,
}
