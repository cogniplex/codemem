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

    // ── A4: Combined Index + Enrich Pipeline ────────────────────────────

    /// Combined result from `index_and_enrich`.
    /// Index a codebase and run all enrichment passes in one call.
    pub fn index_and_enrich(
        &self,
        path: &str,
        namespace: Option<&str>,
        git_days: u64,
    ) -> Result<IndexEnrichResult, CodememError> {
        // 1. Index the codebase
        let mut indexer = Indexer::new();
        let index_results = indexer.index_and_resolve(Path::new(path))?;
        let persist = self.persist_index_results(&index_results, namespace)?;

        // 2. Run all enrichment passes, accumulating total insights
        let root = Path::new(path);
        let project_root = Some(root);
        let mut total_insights = 0usize;

        macro_rules! run_enrich {
            ($label:expr, $call:expr) => {
                match $call {
                    Ok(r) => total_insights += r.insights_stored,
                    Err(e) => tracing::warn!(concat!($label, " enrichment failed: {}"), e),
                }
            };
        }

        run_enrich!(
            "git_history",
            self.enrich_git_history(path, git_days, namespace)
        );
        run_enrich!("security", self.enrich_security(namespace));
        run_enrich!("performance", self.enrich_performance(10, namespace));
        run_enrich!(
            "complexity",
            self.enrich_complexity(namespace, project_root)
        );
        run_enrich!("architecture", self.enrich_architecture(namespace));
        run_enrich!("test_mapping", self.enrich_test_mapping(namespace));
        run_enrich!("api_surface", self.enrich_api_surface(namespace));
        run_enrich!("doc_coverage", self.enrich_doc_coverage(namespace));
        run_enrich!(
            "code_smells",
            self.enrich_code_smells(namespace, project_root)
        );
        run_enrich!("hot_complex", self.enrich_hot_complex(namespace));
        run_enrich!("blame", self.enrich_blame(path, namespace));
        run_enrich!(
            "security_scan",
            self.enrich_security_scan(namespace, project_root)
        );
        run_enrich!("quality", self.enrich_quality_stratification(namespace));
        // change_impact is per-file, not included in run_all

        // Recompute centrality after all changes
        self.lock_graph()?.recompute_centrality();

        Ok(IndexEnrichResult {
            files_indexed: persist.files_created,
            symbols_stored: persist.symbols_stored,
            chunks_stored: persist.chunks_stored,
            edges_resolved: persist.edges_resolved,
            symbols_embedded: persist.symbols_embedded,
            chunks_embedded: persist.chunks_embedded,
            total_insights,
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

/// Combined result from `index_and_enrich`.
#[derive(Debug)]
pub struct IndexEnrichResult {
    pub files_indexed: usize,
    pub symbols_stored: usize,
    pub chunks_stored: usize,
    pub edges_resolved: usize,
    pub symbols_embedded: usize,
    pub chunks_embedded: usize,
    pub total_insights: usize,
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
