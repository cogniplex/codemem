use crate::index::{self, IndexAndResolveResult, Indexer};
use crate::patterns;
use crate::CodememEngine;
use codemem_core::{CodememError, DetectedPattern, MemoryNode};
use std::collections::HashSet;
use std::path::Path;
use std::sync::atomic::Ordering;

/// Check if a file is a spec file (OpenAPI/AsyncAPI) by name or content.
///
/// First checks well-known filenames. For other YAML/JSON files, peeks at
/// the first bytes to look for `"openapi"`, `"swagger"`, or `"asyncapi"` keys.
fn is_spec_file_with_content(path: &str, content: &[u8]) -> bool {
    let filename = path.rsplit('/').next().unwrap_or(path);
    let filename_lower = filename.to_lowercase();

    // Fast path: well-known names
    if matches!(
        filename_lower.as_str(),
        "openapi.yaml"
            | "openapi.yml"
            | "openapi.json"
            | "swagger.yaml"
            | "swagger.yml"
            | "swagger.json"
            | "asyncapi.yaml"
            | "asyncapi.yml"
            | "asyncapi.json"
    ) {
        return true;
    }

    // For any YAML/JSON file, peek at content for spec-identifying keys
    let is_yaml_json = filename_lower.ends_with(".yaml")
        || filename_lower.ends_with(".yml")
        || filename_lower.ends_with(".json");
    if !is_yaml_json {
        return false;
    }

    let peek = std::str::from_utf8(&content[..content.len().min(300)]).unwrap_or("");
    let peek_lower = peek.to_lowercase();
    peek_lower.contains("\"openapi\"")
        || peek_lower.contains("\"swagger\"")
        || peek_lower.contains("\"asyncapi\"")
        || peek_lower.contains("openapi:")
        || peek_lower.contains("swagger:")
        || peek_lower.contains("asyncapi:")
}

impl CodememEngine {
    // ── Index Persistence ────────────────────────────────────────────────

    /// Save the vector and BM25 indexes to disk if a db_path is configured.
    /// Compacts the HNSW index if ghost entries exceed 20% of live entries.
    /// Always clears the dirty flag so `flush_if_dirty()` won't double-save.
    pub fn save_index(&self) {
        if let Some(ref db_path) = self.db_path {
            // Only save vector index if it has been lazily initialized.
            if self.vector_ready() {
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
            }

            // Only save BM25 index if it has been lazily initialized.
            if self.bm25_ready() {
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
        }
        self.dirty.store(false, Ordering::Release);
    }

    /// Reload the in-memory graph from the database.
    pub fn reload_graph(&self) -> Result<(), CodememError> {
        let new_graph = codemem_storage::graph::GraphEngine::from_storage(&*self.storage)?;
        let mut graph = self.lock_graph()?;
        *graph = Box::new(new_graph);
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
    ///
    /// Uses SHA-256 hash dedup to skip re-indexing when content is unchanged.
    /// This prevents duplicate work when both the PostToolUse hook and the
    /// background file watcher fire for the same edit.
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

        // SHA-256 dedup: skip if content unchanged since last index.
        // Uses cached ChangeDetector to avoid reloading all hashes from storage per file.
        let hash = {
            let mut cd_guard = self
                .change_detector
                .lock()
                .map_err(|_| CodememError::LockPoisoned("change_detector".into()))?;
            let ns = namespace.unwrap_or("");
            let cd = cd_guard.get_or_insert_with(|| {
                let mut cd = index::incremental::ChangeDetector::new();
                cd.load_from_storage(&*self.storage, ns);
                cd
            });
            let (changed, hash) = cd.check_changed(&path_str, &content);
            if !changed {
                tracing::debug!("Skipping unchanged file: {path_str}");
                return Ok(());
            }
            // Expire static-analysis memories linked to symbols in this changed file
            if self.config.memory.expire_enrichments_on_reindex {
                match self.storage.expire_memories_for_file(&path_str) {
                    Ok(0) => {}
                    Ok(n) => tracing::debug!("Expired {n} enrichment memories for {path_str}"),
                    Err(e) => tracing::warn!("Failed to expire memories for {path_str}: {e}"),
                }
            }
            hash
        };

        // Check if this is a spec file (OpenAPI/AsyncAPI). If so, re-parse it
        // and update endpoints/channels rather than treating it as code.
        if is_spec_file_with_content(&path_str, &content) {
            self.reparse_spec_file(path, namespace.unwrap_or(""))?;
            // Record hash after successful spec parse
            if let Ok(mut cd_guard) = self.change_detector.lock() {
                if let Some(cd) = cd_guard.as_mut() {
                    cd.record_hash(&path_str, hash);
                    let _ = cd.save_to_storage(&*self.storage, namespace.unwrap_or(""));
                }
            }
            return Ok(());
        }

        let parser = index::CodeParser::new();

        let parse_result = match parser.parse_file(&path_str, &content) {
            Some(pr) => pr,
            None => return Ok(()), // Unsupported file type or parse failure
        };

        // Build a minimal IndexAndResolveResult for this single file
        let mut file_paths = HashSet::new();
        file_paths.insert(parse_result.file_path.clone());

        // Populate the resolver with ALL known symbols from the in-memory graph
        // so cross-file references (calls to functions in other files) can be
        // resolved. Without this, only same-file references would resolve,
        // causing the graph to gradually lose cross-file edges between full
        // re-indexes.
        let mut resolver = index::ReferenceResolver::new();
        resolver.add_symbols(&parse_result.symbols);
        if let Ok(graph) = self.lock_graph() {
            let graph_symbols: Vec<index::Symbol> = graph
                .get_all_nodes()
                .iter()
                .filter(|n| n.id.starts_with("sym:"))
                .filter_map(index::symbol::symbol_from_graph_node)
                .collect();
            resolver.add_symbols(&graph_symbols);
        }
        let resolve_result = resolver.resolve_all_with_unresolved(&parse_result.references);

        let results = IndexAndResolveResult {
            index: index::IndexResult {
                files_scanned: 1,
                files_parsed: 1,
                files_skipped: 0,
                total_symbols: parse_result.symbols.len(),
                total_references: parse_result.references.len(),
                total_chunks: parse_result.chunks.len(),
                total_documents: 0,
                parse_results: Vec::new(),
            },
            symbols: parse_result.symbols,
            references: parse_result.references,
            chunks: parse_result.chunks,
            doc_nodes: Vec::new(),
            file_paths,
            edges: resolve_result.edges,
            unresolved: resolve_result.unresolved,
            root_path: project_root
                .map(|p| p.to_path_buf())
                .unwrap_or_else(|| path.to_path_buf()),
            scip_build: None,
        };

        self.persist_index_results(&results, namespace)?;

        // Record new hash in the cached detector after successful persist
        if let Ok(mut cd_guard) = self.change_detector.lock() {
            if let Some(cd) = cd_guard.as_mut() {
                cd.record_hash(&path_str, hash);
                if let Err(e) = cd.save_to_storage(&*self.storage, namespace.unwrap_or("")) {
                    tracing::warn!("Failed to save file hash for {path_str}: {e}");
                }
            }
        }

        Ok(())
    }

    // ── A2b: Symbol-Level Diff on Re-index ────────────────────────────

    /// Remove symbols that existed for a file before re-indexing but are no
    /// longer present in the new parse results. Returns count of cleaned symbols.
    ///
    /// For code→code edges (CALLS, IMPORTS, etc.), performs a hard delete.
    /// For memory→symbol edges, creates a live redirected edge pointing to the
    /// parent file node, preserving the memory→file connection so recall can
    /// still traverse it. The original edge is then deleted along with the
    /// stale symbol node.
    ///
    /// `old_symbol_ids` should be the set of symbol IDs that existed for this
    /// file before re-indexing (collected from the in-memory graph by the caller
    /// in a single pass across all files).
    pub fn cleanup_stale_symbols(
        &self,
        file_path: &str,
        old_symbol_ids: &HashSet<String>,
        new_symbol_ids: &HashSet<String>,
    ) -> Result<usize, CodememError> {
        // Compute stale set: symbols that existed before but are not in the new parse
        let stale_ids: Vec<&String> = old_symbol_ids
            .iter()
            .filter(|id| !new_symbol_ids.contains(*id))
            .collect();

        if stale_ids.is_empty() {
            return Ok(0);
        }

        let count = stale_ids.len();
        tracing::info!(
            "Cleaning up {count} stale symbols for {file_path}: {:?}",
            stale_ids
        );

        let file_node_id = format!("file:{file_path}");
        let mut redirected_pairs: std::collections::HashSet<(String, String)> =
            std::collections::HashSet::new();
        let mut redirected_edges: Vec<codemem_core::Edge> = Vec::new();
        for sym_id in &stale_ids {
            // Before deleting the symbol, redirect memory→symbol edges to the
            // parent file node, preserving historical context.
            // Memory node IDs are UUIDs (no known prefix like sym:/file:/chunk:).
            let edges = self.storage.get_edges_for_node(sym_id.as_str())?;
            for edge in &edges {
                let other = if edge.src.as_str() == sym_id.as_str() {
                    &edge.dst
                } else {
                    &edge.src
                };
                let is_code_node = other.starts_with("sym:")
                    || other.starts_with("file:")
                    || other.starts_with("chunk:")
                    || other.starts_with("pkg:");
                if !is_code_node {
                    // Skip if we already redirected this memory→file pair
                    let pair = (other.to_string(), file_node_id.clone());
                    if !redirected_pairs.insert(pair) {
                        continue;
                    }
                    let mut redirected = edge.clone();
                    if redirected.src.as_str() == sym_id.as_str() {
                        redirected.src = file_node_id.clone();
                    } else {
                        redirected.dst = file_node_id.clone();
                    }
                    // Don't set valid_to — the redirect should be a live,
                    // queryable edge so recall can still traverse memory→file.
                    redirected.id = format!("{}-redirected", edge.id);
                    if let Err(e) = self.storage.insert_graph_edge(&redirected) {
                        tracing::warn!("Failed to redirect memory edge {}: {e}", edge.id);
                    }
                    redirected_edges.push(redirected);
                }
            }

            // Delete all edges and the node itself
            if let Err(e) = self.storage.delete_graph_edges_for_node(sym_id) {
                tracing::warn!("Failed to delete edges for stale symbol {sym_id}: {e}");
            }
            if let Err(e) = self.storage.delete_graph_node(sym_id) {
                tracing::warn!("Failed to delete stale symbol node {sym_id}: {e}");
            }
            if let Err(e) = self.storage.delete_embedding(sym_id) {
                tracing::warn!("Failed to delete embedding for stale symbol {sym_id}: {e}");
            }
        }

        // Clean up in-memory graph and vector index
        {
            let mut graph = self.lock_graph()?;
            for sym_id in &stale_ids {
                if let Err(e) = graph.remove_node(sym_id.as_str()) {
                    tracing::warn!("Failed to remove stale {sym_id} from in-memory graph: {e}");
                }
            }
            // Add redirected memory→file edges so they're visible to
            // in-memory traversal (BFS, PageRank, recall) during this session.
            for edge in redirected_edges {
                let _ = graph.add_edge(edge);
            }
        }
        {
            let mut vec = self.lock_vector()?;
            for sym_id in &stale_ids {
                if let Err(e) = vec.remove(sym_id.as_str()) {
                    tracing::warn!("Failed to remove stale {sym_id} from vector index: {e}");
                }
            }
        }
        // Remove stale entries from BM25 index so deleted/renamed symbols
        // don't persist in text search results or skew IDF calculations.
        if let Ok(mut bm25) = self.lock_bm25() {
            for sym_id in &stale_ids {
                bm25.remove_document(sym_id);
            }
        }

        Ok(count)
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

        // Remove deleted symbols/chunks from BM25 index
        if let Ok(mut bm25) = self.lock_bm25() {
            for sym_id in &sym_ids {
                bm25.remove_document(sym_id);
            }
            for chunk_id in &chunk_ids {
                bm25.remove_document(chunk_id);
            }
        }

        self.save_index();
        Ok(())
    }

    // ── A3b: Orphan Detection ─────────────────────────────────────────

    /// Scan for orphaned symbol/chunk nodes whose files no longer exist on disk.
    /// Also cleans up dangling edges (src or dst node doesn't exist).
    /// Returns `(symbols_cleaned, edges_cleaned)`.
    ///
    /// When `project_root` is `None`, file-existence checks are skipped
    /// (only dangling edge cleanup runs) to avoid CWD-dependent path
    /// resolution that could cause mass deletion.
    pub fn detect_orphans(
        &self,
        project_root: Option<&Path>,
    ) -> Result<(usize, usize), CodememError> {
        // Use storage for both nodes and edges to avoid in-memory/storage sync races.
        let all_nodes = self.storage.all_graph_nodes()?;
        let node_ids: HashSet<String> = all_nodes.iter().map(|n| n.id.clone()).collect();

        let mut orphan_sym_ids: Vec<String> = Vec::new();

        // Only check file existence when we have a known project root.
        // Without it, relative paths resolve against CWD which may be wrong.
        if let Some(root) = project_root {
            for node in &all_nodes {
                // Collect sym: and chunk: nodes whose backing files are gone
                if node.id.starts_with("sym:") || node.id.starts_with("chunk:") {
                    let file_path = match node.payload.get("file_path").and_then(|v| v.as_str()) {
                        Some(fp) => fp,
                        None => continue,
                    };
                    let abs_path = root.join(file_path);
                    if !abs_path.exists() {
                        orphan_sym_ids.push(node.id.clone());
                    }
                }
                // Collect file: nodes whose backing files are gone
                else if let Some(fp) = node.id.strip_prefix("file:") {
                    let abs_path = root.join(fp);
                    if !abs_path.exists() {
                        orphan_sym_ids.push(node.id.clone());
                    }
                }
                // Collect pkg: nodes whose directories are gone
                else if let Some(dir) = node.id.strip_prefix("pkg:") {
                    let dir_trimmed = dir.trim_end_matches('/');
                    let abs_path = root.join(dir_trimmed);
                    if !abs_path.exists() {
                        orphan_sym_ids.push(node.id.clone());
                    }
                }
            }
        }

        // Also find dangling edges (src or dst doesn't exist in graph)
        let all_edges = self.storage.all_graph_edges()?;
        let mut dangling_edge_ids: Vec<String> = Vec::new();
        for edge in &all_edges {
            if !node_ids.contains(&edge.src) || !node_ids.contains(&edge.dst) {
                dangling_edge_ids.push(edge.id.clone());
            }
        }

        let symbols_cleaned = orphan_sym_ids.len();

        // Clean up orphan nodes
        for sym_id in &orphan_sym_ids {
            if let Err(e) = self.storage.delete_graph_edges_for_node(sym_id) {
                tracing::warn!("Orphan cleanup: failed to delete edges for {sym_id}: {e}");
            }
            if let Err(e) = self.storage.delete_graph_node(sym_id) {
                tracing::warn!("Orphan cleanup: failed to delete node {sym_id}: {e}");
            }
            if let Err(e) = self.storage.delete_embedding(sym_id) {
                tracing::warn!("Orphan cleanup: failed to delete embedding {sym_id}: {e}");
            }
        }

        // Clean up orphan nodes from in-memory graph + vector
        if !orphan_sym_ids.is_empty() {
            if let Ok(mut graph) = self.lock_graph() {
                for sym_id in &orphan_sym_ids {
                    let _ = graph.remove_node(sym_id);
                }
            }
            if let Ok(mut vec) = self.lock_vector() {
                for sym_id in &orphan_sym_ids {
                    let _ = vec.remove(sym_id);
                }
            }
        }

        // Delete dangling edges that weren't already removed by node cleanup
        let mut edges_cleaned = 0usize;
        for edge_id in &dangling_edge_ids {
            match self.storage.delete_graph_edge(edge_id) {
                Ok(true) => edges_cleaned += 1,
                Ok(false) => {} // Already deleted by node cleanup above
                Err(e) => {
                    tracing::warn!("Orphan cleanup: failed to delete dangling edge {edge_id}: {e}");
                }
            }
        }

        if symbols_cleaned > 0 || edges_cleaned > 0 {
            tracing::info!(
                "Orphan scan: cleaned {symbols_cleaned} symbol/chunk nodes, {edges_cleaned} dangling edges"
            );
        }

        Ok((symbols_cleaned, edges_cleaned))
    }

    // ── A3c: Spec File Re-parsing ──────────────────────────────────────

    /// Re-parse a spec file (OpenAPI/AsyncAPI) and update stored endpoints/channels.
    fn reparse_spec_file(&self, path: &Path, namespace: &str) -> Result<(), CodememError> {
        use crate::index::spec_parser::{parse_asyncapi, parse_openapi, SpecFileResult};

        let result = if let Some(openapi) = parse_openapi(path) {
            Some(SpecFileResult::OpenApi(openapi))
        } else {
            parse_asyncapi(path).map(SpecFileResult::AsyncApi)
        };

        match result {
            Some(SpecFileResult::OpenApi(spec)) => {
                for ep in &spec.endpoints {
                    let _ = self.storage.store_api_endpoint(
                        &ep.method,
                        &ep.path,
                        ep.operation_id.as_deref().unwrap_or(""),
                        namespace,
                    );
                }
                tracing::info!(
                    "Re-parsed OpenAPI spec: {} endpoints from {}",
                    spec.endpoints.len(),
                    path.display()
                );
            }
            Some(SpecFileResult::AsyncApi(spec)) => {
                for ch in &spec.channels {
                    let _ = self.storage.store_event_channel(
                        &ch.channel,
                        &ch.direction,
                        ch.protocol.as_deref().unwrap_or(""),
                        ch.operation_id.as_deref().unwrap_or(""),
                        namespace,
                        ch.description.as_deref().unwrap_or(""),
                    );
                }
                tracing::info!(
                    "Re-parsed AsyncAPI spec: {} channels from {}",
                    spec.channels.len(),
                    path.display()
                );
            }
            None => {
                tracing::debug!("Not a recognized spec file: {}", path.display());
            }
        }
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

        // Eagerly initialize embeddings/vector/BM25 for the full analysis pipeline.
        // This triggers lazy init so that embed_and_persist() finds them ready.
        // Skip if embeddings are not needed.
        if !options.skip_embed {
            drop(self.lock_embeddings());
            drop(self.lock_vector());
            drop(self.lock_bm25());
        }

        // 0. SCIP phase: run indexers, parse results, build graph data.
        // Runs BEFORE ast-grep so we know which files SCIP covered.
        let (scip_covered, scip_build) = if !options.skip_scip && self.config.scip.enabled {
            match self.run_scip_phase(root, options.namespace) {
                Ok((covered, build)) => (Some(covered), Some(build)),
                Err(e) => {
                    tracing::warn!("SCIP phase failed, falling back to ast-grep only: {e}");
                    (None, None)
                }
            }
        } else {
            (None, None)
        };

        let scip_nodes_created = scip_build.as_ref().map_or(0, |b| b.nodes.len());
        let scip_edges_created = scip_build.as_ref().map_or(0, |b| b.edges.len());
        let scip_files_covered = scip_covered.as_ref().map_or(0, |s| s.len());

        // 1. Index (ast-grep skips symbol extraction for SCIP-covered files)
        // When force=true, ignore the change detector so all files are re-processed.
        let mut indexer = match options.change_detector {
            Some(cd) if !options.force => Indexer::with_change_detector(cd),
            _ => Indexer::new(),
        };
        let resolved =
            indexer.index_and_resolve_with_scip(root, scip_covered.as_ref(), scip_build)?;

        // 2. Persist (with or without progress callback)
        let persist = if options.skip_embed {
            self.persist_graph_only(&resolved, Some(options.namespace))?
        } else if let Some(ref on_progress) = options.progress {
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

        // 3. Enrich (skip if requested)
        let enrichment = if options.skip_enrich {
            crate::enrichment::EnrichmentPipelineResult {
                results: serde_json::json!({}),
                total_insights: 0,
            }
        } else {
            let path_str = root.to_str().unwrap_or("");
            self.run_enrichments(
                path_str,
                &[],
                options.git_days,
                Some(options.namespace),
                None,
            )
        };

        // 4. Recompute centrality
        self.lock_graph()?.recompute_centrality();

        // 5. Compute summary stats
        let top_nodes = self.find_important_nodes(10, 0.85).unwrap_or_default();
        let community_count = self.louvain_communities(1.0).map(|c| c.len()).unwrap_or(0);

        // 6. Save indexes
        self.save_index();

        // Save incremental state
        indexer
            .change_detector()
            .save_to_storage(self.storage(), options.namespace)?;

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
            scip_nodes_created,
            scip_edges_created,
            scip_files_covered,
        })
    }

    /// Run the SCIP phase: orchestrate indexers, parse results, build graph data.
    fn run_scip_phase(
        &self,
        root: &Path,
        namespace: &str,
    ) -> Result<(HashSet<String>, index::scip::graph_builder::ScipBuildResult), CodememError> {
        let orchestrator =
            index::scip::orchestrate::ScipOrchestrator::new(self.config.scip.clone());
        let orch_result = orchestrator.run(root, namespace)?;

        if orch_result.scip_result.covered_files.is_empty() {
            return Ok((
                HashSet::new(),
                index::scip::graph_builder::ScipBuildResult::default(),
            ));
        }

        for (lang, err) in &orch_result.failed_languages {
            tracing::warn!("SCIP indexer for {:?} failed: {}", lang, err);
        }
        for lang in &orch_result.indexed_languages {
            tracing::info!("SCIP indexed {:?} successfully", lang);
        }

        let build = index::scip::graph_builder::build_graph(
            &orch_result.scip_result,
            Some(namespace),
            &self.config.scip,
        );
        let covered: HashSet<String> = build.files_covered.clone();

        tracing::info!(
            "SCIP phase: {} nodes, {} edges, {} ext nodes, {} files covered, {} doc memories",
            build.nodes.len(),
            build.edges.len(),
            build.ext_nodes_created,
            covered.len(),
            build.doc_memories_created,
        );

        Ok((covered, build))
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
    /// Skip SCIP indexing — use ast-grep only (faster, less accurate).
    pub skip_scip: bool,
    /// Skip embedding phase (graph + chunks stored but not vectorized).
    pub skip_embed: bool,
    /// Skip enrichment phase (no git-history/complexity/etc. analysis).
    pub skip_enrich: bool,
    /// Force re-index even when file SHAs haven't changed.
    pub force: bool,
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
    /// SCIP nodes created (sym: + ext: nodes).
    pub scip_nodes_created: usize,
    /// SCIP edges created (CALLS, IMPORTS, READS, WRITES, IMPLEMENTS, etc.).
    pub scip_edges_created: usize,
    /// Files covered by SCIP indexers (ast-grep skipped symbol extraction for these).
    pub scip_files_covered: usize,
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

#[cfg(test)]
mod tests {
    use super::*;
    use codemem_core::{Edge, GraphNode, NodeKind, RelationshipType};
    use std::collections::{HashMap, HashSet};

    /// Create a test engine backed by a temporary database.
    fn test_engine() -> CodememEngine {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        // Keep the tempdir alive by leaking it (tests are short-lived).
        let _ = Box::leak(Box::new(dir));
        CodememEngine::from_db_path(&db_path).unwrap()
    }

    fn graph_node(id: &str, kind: NodeKind, file_path: Option<&str>) -> GraphNode {
        let mut payload = HashMap::new();
        if let Some(fp) = file_path {
            payload.insert(
                "file_path".to_string(),
                serde_json::Value::String(fp.to_string()),
            );
        }
        GraphNode {
            id: id.to_string(),
            kind,
            label: id.to_string(),
            payload,
            centrality: 0.0,
            memory_id: None,
            namespace: None,
        }
    }

    fn edge(src: &str, dst: &str, rel: RelationshipType) -> Edge {
        Edge {
            id: format!("{rel}:{src}->{dst}"),
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

    // ── cleanup_stale_symbols tests ──────────────────────────────────────

    #[test]
    fn cleanup_stale_symbols_deletes_stale_nodes() {
        let engine = test_engine();

        // Set up: file with two symbols, one will become stale
        let file = graph_node("file:src/a.rs", NodeKind::File, None);
        let sym_keep = graph_node("sym:a::keep", NodeKind::Function, Some("src/a.rs"));
        let sym_stale = graph_node("sym:a::stale", NodeKind::Function, Some("src/a.rs"));

        {
            let mut g = engine.lock_graph().unwrap();
            g.add_node(file).unwrap();
            g.add_node(sym_keep.clone()).unwrap();
            g.add_node(sym_stale.clone()).unwrap();
            g.add_edge(edge(
                "file:src/a.rs",
                "sym:a::keep",
                RelationshipType::Contains,
            ))
            .unwrap();
            g.add_edge(edge(
                "file:src/a.rs",
                "sym:a::stale",
                RelationshipType::Contains,
            ))
            .unwrap();
        }
        // Also persist to storage so cleanup can find edges
        let _ =
            engine
                .storage
                .insert_graph_node(&graph_node("file:src/a.rs", NodeKind::File, None));
        let _ = engine.storage.insert_graph_node(&sym_keep);
        let _ = engine.storage.insert_graph_node(&sym_stale);
        let _ = engine.storage.insert_graph_edge(&edge(
            "file:src/a.rs",
            "sym:a::keep",
            RelationshipType::Contains,
        ));
        let _ = engine.storage.insert_graph_edge(&edge(
            "file:src/a.rs",
            "sym:a::stale",
            RelationshipType::Contains,
        ));

        let old_ids: HashSet<String> = ["sym:a::keep", "sym:a::stale"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let new_ids: HashSet<String> = ["sym:a::keep"].iter().map(|s| s.to_string()).collect();

        let cleaned = engine
            .cleanup_stale_symbols("src/a.rs", &old_ids, &new_ids)
            .unwrap();
        assert_eq!(cleaned, 1);

        // Stale node should be gone from in-memory graph
        let g = engine.lock_graph().unwrap();
        assert!(g.get_node("sym:a::stale").unwrap().is_none());
        assert!(g.get_node("sym:a::keep").unwrap().is_some());
    }

    #[test]
    fn cleanup_stale_symbols_redirects_memory_edges_to_graph() {
        let engine = test_engine();

        let file = graph_node("file:src/a.rs", NodeKind::File, None);
        let sym_stale = graph_node("sym:a::old_fn", NodeKind::Function, Some("src/a.rs"));
        let mem = graph_node("mem-uuid-123", NodeKind::Memory, None);

        {
            let mut g = engine.lock_graph().unwrap();
            g.add_node(file.clone()).unwrap();
            g.add_node(sym_stale.clone()).unwrap();
            g.add_node(mem.clone()).unwrap();
            g.add_edge(edge(
                "file:src/a.rs",
                "sym:a::old_fn",
                RelationshipType::Contains,
            ))
            .unwrap();
            g.add_edge(edge(
                "mem-uuid-123",
                "sym:a::old_fn",
                RelationshipType::RelatesTo,
            ))
            .unwrap();
        }
        let _ = engine.storage.insert_graph_node(&file);
        let _ = engine.storage.insert_graph_node(&sym_stale);
        let _ = engine.storage.insert_graph_node(&mem);
        let _ = engine.storage.insert_graph_edge(&edge(
            "file:src/a.rs",
            "sym:a::old_fn",
            RelationshipType::Contains,
        ));
        let _ = engine.storage.insert_graph_edge(&edge(
            "mem-uuid-123",
            "sym:a::old_fn",
            RelationshipType::RelatesTo,
        ));

        let old_ids: HashSet<String> = ["sym:a::old_fn"].iter().map(|s| s.to_string()).collect();
        let new_ids: HashSet<String> = HashSet::new();

        engine
            .cleanup_stale_symbols("src/a.rs", &old_ids, &new_ids)
            .unwrap();

        // The redirected edge should be in the in-memory graph
        let g = engine.lock_graph().unwrap();
        let file_edges = g.get_edges("file:src/a.rs").unwrap();
        let has_redirect = file_edges.iter().any(|e| {
            (e.src == "mem-uuid-123" || e.dst == "mem-uuid-123") && e.id.contains("-redirected")
        });
        assert!(
            has_redirect,
            "redirected memory→file edge should be in the in-memory graph"
        );
    }

    #[test]
    fn cleanup_stale_symbols_deduplicates_redirects() {
        let engine = test_engine();

        let file = graph_node("file:src/a.rs", NodeKind::File, None);
        let sym1 = graph_node("sym:a::fn1", NodeKind::Function, Some("src/a.rs"));
        let sym2 = graph_node("sym:a::fn2", NodeKind::Function, Some("src/a.rs"));
        let mem = graph_node("mem-uuid-456", NodeKind::Memory, None);

        // Same memory linked to two symbols in the same file
        let _ = engine.storage.insert_graph_node(&file);
        let _ = engine.storage.insert_graph_node(&sym1);
        let _ = engine.storage.insert_graph_node(&sym2);
        let _ = engine.storage.insert_graph_node(&mem);
        let _ = engine.storage.insert_graph_edge(&edge(
            "mem-uuid-456",
            "sym:a::fn1",
            RelationshipType::RelatesTo,
        ));
        let _ = engine.storage.insert_graph_edge(&edge(
            "mem-uuid-456",
            "sym:a::fn2",
            RelationshipType::RelatesTo,
        ));

        {
            let mut g = engine.lock_graph().unwrap();
            g.add_node(file).unwrap();
            g.add_node(sym1).unwrap();
            g.add_node(sym2).unwrap();
            g.add_node(mem).unwrap();
        }

        let old_ids: HashSet<String> = ["sym:a::fn1", "sym:a::fn2"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let new_ids: HashSet<String> = HashSet::new();

        engine
            .cleanup_stale_symbols("src/a.rs", &old_ids, &new_ids)
            .unwrap();

        // Should have exactly one redirect edge, not two
        let g = engine.lock_graph().unwrap();
        let file_edges = g.get_edges("file:src/a.rs").unwrap();
        let redirect_count = file_edges
            .iter()
            .filter(|e| e.id.contains("-redirected"))
            .count();
        assert_eq!(
            redirect_count, 1,
            "should have exactly 1 redirected edge, got {redirect_count}"
        );
    }

    // ── detect_orphans tests ─────────────────────────────────────────────

    #[test]
    fn detect_orphans_skips_file_check_when_no_root() {
        let engine = test_engine();

        // Add a symbol node with a file path that definitely doesn't exist
        let sym = graph_node(
            "sym:nonexistent::fn",
            NodeKind::Function,
            Some("does/not/exist.rs"),
        );
        let _ = engine.storage.insert_graph_node(&sym);
        {
            let mut g = engine.lock_graph().unwrap();
            g.add_node(sym).unwrap();
        }

        // With None, should NOT delete the node (skips file existence check)
        let (symbols_cleaned, _) = engine.detect_orphans(None).unwrap();
        assert_eq!(
            symbols_cleaned, 0,
            "detect_orphans(None) should not delete nodes based on file existence"
        );
    }

    #[test]
    fn detect_orphans_removes_missing_files_with_root() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let engine = CodememEngine::from_db_path(&db_path).unwrap();

        // Add a symbol whose file doesn't exist under the project root
        let sym = graph_node(
            "sym:missing::fn",
            NodeKind::Function,
            Some("src/missing.rs"),
        );
        let _ = engine.storage.insert_graph_node(&sym);
        {
            let mut g = engine.lock_graph().unwrap();
            g.add_node(sym).unwrap();
        }

        let (symbols_cleaned, _) = engine.detect_orphans(Some(dir.path())).unwrap();
        assert_eq!(symbols_cleaned, 1);
    }

    #[test]
    fn detect_orphans_keeps_existing_files() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let engine = CodememEngine::from_db_path(&db_path).unwrap();

        // Create the actual file so it won't be orphaned
        let src_dir = dir.path().join("src");
        std::fs::create_dir_all(&src_dir).unwrap();
        std::fs::write(src_dir.join("exists.rs"), "fn main() {}").unwrap();

        let sym = graph_node(
            "sym:exists::main",
            NodeKind::Function,
            Some("src/exists.rs"),
        );
        let _ = engine.storage.insert_graph_node(&sym);
        {
            let mut g = engine.lock_graph().unwrap();
            g.add_node(sym).unwrap();
        }

        let (symbols_cleaned, _) = engine.detect_orphans(Some(dir.path())).unwrap();
        assert_eq!(symbols_cleaned, 0);
    }

    // Note: dangling edge cleanup in detect_orphans is a defensive no-op
    // because graph_edges has ON DELETE CASCADE foreign keys on src/dst.
    // Deleting a node automatically cascades to its edges in SQLite.
}
