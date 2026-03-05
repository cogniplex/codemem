//! codemem-engine: Domain logic engine for the Codemem memory system.
//!
//! This crate contains all business logic, orchestration, and domain operations:
//! - **index** — ast-grep based code indexing, symbol extraction, reference resolution
//! - **hooks** — Lifecycle hook handlers (PostToolUse, SessionStart, Stop)
//! - **watch** — Real-time file watching with debouncing and .gitignore support
//! - **bm25** — Okapi BM25 scoring with code-aware tokenization
//! - **scoring** — 9-component hybrid scoring for memory recall
//! - **patterns** — Cross-session pattern detection
//! - **compress** — Optional LLM-powered observation compression
//! - **metrics** — Operational metrics collection

use codemem_core::{
    CodememConfig, CodememError, DetectedPattern, Edge, GraphBackend, MemoryNode, MemoryType,
    NodeKind, NodeMemoryResult, RelationshipType, ScoringWeights, StorageBackend, VectorBackend,
};
use codemem_storage::graph::GraphEngine;
use codemem_storage::HnswIndex;
use codemem_storage::Storage;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};

pub mod analysis;
pub mod bm25;
pub mod compress;
pub mod consolidation;
pub mod enrichment;
pub mod hooks;
pub mod index;
pub mod metrics;
pub mod patterns;
pub mod persistence;
pub mod recall;
pub mod scoring;
pub mod search;
pub mod watch;

#[cfg(test)]
#[path = "tests/engine_integration_tests.rs"]
mod integration_tests;

#[cfg(test)]
#[path = "tests/enrichment_tests.rs"]
mod enrichment_tests;

// Re-export key index types at crate root for convenience
pub use index::{
    ChunkConfig, CodeChunk, CodeParser, Dependency, IndexAndResolveResult, IndexProgress,
    IndexResult, Indexer, ManifestResult, ParseResult, Reference, ReferenceKind, ReferenceResolver,
    ResolvedEdge, Symbol, SymbolKind, Visibility, Workspace,
};

// Re-export key domain types for convenience
pub use bm25::Bm25Index;
pub use metrics::InMemoryMetrics;

// Re-export enrichment types
pub use enrichment::EnrichResult;

// Re-export persistence types
pub use persistence::{edge_weight_for, IndexPersistResult};

// Re-export recall types
pub use recall::{ExpandedResult, NamespaceStats};

// Re-export search types
pub use search::{CodeSearchResult, SummaryTreeNode, SymbolSearchResult};

// Re-export analysis types
pub use analysis::{
    DecisionChain, DecisionConnection, DecisionEntry, ImpactResult, SessionCheckpointReport,
};

/// A part descriptor for `split_memory()`.
#[derive(Debug, Clone)]
pub struct SplitPart {
    pub content: String,
    pub tags: Option<Vec<String>>,
    pub importance: Option<f64>,
}

// ── Index Cache ──────────────────────────────────────────────────────────────

/// Cached code-index results for structural queries.
pub struct IndexCache {
    pub symbols: Vec<Symbol>,
    pub chunks: Vec<CodeChunk>,
    pub root_path: String,
}

// ── CodememEngine ────────────────────────────────────────────────────────────

/// Core domain engine holding all backends and domain state.
///
/// This struct contains all the business logic for the Codemem memory system.
/// Transport layers (MCP, REST API, CLI) hold a `CodememEngine` and delegate
/// domain operations to it, keeping transport concerns separate.
///
/// **Concrete types are intentional**: `CodememEngine` uses concrete backend types
/// (`Storage`, `HnswIndex`, `GraphEngine`) rather than trait objects (`dyn StorageBackend`,
/// `dyn VectorBackend`, `dyn GraphBackend`) for performance. This enables monomorphization
/// (the compiler generates specialized code for each concrete type), eliminates vtable
/// indirection overhead on every call, and provides predictable memory layout for
/// cache-friendly access patterns. The trait abstractions exist for testing and
/// alternative implementations, but the engine itself benefits from static dispatch.
pub struct CodememEngine {
    // TODO(C7): Fields should be `pub(crate)` with accessor methods, but many files
    // in `crates/codemem/src/` (a different crate) access these fields directly
    // (e.g., `engine.storage`, `engine.graph.lock()`, `engine.metrics`). These must
    // remain `pub` until all external callers are migrated to use accessor methods.
    pub storage: Box<dyn StorageBackend>,
    pub vector: Mutex<HnswIndex>,
    pub graph: Mutex<GraphEngine>,
    /// Optional embedding provider (None if not configured).
    pub embeddings: Option<Mutex<Box<dyn codemem_embeddings::EmbeddingProvider>>>,
    /// Path to the database file, used to derive the index save path.
    pub db_path: Option<PathBuf>,
    /// Cached index results for structural queries.
    pub index_cache: Mutex<Option<IndexCache>>,
    /// Configurable scoring weights for the 9-component hybrid scoring system.
    pub scoring_weights: RwLock<ScoringWeights>,
    /// BM25 index for code-aware token overlap scoring.
    pub bm25_index: Mutex<Bm25Index>,
    /// Loaded configuration.
    pub config: CodememConfig,
    /// Operational metrics collector.
    pub metrics: Arc<InMemoryMetrics>,
    /// Dirty flag for batch saves: set after `persist_memory_no_save()`,
    /// cleared by `save_index()`.
    dirty: AtomicBool,
}

impl CodememEngine {
    /// Create an engine with storage, vector, graph, and optional embeddings backends.
    pub fn new(
        storage: Box<dyn StorageBackend>,
        vector: HnswIndex,
        graph: GraphEngine,
        embeddings: Option<Box<dyn codemem_embeddings::EmbeddingProvider>>,
    ) -> Self {
        let config = CodememConfig::load_or_default();
        Self::new_with_config(storage, vector, graph, embeddings, config)
    }

    /// Create an engine with an explicit config (avoids double-loading from disk).
    pub fn new_with_config(
        storage: Box<dyn StorageBackend>,
        vector: HnswIndex,
        graph: GraphEngine,
        embeddings: Option<Box<dyn codemem_embeddings::EmbeddingProvider>>,
        config: CodememConfig,
    ) -> Self {
        Self {
            storage,
            vector: Mutex::new(vector),
            graph: Mutex::new(graph),
            embeddings: embeddings.map(Mutex::new),
            db_path: None,
            index_cache: Mutex::new(None),
            scoring_weights: RwLock::new(config.scoring.clone()),
            bm25_index: Mutex::new(Bm25Index::new()),
            config,
            metrics: Arc::new(InMemoryMetrics::new()),
            dirty: AtomicBool::new(false),
        }
    }

    /// Create an engine from a database path, loading all backends.
    pub fn from_db_path(db_path: &Path) -> Result<Self, CodememError> {
        let config = CodememConfig::load_or_default();

        // Wire StorageConfig into Storage::open
        let storage = Storage::open_with_config(
            db_path,
            Some(config.storage.cache_size_mb),
            Some(config.storage.busy_timeout_secs),
        )?;
        let mut vector = HnswIndex::with_defaults()?;

        // Load existing vector index if it exists
        let index_path = db_path.with_extension("idx");
        if index_path.exists() {
            vector.load(&index_path)?;
        }

        // C6: Vector index consistency check — compare vector index count vs DB embedding count.
        // If they mismatch, rebuild the vector index from SQLite embeddings.
        let vector_count = vector.stats().count;
        let db_stats = storage.stats()?;
        let db_embed_count = db_stats.embedding_count;
        if vector_count != db_embed_count {
            tracing::warn!(
                "Vector index ({vector_count}) out of sync with DB ({db_embed_count}), rebuilding..."
            );
            // Rebuild: create a fresh index and re-insert all embeddings from DB
            let mut fresh_vector = HnswIndex::with_defaults()?;
            if let Ok(embeddings) = storage.list_all_embeddings() {
                for (id, embedding) in &embeddings {
                    if let Err(e) = fresh_vector.insert(id, embedding) {
                        tracing::warn!("Failed to re-insert embedding {id}: {e}");
                    }
                }
            }
            vector = fresh_vector;
            // Save the rebuilt index
            if let Err(e) = vector.save(&index_path) {
                tracing::warn!("Failed to save rebuilt vector index: {e}");
            }
        }

        // Load graph from storage
        let graph = GraphEngine::from_storage(&storage)?;

        // Wire EmbeddingConfig into from_env as fallback
        let embeddings = codemem_embeddings::from_env(Some(&config.embedding)).ok();

        let mut engine =
            Self::new_with_config(Box::new(storage), vector, graph, embeddings, config);
        engine.db_path = Some(db_path.to_path_buf());

        // H7: Only compute PageRank at startup; betweenness is computed lazily
        // via `ensure_betweenness_computed()` when first needed.
        engine
            .lock_graph()?
            .recompute_centrality_with_options(false);

        // Try loading persisted BM25 index; fall back to rebuilding from memories.
        let bm25_path = db_path.with_extension("bm25");
        let mut bm25_loaded = false;
        if bm25_path.exists() {
            match std::fs::read(&bm25_path) {
                Ok(data) => match Bm25Index::deserialize(&data) {
                    Ok(index) => {
                        let mut bm25 = engine.lock_bm25()?;
                        *bm25 = index;
                        bm25_loaded = true;
                        tracing::info!(
                            "Loaded BM25 index from disk ({} documents)",
                            bm25.doc_count
                        );
                    }
                    Err(e) => {
                        tracing::warn!("Failed to deserialize BM25 index, rebuilding: {e}");
                    }
                },
                Err(e) => {
                    tracing::warn!("Failed to read BM25 index file, rebuilding: {e}");
                }
            }
        }

        if !bm25_loaded {
            // Rebuild BM25 index from all existing memories (batch load)
            if let Ok(ids) = engine.storage.list_memory_ids() {
                let id_refs: Vec<&str> = ids.iter().map(|s| s.as_str()).collect();
                if let Ok(memories) = engine.storage.get_memories_batch(&id_refs) {
                    let mut bm25 = engine.lock_bm25()?;
                    for memory in &memories {
                        bm25.add_document(&memory.id, &memory.content);
                    }
                    tracing::info!("Rebuilt BM25 index from {} memories", bm25.doc_count);
                }
            }
        }

        Ok(engine)
    }

    /// Create a minimal engine for testing.
    pub fn for_testing() -> Self {
        let storage = Storage::open_in_memory().unwrap();
        let vector = HnswIndex::with_defaults().unwrap();
        let graph = GraphEngine::new();
        let config = CodememConfig::default();
        Self {
            storage: Box::new(storage),
            vector: Mutex::new(vector),
            graph: Mutex::new(graph),
            embeddings: None,
            db_path: None,
            index_cache: Mutex::new(None),
            scoring_weights: RwLock::new(config.scoring.clone()),
            bm25_index: Mutex::new(Bm25Index::new()),
            config,
            metrics: Arc::new(InMemoryMetrics::new()),
            dirty: AtomicBool::new(false),
        }
    }

    // ── Lock Helpers ─────────────────────────────────────────────────────────

    pub fn lock_vector(&self) -> Result<std::sync::MutexGuard<'_, HnswIndex>, CodememError> {
        self.vector
            .lock()
            .map_err(|e| CodememError::LockPoisoned(format!("vector: {e}")))
    }

    pub fn lock_graph(&self) -> Result<std::sync::MutexGuard<'_, GraphEngine>, CodememError> {
        self.graph
            .lock()
            .map_err(|e| CodememError::LockPoisoned(format!("graph: {e}")))
    }

    pub fn lock_bm25(&self) -> Result<std::sync::MutexGuard<'_, Bm25Index>, CodememError> {
        self.bm25_index
            .lock()
            .map_err(|e| CodememError::LockPoisoned(format!("bm25: {e}")))
    }

    pub fn lock_embeddings(
        &self,
    ) -> Result<
        Option<std::sync::MutexGuard<'_, Box<dyn codemem_embeddings::EmbeddingProvider>>>,
        CodememError,
    > {
        match &self.embeddings {
            Some(m) => Ok(Some(m.lock().map_err(|e| {
                CodememError::LockPoisoned(format!("embeddings: {e}"))
            })?)),
            None => Ok(None),
        }
    }

    pub fn lock_index_cache(
        &self,
    ) -> Result<std::sync::MutexGuard<'_, Option<IndexCache>>, CodememError> {
        self.index_cache
            .lock()
            .map_err(|e| CodememError::LockPoisoned(format!("index_cache: {e}")))
    }

    pub fn scoring_weights(
        &self,
    ) -> Result<std::sync::RwLockReadGuard<'_, ScoringWeights>, CodememError> {
        self.scoring_weights
            .read()
            .map_err(|e| CodememError::LockPoisoned(format!("scoring_weights read: {e}")))
    }

    pub fn scoring_weights_mut(
        &self,
    ) -> Result<std::sync::RwLockWriteGuard<'_, ScoringWeights>, CodememError> {
        self.scoring_weights
            .write()
            .map_err(|e| CodememError::LockPoisoned(format!("scoring_weights write: {e}")))
    }

    /// Check if the engine has unsaved changes (dirty flag is set).
    #[cfg(test)]
    pub(crate) fn is_dirty(&self) -> bool {
        self.dirty.load(Ordering::Acquire)
    }

    // ── Contextual Enrichment ────────────────────────────────────────────────

    /// Build contextual text for a memory node.
    ///
    /// NOTE: Acquires the graph lock on each call. For batch operations,
    /// consider passing a pre-acquired guard or caching results.
    pub fn enrich_memory_text(
        &self,
        content: &str,
        memory_type: MemoryType,
        tags: &[String],
        namespace: Option<&str>,
        node_id: Option<&str>,
    ) -> String {
        let mut ctx = String::new();
        ctx.push_str(&format!("[{}]", memory_type));

        if let Some(ns) = namespace {
            ctx.push_str(&format!(" [namespace:{}]", ns));
        }

        if !tags.is_empty() {
            ctx.push_str(&format!(" [tags:{}]", tags.join(",")));
        }

        if let Some(nid) = node_id {
            let graph = match self.lock_graph() {
                Ok(g) => g,
                Err(_) => return format!("{ctx}\n{content}"),
            };
            if let Ok(edges) = graph.get_edges(nid) {
                let mut rels: Vec<String> = Vec::new();
                for edge in edges.iter().take(8) {
                    let other = if edge.src == nid {
                        &edge.dst
                    } else {
                        &edge.src
                    };
                    let label = graph
                        .get_node(other)
                        .ok()
                        .flatten()
                        .map(|n| n.label.clone())
                        .unwrap_or_else(|| other.to_string());
                    let dir = if edge.src == nid { "->" } else { "<-" };
                    rels.push(format!("{dir} {} ({})", label, edge.relationship));
                }
                if !rels.is_empty() {
                    ctx.push_str(&format!("\nRelated: {}", rels.join("; ")));
                }
            }
        }

        format!("{ctx}\n{content}")
    }

    /// Build contextual text for a code symbol.
    pub fn enrich_symbol_text(&self, sym: &Symbol, edges: &[ResolvedEdge]) -> String {
        let mut ctx = String::new();
        ctx.push_str(&format!("[{} {}]", sym.visibility, sym.kind));
        ctx.push_str(&format!(" File: {}", sym.file_path));

        if let Some(ref parent) = sym.parent {
            ctx.push_str(&format!(" Parent: {}", parent));
        }

        let related: Vec<String> = edges
            .iter()
            .filter(|e| {
                e.source_qualified_name == sym.qualified_name
                    || e.target_qualified_name == sym.qualified_name
            })
            .take(8)
            .map(|e| {
                if e.source_qualified_name == sym.qualified_name {
                    format!("-> {} ({})", e.target_qualified_name, e.relationship)
                } else {
                    format!("<- {} ({})", e.source_qualified_name, e.relationship)
                }
            })
            .collect();
        if !related.is_empty() {
            ctx.push_str(&format!("\nRelated: {}", related.join("; ")));
        }

        let mut body = format!("{}: {}", sym.qualified_name, sym.signature);
        if let Some(ref doc) = sym.doc_comment {
            body.push('\n');
            body.push_str(doc);
        }

        format!("{ctx}\n{body}")
    }

    /// Build contextual text for a code chunk before embedding.
    pub fn enrich_chunk_text(&self, chunk: &CodeChunk) -> String {
        let mut ctx = String::new();
        ctx.push_str(&format!("[chunk:{}]", chunk.node_kind));
        ctx.push_str(&format!(" File: {}", chunk.file_path));
        ctx.push_str(&format!(" Lines: {}-{}", chunk.line_start, chunk.line_end));
        if let Some(ref parent) = chunk.parent_symbol {
            ctx.push_str(&format!(" Parent: {}", parent));
        }

        let body = scoring::truncate_content(&chunk.text, 4000);

        format!("{ctx}\n{body}")
    }

    // ── Auto-linking ─────────────────────────────────────────────────────

    /// Scan memory content for file paths and qualified symbol names that exist
    /// as graph nodes, and create RELATES_TO edges.
    pub fn auto_link_to_code_nodes(
        &self,
        memory_id: &str,
        content: &str,
        existing_links: &[String],
    ) -> usize {
        let mut graph = match self.lock_graph() {
            Ok(g) => g,
            Err(_) => return 0,
        };

        let existing_set: std::collections::HashSet<&str> =
            existing_links.iter().map(|s| s.as_str()).collect();

        let mut candidates: Vec<String> = Vec::new();

        for word in content.split_whitespace() {
            let cleaned = word.trim_matches(|c: char| {
                !c.is_alphanumeric() && c != '/' && c != '.' && c != '_' && c != '-' && c != ':'
            });
            if cleaned.is_empty() {
                continue;
            }
            if cleaned.contains('/') || cleaned.contains('.') {
                let file_id = format!("file:{cleaned}");
                if !existing_set.contains(file_id.as_str()) {
                    candidates.push(file_id);
                }
            }
            if cleaned.contains("::") {
                let sym_id = format!("sym:{cleaned}");
                if !existing_set.contains(sym_id.as_str()) {
                    candidates.push(sym_id);
                }
            }
        }

        let now = chrono::Utc::now();
        let mut created = 0;
        let mut seen = std::collections::HashSet::new();

        for candidate_id in &candidates {
            if !seen.insert(candidate_id.clone()) {
                continue;
            }
            if graph.get_node(candidate_id).ok().flatten().is_none() {
                continue;
            }
            let edge = Edge {
                id: format!("{memory_id}-RELATES_TO-{candidate_id}"),
                src: memory_id.to_string(),
                dst: candidate_id.clone(),
                relationship: RelationshipType::RelatesTo,
                weight: 0.5,
                properties: std::collections::HashMap::from([(
                    "auto_linked".to_string(),
                    serde_json::json!(true),
                )]),
                created_at: now,
                valid_from: None,
                valid_to: None,
            };
            if self.storage.insert_graph_edge(&edge).is_ok() && graph.add_edge(edge).is_ok() {
                created += 1;
            }
        }

        created
    }

    // ── Node Memory Queries ──────────────────────────────────────────────

    /// Retrieve all memories connected to a graph node via BFS traversal.
    ///
    /// Performs level-by-level BFS to track actual hop distance. For each
    /// Memory node found, reports the relationship type from the edge that
    /// connected it (or the edge leading into the path toward it).
    pub fn get_node_memories(
        &self,
        node_id: &str,
        max_depth: usize,
        include_relationships: Option<&[RelationshipType]>,
    ) -> Result<Vec<NodeMemoryResult>, CodememError> {
        let graph = self.lock_graph()?;

        // Manual BFS tracking (node_id, depth, relationship_from_parent_edge)
        let mut results: Vec<NodeMemoryResult> = Vec::new();
        let mut seen_memory_ids = HashSet::new();
        let mut visited = HashSet::new();
        let mut queue: std::collections::VecDeque<(String, usize, String)> =
            std::collections::VecDeque::new();

        visited.insert(node_id.to_string());
        queue.push_back((node_id.to_string(), 0, String::new()));

        while let Some((current_id, depth, parent_rel)) = queue.pop_front() {
            // Collect Memory nodes (skip the start node itself)
            if current_id != node_id {
                if let Some(node) = graph.get_node_ref(&current_id) {
                    if node.kind == NodeKind::Memory {
                        let memory_id = node.memory_id.as_deref().unwrap_or(&node.id);
                        if seen_memory_ids.insert(memory_id.to_string()) {
                            if let Ok(Some(memory)) = self.storage.get_memory_no_touch(memory_id) {
                                results.push(NodeMemoryResult {
                                    memory,
                                    relationship: parent_rel.clone(),
                                    depth,
                                });
                            }
                        }
                    }
                }
            }

            if depth >= max_depth {
                continue;
            }

            // Expand neighbors via edges, skipping Chunk nodes
            for edge in graph.get_edges_ref(&current_id) {
                let neighbor_id = if edge.src == current_id {
                    &edge.dst
                } else {
                    &edge.src
                };

                if visited.contains(neighbor_id.as_str()) {
                    continue;
                }

                // Apply relationship filter
                if let Some(allowed) = include_relationships {
                    if !allowed.contains(&edge.relationship) {
                        continue;
                    }
                }

                // Skip Chunk nodes (noisy, low-value for memory discovery)
                if let Some(neighbor) = graph.get_node_ref(neighbor_id) {
                    if neighbor.kind == NodeKind::Chunk {
                        continue;
                    }
                }

                visited.insert(neighbor_id.clone());
                queue.push_back((
                    neighbor_id.clone(),
                    depth + 1,
                    edge.relationship.to_string(),
                ));
            }
        }

        Ok(results)
    }

    // ── Persistence ─────────────────────────────────────────────────────

    /// Persist a memory through the full pipeline: storage → BM25 → graph → embedding → vector.
    pub fn persist_memory(&self, memory: &MemoryNode) -> Result<(), CodememError> {
        self.persist_memory_inner(memory, true)
    }

    /// Persist a memory without saving the vector index to disk.
    /// Use this in batch operations, then call `save_index()` once at the end.
    pub(crate) fn persist_memory_no_save(&self, memory: &MemoryNode) -> Result<(), CodememError> {
        self.persist_memory_inner(memory, false)
    }

    /// Inner persist implementation with optional index save.
    ///
    /// H3: Lock ordering is enforced to prevent deadlocks:
    /// 1. Embeddings lock (acquire, embed, drop)
    /// 2. BM25 lock
    /// 3. Graph lock
    /// 4. Vector lock
    fn persist_memory_inner(&self, memory: &MemoryNode, save: bool) -> Result<(), CodememError> {
        // 1. Store in SQLite
        self.storage.insert_memory(memory)?;

        // H3: Step 1 — Acquire embeddings lock first, embed, save result, drop lock.
        // This prevents holding the embeddings lock while acquiring vector/graph locks.
        let embedding_result = match self.lock_embeddings() {
            Ok(Some(emb)) => {
                let enriched = self.enrich_memory_text(
                    &memory.content,
                    memory.memory_type,
                    &memory.tags,
                    memory.namespace.as_deref(),
                    Some(&memory.id),
                );
                let result = emb.embed(&enriched).ok();
                drop(emb);
                result
            }
            Ok(None) => None,
            Err(e) => {
                tracing::warn!("Embeddings lock failed during persist: {e}");
                None
            }
        };

        // 2. Update BM25 index
        match self.lock_bm25() {
            Ok(mut bm25) => {
                bm25.add_document(&memory.id, &memory.content);
            }
            Err(e) => tracing::warn!("BM25 lock failed during persist: {e}"),
        }

        // 3. Add memory node to graph (separate lock scope)
        match self.lock_graph() {
            Ok(mut graph) => {
                let node = codemem_core::GraphNode {
                    id: memory.id.clone(),
                    kind: codemem_core::NodeKind::Memory,
                    label: scoring::truncate_content(&memory.content, 80),
                    payload: std::collections::HashMap::new(),
                    centrality: 0.0,
                    memory_id: Some(memory.id.clone()),
                    namespace: memory.namespace.clone(),
                };
                if let Err(e) = self.storage.insert_graph_node(&node) {
                    tracing::warn!("Failed to insert graph node for memory {}: {e}", memory.id);
                }
                if let Err(e) = graph.add_node(node) {
                    tracing::warn!(
                        "Failed to add graph node in-memory for memory {}: {e}",
                        memory.id
                    );
                }
            }
            Err(e) => tracing::warn!("Graph lock failed during persist: {e}"),
        }

        // H3: Step 4 — Insert embedding into vector index (separate lock scope from embeddings).
        if let Some(vec) = &embedding_result {
            if let Ok(mut vi) = self.lock_vector() {
                if let Err(e) = vi.insert(&memory.id, vec) {
                    tracing::warn!("Failed to insert into vector index for {}: {e}", memory.id);
                }
            }
            if let Err(e) = self.storage.store_embedding(&memory.id, vec) {
                tracing::warn!("Failed to store embedding for {}: {e}", memory.id);
            }
        }

        // C5: Set dirty flag instead of calling save_index() after each persist.
        // Callers should use flush_if_dirty() to batch save the index.
        if save {
            self.save_index(); // save_index() clears dirty flag
        } else {
            self.dirty.store(true, Ordering::Release);
        }

        Ok(())
    }

    // ── Edge Helpers ─────────────────────────────────────────────────────

    /// Add an edge to both storage and in-memory graph.
    pub fn add_edge(&self, edge: Edge) -> Result<(), CodememError> {
        self.storage.insert_graph_edge(&edge)?;
        let mut graph = self.lock_graph()?;
        graph.add_edge(edge)?;
        Ok(())
    }

    // ── Self-Editing ────────────────────────────────────────────────────

    /// Refine a memory: create a new version with an EVOLVED_INTO edge from old to new.
    pub fn refine_memory(
        &self,
        old_id: &str,
        content: Option<&str>,
        tags: Option<Vec<String>>,
        importance: Option<f64>,
    ) -> Result<(MemoryNode, String), CodememError> {
        let old_memory = self
            .storage
            .get_memory(old_id)?
            .ok_or_else(|| CodememError::NotFound(format!("Memory not found: {old_id}")))?;

        let new_content = content.unwrap_or(&old_memory.content);
        let new_tags = tags.unwrap_or_else(|| old_memory.tags.clone());
        let new_importance = importance.unwrap_or(old_memory.importance);

        let now = chrono::Utc::now();
        let new_id = uuid::Uuid::new_v4().to_string();
        let hash = codemem_storage::Storage::content_hash(new_content);

        let memory = MemoryNode {
            id: new_id.clone(),
            content: new_content.to_string(),
            memory_type: old_memory.memory_type,
            importance: new_importance,
            confidence: old_memory.confidence,
            access_count: 0,
            content_hash: hash,
            tags: new_tags,
            metadata: old_memory.metadata.clone(),
            namespace: old_memory.namespace.clone(),
            created_at: now,
            updated_at: now,
            last_accessed_at: now,
        };

        self.persist_memory(&memory)?;

        // Create EVOLVED_INTO edge from old -> new
        let edge = Edge {
            id: format!("{old_id}-EVOLVED_INTO-{new_id}"),
            src: old_id.to_string(),
            dst: new_id.clone(),
            relationship: RelationshipType::EvolvedInto,
            weight: 1.0,
            properties: std::collections::HashMap::new(),
            created_at: now,
            valid_from: Some(now),
            valid_to: None,
        };
        if let Err(e) = self.add_edge(edge) {
            tracing::warn!("Failed to add EVOLVED_INTO edge: {e}");
        }

        Ok((memory, new_id))
    }

    /// Split a memory into multiple parts, each linked via PART_OF edges.
    pub fn split_memory(
        &self,
        source_id: &str,
        parts: &[SplitPart],
    ) -> Result<Vec<String>, CodememError> {
        let source_memory = self
            .storage
            .get_memory(source_id)?
            .ok_or_else(|| CodememError::NotFound(format!("Memory not found: {source_id}")))?;

        if parts.is_empty() {
            return Err(CodememError::InvalidInput(
                "'parts' array must not be empty".to_string(),
            ));
        }

        // Validate all parts upfront before persisting anything
        for part in parts {
            if part.content.is_empty() {
                return Err(CodememError::InvalidInput(
                    "Each part must have a non-empty 'content' field".to_string(),
                ));
            }
        }

        let now = chrono::Utc::now();
        let mut child_ids: Vec<String> = Vec::new();

        for part in parts {
            let tags = part
                .tags
                .clone()
                .unwrap_or_else(|| source_memory.tags.clone());
            let importance = part.importance.unwrap_or(source_memory.importance);

            let child_id = uuid::Uuid::new_v4().to_string();
            let hash = codemem_storage::Storage::content_hash(&part.content);

            let memory = MemoryNode {
                id: child_id.clone(),
                content: part.content.clone(),
                memory_type: source_memory.memory_type,
                importance,
                confidence: source_memory.confidence,
                access_count: 0,
                content_hash: hash,
                tags,
                metadata: std::collections::HashMap::new(),
                namespace: source_memory.namespace.clone(),
                created_at: now,
                updated_at: now,
                last_accessed_at: now,
            };

            if let Err(e) = self.persist_memory_no_save(&memory) {
                // Clean up already-created child memories
                for created_id in &child_ids {
                    if let Err(del_err) = self.delete_memory(created_id) {
                        tracing::warn!(
                            "Failed to clean up child memory {created_id} after split failure: {del_err}"
                        );
                    }
                }
                return Err(e);
            }

            // Create PART_OF edge: child -> source
            let edge = Edge {
                id: format!("{child_id}-PART_OF-{source_id}"),
                src: child_id.clone(),
                dst: source_id.to_string(),
                relationship: RelationshipType::PartOf,
                weight: 1.0,
                properties: std::collections::HashMap::new(),
                created_at: now,
                valid_from: Some(now),
                valid_to: None,
            };
            if let Err(e) = self.add_edge(edge) {
                tracing::warn!("Failed to add PART_OF edge: {e}");
            }

            child_ids.push(child_id);
        }

        self.save_index();
        Ok(child_ids)
    }

    /// Merge multiple memories into one, linked via SUMMARIZES edges.
    pub fn merge_memories(
        &self,
        source_ids: &[String],
        content: &str,
        memory_type: MemoryType,
        importance: f64,
        tags: Vec<String>,
    ) -> Result<String, CodememError> {
        if source_ids.len() < 2 {
            return Err(CodememError::InvalidInput(
                "'source_ids' must contain at least 2 IDs".to_string(),
            ));
        }

        // Verify all sources exist
        let id_refs: Vec<&str> = source_ids.iter().map(|s| s.as_str()).collect();
        let found = self.storage.get_memories_batch(&id_refs)?;
        if found.len() != source_ids.len() {
            let found_ids: std::collections::HashSet<&str> =
                found.iter().map(|m| m.id.as_str()).collect();
            let missing: Vec<&str> = id_refs
                .iter()
                .filter(|id| !found_ids.contains(**id))
                .copied()
                .collect();
            return Err(CodememError::NotFound(format!(
                "Source memories not found: {}",
                missing.join(", ")
            )));
        }

        let now = chrono::Utc::now();
        let merged_id = uuid::Uuid::new_v4().to_string();
        let hash = codemem_storage::Storage::content_hash(content);

        let memory = MemoryNode {
            id: merged_id.clone(),
            content: content.to_string(),
            memory_type,
            importance,
            confidence: found.iter().map(|m| m.confidence).sum::<f64>() / found.len() as f64,
            access_count: 0,
            content_hash: hash,
            tags,
            metadata: std::collections::HashMap::new(),
            namespace: found.iter().find_map(|m| m.namespace.clone()),
            created_at: now,
            updated_at: now,
            last_accessed_at: now,
        };

        self.persist_memory_no_save(&memory)?;

        // Create SUMMARIZES edges: merged -> each source
        for source_id in source_ids {
            let edge = Edge {
                id: format!("{merged_id}-SUMMARIZES-{source_id}"),
                src: merged_id.clone(),
                dst: source_id.clone(),
                relationship: RelationshipType::Summarizes,
                weight: 1.0,
                properties: std::collections::HashMap::new(),
                created_at: now,
                valid_from: Some(now),
                valid_to: None,
            };
            if let Err(e) = self.add_edge(edge) {
                tracing::warn!("Failed to add SUMMARIZES edge to {source_id}: {e}");
            }
        }

        self.save_index();
        Ok(merged_id)
    }

    /// Update a memory's content and/or importance, re-embedding if needed.
    pub fn update_memory(
        &self,
        id: &str,
        content: &str,
        importance: Option<f64>,
    ) -> Result<(), CodememError> {
        self.storage.update_memory(id, content, importance)?;

        // Update BM25 index
        self.lock_bm25()?.add_document(id, content);

        // Update graph node label
        if let Ok(mut graph) = self.lock_graph() {
            if let Ok(Some(mut node)) = graph.get_node(id) {
                node.label = scoring::truncate_content(content, 80);
                if let Err(e) = graph.add_node(node) {
                    tracing::warn!("Failed to update graph node for {id}: {e}");
                }
            }
        }

        // Re-embed with contextual enrichment
        // H3: Acquire embeddings lock, embed, drop lock before acquiring vector lock.
        if let Some(emb_guard) = self.lock_embeddings()? {
            let (mem_type, tags, namespace) =
                if let Ok(Some(mem)) = self.storage.get_memory_no_touch(id) {
                    (mem.memory_type, mem.tags, mem.namespace)
                } else {
                    (MemoryType::Context, vec![], None)
                };
            let enriched =
                self.enrich_memory_text(content, mem_type, &tags, namespace.as_deref(), Some(id));
            let emb_result = emb_guard.embed(&enriched);
            drop(emb_guard);
            if let Ok(embedding) = emb_result {
                if let Err(e) = self.storage.store_embedding(id, &embedding) {
                    tracing::warn!("Failed to store embedding for {id}: {e}");
                }
                let mut vec = self.lock_vector()?;
                if let Err(e) = vec.remove(id) {
                    tracing::warn!("Failed to remove old vector for {id}: {e}");
                }
                if let Err(e) = vec.insert(id, &embedding) {
                    tracing::warn!("Failed to insert new vector for {id}: {e}");
                }
            }
        }

        self.save_index();
        Ok(())
    }

    /// Delete a memory from all subsystems.
    ///
    /// M1: Uses `delete_memory_cascade` on the storage backend to wrap all
    /// SQLite deletes (memory + graph nodes/edges + embedding) in a single
    /// transaction when the backend supports it. In-memory structures
    /// (vector, graph, BM25) are cleaned up separately with proper lock ordering.
    pub fn delete_memory(&self, id: &str) -> Result<bool, CodememError> {
        // Use cascade delete for all storage-side operations in a single transaction.
        let deleted = self.storage.delete_memory_cascade(id)?;
        if !deleted {
            return Ok(false);
        }

        // Clean up in-memory structures with proper lock ordering:
        // vector first, then graph, then BM25.
        let mut vec = self.lock_vector()?;
        if let Err(e) = vec.remove(id) {
            tracing::warn!("Failed to remove {id} from vector index: {e}");
        }
        drop(vec);

        let mut graph = self.lock_graph()?;
        if let Err(e) = graph.remove_node(id) {
            tracing::warn!("Failed to remove {id} from in-memory graph: {e}");
        }
        drop(graph);

        self.lock_bm25()?.remove_document(id);

        // Persist vector index to disk
        self.save_index();
        Ok(true)
    }

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
        let new_graph = GraphEngine::from_storage(&*self.storage)?;
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
        event: &watch::WatchEvent,
        namespace: Option<&str>,
        project_root: Option<&Path>,
    ) -> Result<(), CodememError> {
        match event {
            watch::WatchEvent::FileChanged(path) | watch::WatchEvent::FileCreated(path) => {
                self.index_single_file(path, namespace, project_root)?;
            }
            watch::WatchEvent::FileDeleted(path) => {
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

    /// Compare files on disk vs file nodes in the graph and clean up stale entries.
    /// Call this after indexing or on watcher delete events.
    pub fn cleanup_deleted_files(&self, dir_path: &str) -> Result<usize, CodememError> {
        let dir = Path::new(dir_path);
        if !dir.is_dir() {
            return Ok(0);
        }

        // Collect file: nodes from the graph
        let graph = self.lock_graph()?;
        let file_nodes: Vec<String> = graph
            .get_all_nodes()
            .into_iter()
            .filter(|n| n.kind == NodeKind::File && n.label.starts_with(dir_path))
            .map(|n| n.label.clone())
            .collect();
        drop(graph);

        let mut cleaned = 0usize;
        for file_path in &file_nodes {
            if !Path::new(file_path).exists() {
                self.cleanup_file_nodes(file_path)?;
                cleaned += 1;
            }
        }

        if cleaned > 0 {
            self.lock_graph()?.recompute_centrality();
        }

        Ok(cleaned)
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
