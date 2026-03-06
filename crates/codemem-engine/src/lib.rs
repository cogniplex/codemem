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

use codemem_core::{CodememConfig, CodememError, ScoringWeights, StorageBackend, VectorBackend};
use codemem_storage::graph::GraphEngine;
use codemem_storage::HnswIndex;
use codemem_storage::Storage;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
#[cfg(test)]
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, RwLock};

pub mod analysis;
pub mod bm25;
pub mod compress;
pub mod consolidation;
pub mod enrichment;
mod enrichment_text;
mod file_indexing;
mod graph_linking;
pub mod hooks;
pub mod index;
mod memory_ops;
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

#[cfg(test)]
#[path = "tests/recall_tests.rs"]
mod recall_tests;

#[cfg(test)]
#[path = "tests/search_tests.rs"]
mod search_tests;

#[cfg(test)]
#[path = "tests/consolidation_tests.rs"]
mod consolidation_tests;

#[cfg(test)]
#[path = "tests/analysis_tests.rs"]
mod analysis_tests;

#[cfg(test)]
#[path = "tests/persistence_tests.rs"]
mod persistence_tests;

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
pub use enrichment::{EnrichResult, EnrichmentPipelineResult};

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
    pub(crate) storage: Box<dyn StorageBackend>,
    pub(crate) vector: Mutex<HnswIndex>,
    pub(crate) graph: Mutex<GraphEngine>,
    /// Optional embedding provider (None if not configured).
    pub(crate) embeddings: Option<Mutex<Box<dyn codemem_embeddings::EmbeddingProvider>>>,
    /// Path to the database file, used to derive the index save path.
    pub(crate) db_path: Option<PathBuf>,
    /// Cached index results for structural queries.
    pub(crate) index_cache: Mutex<Option<IndexCache>>,
    /// Configurable scoring weights for the 9-component hybrid scoring system.
    pub(crate) scoring_weights: RwLock<ScoringWeights>,
    /// BM25 index for code-aware token overlap scoring.
    pub(crate) bm25_index: Mutex<Bm25Index>,
    /// Loaded configuration.
    pub(crate) config: CodememConfig,
    /// Operational metrics collector.
    pub(crate) metrics: Arc<InMemoryMetrics>,
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
        // Ensure parent directory exists (e.g. ~/.codemem/)
        if let Some(parent) = db_path.parent() {
            if !parent.exists() {
                std::fs::create_dir_all(parent).map_err(|e| {
                    CodememError::Storage(format!(
                        "Failed to create database directory {}: {e}",
                        parent.display()
                    ))
                })?;
            }
        }

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

    // ── Public Accessors ──────────────────────────────────────────────────

    /// Access the storage backend.
    pub fn storage(&self) -> &dyn StorageBackend {
        &*self.storage
    }

    /// Whether an embedding provider is configured.
    pub fn has_embeddings(&self) -> bool {
        self.embeddings.is_some()
    }

    /// Access the database path (if backed by a file).
    pub fn db_path(&self) -> Option<&Path> {
        self.db_path.as_deref()
    }

    /// Access the loaded configuration.
    pub fn config(&self) -> &CodememConfig {
        &self.config
    }

    /// Access the metrics collector.
    pub fn metrics(&self) -> &Arc<InMemoryMetrics> {
        &self.metrics
    }

    /// Access the raw graph Mutex (for callers that need `&Mutex<GraphEngine>`).
    pub fn graph_mutex(&self) -> &Mutex<GraphEngine> {
        &self.graph
    }

    /// Access the raw vector Mutex (for callers that need `&Mutex<HnswIndex>`).
    pub fn vector_mutex(&self) -> &Mutex<HnswIndex> {
        &self.vector
    }

    /// Access the raw BM25 Mutex (for callers that need `&Mutex<Bm25Index>`).
    pub fn bm25_mutex(&self) -> &Mutex<Bm25Index> {
        &self.bm25_index
    }

    /// Access the raw embeddings Mutex (for callers that need the `Option<&Mutex<...>>`).
    pub fn embeddings_mutex(
        &self,
    ) -> Option<&Mutex<Box<dyn codemem_embeddings::EmbeddingProvider>>> {
        self.embeddings.as_ref()
    }

    /// Check if the engine has unsaved changes (dirty flag is set).
    #[cfg(test)]
    pub(crate) fn is_dirty(&self) -> bool {
        self.dirty.load(Ordering::Acquire)
    }
}

// Re-export types from file_indexing at crate root for API compatibility
pub use file_indexing::{IndexEnrichResult, SessionContext};
