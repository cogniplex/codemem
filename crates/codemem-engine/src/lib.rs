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
    CodememConfig, CodememError, GraphBackend, ScoringWeights, StorageBackend, VectorBackend,
};
pub use codemem_storage::graph::GraphEngine;
pub use codemem_storage::HnswIndex;
pub use codemem_storage::Storage;
use std::path::{Path, PathBuf};
#[cfg(test)]
use std::sync::atomic::Ordering;
use std::sync::atomic::{AtomicBool, AtomicI64};
use std::sync::{Arc, Mutex, OnceLock, RwLock};

pub mod analysis;
pub mod bm25;
pub mod compress;
pub mod consolidation;
pub mod enrichment;
mod enrichment_text;
mod file_indexing;
mod graph_linking;
pub mod graph_ops;
pub mod hooks;
pub mod index;
pub mod insights;
mod memory_ops;
pub mod metrics;
pub mod patterns;
pub mod pca;
pub mod persistence;
pub mod recall;
pub mod review;
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

#[cfg(test)]
#[path = "tests/memory_expiry_tests.rs"]
mod memory_expiry_tests;

#[cfg(test)]
#[path = "tests/scope_tests.rs"]
mod scope_tests;

// Re-export key index types at crate root for convenience
pub use index::{
    ChunkConfig, CodeChunk, CodeParser, Dependency, IndexAndResolveResult, IndexProgress,
    IndexResult, Indexer, ManifestResult, ParseResult, Reference, ReferenceKind, ReferenceResolver,
    ResolvedEdge, Symbol, SymbolKind, Visibility, Workspace,
};

// Re-export key domain types for convenience
pub use bm25::Bm25Index;
pub use metrics::InMemoryMetrics;
pub use review::{BlastRadiusReport, DiffSymbolMapping};

// Re-export enrichment types
pub use enrichment::{EnrichResult, EnrichmentPipelineResult};

// Re-export persistence types
pub use persistence::{edge_weight_for, CrossRepoPersistResult, IndexPersistResult};

// Re-export recall types
pub use recall::{ExpandedResult, NamespaceStats, RecallQuery};

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
/// **Trait-object backends**: `CodememEngine` uses `Box<dyn Trait>` for all three
/// backends (storage, vector, graph). This enables pluggable backends (Postgres,
/// Qdrant, Neo4j) at the cost of vtable indirection. The default build uses
/// SQLite + usearch HNSW + petgraph, and the vtable overhead is negligible
/// compared to I/O latency.
pub struct CodememEngine {
    pub(crate) storage: Box<dyn StorageBackend>,
    /// Lazily initialized vector index. Loaded on first `lock_vector()` call.
    pub(crate) vector: OnceLock<Mutex<Box<dyn VectorBackend>>>,
    pub(crate) graph: Mutex<Box<dyn GraphBackend>>,
    /// Lazily initialized embedding provider. Loaded on first `lock_embeddings()` call.
    pub(crate) embeddings: OnceLock<Option<Mutex<Box<dyn codemem_embeddings::EmbeddingProvider>>>>,
    /// Path to the database file, used to derive the index save path.
    pub(crate) db_path: Option<PathBuf>,
    /// Cached index results for structural queries.
    pub(crate) index_cache: Mutex<Option<IndexCache>>,
    /// Configurable scoring weights for the 9-component hybrid scoring system.
    pub(crate) scoring_weights: RwLock<ScoringWeights>,
    /// Lazily initialized BM25 index. Loaded on first `lock_bm25()` call.
    pub(crate) bm25_index: OnceLock<Mutex<Bm25Index>>,
    /// Loaded configuration.
    pub(crate) config: CodememConfig,
    /// Operational metrics collector.
    pub(crate) metrics: Arc<InMemoryMetrics>,
    /// Dirty flag for batch saves: set after `persist_memory_no_save()`,
    /// cleared by `save_index()`.
    dirty: AtomicBool,
    /// Active session ID for auto-populating `session_id` on persisted memories.
    active_session_id: RwLock<Option<String>>,
    /// Active scope context for repo/branch/user-aware operations.
    scope: RwLock<Option<codemem_core::ScopeContext>>,
    /// Cached change detector for incremental single-file indexing.
    /// Loaded lazily from storage on first use.
    change_detector: Mutex<Option<index::incremental::ChangeDetector>>,
    /// Unix timestamp of the last expired-memory sweep. Used to rate-limit
    /// opportunistic cleanup to at most once per 60 seconds.
    last_expiry_sweep: AtomicI64,
}

impl CodememEngine {
    /// Create an engine with storage, vector, graph, and optional embeddings backends.
    pub fn new(
        storage: Box<dyn StorageBackend>,
        vector: Box<dyn VectorBackend>,
        graph: Box<dyn GraphBackend>,
        embeddings: Option<Box<dyn codemem_embeddings::EmbeddingProvider>>,
    ) -> Self {
        let config = CodememConfig::load_or_default();
        Self::new_with_config(storage, vector, graph, embeddings, config)
    }

    /// Create an engine with an explicit config (avoids double-loading from disk).
    pub fn new_with_config(
        storage: Box<dyn StorageBackend>,
        vector: Box<dyn VectorBackend>,
        graph: Box<dyn GraphBackend>,
        embeddings: Option<Box<dyn codemem_embeddings::EmbeddingProvider>>,
        config: CodememConfig,
    ) -> Self {
        let vector_lock = OnceLock::new();
        let _ = vector_lock.set(Mutex::new(vector));
        let embeddings_lock = OnceLock::new();
        let _ = embeddings_lock.set(embeddings.map(Mutex::new));
        let bm25_lock = OnceLock::new();
        let _ = bm25_lock.set(Mutex::new(Bm25Index::new()));
        Self {
            storage,
            vector: vector_lock,
            graph: Mutex::new(graph),
            embeddings: embeddings_lock,
            db_path: None,
            index_cache: Mutex::new(None),
            scoring_weights: RwLock::new(config.scoring.clone()),
            bm25_index: bm25_lock,
            config,
            metrics: Arc::new(InMemoryMetrics::new()),
            dirty: AtomicBool::new(false),
            active_session_id: RwLock::new(None),
            scope: RwLock::new(None),
            change_detector: Mutex::new(None),
            last_expiry_sweep: AtomicI64::new(0),
        }
    }

    /// Create an engine from a database path.
    ///
    /// Only loads SQLite storage and the in-memory graph eagerly. The vector index,
    /// BM25 index, and embedding provider are lazily initialized on first access
    /// via `lock_vector()`, `lock_bm25()`, and `lock_embeddings()`. This makes
    /// lightweight callers (lifecycle hooks) fast (~200ms) while full operations
    /// (recall, search, analyze) pay the init cost once on first use.
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

        // Validate backend config — only built-in backends are supported without
        // feature-flagged crates (codemem-postgres, codemem-qdrant, codemem-neo4j).
        if !config.storage.backend.eq_ignore_ascii_case("sqlite") {
            return Err(CodememError::Config(format!(
                "Unsupported storage backend '{}'. Only 'sqlite' is available in this build.",
                config.storage.backend
            )));
        }
        if !config.vector.backend.eq_ignore_ascii_case("hnsw") {
            return Err(CodememError::Config(format!(
                "Unsupported vector backend '{}'. Only 'hnsw' is available in this build.",
                config.vector.backend
            )));
        }
        if !config.graph.backend.eq_ignore_ascii_case("petgraph") {
            return Err(CodememError::Config(format!(
                "Unsupported graph backend '{}'. Only 'petgraph' is available in this build.",
                config.graph.backend
            )));
        }

        // Wire StorageConfig into Storage::open
        let storage = Storage::open_with_config(
            db_path,
            Some(config.storage.cache_size_mb),
            Some(config.storage.busy_timeout_secs),
        )?;

        // Load graph from storage (needed for centrality and graph queries)
        let graph = GraphEngine::from_storage(&storage)?;

        let engine = Self {
            storage: Box::new(storage),
            vector: OnceLock::new(),
            graph: Mutex::new(Box::new(graph)),
            embeddings: OnceLock::new(),
            db_path: Some(db_path.to_path_buf()),
            index_cache: Mutex::new(None),
            scoring_weights: RwLock::new(config.scoring.clone()),
            bm25_index: OnceLock::new(),
            config,
            metrics: Arc::new(InMemoryMetrics::new()),
            dirty: AtomicBool::new(false),
            active_session_id: RwLock::new(None),
            scope: RwLock::new(None),
            change_detector: Mutex::new(None),
            last_expiry_sweep: AtomicI64::new(0),
        };

        // H7: Only compute PageRank at startup; betweenness is computed lazily
        // via `ensure_betweenness_computed()` when first needed.
        engine
            .lock_graph()?
            .recompute_centrality_with_options(false);

        Ok(engine)
    }

    /// Create a minimal engine for testing.
    pub fn for_testing() -> Self {
        let storage = Storage::open_in_memory().unwrap();
        let graph = GraphEngine::new();
        let config = CodememConfig::default();
        let vector_lock = OnceLock::new();
        let _ = vector_lock.set(Mutex::new(
            Box::new(HnswIndex::with_defaults().unwrap()) as Box<dyn VectorBackend>
        ));
        let embeddings_lock = OnceLock::new();
        let _ = embeddings_lock.set(None);
        let bm25_lock = OnceLock::new();
        let _ = bm25_lock.set(Mutex::new(Bm25Index::new()));
        Self {
            storage: Box::new(storage),
            vector: vector_lock,
            graph: Mutex::new(Box::new(graph)),
            embeddings: embeddings_lock,
            db_path: None,
            index_cache: Mutex::new(None),
            scoring_weights: RwLock::new(config.scoring.clone()),
            bm25_index: bm25_lock,
            config,
            metrics: Arc::new(InMemoryMetrics::new()),
            dirty: AtomicBool::new(false),
            active_session_id: RwLock::new(None),
            scope: RwLock::new(None),
            change_detector: Mutex::new(None),
            last_expiry_sweep: AtomicI64::new(0),
        }
    }

    // ── Lock Helpers ─────────────────────────────────────────────────────────

    pub fn lock_vector(
        &self,
    ) -> Result<std::sync::MutexGuard<'_, Box<dyn VectorBackend>>, CodememError> {
        self.vector
            .get_or_init(|| self.init_vector())
            .lock()
            .map_err(|e| CodememError::LockPoisoned(format!("vector: {e}")))
    }

    pub fn lock_graph(
        &self,
    ) -> Result<std::sync::MutexGuard<'_, Box<dyn GraphBackend>>, CodememError> {
        self.graph
            .lock()
            .map_err(|e| CodememError::LockPoisoned(format!("graph: {e}")))
    }

    pub fn lock_bm25(&self) -> Result<std::sync::MutexGuard<'_, Bm25Index>, CodememError> {
        self.bm25_index
            .get_or_init(|| self.init_bm25())
            .lock()
            .map_err(|e| CodememError::LockPoisoned(format!("bm25: {e}")))
    }

    /// Lock the embedding provider, lazily initializing it on first access.
    ///
    /// Returns `Ok(None)` if no provider is configured (e.g. `from_env()` fails).
    pub fn lock_embeddings(
        &self,
    ) -> Result<
        Option<std::sync::MutexGuard<'_, Box<dyn codemem_embeddings::EmbeddingProvider>>>,
        CodememError,
    > {
        match self.embeddings.get_or_init(|| self.init_embeddings()) {
            Some(m) => Ok(Some(m.lock().map_err(|e| {
                CodememError::LockPoisoned(format!("embeddings: {e}"))
            })?)),
            None => Ok(None),
        }
    }

    /// Check if embeddings are already initialized (without triggering lazy init).
    fn embeddings_ready(&self) -> bool {
        self.embeddings.get().is_some_and(|opt| opt.is_some())
    }

    /// Check if the vector index is already initialized (without triggering lazy init).
    fn vector_ready(&self) -> bool {
        self.vector.get().is_some()
    }

    /// Check if the BM25 index is already initialized (without triggering lazy init).
    fn bm25_ready(&self) -> bool {
        self.bm25_index.get().is_some()
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

    // ── Lazy Initialization ────────────────────────────────────────────

    /// Initialize the HNSW vector index: load from disk, run consistency check.
    fn init_vector(&self) -> Mutex<Box<dyn VectorBackend>> {
        let vector_config = self.config.vector.clone();
        let mut vector = HnswIndex::new(vector_config.clone())
            .unwrap_or_else(|_| HnswIndex::with_defaults().expect("default vector index"));

        if let Some(ref db_path) = self.db_path {
            let index_path = db_path.with_extension("idx");
            if index_path.exists() {
                if let Err(e) = vector.load(&index_path) {
                    tracing::warn!("Stale or corrupt vector index, will rebuild: {e}");
                }
            }

            // C6: Consistency check — rebuild if count mismatches DB embedding count.
            let vector_count = vector.stats().count;
            if let Ok(db_stats) = self.storage.stats() {
                let db_embed_count = db_stats.embedding_count;
                if vector_count != db_embed_count {
                    tracing::warn!(
                        "Vector index ({vector_count}) out of sync with DB ({db_embed_count}), rebuilding..."
                    );
                    if let Ok(mut fresh) = HnswIndex::new(vector_config) {
                        if let Ok(embeddings) = self.storage.list_all_embeddings() {
                            for (id, emb) in &embeddings {
                                if let Err(e) = fresh.insert(id, emb) {
                                    tracing::warn!("Failed to re-insert embedding {id}: {e}");
                                }
                            }
                        }
                        vector = fresh;
                        if let Err(e) = vector.save(&index_path) {
                            tracing::warn!("Failed to save rebuilt vector index: {e}");
                        }
                    }
                }
            }
        }

        Mutex::new(Box::new(vector))
    }

    /// Initialize the BM25 index: load from disk or rebuild from memories.
    fn init_bm25(&self) -> Mutex<Bm25Index> {
        let mut bm25 = Bm25Index::new();

        if let Some(ref db_path) = self.db_path {
            let bm25_path = db_path.with_extension("bm25");
            let mut loaded = false;
            if bm25_path.exists() {
                if let Ok(data) = std::fs::read(&bm25_path) {
                    if let Ok(index) = Bm25Index::deserialize(&data) {
                        tracing::info!(
                            "Loaded BM25 index from disk ({} documents)",
                            index.doc_count
                        );
                        bm25 = index;
                        loaded = true;
                    }
                }
            }
            if !loaded {
                if let Ok(ids) = self.storage.list_memory_ids() {
                    let id_refs: Vec<&str> = ids.iter().map(|s| s.as_str()).collect();
                    if let Ok(memories) = self.storage.get_memories_batch(&id_refs) {
                        for m in &memories {
                            bm25.add_document(&m.id, &m.content);
                        }
                        tracing::info!("Rebuilt BM25 index from {} memories", bm25.doc_count);
                    }
                }
            }
        }

        Mutex::new(bm25)
    }

    /// Initialize the embedding provider from environment/config.
    ///
    /// Also backfills embeddings for any memories that were stored without them
    /// (e.g. by lifecycle hooks that skipped embedding for speed).
    fn init_embeddings(&self) -> Option<Mutex<Box<dyn codemem_embeddings::EmbeddingProvider>>> {
        let provider = match codemem_embeddings::from_env(Some(&self.config.embedding)) {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!("Failed to initialize embedding provider: {e}");
                return None;
            }
        };

        // Backfill un-embedded memories (from hooks that skipped embedding)
        self.backfill_embeddings(&*provider);

        Some(Mutex::new(provider))
    }

    /// Embed any memories that lack embeddings in SQLite.
    ///
    /// This runs during lazy init of the embedding provider to pick up memories
    /// stored by lightweight hooks without embedding.
    fn backfill_embeddings(&self, provider: &dyn codemem_embeddings::EmbeddingProvider) {
        let ids = match self.storage.list_memory_ids() {
            Ok(ids) => ids,
            Err(_) => return,
        };

        let mut to_embed: Vec<(String, String)> = Vec::new();
        for id in &ids {
            if self.storage.get_embedding(id).ok().flatten().is_none() {
                if let Ok(Some(mem)) = self.storage.get_memory_no_touch(id) {
                    let text = self.enrich_memory_text(
                        &mem.content,
                        mem.memory_type,
                        &mem.tags,
                        mem.namespace.as_deref(),
                        Some(&mem.id),
                    );
                    to_embed.push((id.clone(), text));
                }
            }
        }

        if to_embed.is_empty() {
            return;
        }

        tracing::info!("Backfilling {} un-embedded memories", to_embed.len());
        let text_refs: Vec<&str> = to_embed.iter().map(|(_, t)| t.as_str()).collect();
        match provider.embed_batch(&text_refs) {
            Ok(embeddings) => {
                for ((id, _), emb) in to_embed.iter().zip(embeddings.iter()) {
                    let _ = self.storage.store_embedding(id, emb);
                    // Insert into vector index if already loaded
                    if let Some(vi_mutex) = self.vector.get() {
                        if let Ok(mut vi) = vi_mutex.lock().map_err(|e| {
                            tracing::warn!("Vector lock failed during backfill: {e}");
                            e
                        }) {
                            let _ = vi.insert(id, emb);
                        }
                    }
                }
                tracing::info!("Backfilled {} embeddings", to_embed.len());
            }
            Err(e) => tracing::warn!("Backfill embedding failed: {e}"),
        }
    }

    // ── Active Session ───────────────────────────────────────────────────

    /// Set the active session ID for auto-populating `session_id` on persisted memories.
    pub fn set_active_session(&self, id: Option<String>) {
        match self.active_session_id.write() {
            Ok(mut guard) => *guard = id,
            Err(e) => *e.into_inner() = id,
        }
    }

    /// Get the current active session ID.
    pub fn active_session_id(&self) -> Option<String> {
        match self.active_session_id.read() {
            Ok(guard) => guard.clone(),
            Err(e) => e.into_inner().clone(),
        }
    }

    // ── Scope Context ─────────────────────────────────────────────────────

    /// Set the active scope context for repo/branch/user-aware operations.
    pub fn set_scope(&self, scope: Option<codemem_core::ScopeContext>) {
        match self.scope.write() {
            Ok(mut guard) => *guard = scope,
            Err(e) => *e.into_inner() = scope,
        }
    }

    /// Get the current scope context.
    pub fn scope(&self) -> Option<codemem_core::ScopeContext> {
        match self.scope.read() {
            Ok(guard) => guard.clone(),
            Err(e) => e.into_inner().clone(),
        }
    }

    /// Derive namespace from the active scope, falling back to None.
    pub fn scope_namespace(&self) -> Option<String> {
        self.scope().map(|s| s.namespace().to_string())
    }

    // ── Public Accessors ──────────────────────────────────────────────────

    /// Access the storage backend.
    pub fn storage(&self) -> &dyn StorageBackend {
        &*self.storage
    }

    /// Whether an embedding provider is configured.
    ///
    /// Returns `true` if embeddings are already loaded, or if the config suggests
    /// a provider is available (without triggering lazy init).
    pub fn has_embeddings(&self) -> bool {
        match self.embeddings.get() {
            Some(opt) => opt.is_some(),
            None => !self.config.embedding.provider.is_empty(),
        }
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

    // ── Closure Accessors (safe read-only access for transport layers) ──

    /// Execute a closure with a locked reference to the graph engine.
    /// Provides safe read-only access without exposing raw mutexes.
    pub fn with_graph<F, R>(&self, f: F) -> Result<R, CodememError>
    where
        F: FnOnce(&dyn GraphBackend) -> R,
    {
        let guard = self.lock_graph()?;
        Ok(f(&**guard))
    }

    /// Execute a closure with a locked reference to the vector index.
    /// Provides safe read-only access without exposing raw mutexes.
    pub fn with_vector<F, R>(&self, f: F) -> Result<R, CodememError>
    where
        F: FnOnce(&dyn VectorBackend) -> R,
    {
        let guard = self.lock_vector()?;
        Ok(f(&**guard))
    }

    /// Check if the engine has unsaved changes (dirty flag is set).
    #[cfg(test)]
    pub(crate) fn is_dirty(&self) -> bool {
        self.dirty.load(Ordering::Acquire)
    }

    // ── Repository Management (delegates to storage) ─────────────────

    /// List all registered repositories.
    pub fn list_repos(&self) -> Result<Vec<codemem_core::Repository>, CodememError> {
        self.storage.list_repos()
    }

    /// Add a new repository.
    pub fn add_repo(&self, repo: &codemem_core::Repository) -> Result<(), CodememError> {
        self.storage.add_repo(repo)
    }

    /// Get a repository by ID.
    pub fn get_repo(&self, id: &str) -> Result<Option<codemem_core::Repository>, CodememError> {
        self.storage.get_repo(id)
    }

    /// Remove a repository by ID.
    pub fn remove_repo(&self, id: &str) -> Result<bool, CodememError> {
        self.storage.remove_repo(id)
    }

    /// Update a repository's status and optionally its last-indexed timestamp.
    pub fn update_repo_status(
        &self,
        id: &str,
        status: &str,
        indexed_at: Option<&str>,
    ) -> Result<(), CodememError> {
        self.storage.update_repo_status(id, status, indexed_at)
    }
}

// Re-export types from file_indexing at crate root for API compatibility
pub use file_indexing::{AnalyzeOptions, AnalyzeProgress, AnalyzeResult, SessionContext};

// Re-export embeddings types so downstream crates need not depend on codemem-embeddings directly.
/// Create an embedding provider from environment configuration.
pub use codemem_embeddings::from_env as embeddings_from_env;
pub use codemem_embeddings::{EmbeddingProvider, EmbeddingService};
