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
    CodememConfig, CodememError, Edge, GraphBackend, MemoryNode, MemoryType, RelationshipType,
    ScoringWeights, StorageBackend, VectorBackend,
};
use codemem_storage::graph::GraphEngine;
use codemem_storage::HnswIndex;
use codemem_storage::Storage;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};

pub mod bm25;
pub mod compress;
pub mod hooks;
pub mod index;
pub mod metrics;
pub mod patterns;
pub mod scoring;
pub mod watch;

// Re-export key index types at crate root for convenience
pub use index::{
    ChunkConfig, CodeChunk, CodeParser, Dependency, IndexAndResolveResult, IndexProgress,
    IndexResult, Indexer, ManifestResult, ParseResult, Reference, ReferenceKind, ReferenceResolver,
    ResolvedEdge, Symbol, SymbolKind, Visibility, Workspace,
};

// Re-export key domain types for convenience
pub use bm25::Bm25Index;
pub use metrics::InMemoryMetrics;

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
pub struct CodememEngine {
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
        }
    }

    /// Create an engine from a database path, loading all backends.
    pub fn from_db_path(db_path: &Path) -> Result<Self, CodememError> {
        let storage = Storage::open(db_path)?;
        let mut vector = HnswIndex::with_defaults()?;

        // Load existing vector index if it exists
        let index_path = db_path.with_extension("idx");
        if index_path.exists() {
            vector.load(&index_path)?;
        }

        // Load graph from storage
        let graph = GraphEngine::from_storage(&storage)?;

        // Try loading embeddings (optional)
        let embeddings = codemem_embeddings::from_env().ok();

        let mut engine = Self::new(Box::new(storage), vector, graph, embeddings);
        engine.db_path = Some(db_path.to_path_buf());

        // Recompute centrality metrics on startup
        engine.lock_graph()?.recompute_centrality();

        // Populate BM25 index from all existing memories
        if let Ok(ids) = engine.storage.list_memory_ids() {
            let mut bm25 = engine.lock_bm25()?;
            for id in &ids {
                if let Ok(Some(memory)) = engine.storage.get_memory(id) {
                    bm25.add_document(id, &memory.content);
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

    // ── Contextual Enrichment ────────────────────────────────────────────────

    /// Build contextual text for a memory node.
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

        let body = if chunk.text.len() > 4000 {
            &chunk.text[..4000]
        } else {
            &chunk.text
        };

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

    // ── Persistence ─────────────────────────────────────────────────────

    /// Persist a memory through the full pipeline: storage → BM25 → graph → embedding → vector.
    pub fn persist_memory(&self, memory: &MemoryNode) -> Result<(), CodememError> {
        // 1. Store in SQLite
        self.storage.insert_memory(memory)?;

        // 2. Update BM25 index
        if let Ok(mut bm25) = self.lock_bm25() {
            bm25.add_document(&memory.id, &memory.content);
        }

        // 3. Add memory node to graph
        if let Ok(mut graph) = self.lock_graph() {
            let node = codemem_core::GraphNode {
                id: memory.id.clone(),
                kind: codemem_core::NodeKind::Memory,
                label: scoring::truncate_content(&memory.content, 80),
                payload: std::collections::HashMap::new(),
                centrality: 0.0,
                memory_id: None,
                namespace: memory.namespace.clone(),
            };
            let _ = graph.add_node(node.clone());
            let _ = self.storage.insert_graph_node(&node);
        }

        // 4. Embed and store in vector index
        if let Ok(Some(emb)) = self.lock_embeddings() {
            let enriched = self.enrich_memory_text(
                &memory.content,
                memory.memory_type,
                &memory.tags,
                memory.namespace.as_deref(),
                Some(&memory.id),
            );
            if let Ok(vec) = emb.embed(&enriched) {
                if let Ok(mut vi) = self.lock_vector() {
                    let _ = vi.insert(&memory.id, &vec);
                }
                let _ = self.storage.store_embedding(&memory.id, &vec);
            }
        }

        // 5. Auto-link to code nodes
        self.auto_link_to_code_nodes(&memory.id, &memory.content, &[]);

        // 6. Save vector index to disk
        self.save_index();

        Ok(())
    }

    // ── Index Persistence ────────────────────────────────────────────────

    /// Save the vector index to disk if a db_path is configured.
    pub fn save_index(&self) {
        if let Some(ref db_path) = self.db_path {
            let idx_path = db_path.with_extension("idx");
            if let Ok(vi) = self.lock_vector() {
                if let Err(e) = vi.save(&idx_path) {
                    tracing::warn!("Failed to save vector index: {e}");
                }
            }
        }
    }

    /// Reload the in-memory graph from the database.
    pub fn reload_graph(&self) -> Result<(), CodememError> {
        let new_graph = GraphEngine::from_storage(&*self.storage)?;
        let mut graph = self.lock_graph()?;
        *graph = new_graph;
        graph.recompute_centrality();
        Ok(())
    }
}
