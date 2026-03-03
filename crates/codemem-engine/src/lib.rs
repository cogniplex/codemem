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
        self.persist_memory_inner(memory, true)
    }

    /// Persist a memory without saving the vector index to disk.
    /// Use this in batch operations, then call `save_index()` once at the end.
    fn persist_memory_no_save(&self, memory: &MemoryNode) -> Result<(), CodememError> {
        self.persist_memory_inner(memory, false)
    }

    /// Inner persist implementation with optional index save.
    fn persist_memory_inner(&self, memory: &MemoryNode, save: bool) -> Result<(), CodememError> {
        // 1. Store in SQLite
        self.storage.insert_memory(memory)?;

        // 2. Update BM25 index
        match self.lock_bm25() {
            Ok(mut bm25) => {
                bm25.add_document(&memory.id, &memory.content);
            }
            Err(e) => tracing::warn!("BM25 lock failed during persist: {e}"),
        }

        // 3. Add memory node to graph
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
                let _ = self.storage.insert_graph_node(&node);
                let _ = graph.add_node(node);
            }
            Err(e) => tracing::warn!("Graph lock failed during persist: {e}"),
        }

        // 4. Embed and store in vector index
        match self.lock_embeddings() {
            Ok(Some(emb)) => {
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
            Ok(None) => {} // No embeddings provider configured
            Err(e) => tracing::warn!("Embeddings lock failed during persist: {e}"),
        }

        // 5. Save vector index to disk (if requested)
        if save {
            self.save_index();
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

        self.save_index();

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
                let _ = graph.add_node(node);
            }
        }

        // Re-embed with contextual enrichment
        if let Some(emb_guard) = self.lock_embeddings()? {
            let (mem_type, tags, namespace) = if let Ok(Some(mem)) = self.storage.get_memory(id) {
                (mem.memory_type, mem.tags, mem.namespace)
            } else {
                (MemoryType::Context, vec![], None)
            };
            let enriched =
                self.enrich_memory_text(content, mem_type, &tags, namespace.as_deref(), Some(id));
            let emb_result = emb_guard.embed(&enriched);
            drop(emb_guard);
            if let Ok(embedding) = emb_result {
                let _ = self.storage.store_embedding(id, &embedding);
                let mut vec = self.lock_vector()?;
                let _ = vec.remove(id);
                let _ = vec.insert(id, &embedding);
            }
        }

        self.save_index();
        Ok(())
    }

    /// Delete a memory from all subsystems.
    pub fn delete_memory(&self, id: &str) -> Result<bool, CodememError> {
        match self.storage.delete_memory(id)? {
            true => {
                // Remove from vector index
                let mut vec = self.lock_vector()?;
                let _ = vec.remove(id);
                drop(vec);
                // Remove from in-memory graph
                let mut graph = self.lock_graph()?;
                let _ = graph.remove_node(id);
                drop(graph);
                // Remove graph node and edges from SQLite
                let _ = self.storage.delete_graph_edges_for_node(id);
                let _ = self.storage.delete_graph_node(id);
                // Remove embedding from SQLite
                let _ = self.storage.delete_embedding(id);
                // Remove from BM25 index
                self.lock_bm25()?.remove_document(id);
                // Persist vector index to disk
                self.save_index();
                Ok(true)
            }
            false => Ok(false),
        }
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
