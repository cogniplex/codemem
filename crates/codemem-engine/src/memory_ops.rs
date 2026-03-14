use crate::scoring;
use crate::CodememEngine;
use crate::SplitPart;
use codemem_core::{CodememError, Edge, MemoryNode, MemoryType, RelationshipType};
use std::collections::HashMap;
use std::sync::atomic::Ordering;

impl CodememEngine {
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
        // Auto-populate session_id from the engine's active session if not already set
        let memory = if memory.session_id.is_none() {
            if let Some(active_sid) = self.active_session_id() {
                let mut m = memory.clone();
                m.session_id = Some(active_sid);
                std::borrow::Cow::Owned(m)
            } else {
                std::borrow::Cow::Borrowed(memory)
            }
        } else {
            std::borrow::Cow::Borrowed(memory)
        };

        // Auto-set expires_at for session memories if not explicitly set
        let memory = if memory.expires_at.is_none() && memory.session_id.is_some() {
            let ttl_hours = self.config.memory.default_session_ttl_hours;
            if ttl_hours > 0 {
                let mut m = memory.into_owned();
                m.expires_at = Some(chrono::Utc::now() + chrono::Duration::hours(ttl_hours as i64));
                std::borrow::Cow::Owned(m)
            } else {
                memory
            }
        } else {
            memory
        };

        // Auto-populate repo/git_ref from engine scope if not already set
        let memory = if memory.repo.is_none() {
            if let Some(scope) = self.scope() {
                let mut m = memory.into_owned();
                m.repo = Some(scope.repo.clone());
                m.git_ref = Some(scope.git_ref.clone());
                std::borrow::Cow::Owned(m)
            } else {
                memory
            }
        } else {
            memory
        };
        let memory = memory.as_ref();

        // H3: Step 1 — Embed if the provider is already loaded (don't trigger lazy init).
        // Lifecycle hooks skip embedding for speed; the provider gets initialized on
        // first recall/search, and backfill_embeddings() picks up any gaps.
        let embedding_result = if self.embeddings_ready() {
            match self.lock_embeddings() {
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
            }
        } else {
            None
        };

        // 2F: Wrap all SQLite mutations in a single transaction so that the
        // database cannot be left in an inconsistent state if one step fails.
        // The HNSW vector index is NOT in SQLite, so vector insertion happens
        // after commit — if it fails, the memory is still persisted without
        // its embedding, which is recoverable.
        self.storage.begin_transaction()?;

        let result = self.persist_memory_sqlite(memory, &embedding_result);

        match result {
            Ok(()) => {
                self.storage.commit_transaction()?;
            }
            Err(e) => {
                if let Err(rb_err) = self.storage.rollback_transaction() {
                    tracing::error!("Failed to rollback transaction after persist error: {rb_err}");
                }
                return Err(e);
            }
        }

        // 2. Update BM25 index if already loaded (don't trigger lazy init).
        // The BM25 index rebuilds from all memories on first access anyway.
        if self.bm25_ready() {
            match self.lock_bm25() {
                Ok(mut bm25) => {
                    bm25.add_document(&memory.id, &memory.content);
                }
                Err(e) => tracing::warn!("BM25 lock failed during persist: {e}"),
            }
        }

        // 3. Add memory node to in-memory graph (already persisted to SQLite above)
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
                if let Err(e) = graph.add_node(node) {
                    tracing::warn!(
                        "Failed to add graph node in-memory for memory {}: {e}",
                        memory.id
                    );
                }
            }
            Err(e) => tracing::warn!("Graph lock failed during persist: {e}"),
        }

        // 3b. Auto-link to memories with shared tags (session co-membership, topic overlap)
        self.auto_link_by_tags(memory);

        // H3: Step 4 — Insert embedding into HNSW vector index if already loaded.
        if let Some(vec) = &embedding_result {
            if self.vector_ready() {
                if let Ok(mut vi) = self.lock_vector() {
                    if let Err(e) = vi.insert(&memory.id, vec) {
                        tracing::warn!("Failed to insert into vector index for {}: {e}", memory.id);
                    }
                }
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

    /// Execute all SQLite mutations for a memory persist.
    ///
    /// Called within the transaction opened by `persist_memory_inner`.
    /// Inserts the memory row, graph node, and embedding (if available).
    fn persist_memory_sqlite(
        &self,
        memory: &MemoryNode,
        embedding: &Option<Vec<f32>>,
    ) -> Result<(), CodememError> {
        // 1. Store memory in SQLite
        self.storage.insert_memory(memory)?;

        // 2. Insert graph node in SQLite
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

        // 3. Store embedding in SQLite (vector blob, not HNSW index)
        if let Some(vec) = embedding {
            if let Err(e) = self.storage.store_embedding(&memory.id, vec) {
                tracing::warn!("Failed to store embedding for {}: {e}", memory.id);
            }
        }

        Ok(())
    }

    // ── Store with Links ──────────────────────────────────────────────────

    /// Store a memory with optional explicit link IDs.
    ///
    /// Runs the full pipeline: persist → explicit RELATES_TO edges → auto-link
    /// to code nodes → save index. This consolidates domain logic that was
    /// previously spread across the MCP transport layer.
    pub fn store_memory_with_links(
        &self,
        memory: &MemoryNode,
        links: &[String],
    ) -> Result<(), CodememError> {
        self.persist_memory(memory)?;

        // Create RELATES_TO edges for explicit links
        if !links.is_empty() {
            let now = chrono::Utc::now();
            let mut graph = self.lock_graph()?;
            for link_id in links {
                let edge = Edge {
                    id: format!("{}-RELATES_TO-{link_id}", memory.id),
                    src: memory.id.clone(),
                    dst: link_id.clone(),
                    relationship: RelationshipType::RelatesTo,
                    weight: 1.0,
                    properties: HashMap::new(),
                    created_at: now,
                    valid_from: None,
                    valid_to: None,
                };
                if let Err(e) = self.storage.insert_graph_edge(&edge) {
                    tracing::warn!("Failed to persist link edge to {link_id}: {e}");
                }
                if let Err(e) = graph.add_edge(edge) {
                    tracing::warn!("Failed to add link edge to {link_id}: {e}");
                }
            }
        }

        // Auto-link to code nodes mentioned in content
        self.auto_link_to_code_nodes(&memory.id, &memory.content, links);

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

        let mut memory = MemoryNode::new(new_content, old_memory.memory_type);
        let new_id = memory.id.clone();
        memory.importance = new_importance;
        memory.confidence = old_memory.confidence;
        memory.tags = new_tags;
        memory.metadata = old_memory.metadata.clone();
        memory.namespace = old_memory.namespace.clone();

        self.persist_memory(&memory)?;

        // Create EVOLVED_INTO edge from old -> new
        let now = chrono::Utc::now();
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

            let mut memory = MemoryNode::new(part.content.clone(), source_memory.memory_type);
            let child_id = memory.id.clone();
            memory.importance = importance;
            memory.confidence = source_memory.confidence;
            memory.tags = tags;
            memory.namespace = source_memory.namespace.clone();

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

        let mut memory = MemoryNode::new(content, memory_type);
        let merged_id = memory.id.clone();
        memory.importance = importance;
        memory.confidence = found.iter().map(|m| m.confidence).sum::<f64>() / found.len() as f64;
        memory.tags = tags;
        memory.namespace = found.iter().find_map(|m| m.namespace.clone());

        self.persist_memory_no_save(&memory)?;

        // Create SUMMARIZES edges: merged -> each source
        let now = chrono::Utc::now();
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

    /// Update only the importance of a memory.
    /// Routes through the engine to maintain the transport → engine → storage boundary.
    pub fn update_importance(&self, id: &str, importance: f64) -> Result<(), CodememError> {
        self.storage
            .batch_update_importance(&[(id.to_string(), importance)])?;
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

    /// Opportunistic sweep of expired memories. Rate-limited to once per 60 seconds
    /// to avoid adding overhead to every recall call.
    pub(crate) fn sweep_expired_memories(&self) {
        let now = chrono::Utc::now().timestamp();
        let last = self.last_expiry_sweep.load(Ordering::Relaxed);
        if now - last < 60 {
            return;
        }
        // CAS to avoid concurrent sweeps
        if self
            .last_expiry_sweep
            .compare_exchange(last, now, Ordering::Relaxed, Ordering::Relaxed)
            .is_err()
        {
            return;
        }
        match self.storage.delete_expired_memories() {
            Ok(0) => {}
            Ok(n) => tracing::debug!("Swept {n} expired memories"),
            Err(e) => tracing::warn!("Expired memory sweep failed: {e}"),
        }
    }
}
