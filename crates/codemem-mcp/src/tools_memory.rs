//! Memory CRUD tools: store, recall, update, delete, associate.

use crate::scoring::{compute_score, format_recall_results, truncate_str};
use crate::types::ToolResult;
use crate::McpServer;
use codemem_core::{
    CodememError, Edge, GraphBackend, GraphNode, MemoryNode, MemoryType, NodeKind,
    RelationshipType, SearchResult, VectorBackend,
};
use codemem_storage::Storage;
use serde_json::{json, Value};
use std::collections::HashMap;

impl McpServer {
    pub(crate) fn tool_store_memory(&self, args: &Value) -> ToolResult {
        let content = match args.get("content").and_then(|v| v.as_str()) {
            Some(c) if !c.is_empty() => c,
            _ => return ToolResult::tool_error("Missing or empty 'content' parameter"),
        };

        let memory_type: MemoryType = args
            .get("memory_type")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse().ok())
            .unwrap_or(MemoryType::Context);

        let importance = args
            .get("importance")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.5);

        let tags: Vec<String> = args
            .get("tags")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        let now = chrono::Utc::now();
        let id = uuid::Uuid::new_v4().to_string();
        let hash = Storage::content_hash(content);

        let namespace = args
            .get("namespace")
            .and_then(|v| v.as_str())
            .map(String::from);

        let memory = MemoryNode {
            id: id.clone(),
            content: content.to_string(),
            memory_type,
            importance,
            confidence: 1.0,
            access_count: 0,
            content_hash: hash,
            tags,
            metadata: HashMap::new(),
            namespace,
            created_at: now,
            updated_at: now,
            last_accessed_at: now,
        };

        // Insert into storage
        match self.storage.insert_memory(&memory) {
            Ok(()) => {}
            Err(CodememError::Duplicate(h)) => {
                return ToolResult::text(format!("Memory already exists (hash: {h})"));
            }
            Err(e) => return ToolResult::tool_error(format!("Storage error: {e}")),
        }

        // Update BM25 index
        match self.lock_bm25() {
            Ok(mut bm25) => bm25.add_document(&id, content),
            Err(e) => return ToolResult::tool_error(format!("Lock error: {e}")),
        }

        // Create graph node for the memory (before embedding so graph context is available)
        let graph_node = GraphNode {
            id: id.clone(),
            kind: NodeKind::Memory,
            label: truncate_str(content, 80),
            payload: HashMap::new(),
            centrality: 0.0,
            memory_id: Some(id.clone()),
            namespace: None,
        };
        // Persist to SQLite (needed for FK constraints on graph_edges)
        if let Err(e) = self.storage.insert_graph_node(&graph_node) {
            tracing::warn!("Failed to persist graph node: {e}");
        }
        match self.lock_graph() {
            Ok(mut graph) => {
                if let Err(e) = graph.add_node(graph_node) {
                    tracing::warn!("Failed to add graph node: {e}");
                }
            }
            Err(e) => return ToolResult::tool_error(format!("Lock error: {e}")),
        }

        // Handle optional `links` parameter: create RELATES_TO edges to linked nodes
        if let Some(links) = args.get("links").and_then(|v| v.as_array()) {
            let mut graph = match self.lock_graph() {
                Ok(g) => g,
                Err(e) => return ToolResult::tool_error(format!("Lock error: {e}")),
            };
            for link_val in links {
                if let Some(link_id) = link_val.as_str() {
                    let edge = Edge {
                        id: format!("{id}-RELATES_TO-{link_id}"),
                        src: id.clone(),
                        dst: link_id.to_string(),
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
        }

        // Auto-link memory to code nodes mentioned in content
        let explicit_links: Vec<String> = args
            .get("links")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();
        self.auto_link_to_code_nodes(&id, content, &explicit_links);

        // Generate contextual embedding and insert into vector index
        // (after graph node + links so enrichment can reference them)
        if let Some(emb_guard) = match self.lock_embeddings() {
            Ok(g) => g,
            Err(e) => return ToolResult::tool_error(format!("Lock error: {e}")),
        } {
            let enriched = self.enrich_memory_text(
                content,
                memory_type,
                &memory.tags,
                memory.namespace.as_deref(),
                Some(&id),
            );
            match emb_guard.embed(&enriched) {
                Ok(embedding) => {
                    drop(emb_guard);
                    if let Err(e) = self.storage.store_embedding(&id, &embedding) {
                        tracing::warn!("Failed to store embedding: {e}");
                    }
                    match self.lock_vector() {
                        Ok(mut vec) => {
                            if let Err(e) = vec.insert(&id, &embedding) {
                                tracing::warn!("Failed to index vector: {e}");
                            }
                        }
                        Err(e) => return ToolResult::tool_error(format!("Lock error: {e}")),
                    }
                }
                Err(e) => {
                    tracing::warn!("Embedding failed: {e}");
                }
            }
        }

        // Persist vector index to disk
        self.save_index();

        ToolResult::text(
            serde_json::to_string_pretty(&json!({
                "id": id,
                "memory_type": memory_type.to_string(),
                "importance": importance,
                "embedded": self.embeddings.is_some(),
            }))
            .expect("JSON serialization of literal"),
        )
    }

    pub(crate) fn tool_recall_memory(&self, args: &Value) -> ToolResult {
        let query = match args.get("query").and_then(|v| v.as_str()) {
            Some(q) if !q.is_empty() => q,
            _ => return ToolResult::tool_error("Missing or empty 'query' parameter"),
        };

        let k = args.get("k").and_then(|v| v.as_u64()).unwrap_or(10) as usize;

        // Parse optional memory_type filter
        let memory_type_filter: Option<MemoryType> = args
            .get("memory_type")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse().ok());

        // Parse optional namespace filter
        let namespace_filter: Option<&str> = args.get("namespace").and_then(|v| v.as_str());

        // Parse optional quality filters
        let exclude_tags: Vec<String> = args
            .get("exclude_tags")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();
        let min_importance: Option<f64> = args.get("min_importance").and_then(|v| v.as_f64());
        let min_confidence: Option<f64> = args.get("min_confidence").and_then(|v| v.as_f64());

        self.recall_memories(
            query,
            k,
            &memory_type_filter,
            namespace_filter,
            &exclude_tags,
            min_importance,
            min_confidence,
        )
    }

    /// Search the server's storage with optional type, namespace, and quality filters.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn recall_memories(
        &self,
        query: &str,
        k: usize,
        memory_type_filter: &Option<MemoryType>,
        namespace_filter: Option<&str>,
        exclude_tags: &[String],
        min_importance: Option<f64>,
        min_confidence: Option<f64>,
    ) -> ToolResult {
        // Try vector search first (if embeddings available)
        let vector_results: Vec<(String, f32)> = if let Some(emb_guard) =
            match self.lock_embeddings() {
                Ok(g) => g,
                Err(e) => return ToolResult::tool_error(format!("Lock error: {e}")),
            } {
            match emb_guard.embed(query) {
                Ok(query_embedding) => {
                    drop(emb_guard);
                    let vec = match self.lock_vector() {
                        Ok(v) => v,
                        Err(e) => return ToolResult::tool_error(format!("Lock error: {e}")),
                    };
                    vec.search(&query_embedding, k * 2) // over-fetch for re-ranking
                        .unwrap_or_default()
                }
                Err(e) => {
                    tracing::warn!("Query embedding failed: {e}");
                    vec![]
                }
            }
        } else {
            vec![]
        };

        let query_lower = query.to_lowercase();
        let query_tokens: Vec<&str> = query_lower.split_whitespace().collect();

        let graph = match self.lock_graph() {
            Ok(g) => g,
            Err(e) => return ToolResult::tool_error(format!("Lock error: {e}")),
        };
        let bm25 = match self.lock_bm25() {
            Ok(b) => b,
            Err(e) => return ToolResult::tool_error(format!("Lock error: {e}")),
        };

        // Build scored results
        let mut results: Vec<SearchResult> = Vec::new();

        if vector_results.is_empty() {
            // Fallback: text search over all memories
            let ids = match self.storage.list_memory_ids() {
                Ok(ids) => ids,
                Err(e) => return ToolResult::tool_error(format!("Storage error: {e}")),
            };

            for id in &ids {
                if let Ok(Some(memory)) = self.storage.get_memory(id) {
                    // Apply memory_type filter
                    if let Some(ref filter_type) = memory_type_filter {
                        if memory.memory_type != *filter_type {
                            continue;
                        }
                    }
                    // Apply namespace filter
                    if let Some(ns) = namespace_filter {
                        if memory.namespace.as_deref() != Some(ns) {
                            continue;
                        }
                    }
                    // Apply quality filters
                    if !exclude_tags.is_empty()
                        && memory.tags.iter().any(|t| exclude_tags.contains(t))
                    {
                        continue;
                    }
                    if let Some(min) = min_importance {
                        if memory.importance < min {
                            continue;
                        }
                    }
                    if let Some(min) = min_confidence {
                        if memory.confidence < min {
                            continue;
                        }
                    }

                    let breakdown =
                        compute_score(&memory, query, &query_tokens, 0.0, &graph, &bm25);
                    let weights = match self.scoring_weights() {
                        Ok(w) => w,
                        Err(e) => return ToolResult::tool_error(format!("Lock error: {e}")),
                    };
                    let score = breakdown.total_with_weights(&weights);
                    drop(weights);
                    if score > 0.01 {
                        results.push(SearchResult {
                            memory,
                            score,
                            score_breakdown: breakdown,
                        });
                    }
                }
            }
        } else {
            // Vector search + hybrid scoring
            for (id, distance) in &vector_results {
                if let Ok(Some(memory)) = self.storage.get_memory(id) {
                    // Apply memory_type filter
                    if let Some(ref filter_type) = memory_type_filter {
                        if memory.memory_type != *filter_type {
                            continue;
                        }
                    }
                    // Apply namespace filter
                    if let Some(ns) = namespace_filter {
                        if memory.namespace.as_deref() != Some(ns) {
                            continue;
                        }
                    }
                    // Apply quality filters
                    if !exclude_tags.is_empty()
                        && memory.tags.iter().any(|t| exclude_tags.contains(t))
                    {
                        continue;
                    }
                    if let Some(min) = min_importance {
                        if memory.importance < min {
                            continue;
                        }
                    }
                    if let Some(min) = min_confidence {
                        if memory.confidence < min {
                            continue;
                        }
                    }

                    // Convert cosine distance to similarity (1.0 - distance for cosine)
                    let similarity = 1.0 - (*distance as f64);
                    let breakdown =
                        compute_score(&memory, query, &query_tokens, similarity, &graph, &bm25);
                    let weights = match self.scoring_weights() {
                        Ok(w) => w,
                        Err(e) => return ToolResult::tool_error(format!("Lock error: {e}")),
                    };
                    let score = breakdown.total_with_weights(&weights);
                    drop(weights);
                    results.push(SearchResult {
                        memory,
                        score,
                        score_breakdown: breakdown,
                    });
                }
            }
        }

        // Sort by score descending, take top k
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(k);

        format_recall_results(&results, None)
    }

    pub(crate) fn tool_update_memory(&self, args: &Value) -> ToolResult {
        let id = match args.get("id").and_then(|v| v.as_str()) {
            Some(id) => id,
            None => return ToolResult::tool_error("Missing 'id' parameter"),
        };
        let content = match args.get("content").and_then(|v| v.as_str()) {
            Some(c) => c,
            None => return ToolResult::tool_error("Missing 'content' parameter"),
        };
        let importance = args.get("importance").and_then(|v| v.as_f64());

        if let Err(e) = self.storage.update_memory(id, content, importance) {
            return ToolResult::tool_error(format!("Update failed: {e}"));
        }

        // Update BM25 index with new content
        match self.lock_bm25() {
            Ok(mut bm25) => bm25.add_document(id, content),
            Err(e) => return ToolResult::tool_error(format!("Lock error: {e}")),
        }

        // Re-embed with contextual enrichment
        if let Some(emb_guard) = match self.lock_embeddings() {
            Ok(g) => g,
            Err(e) => return ToolResult::tool_error(format!("Lock error: {e}")),
        } {
            // Fetch the updated memory to get its metadata for enrichment
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
                match self.lock_vector() {
                    Ok(mut vec) => {
                        let _ = vec.remove(id);
                        let _ = vec.insert(id, &embedding);
                    }
                    Err(e) => return ToolResult::tool_error(format!("Lock error: {e}")),
                }
            }
        }

        // Persist vector index to disk
        self.save_index();

        ToolResult::text(json!({"id": id, "updated": true}).to_string())
    }

    pub(crate) fn tool_delete_memory(&self, args: &Value) -> ToolResult {
        let id = match args.get("id").and_then(|v| v.as_str()) {
            Some(id) => id,
            None => return ToolResult::tool_error("Missing 'id' parameter"),
        };

        match self.storage.delete_memory(id) {
            Ok(true) => {
                // Remove from vector index
                match self.lock_vector() {
                    Ok(mut vec) => {
                        let _ = vec.remove(id);
                    }
                    Err(e) => return ToolResult::tool_error(format!("Lock error: {e}")),
                }
                // Remove from in-memory graph
                match self.lock_graph() {
                    Ok(mut graph) => {
                        let _ = graph.remove_node(id);
                    }
                    Err(e) => return ToolResult::tool_error(format!("Lock error: {e}")),
                }
                // Remove graph node and edges from SQLite
                let _ = self.storage.delete_graph_edges_for_node(id);
                let _ = self.storage.delete_graph_node(id);
                // Remove embedding from SQLite
                let _ = self.storage.delete_embedding(id);
                // Remove from BM25 index
                match self.lock_bm25() {
                    Ok(mut bm25) => bm25.remove_document(id),
                    Err(e) => return ToolResult::tool_error(format!("Lock error: {e}")),
                }
                // Persist vector index to disk
                self.save_index();
                ToolResult::text(json!({"id": id, "deleted": true}).to_string())
            }
            Ok(false) => ToolResult::tool_error(format!("Memory not found: {id}")),
            Err(e) => ToolResult::tool_error(format!("Delete failed: {e}")),
        }
    }

    pub(crate) fn tool_refine_memory(&self, args: &Value) -> ToolResult {
        let old_id = match args.get("id").and_then(|v| v.as_str()) {
            Some(id) => id,
            None => return ToolResult::tool_error("Missing 'id' parameter"),
        };

        // Fetch old memory
        let old_memory = match self.storage.get_memory(old_id) {
            Ok(Some(m)) => m,
            Ok(None) => return ToolResult::tool_error(format!("Memory not found: {old_id}")),
            Err(e) => return ToolResult::tool_error(format!("Storage error: {e}")),
        };

        // Build new memory inheriting unchanged fields from old
        let content = args
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or(&old_memory.content);

        let tags: Vec<String> = args
            .get("tags")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_else(|| old_memory.tags.clone());

        let importance = args
            .get("importance")
            .and_then(|v| v.as_f64())
            .unwrap_or(old_memory.importance);

        let now = chrono::Utc::now();
        let new_id = uuid::Uuid::new_v4().to_string();
        let hash = Storage::content_hash(content);

        let memory = MemoryNode {
            id: new_id.clone(),
            content: content.to_string(),
            memory_type: old_memory.memory_type,
            importance,
            confidence: old_memory.confidence,
            access_count: 0,
            content_hash: hash,
            tags,
            metadata: HashMap::new(),
            namespace: old_memory.namespace.clone(),
            created_at: now,
            updated_at: now,
            last_accessed_at: now,
        };

        // Insert into storage
        match self.storage.insert_memory(&memory) {
            Ok(()) => {}
            Err(CodememError::Duplicate(h)) => {
                return ToolResult::text(format!("Memory already exists (hash: {h})"));
            }
            Err(e) => return ToolResult::tool_error(format!("Storage error: {e}")),
        }

        // Update BM25 index
        match self.lock_bm25() {
            Ok(mut bm25) => bm25.add_document(&new_id, content),
            Err(e) => return ToolResult::tool_error(format!("Lock error: {e}")),
        }

        // Create graph node
        let graph_node = GraphNode {
            id: new_id.clone(),
            kind: NodeKind::Memory,
            label: truncate_str(content, 80),
            payload: HashMap::new(),
            centrality: 0.0,
            memory_id: Some(new_id.clone()),
            namespace: None,
        };
        if let Err(e) = self.storage.insert_graph_node(&graph_node) {
            tracing::warn!("Failed to persist graph node: {e}");
        }
        match self.lock_graph() {
            Ok(mut graph) => {
                if let Err(e) = graph.add_node(graph_node) {
                    tracing::warn!("Failed to add graph node: {e}");
                }
            }
            Err(e) => return ToolResult::tool_error(format!("Lock error: {e}")),
        }

        // Generate contextual embedding and insert into vector index
        if let Some(emb_guard) = match self.lock_embeddings() {
            Ok(g) => g,
            Err(e) => return ToolResult::tool_error(format!("Lock error: {e}")),
        } {
            let enriched = self.enrich_memory_text(
                content,
                memory.memory_type,
                &memory.tags,
                memory.namespace.as_deref(),
                Some(&new_id),
            );
            match emb_guard.embed(&enriched) {
                Ok(embedding) => {
                    drop(emb_guard);
                    if let Err(e) = self.storage.store_embedding(&new_id, &embedding) {
                        tracing::warn!("Failed to store embedding: {e}");
                    }
                    match self.lock_vector() {
                        Ok(mut vec) => {
                            if let Err(e) = vec.insert(&new_id, &embedding) {
                                tracing::warn!("Failed to index vector: {e}");
                            }
                        }
                        Err(e) => return ToolResult::tool_error(format!("Lock error: {e}")),
                    }
                }
                Err(e) => {
                    tracing::warn!("Embedding failed: {e}");
                }
            }
        }

        // Create EVOLVED_INTO edge from old → new
        let edge = Edge {
            id: format!("{old_id}-EVOLVED_INTO-{new_id}"),
            src: old_id.to_string(),
            dst: new_id.clone(),
            relationship: RelationshipType::EvolvedInto,
            weight: 1.0,
            properties: HashMap::new(),
            created_at: now,
            valid_from: Some(now),
            valid_to: None,
        };
        if let Err(e) = self.storage.insert_graph_edge(&edge) {
            tracing::warn!("Failed to persist EVOLVED_INTO edge: {e}");
        }
        match self.lock_graph() {
            Ok(mut graph) => {
                if let Err(e) = graph.add_edge(edge) {
                    tracing::warn!("Failed to add EVOLVED_INTO edge: {e}");
                }
            }
            Err(e) => return ToolResult::tool_error(format!("Lock error: {e}")),
        }

        // Persist vector index to disk
        self.save_index();

        ToolResult::text(
            serde_json::to_string_pretty(&json!({
                "old_id": old_id,
                "new_id": new_id,
                "relationship": "EVOLVED_INTO",
            }))
            .expect("JSON serialization of literal"),
        )
    }

    pub(crate) fn tool_split_memory(&self, args: &Value) -> ToolResult {
        let source_id = match args.get("id").and_then(|v| v.as_str()) {
            Some(id) => id,
            None => return ToolResult::tool_error("Missing 'id' parameter"),
        };

        // Verify source memory exists
        let source_memory = match self.storage.get_memory(source_id) {
            Ok(Some(m)) => m,
            Ok(None) => return ToolResult::tool_error(format!("Memory not found: {source_id}")),
            Err(e) => return ToolResult::tool_error(format!("Storage error: {e}")),
        };

        let parts = match args.get("parts").and_then(|v| v.as_array()) {
            Some(arr) if !arr.is_empty() => arr,
            Some(_) => return ToolResult::tool_error("'parts' array must not be empty"),
            None => return ToolResult::tool_error("Missing 'parts' parameter"),
        };

        let now = chrono::Utc::now();
        let mut child_ids: Vec<String> = Vec::new();

        for part in parts {
            let content = match part.get("content").and_then(|v| v.as_str()) {
                Some(c) if !c.is_empty() => c,
                _ => {
                    return ToolResult::tool_error(
                        "Each part must have a non-empty 'content' field",
                    )
                }
            };

            let tags: Vec<String> = part
                .get("tags")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_else(|| source_memory.tags.clone());

            let importance = part
                .get("importance")
                .and_then(|v| v.as_f64())
                .unwrap_or(source_memory.importance);

            let child_id = uuid::Uuid::new_v4().to_string();
            let hash = Storage::content_hash(content);

            let memory = MemoryNode {
                id: child_id.clone(),
                content: content.to_string(),
                memory_type: source_memory.memory_type,
                importance,
                confidence: source_memory.confidence,
                access_count: 0,
                content_hash: hash,
                tags,
                metadata: HashMap::new(),
                namespace: source_memory.namespace.clone(),
                created_at: now,
                updated_at: now,
                last_accessed_at: now,
            };

            // Insert into storage
            match self.storage.insert_memory(&memory) {
                Ok(()) => {}
                Err(CodememError::Duplicate(h)) => {
                    return ToolResult::text(format!("Memory already exists (hash: {h})"));
                }
                Err(e) => return ToolResult::tool_error(format!("Storage error: {e}")),
            }

            // Update BM25 index
            match self.lock_bm25() {
                Ok(mut bm25) => bm25.add_document(&child_id, content),
                Err(e) => return ToolResult::tool_error(format!("Lock error: {e}")),
            }

            // Create graph node
            let graph_node = GraphNode {
                id: child_id.clone(),
                kind: NodeKind::Memory,
                label: truncate_str(content, 80),
                payload: HashMap::new(),
                centrality: 0.0,
                memory_id: Some(child_id.clone()),
                namespace: None,
            };
            if let Err(e) = self.storage.insert_graph_node(&graph_node) {
                tracing::warn!("Failed to persist graph node: {e}");
            }
            match self.lock_graph() {
                Ok(mut graph) => {
                    if let Err(e) = graph.add_node(graph_node) {
                        tracing::warn!("Failed to add graph node: {e}");
                    }
                }
                Err(e) => return ToolResult::tool_error(format!("Lock error: {e}")),
            }

            // Generate contextual embedding and insert into vector index
            if let Some(emb_guard) = match self.lock_embeddings() {
                Ok(g) => g,
                Err(e) => return ToolResult::tool_error(format!("Lock error: {e}")),
            } {
                let enriched = self.enrich_memory_text(
                    content,
                    memory.memory_type,
                    &memory.tags,
                    memory.namespace.as_deref(),
                    Some(&child_id),
                );
                match emb_guard.embed(&enriched) {
                    Ok(embedding) => {
                        drop(emb_guard);
                        if let Err(e) = self.storage.store_embedding(&child_id, &embedding) {
                            tracing::warn!("Failed to store embedding: {e}");
                        }
                        match self.lock_vector() {
                            Ok(mut vec) => {
                                if let Err(e) = vec.insert(&child_id, &embedding) {
                                    tracing::warn!("Failed to index vector: {e}");
                                }
                            }
                            Err(e) => return ToolResult::tool_error(format!("Lock error: {e}")),
                        }
                    }
                    Err(e) => {
                        tracing::warn!("Embedding failed: {e}");
                    }
                }
            }

            // Create PART_OF edge: child → source
            let edge = Edge {
                id: format!("{child_id}-PART_OF-{source_id}"),
                src: child_id.clone(),
                dst: source_id.to_string(),
                relationship: RelationshipType::PartOf,
                weight: 1.0,
                properties: HashMap::new(),
                created_at: now,
                valid_from: Some(now),
                valid_to: None,
            };
            if let Err(e) = self.storage.insert_graph_edge(&edge) {
                tracing::warn!("Failed to persist PART_OF edge: {e}");
            }
            match self.lock_graph() {
                Ok(mut graph) => {
                    if let Err(e) = graph.add_edge(edge) {
                        tracing::warn!("Failed to add PART_OF edge: {e}");
                    }
                }
                Err(e) => return ToolResult::tool_error(format!("Lock error: {e}")),
            }

            child_ids.push(child_id);
        }

        // Persist vector index to disk
        self.save_index();

        ToolResult::text(
            serde_json::to_string_pretty(&json!({
                "source_id": source_id,
                "parts": child_ids,
                "relationship": "PART_OF",
            }))
            .expect("JSON serialization of literal"),
        )
    }

    pub(crate) fn tool_merge_memories(&self, args: &Value) -> ToolResult {
        let source_ids: Vec<String> = match args.get("source_ids").and_then(|v| v.as_array()) {
            Some(arr) => arr
                .iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect(),
            None => return ToolResult::tool_error("Missing 'source_ids' parameter"),
        };

        if source_ids.len() < 2 {
            return ToolResult::tool_error("'source_ids' must contain at least 2 IDs");
        }

        // Verify all sources exist
        let id_refs: Vec<&str> = source_ids.iter().map(|s| s.as_str()).collect();
        let found = match self.storage.get_memories_batch(&id_refs) {
            Ok(m) => m,
            Err(e) => return ToolResult::tool_error(format!("Storage error: {e}")),
        };
        if found.len() != source_ids.len() {
            let found_ids: Vec<&str> = found.iter().map(|m| m.id.as_str()).collect();
            let missing: Vec<&str> = id_refs
                .iter()
                .filter(|id| !found_ids.contains(id))
                .copied()
                .collect();
            return ToolResult::tool_error(format!(
                "Source memories not found: {}",
                missing.join(", ")
            ));
        }

        let content = match args.get("content").and_then(|v| v.as_str()) {
            Some(c) if !c.is_empty() => c,
            _ => return ToolResult::tool_error("Missing or empty 'content' parameter"),
        };

        let memory_type: MemoryType = args
            .get("memory_type")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse().ok())
            .unwrap_or(MemoryType::Insight);

        let importance = args
            .get("importance")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.7);

        let tags: Vec<String> = args
            .get("tags")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        let now = chrono::Utc::now();
        let merged_id = uuid::Uuid::new_v4().to_string();
        let hash = Storage::content_hash(content);

        let memory = MemoryNode {
            id: merged_id.clone(),
            content: content.to_string(),
            memory_type,
            importance,
            confidence: 1.0,
            access_count: 0,
            content_hash: hash,
            tags,
            metadata: HashMap::new(),
            namespace: None,
            created_at: now,
            updated_at: now,
            last_accessed_at: now,
        };

        // Insert into storage
        match self.storage.insert_memory(&memory) {
            Ok(()) => {}
            Err(CodememError::Duplicate(h)) => {
                return ToolResult::text(format!("Memory already exists (hash: {h})"));
            }
            Err(e) => return ToolResult::tool_error(format!("Storage error: {e}")),
        }

        // Update BM25 index
        match self.lock_bm25() {
            Ok(mut bm25) => bm25.add_document(&merged_id, content),
            Err(e) => return ToolResult::tool_error(format!("Lock error: {e}")),
        }

        // Create graph node
        let graph_node = GraphNode {
            id: merged_id.clone(),
            kind: NodeKind::Memory,
            label: truncate_str(content, 80),
            payload: HashMap::new(),
            centrality: 0.0,
            memory_id: Some(merged_id.clone()),
            namespace: None,
        };
        if let Err(e) = self.storage.insert_graph_node(&graph_node) {
            tracing::warn!("Failed to persist graph node: {e}");
        }
        match self.lock_graph() {
            Ok(mut graph) => {
                if let Err(e) = graph.add_node(graph_node) {
                    tracing::warn!("Failed to add graph node: {e}");
                }
            }
            Err(e) => return ToolResult::tool_error(format!("Lock error: {e}")),
        }

        // Generate contextual embedding and insert into vector index
        if let Some(emb_guard) = match self.lock_embeddings() {
            Ok(g) => g,
            Err(e) => return ToolResult::tool_error(format!("Lock error: {e}")),
        } {
            let enriched = self.enrich_memory_text(
                content,
                memory.memory_type,
                &memory.tags,
                memory.namespace.as_deref(),
                Some(&merged_id),
            );
            match emb_guard.embed(&enriched) {
                Ok(embedding) => {
                    drop(emb_guard);
                    if let Err(e) = self.storage.store_embedding(&merged_id, &embedding) {
                        tracing::warn!("Failed to store embedding: {e}");
                    }
                    match self.lock_vector() {
                        Ok(mut vec) => {
                            if let Err(e) = vec.insert(&merged_id, &embedding) {
                                tracing::warn!("Failed to index vector: {e}");
                            }
                        }
                        Err(e) => return ToolResult::tool_error(format!("Lock error: {e}")),
                    }
                }
                Err(e) => {
                    tracing::warn!("Embedding failed: {e}");
                }
            }
        }

        // Create SUMMARIZES edges: merged → each source
        for source_id in &source_ids {
            let edge = Edge {
                id: format!("{merged_id}-SUMMARIZES-{source_id}"),
                src: merged_id.clone(),
                dst: source_id.clone(),
                relationship: RelationshipType::Summarizes,
                weight: 1.0,
                properties: HashMap::new(),
                created_at: now,
                valid_from: Some(now),
                valid_to: None,
            };
            if let Err(e) = self.storage.insert_graph_edge(&edge) {
                tracing::warn!("Failed to persist SUMMARIZES edge to {source_id}: {e}");
            }
            match self.lock_graph() {
                Ok(mut graph) => {
                    if let Err(e) = graph.add_edge(edge) {
                        tracing::warn!("Failed to add SUMMARIZES edge to {source_id}: {e}");
                    }
                }
                Err(e) => return ToolResult::tool_error(format!("Lock error: {e}")),
            }
        }

        // Persist vector index to disk
        self.save_index();

        ToolResult::text(
            serde_json::to_string_pretty(&json!({
                "merged_id": merged_id,
                "source_ids": source_ids,
                "relationship": "SUMMARIZES",
            }))
            .expect("JSON serialization of literal"),
        )
    }

    pub(crate) fn tool_associate_memories(&self, args: &Value) -> ToolResult {
        let src = match args.get("source_id").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => return ToolResult::tool_error("Missing 'source_id' parameter"),
        };
        let dst = match args.get("target_id").and_then(|v| v.as_str()) {
            Some(d) => d,
            None => return ToolResult::tool_error("Missing 'target_id' parameter"),
        };
        let rel_str = args
            .get("relationship")
            .and_then(|v| v.as_str())
            .unwrap_or("RELATES_TO");
        let weight = args.get("weight").and_then(|v| v.as_f64()).unwrap_or(1.0);

        let relationship: RelationshipType = match rel_str.parse() {
            Ok(r) => r,
            Err(e) => return ToolResult::tool_error(format!("Invalid relationship: {e}")),
        };

        let edge = Edge {
            id: format!("{src}-{}-{dst}", rel_str),
            src: src.to_string(),
            dst: dst.to_string(),
            relationship,
            weight,
            properties: HashMap::new(),
            created_at: chrono::Utc::now(),
            valid_from: None,
            valid_to: None,
        };

        // Store in SQLite
        if let Err(e) = self.storage.insert_graph_edge(&edge) {
            return ToolResult::tool_error(format!("Failed to store edge: {e}"));
        }

        // Add to in-memory graph
        match self.lock_graph() {
            Ok(mut graph) => {
                if let Err(e) = graph.add_edge(edge) {
                    tracing::warn!("Failed to add edge to graph: {e}");
                }
            }
            Err(e) => return ToolResult::tool_error(format!("Lock error: {e}")),
        }

        ToolResult::text(
            json!({
                "source": src,
                "target": dst,
                "relationship": rel_str,
                "weight": weight,
            })
            .to_string(),
        )
    }
}

#[cfg(test)]
#[path = "tests/tools_memory_tests.rs"]
mod tests;
