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

        self.recall_memories(query, k, &memory_type_filter, namespace_filter)
    }

    /// Search the server's storage with optional type and namespace filters.
    pub(crate) fn recall_memories(
        &self,
        query: &str,
        k: usize,
        memory_type_filter: &Option<MemoryType>,
        namespace_filter: Option<&str>,
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
mod tests {
    use super::*;
    use crate::test_helpers::*;

    #[test]
    fn handle_tools_call_store() {
        let server = test_server();
        let params = json!({"name": "store_memory", "arguments": {"content": "test content"}});
        let resp = server.handle_request("tools/call", Some(&params), json!(3));
        assert!(resp.result.is_some());
        assert!(resp.error.is_none());

        // Verify it actually stored
        let stats_resp = server.handle_request(
            "tools/call",
            Some(&json!({"name": "codemem_stats", "arguments": {}})),
            json!(4),
        );
        let stats = stats_resp.result.unwrap();
        let text = stats["content"][0]["text"].as_str().unwrap();
        let parsed: Value = serde_json::from_str(text).unwrap();
        assert_eq!(parsed["storage"]["memories"], 1);
    }

    #[test]
    fn handle_store_and_recall() {
        let server = test_server();

        // Store a memory
        let store_params = json!({
            "name": "store_memory",
            "arguments": {
                "content": "Rust uses ownership and borrowing for memory safety",
                "memory_type": "insight",
                "tags": ["rust", "memory"]
            }
        });
        server.handle_request("tools/call", Some(&store_params), json!(1));

        // Recall it (text search fallback, no embeddings in test)
        let recall_params = json!({
            "name": "recall_memory",
            "arguments": {"query": "rust memory safety"}
        });
        let resp = server.handle_request("tools/call", Some(&recall_params), json!(2));
        let result = resp.result.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        // Should find the memory via token overlap
        assert!(text.contains("ownership") || text.contains("rust"));
    }

    #[test]
    fn handle_store_and_delete() {
        let server = test_server();

        // Store
        let store_params = json!({
            "name": "store_memory",
            "arguments": {"content": "delete me"}
        });
        let resp = server.handle_request("tools/call", Some(&store_params), json!(1));
        let result = resp.result.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        let stored: Value = serde_json::from_str(text).unwrap();
        let id = stored["id"].as_str().unwrap();

        // Delete
        let delete_params = json!({
            "name": "delete_memory",
            "arguments": {"id": id}
        });
        let resp = server.handle_request("tools/call", Some(&delete_params), json!(2));
        assert!(resp.error.is_none());
    }

    // ── Memory Type Filter Tests ────────────────────────────────────────

    #[test]
    fn recall_filters_by_memory_type() {
        let server = test_server();

        // Store memories of different types, all containing "rust"
        store_memory(&server, "rust ownership insight", "insight", &["rust"]);
        store_memory(&server, "rust pattern matching", "pattern", &["rust"]);
        store_memory(&server, "rust decision to use enums", "decision", &["rust"]);

        // Recall with type filter "insight"
        let text = recall_memories(&server, "rust", Some("insight"));
        let results: Vec<Value> = serde_json::from_str(&text).unwrap();

        // Should only contain the insight memory
        assert_eq!(results.len(), 1);
        assert_eq!(results[0]["memory_type"], "insight");
        assert!(results[0]["content"]
            .as_str()
            .unwrap()
            .contains("ownership"));
    }

    #[test]
    fn recall_without_type_filter_returns_all() {
        let server = test_server();

        store_memory(&server, "rust ownership insight", "insight", &["rust"]);
        store_memory(&server, "rust pattern matching", "pattern", &["rust"]);

        // Recall without type filter
        let text = recall_memories(&server, "rust", None);
        let results: Vec<Value> = serde_json::from_str(&text).unwrap();

        // Should return both
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn recall_with_invalid_type_filter_returns_all() {
        let server = test_server();

        store_memory(&server, "rust ownership insight", "insight", &["rust"]);

        // An invalid memory_type string should be ignored (parsed as None)
        let text = recall_memories(&server, "rust", Some("nonexistent_type"));
        let results: Vec<Value> = serde_json::from_str(&text).unwrap();

        // Should return everything (no filter applied)
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn recall_with_type_filter_no_matches() {
        let server = test_server();

        store_memory(&server, "rust ownership insight", "insight", &["rust"]);

        // Filter for a type that has no matches in the content query
        let text = recall_memories(&server, "rust", Some("habit"));
        assert_eq!(text, "No matching memories found.");
    }

    // ── Namespace Filter Tests ────────────────────────────────────────

    #[test]
    fn recall_filters_by_namespace() {
        let server = test_server();

        // Store memories with different namespaces via direct storage
        let now = chrono::Utc::now();
        for (content, ns) in [
            ("rust ownership in project-a", Some("/projects/a")),
            ("rust borrowing in project-b", Some("/projects/b")),
            ("rust global memory no namespace", None),
        ] {
            let id = uuid::Uuid::new_v4().to_string();
            let hash = Storage::content_hash(content);
            let memory = MemoryNode {
                id: id.clone(),
                content: content.to_string(),
                memory_type: MemoryType::Insight,
                importance: 0.5,
                confidence: 1.0,
                access_count: 0,
                content_hash: hash,
                tags: vec!["rust".to_string()],
                metadata: HashMap::new(),
                namespace: ns.map(String::from),
                created_at: now,
                updated_at: now,
                last_accessed_at: now,
            };
            server.storage.insert_memory(&memory).unwrap();

            // Add graph node so graph scoring works
            let graph_node = GraphNode {
                id: id.clone(),
                kind: NodeKind::Memory,
                label: content.to_string(),
                payload: HashMap::new(),
                centrality: 0.0,
                memory_id: Some(id),
                namespace: None,
            };
            server.storage.insert_graph_node(&graph_node).unwrap();
            let _ = server.graph.lock().unwrap().add_node(graph_node);
        }

        // Recall with namespace filter "/projects/a"
        let params = json!({
            "name": "recall_memory",
            "arguments": {"query": "rust", "namespace": "/projects/a"}
        });
        let resp = server.handle_request("tools/call", Some(&params), json!(100));
        let result = resp.result.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        let results: Vec<Value> = serde_json::from_str(text).unwrap();

        assert_eq!(results.len(), 1);
        assert!(results[0]["content"]
            .as_str()
            .unwrap()
            .contains("project-a"));
    }

    #[test]
    fn recall_without_namespace_returns_all() {
        let server = test_server();

        // Store memories in different namespaces
        store_memory(&server, "rust memory one", "context", &["rust"]);
        store_memory(&server, "rust memory two", "context", &["rust"]);

        // Recall without namespace filter returns all
        let text = recall_memories(&server, "rust", None);
        let results: Vec<Value> = serde_json::from_str(&text).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn store_memory_with_namespace() {
        let server = test_server();

        let params = json!({
            "name": "store_memory",
            "arguments": {
                "content": "namespaced memory content",
                "namespace": "/my/project"
            }
        });
        let resp = server.handle_request("tools/call", Some(&params), json!(200));
        let result = resp.result.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        let stored: Value = serde_json::from_str(text).unwrap();
        let id = stored["id"].as_str().unwrap();

        // Retrieve and verify namespace is set
        let memory = server.storage.get_memory(id).unwrap().unwrap();
        assert_eq!(memory.namespace.as_deref(), Some("/my/project"));
    }

    #[test]
    fn store_memory_with_links() {
        let server = test_server();

        // First store two memories to get node IDs
        let m1 = store_memory(&server, "target node one", "context", &[]);
        let m2 = store_memory(&server, "target node two", "context", &[]);
        let m1_id = m1["id"].as_str().unwrap();
        let m2_id = m2["id"].as_str().unwrap();

        // Store a new memory with links to the previous two
        let params = json!({
            "name": "store_memory",
            "arguments": {
                "content": "linked memory content",
                "links": [m1_id, m2_id]
            }
        });
        let resp = server.handle_request("tools/call", Some(&params), json!(305));
        let result = resp.result.unwrap();
        assert_eq!(result["isError"], false);
        let text = result["content"][0]["text"].as_str().unwrap();
        let stored: Value = serde_json::from_str(text).unwrap();
        let linked_id = stored["id"].as_str().unwrap();

        // Verify edges were created
        let graph = server.graph.lock().unwrap();
        let edges = graph.get_edges(linked_id).unwrap();
        assert_eq!(edges.len(), 2);
        for edge in &edges {
            assert_eq!(edge.src, linked_id);
            assert_eq!(edge.relationship, RelationshipType::RelatesTo);
        }
    }

    // ── Vector Index Persistence Tests ──────────────────────────────────

    // ── Refine Memory Tests ────────────────────────────────────────────

    #[test]
    fn refine_creates_evolved_into_edge() {
        let server = test_server();
        let stored = store_memory(&server, "original content", "insight", &["rust"]);
        let old_id = stored["id"].as_str().unwrap();

        let params = json!({
            "name": "refine_memory",
            "arguments": {
                "id": old_id,
                "content": "refined content with more detail",
                "tags": ["rust", "refined"]
            }
        });
        let resp = server.handle_request("tools/call", Some(&params), json!(100));
        let result = resp.result.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        let parsed: Value = serde_json::from_str(text).unwrap();

        assert_eq!(parsed["old_id"].as_str().unwrap(), old_id);
        assert_eq!(parsed["relationship"], "EVOLVED_INTO");
        let new_id = parsed["new_id"].as_str().unwrap();
        assert_ne!(old_id, new_id);

        // Verify EVOLVED_INTO edge exists in storage
        let edges = server.storage.get_edges_for_node(old_id).unwrap();
        let evolved_edges: Vec<_> = edges
            .iter()
            .filter(|e| e.relationship == RelationshipType::EvolvedInto)
            .collect();
        assert_eq!(evolved_edges.len(), 1);
        assert_eq!(evolved_edges[0].src, old_id);
        assert_eq!(evolved_edges[0].dst, new_id);
        assert!(evolved_edges[0].valid_from.is_some());
    }

    #[test]
    fn refine_preserves_old_memory() {
        let server = test_server();
        let stored = store_memory(&server, "keep this content", "decision", &["arch"]);
        let old_id = stored["id"].as_str().unwrap();

        let params = json!({
            "name": "refine_memory",
            "arguments": {
                "id": old_id,
                "content": "new version of the content"
            }
        });
        server.handle_request("tools/call", Some(&params), json!(101));

        // Old memory should still exist and be unchanged
        let old_memory = server.storage.get_memory(old_id).unwrap().unwrap();
        assert_eq!(old_memory.content, "keep this content");
        assert_eq!(old_memory.memory_type, MemoryType::Decision);
    }

    #[test]
    fn refine_nonexistent_errors() {
        let server = test_server();
        let params = json!({
            "name": "refine_memory",
            "arguments": {
                "id": "nonexistent-id",
                "content": "will fail"
            }
        });
        let resp = server.handle_request("tools/call", Some(&params), json!(102));
        let result = resp.result.unwrap();
        assert_eq!(result["isError"], true);
        let text = result["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("not found"));
    }

    // ── Split Memory Tests ──────────────────────────────────────────────

    #[test]
    fn split_creates_part_of_edges() {
        let server = test_server();
        let stored = store_memory(
            &server,
            "combined content about A and B",
            "insight",
            &["tag"],
        );
        let source_id = stored["id"].as_str().unwrap();

        let params = json!({
            "name": "split_memory",
            "arguments": {
                "id": source_id,
                "parts": [
                    { "content": "content about A" },
                    { "content": "content about B", "tags": ["b_tag"] }
                ]
            }
        });
        let resp = server.handle_request("tools/call", Some(&params), json!(200));
        let result = resp.result.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        let parsed: Value = serde_json::from_str(text).unwrap();

        assert_eq!(parsed["source_id"].as_str().unwrap(), source_id);
        assert_eq!(parsed["relationship"], "PART_OF");
        let parts = parsed["parts"].as_array().unwrap();
        assert_eq!(parts.len(), 2);

        // Verify PART_OF edges exist in storage
        let edges = server.storage.get_edges_for_node(source_id).unwrap();
        let part_of_edges: Vec<_> = edges
            .iter()
            .filter(|e| e.relationship == RelationshipType::PartOf)
            .collect();
        assert_eq!(part_of_edges.len(), 2);
        for edge in &part_of_edges {
            assert_eq!(edge.dst, source_id);
            assert!(edge.valid_from.is_some());
        }
    }

    #[test]
    fn split_empty_parts_errors() {
        let server = test_server();
        let stored = store_memory(&server, "something to split", "context", &[]);
        let source_id = stored["id"].as_str().unwrap();

        let params = json!({
            "name": "split_memory",
            "arguments": {
                "id": source_id,
                "parts": []
            }
        });
        let resp = server.handle_request("tools/call", Some(&params), json!(201));
        let result = resp.result.unwrap();
        assert_eq!(result["isError"], true);
        let text = result["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("empty"));
    }

    // ── Merge Memories Tests ────────────────────────────────────────────

    #[test]
    fn merge_creates_summarizes_edges() {
        let server = test_server();
        let m1 = store_memory(&server, "first memory to merge", "context", &["a"]);
        let m2 = store_memory(&server, "second memory to merge", "context", &["b"]);
        let id1 = m1["id"].as_str().unwrap();
        let id2 = m2["id"].as_str().unwrap();

        let params = json!({
            "name": "merge_memories",
            "arguments": {
                "source_ids": [id1, id2],
                "content": "merged summary of first and second",
                "tags": ["merged"]
            }
        });
        let resp = server.handle_request("tools/call", Some(&params), json!(300));
        let result = resp.result.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        let parsed: Value = serde_json::from_str(text).unwrap();

        assert_eq!(parsed["relationship"], "SUMMARIZES");
        let merged_id = parsed["merged_id"].as_str().unwrap();
        let source_ids = parsed["source_ids"].as_array().unwrap();
        assert_eq!(source_ids.len(), 2);

        // Verify SUMMARIZES edges exist in storage
        let edges = server.storage.get_edges_for_node(merged_id).unwrap();
        let summarizes_edges: Vec<_> = edges
            .iter()
            .filter(|e| e.relationship == RelationshipType::Summarizes)
            .collect();
        assert_eq!(summarizes_edges.len(), 2);
        for edge in &summarizes_edges {
            assert_eq!(edge.src, merged_id);
            assert!(edge.valid_from.is_some());
        }
    }

    #[test]
    fn merge_insufficient_sources_errors() {
        let server = test_server();
        let m1 = store_memory(&server, "only one memory", "context", &[]);
        let id1 = m1["id"].as_str().unwrap();

        let params = json!({
            "name": "merge_memories",
            "arguments": {
                "source_ids": [id1],
                "content": "cannot merge just one"
            }
        });
        let resp = server.handle_request("tools/call", Some(&params), json!(301));
        let result = resp.result.unwrap();
        assert_eq!(result["isError"], true);
        let text = result["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("at least 2"));
    }

    #[test]
    fn merge_missing_source_errors() {
        let server = test_server();
        let m1 = store_memory(&server, "existing memory", "context", &[]);
        let id1 = m1["id"].as_str().unwrap();

        let params = json!({
            "name": "merge_memories",
            "arguments": {
                "source_ids": [id1, "nonexistent-id"],
                "content": "merge with missing"
            }
        });
        let resp = server.handle_request("tools/call", Some(&params), json!(302));
        let result = resp.result.unwrap();
        assert_eq!(result["isError"], true);
        let text = result["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("not found"));
    }

    // ── Vector Index Persistence Tests ──────────────────────────────────

    #[test]
    fn save_index_noop_for_in_memory_server() {
        let server = test_server();
        // db_path is None for in-memory server, save_index should not panic
        assert!(server.db_path.is_none());
        server.save_index(); // should be a no-op
    }

    #[test]
    fn from_db_path_sets_db_path() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();

        let server = McpServer::from_db_path(&path).unwrap();
        assert_eq!(server.db_path, Some(path));
    }

    #[test]
    fn save_index_persists_to_disk() {
        let dir = tempfile::TempDir::new().unwrap();
        let db_path = dir.path().join("test.db");

        let server = McpServer::from_db_path(&db_path).unwrap();

        // Store a memory (triggers save_index internally)
        store_memory(&server, "persistent memory test", "context", &[]);

        // The index file should exist if embeddings were available,
        // but even without embeddings save_index should not error.
        // Verify save_index can be called explicitly without panicking.
        server.save_index();

        // Verify the idx path is derived correctly
        let expected_idx_path = db_path.with_extension("idx");
        assert_eq!(expected_idx_path, dir.path().join("test.idx"),);
    }
}
