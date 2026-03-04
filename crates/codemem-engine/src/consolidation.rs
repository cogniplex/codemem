//! Consolidation logic for the Codemem memory engine.
//!
//! Contains all 5 consolidation cycles (decay, creative, cluster, forget, summarize),
//! helper data structures (UnionFind), and consolidation status queries.
//!
//! Lock ordering: always graph -> vector -> bm25 (to prevent deadlocks).

use crate::CodememEngine;
use codemem_core::{
    CodememError, Edge, GraphBackend, GraphNode, MemoryNode, MemoryType, NodeKind,
    RelationshipType, VectorBackend,
};
use codemem_storage::vector::cosine_similarity;
use serde_json::json;
use std::collections::{HashMap, HashSet};
use std::iter;

/// Union-Find (disjoint set) data structure for transitive clustering.
pub struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    pub fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    pub fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    pub fn union(&mut self, x: usize, y: usize) {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx == ry {
            return;
        }
        match self.rank[rx].cmp(&self.rank[ry]) {
            std::cmp::Ordering::Less => self.parent[rx] = ry,
            std::cmp::Ordering::Greater => self.parent[ry] = rx,
            std::cmp::Ordering::Equal => {
                self.parent[ry] = rx;
                self.rank[rx] += 1;
            }
        }
    }

    pub fn groups(&mut self, n: usize) -> Vec<Vec<usize>> {
        let mut map: HashMap<usize, Vec<usize>> = HashMap::new();
        for i in 0..n {
            let root = self.find(i);
            map.entry(root).or_default().push(i);
        }
        map.into_values().collect()
    }
}

/// Result of a consolidation cycle.
pub struct ConsolidationResult {
    /// Name of the cycle (decay, creative, cluster, forget, summarize).
    pub cycle: String,
    /// Number of affected items (meaning depends on cycle type).
    pub affected: usize,
    /// Additional details as JSON.
    pub details: serde_json::Value,
}

/// Status of a single consolidation cycle.
pub struct ConsolidationStatusEntry {
    pub cycle_type: String,
    pub last_run: String,
    pub affected_count: usize,
}

impl CodememEngine {
    /// Consolidate decay: power-law decay that rewards access frequency.
    pub fn consolidate_decay(
        &self,
        threshold_days: Option<i64>,
    ) -> Result<ConsolidationResult, CodememError> {
        let threshold_days = threshold_days.unwrap_or(30);
        let now = chrono::Utc::now();
        let threshold_ts = (now - chrono::Duration::days(threshold_days)).timestamp();

        let stale = self.storage.get_stale_memories_for_decay(threshold_ts)?;

        if stale.is_empty() {
            if let Err(e) = self.storage.insert_consolidation_log("decay", 0) {
                tracing::warn!("Failed to log decay consolidation: {e}");
            }
            return Ok(ConsolidationResult {
                cycle: "decay".to_string(),
                affected: 0,
                details: json!({
                    "threshold_days": threshold_days,
                }),
            });
        }

        // Compute power-law decay: importance * 0.9^(days_since/30) * (1 + log2(max(access_count,1)) * 0.1)
        let now_ts = now.timestamp();
        let updates: Vec<(String, f64)> = stale
            .iter()
            .map(|(id, importance, access_count, last_accessed_at)| {
                let days_since = (now_ts - last_accessed_at) as f64 / 86400.0;
                let time_decay = 0.9_f64.powf(days_since / 30.0);
                let access_boost = 1.0 + ((*access_count).max(1) as f64).log2() * 0.1;
                let new_importance = (importance * time_decay * access_boost).clamp(0.0, 1.0);
                (id.clone(), new_importance)
            })
            .collect();

        let affected = self.storage.batch_update_importance(&updates)?;

        if let Err(e) = self.storage.insert_consolidation_log("decay", affected) {
            tracing::warn!("Failed to log decay consolidation: {e}");
        }

        Ok(ConsolidationResult {
            cycle: "decay".to_string(),
            affected,
            details: json!({
                "threshold_days": threshold_days,
                "algorithm": "power_law",
            }),
        })
    }

    /// Consolidate creative: O(n log n) semantic creative consolidation.
    /// Uses vector KNN search per memory to find cross-type neighbors and creates
    /// SHARES_THEME edges.
    ///
    /// Memory usage: O(K*768) per query instead of O(N*768) for all embeddings,
    /// where K is the number of nearest neighbors searched (7). Only the memory
    /// metadata (IDs + types) is kept in RAM, not the full embedding vectors.
    pub fn consolidate_creative(&self) -> Result<ConsolidationResult, CodememError> {
        // Load all memories with their types
        let parsed = self.storage.list_memories_for_creative()?;

        let ids_refs: Vec<&str> = parsed.iter().map(|(id, _, _)| id.as_str()).collect();
        let memories = self.storage.get_memories_batch(&ids_refs)?;

        // Build type lookup
        let type_map: HashMap<String, String> = memories
            .iter()
            .map(|m| (m.id.clone(), m.memory_type.to_string()))
            .collect();

        // Load existing SHARES_THEME edges to avoid duplicates.
        // Use a combined key string to allow borrowed lookups without cloning per check (L19).
        let all_edges = self.storage.all_graph_edges()?;
        let existing_edges: HashSet<String> = all_edges
            .iter()
            .filter(|e| {
                e.relationship == RelationshipType::SharesTheme
                    || e.relationship == RelationshipType::RelatesTo
            })
            .flat_map(|e| {
                // H6: Use iter::once chains instead of vec![] to avoid per-edge heap alloc
                iter::once(format!("{}\0{}", e.src, e.dst))
                    .chain(iter::once(format!("{}\0{}", e.dst, e.src)))
            })
            .collect();

        // X2: Instead of loading ALL embeddings into a HashMap (O(N*768) memory),
        // we iterate over memory IDs and fetch each embedding individually from
        // storage, then use vector KNN to find neighbors. This uses O(K*768)
        // memory per query where K=7.
        let memory_ids: Vec<String> = type_map.keys().cloned().collect();

        let now = chrono::Utc::now();
        let mut new_connections = 0usize;
        // C1: Lock ordering: graph first, then vector
        let mut graph = self.lock_graph()?;
        let mut vector = self.lock_vector()?;

        // H1: For each memory, load its embedding on demand and find 6 nearest neighbors.
        // Drop and re-acquire graph+vector locks every 50 iterations to yield to other threads.
        for (iter_idx, id) in memory_ids.iter().enumerate() {
            // H1: Yield locks every 50 iterations to avoid long lock holds
            if iter_idx > 0 && iter_idx % 50 == 0 {
                // Must drop in reverse acquisition order
                drop(vector);
                drop(graph);
                graph = self.lock_graph()?;
                vector = self.lock_vector()?;
            }

            let my_type = match type_map.get(id) {
                Some(t) => t,
                None => continue,
            };

            // Load embedding for this single memory from storage (not kept in RAM)
            let embedding = match self.storage.get_embedding(id) {
                Ok(Some(emb)) => emb,
                _ => continue,
            };

            let neighbors = vector.search(&embedding, 7).unwrap_or_default();

            for (neighbor_id, sim) in &neighbors {
                if neighbor_id == id {
                    continue;
                }

                let neighbor_type = match type_map.get(neighbor_id) {
                    Some(t) => t,
                    None => continue,
                };

                // Cross-type only
                if my_type == neighbor_type {
                    continue;
                }

                // L19: Use combined key to avoid cloning both ID strings per check
                let edge_key = format!("{id}\0{neighbor_id}");
                if existing_edges.contains(&edge_key) {
                    continue;
                }

                let similarity = *sim as f64;
                if similarity < 0.5 {
                    continue;
                }

                // M10: Ensure both nodes exist, using memory content as label when available
                for nid in [id, neighbor_id] {
                    if graph.get_node(nid).ok().flatten().is_some() {
                        continue; // Node already exists, don't overwrite
                    }
                    let label = memories
                        .iter()
                        .find(|m| m.id == *nid)
                        .map(|m| crate::scoring::truncate_content(&m.content, 80))
                        .unwrap_or_else(|| nid.clone());
                    let node = GraphNode {
                        id: nid.clone(),
                        kind: NodeKind::Memory,
                        label,
                        payload: HashMap::new(),
                        centrality: 0.0,
                        memory_id: Some(nid.clone()),
                        namespace: None,
                    };
                    if let Err(e) = self.storage.insert_graph_node(&node) {
                        tracing::warn!(
                            "Failed to insert graph node {nid} during creative consolidation: {e}"
                        );
                    }
                    let _ = graph.add_node(node);
                }

                let edge_id = format!("{id}-SHARES_THEME-{neighbor_id}");
                let edge = Edge {
                    id: edge_id,
                    src: id.clone(),
                    dst: neighbor_id.clone(),
                    relationship: RelationshipType::SharesTheme,
                    weight: similarity,
                    properties: HashMap::new(),
                    created_at: now,
                    valid_from: Some(now),
                    valid_to: None,
                };

                if self.storage.insert_graph_edge(&edge).is_ok() {
                    let _ = graph.add_edge(edge);
                    new_connections += 1;
                }
            }
        }
        drop(vector);
        drop(graph);

        if let Err(e) = self
            .storage
            .insert_consolidation_log("creative", new_connections)
        {
            tracing::warn!("Failed to log creative consolidation: {e}");
        }

        Ok(ConsolidationResult {
            cycle: "creative".to_string(),
            affected: new_connections,
            details: json!({
                "algorithm": "vector_knn",
            }),
        })
    }

    /// Consolidate cluster: semantic deduplication using vector KNN + cosine similarity.
    ///
    /// Groups memories by namespace and memory_type for candidate grouping. Within each
    /// group, uses vector KNN search to find candidate duplicates (avoiding O(n^2) pairwise
    /// comparison), then verifies with cosine similarity + union-find to cluster
    /// transitively-similar memories. Keeps the highest-importance memory per cluster.
    ///
    /// For small groups (<=50 members), falls back to pairwise comparison since the
    /// overhead of KNN setup is not worth it.
    pub fn consolidate_cluster(
        &self,
        similarity_threshold: Option<f64>,
    ) -> Result<ConsolidationResult, CodememError> {
        let similarity_threshold = similarity_threshold.unwrap_or(0.92);

        let ids = self.storage.list_memory_ids()?;
        let id_refs: Vec<&str> = ids.iter().map(|s| s.as_str()).collect();
        let memories = self.storage.get_memories_batch(&id_refs)?;

        // M11: Group by namespace+memory_type instead of hash prefix (SHA-256 prefix is
        // uniformly distributed, making it a no-op as a pre-filter). Grouping by semantic
        // attributes ensures we only compare memories that could plausibly be duplicates.
        let mut groups: HashMap<String, Vec<usize>> = HashMap::new();
        for (idx, m) in memories.iter().enumerate() {
            let key = format!(
                "{}:{}",
                m.namespace.as_deref().unwrap_or("default"),
                m.memory_type
            );
            groups.entry(key).or_default().push(idx);
        }

        // Union-find for transitive clustering
        let n = memories.len();
        let mut uf = UnionFind::new(n);

        // X3: For large groups, use vector KNN to find candidate duplicates
        // instead of O(n^2) pairwise comparison. For small groups (<=50),
        // pairwise is fine and avoids KNN overhead.
        let vector = self.lock_vector()?;

        // Build index from memory idx to id for quick lookup
        let id_to_idx: HashMap<&str, usize> = memories
            .iter()
            .enumerate()
            .map(|(i, m)| (m.id.as_str(), i))
            .collect();

        for member_indices in groups.values() {
            if member_indices.len() <= 1 {
                continue;
            }

            if member_indices.len() <= 50 {
                // O(n^2) pairwise comparison is acceptable here — groups are capped at <=50 members,
                // so worst case is ~1250 comparisons which completes in microseconds.
                for i in 0..member_indices.len() {
                    for j in (i + 1)..member_indices.len() {
                        let idx_a = member_indices[i];
                        let idx_b = member_indices[j];

                        let id_a = &memories[idx_a].id;
                        let id_b = &memories[idx_b].id;

                        let sim = match (
                            self.storage.get_embedding(id_a).ok().flatten(),
                            self.storage.get_embedding(id_b).ok().flatten(),
                        ) {
                            (Some(emb_a), Some(emb_b)) => cosine_similarity(&emb_a, &emb_b),
                            _ => {
                                if memories[idx_a].content_hash == memories[idx_b].content_hash {
                                    1.0
                                } else {
                                    0.0
                                }
                            }
                        };

                        if sim >= similarity_threshold {
                            uf.union(idx_a, idx_b);
                        }
                    }
                }
            } else {
                // Large group: use vector KNN per member to find candidates
                // Search for K nearest neighbors where K is small (e.g. 10)
                let k_neighbors = 10.min(member_indices.len());

                // Build a set of IDs in this group for filtering
                let group_ids: HashSet<&str> = member_indices
                    .iter()
                    .map(|&idx| memories[idx].id.as_str())
                    .collect();

                for &idx_a in member_indices {
                    let id_a = &memories[idx_a].id;
                    let embedding = match self.storage.get_embedding(id_a).ok().flatten() {
                        Some(e) => e,
                        None => continue,
                    };

                    // Use vector KNN to find nearest neighbors
                    let neighbors = vector
                        .search(&embedding, k_neighbors + 1)
                        .unwrap_or_default();

                    for (neighbor_id, _) in &neighbors {
                        if neighbor_id == id_a {
                            continue;
                        }
                        // Only consider neighbors within the same group
                        if !group_ids.contains(neighbor_id.as_str()) {
                            continue;
                        }

                        let idx_b = match id_to_idx.get(neighbor_id.as_str()) {
                            Some(&idx) => idx,
                            None => continue,
                        };

                        // Verify with cosine similarity
                        let sim = match self.storage.get_embedding(neighbor_id).ok().flatten() {
                            Some(emb_b) => cosine_similarity(&embedding, &emb_b),
                            None => {
                                if memories[idx_a].content_hash == memories[idx_b].content_hash {
                                    1.0
                                } else {
                                    0.0
                                }
                            }
                        };

                        if sim >= similarity_threshold {
                            uf.union(idx_a, idx_b);
                        }
                    }
                }
            }
        }
        drop(vector);

        let clusters = uf.groups(n);

        let mut merged_count = 0usize;
        let mut kept_count = 0usize;
        let mut ids_to_delete: Vec<String> = Vec::new();

        for cluster in &clusters {
            if cluster.len() <= 1 {
                kept_count += 1;
                continue;
            }

            let mut members: Vec<(usize, f64)> = cluster
                .iter()
                .map(|&idx| (idx, memories[idx].importance))
                .collect();
            members.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            kept_count += 1;

            for &(idx, _) in members.iter().skip(1) {
                ids_to_delete.push(memories[idx].id.clone());
                merged_count += 1;
            }
        }

        // H2: Batch deletes in groups of 100, releasing all locks between batches
        for batch in ids_to_delete.chunks(100) {
            // C1: Lock ordering: graph first, then vector, then bm25
            let mut graph = self.lock_graph()?;
            let mut vector = self.lock_vector()?;
            let mut bm25 = self.lock_bm25()?;
            for id in batch {
                if let Err(e) = self.storage.delete_memory(id) {
                    tracing::warn!(
                        "Failed to delete memory {id} during cluster consolidation: {e}"
                    );
                }
                if let Err(e) = self.storage.delete_embedding(id) {
                    tracing::warn!(
                        "Failed to delete embedding {id} during cluster consolidation: {e}"
                    );
                }
                if let Err(e) = self.storage.delete_graph_edges_for_node(id) {
                    tracing::warn!(
                        "Failed to delete graph edges for {id} during cluster consolidation: {e}"
                    );
                }
                if let Err(e) = self.storage.delete_graph_node(id) {
                    tracing::warn!(
                        "Failed to delete graph node {id} during cluster consolidation: {e}"
                    );
                }
                if let Err(e) = vector.remove(id) {
                    tracing::warn!(
                        "Failed to remove {id} from vector index during cluster consolidation: {e}"
                    );
                }
                if let Err(e) = graph.remove_node(id) {
                    tracing::warn!(
                        "Failed to remove {id} from graph during cluster consolidation: {e}"
                    );
                }
                // M15: Clean up BM25 index for deleted memories (was missing)
                bm25.remove_document(id);
            }
            drop(bm25);
            drop(vector);
            drop(graph);
        }

        // Rebuild vector index if we deleted anything
        if merged_count > 0 {
            let mut vector = self.lock_vector()?;
            self.rebuild_vector_index_internal(&mut vector);
            drop(vector);
        }

        self.save_index();

        if let Err(e) = self
            .storage
            .insert_consolidation_log("cluster", merged_count)
        {
            tracing::warn!("Failed to log cluster consolidation: {e}");
        }

        Ok(ConsolidationResult {
            cycle: "cluster".to_string(),
            affected: merged_count,
            details: json!({
                "merged": merged_count,
                "kept": kept_count,
                "similarity_threshold": similarity_threshold,
                "algorithm": "semantic_cosine",
            }),
        })
    }

    /// Consolidate forget: delete low-importance, never-accessed memories.
    pub fn consolidate_forget(
        &self,
        importance_threshold: Option<f64>,
        target_tags: Option<&[String]>,
        max_access_count: Option<u32>,
    ) -> Result<ConsolidationResult, CodememError> {
        let importance_threshold = importance_threshold.unwrap_or(0.1);
        let max_access_count = max_access_count.unwrap_or(0);

        // M13: When max_access_count > 0 (non-default), filter in Rust since
        // find_forgettable() hardcodes access_count = 0 in SQL.
        let ids = match target_tags {
            Some(tags) if !tags.is_empty() => {
                self.find_forgettable_by_tags(importance_threshold, tags, max_access_count)?
            }
            _ if max_access_count > 0 => {
                // find_forgettable only returns access_count=0; fall back to manual filtering
                let all = self.storage.list_memories_filtered(None, None)?;
                all.into_iter()
                    .filter(|m| {
                        m.importance < importance_threshold && m.access_count <= max_access_count
                    })
                    .map(|m| m.id)
                    .collect()
            }
            _ => self.storage.find_forgettable(importance_threshold)?,
        };

        let deleted = ids.len();

        // H2: Batch deletes in groups of 100, releasing all locks between batches
        for batch in ids.chunks(100) {
            // C1: Lock ordering: graph first, then vector, then bm25
            let mut graph = self.lock_graph()?;
            let mut vector = self.lock_vector()?;
            let mut bm25 = self.lock_bm25()?;
            for id in batch {
                if let Err(e) = self.storage.delete_memory(id) {
                    tracing::warn!("Failed to delete memory {id} during forget consolidation: {e}");
                }
                if let Err(e) = self.storage.delete_embedding(id) {
                    tracing::warn!(
                        "Failed to delete embedding {id} during forget consolidation: {e}"
                    );
                }
                if let Err(e) = self.storage.delete_graph_edges_for_node(id) {
                    tracing::warn!(
                        "Failed to delete graph edges for {id} during forget consolidation: {e}"
                    );
                }
                if let Err(e) = self.storage.delete_graph_node(id) {
                    tracing::warn!(
                        "Failed to delete graph node {id} during forget consolidation: {e}"
                    );
                }
                if let Err(e) = graph.remove_node(id) {
                    tracing::warn!(
                        "Failed to remove {id} from graph during forget consolidation: {e}"
                    );
                }
                if let Err(e) = vector.remove(id) {
                    tracing::warn!(
                        "Failed to remove {id} from vector index during forget consolidation: {e}"
                    );
                }
                bm25.remove_document(id);
            }
            drop(bm25);
            drop(vector);
            drop(graph);
        }

        // Rebuild vector index if we deleted anything
        if deleted > 0 {
            let mut vector = self.lock_vector()?;
            self.rebuild_vector_index_internal(&mut vector);
            drop(vector);
        }

        self.save_index();

        if let Err(e) = self.storage.insert_consolidation_log("forget", deleted) {
            tracing::warn!("Failed to log forget consolidation: {e}");
        }

        Ok(ConsolidationResult {
            cycle: "forget".to_string(),
            affected: deleted,
            details: json!({
                "threshold": importance_threshold,
            }),
        })
    }

    /// Consolidate summarize: LLM-powered consolidation that finds
    /// connected components, summarizes large clusters into Insight memories
    /// linked via SUMMARIZES edges.
    pub fn consolidate_summarize(
        &self,
        min_cluster_size: Option<usize>,
    ) -> Result<ConsolidationResult, CodememError> {
        let min_cluster_size = min_cluster_size.unwrap_or(5);

        let provider = crate::compress::CompressProvider::from_env();
        if !provider.is_enabled() {
            return Err(CodememError::Config(
                "CODEMEM_COMPRESS_PROVIDER env var not set. \
                 Set it to 'ollama', 'openai', or 'anthropic' to enable LLM-powered consolidation."
                    .to_string(),
            ));
        }

        // Find connected components via the graph
        let graph = self.lock_graph()?;
        let components = graph.connected_components();
        drop(graph);

        let large_clusters: Vec<&Vec<String>> = components
            .iter()
            .filter(|c| c.len() >= min_cluster_size)
            .collect();

        if large_clusters.is_empty() {
            return Ok(ConsolidationResult {
                cycle: "summarize".to_string(),
                affected: 0,
                details: json!({
                    "clusters_found": 0,
                    "min_cluster_size": min_cluster_size,
                    "message": format!("No clusters with {} or more members found", min_cluster_size),
                }),
            });
        }

        let mut summarized_count = 0usize;
        let mut created_ids: Vec<String> = Vec::new();

        for cluster in &large_clusters {
            let mut contents: Vec<String> = Vec::new();
            let mut source_ids: Vec<String> = Vec::new();
            let mut all_tags: Vec<String> = Vec::new();

            // M12: Acquire graph lock once before the inner loop, collect all memory IDs,
            // then drop the lock before batch-fetching memories from storage.
            let memory_ids: Vec<String> = {
                let graph = self.lock_graph()?;
                cluster
                    .iter()
                    .filter_map(|node_id| {
                        graph
                            .get_node(node_id)
                            .ok()
                            .flatten()
                            .and_then(|node| node.memory_id.clone())
                    })
                    .collect()
            };

            for mid in &memory_ids {
                if let Ok(Some(mem)) = self.storage.get_memory_no_touch(mid) {
                    contents.push(mem.content.clone());
                    source_ids.push(mid.clone());
                    all_tags.extend(mem.tags.clone());
                }
            }

            if contents.len() < 2 {
                continue;
            }

            let combined = contents.join("\n---\n");
            let summary = match provider.compress(&combined, "consolidate_summarize", None) {
                Some(s) => s,
                None => continue,
            };

            all_tags.sort();
            all_tags.dedup();

            let now = chrono::Utc::now();
            let new_id = uuid::Uuid::new_v4().to_string();
            let hash = codemem_storage::Storage::content_hash(&summary);

            let mem = MemoryNode {
                id: new_id.clone(),
                content: summary,
                memory_type: MemoryType::Insight,
                importance: 0.7,
                confidence: 1.0,
                access_count: 0,
                content_hash: hash,
                tags: all_tags,
                metadata: HashMap::new(),
                namespace: None,
                created_at: now,
                updated_at: now,
                last_accessed_at: now,
            };

            // M4: Use persist_memory_no_save to skip per-memory index save,
            // then call save_index() once after the entire loop.
            if self.persist_memory_no_save(&mem).is_err() {
                tracing::warn!("Failed to persist summary memory: {new_id}");
                continue;
            }

            if let Ok(mut graph) = self.lock_graph() {
                for sid in &source_ids {
                    let edge = Edge {
                        id: format!("{new_id}-SUMMARIZES-{sid}"),
                        src: new_id.clone(),
                        dst: sid.clone(),
                        relationship: RelationshipType::Summarizes,
                        weight: 1.0,
                        properties: HashMap::new(),
                        created_at: now,
                        valid_from: Some(now),
                        valid_to: None,
                    };
                    if let Err(e) = self.storage.insert_graph_edge(&edge) {
                        tracing::warn!("Failed to persist SUMMARIZES edge: {e}");
                    }
                    let _ = graph.add_edge(edge);
                }
            }

            summarized_count += 1;
            created_ids.push(new_id);
        }

        // M4: Save vector index once after all summaries are persisted
        if summarized_count > 0 {
            self.save_index();
        }

        if let Err(e) = self
            .storage
            .insert_consolidation_log("summarize", summarized_count)
        {
            tracing::warn!("Failed to log summarize consolidation: {e}");
        }

        Ok(ConsolidationResult {
            cycle: "summarize".to_string(),
            affected: summarized_count,
            details: json!({
                "clusters_found": large_clusters.len(),
                "summarized": summarized_count,
                "created_ids": created_ids,
                "min_cluster_size": min_cluster_size,
            }),
        })
    }

    /// Get the status of all consolidation cycles.
    pub fn consolidation_status(&self) -> Result<Vec<ConsolidationStatusEntry>, CodememError> {
        let runs = self.storage.last_consolidation_runs()?;
        let mut entries = Vec::new();
        for entry in &runs {
            let dt = chrono::DateTime::from_timestamp(entry.run_at, 0)
                .map(|t| t.format("%Y-%m-%d %H:%M:%S UTC").to_string())
                .unwrap_or_else(|| "unknown".to_string());
            entries.push(ConsolidationStatusEntry {
                cycle_type: entry.cycle_type.clone(),
                last_run: dt,
                affected_count: entry.affected_count,
            });
        }
        Ok(entries)
    }

    /// Find memories matching any of the target tags below importance threshold
    /// and with access_count <= max_access_count.
    ///
    /// M14: Note — this loads all memories and filters in Rust. For large databases,
    /// a storage method with SQL WHERE clauses for importance/access_count/tags would
    /// be more efficient, but that requires adding a new StorageBackend trait method.
    pub fn find_forgettable_by_tags(
        &self,
        importance_threshold: f64,
        target_tags: &[String],
        max_access_count: u32,
    ) -> Result<Vec<String>, CodememError> {
        let all_memories = self.storage.list_memories_filtered(None, None)?;
        let mut forgettable = Vec::new();

        for memory in &all_memories {
            if memory.importance >= importance_threshold {
                continue;
            }
            if memory.access_count > max_access_count {
                continue;
            }
            if memory.tags.iter().any(|t| target_tags.contains(t)) {
                forgettable.push(memory.id.clone());
            }
        }

        Ok(forgettable)
    }

    /// Internal helper: rebuild vector index from all stored embeddings.
    pub fn rebuild_vector_index_internal(&self, vector: &mut codemem_storage::HnswIndex) {
        let embeddings = match self.storage.list_all_embeddings() {
            Ok(e) => e,
            Err(e) => {
                tracing::warn!("Failed to rebuild vector index: {e}");
                return;
            }
        };

        if let Ok(mut fresh) = codemem_storage::HnswIndex::with_defaults() {
            for (id, floats) in &embeddings {
                let _ = fresh.insert(id, floats);
            }
            *vector = fresh;
        }
    }
}
