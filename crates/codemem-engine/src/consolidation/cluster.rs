use super::union_find::UnionFind;
use super::ConsolidationResult;
use crate::CodememEngine;
use codemem_core::{CodememError, GraphBackend, VectorBackend};
use codemem_storage::vector::cosine_similarity;
use serde_json::json;
use std::collections::{HashMap, HashSet};

impl CodememEngine {
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
}
