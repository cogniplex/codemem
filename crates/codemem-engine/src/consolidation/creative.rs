use super::ConsolidationResult;
use crate::CodememEngine;
use codemem_core::{CodememError, Edge, GraphNode, NodeKind, RelationshipType};
use serde_json::json;
use std::collections::{HashMap, HashSet};
use std::iter;

impl CodememEngine {
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

        // Collect nodes and edges to batch-insert after the loop
        let mut pending_nodes: Vec<GraphNode> = Vec::new();
        let mut pending_edges: Vec<Edge> = Vec::new();
        // Track which node IDs we've already queued for insertion to avoid duplicates
        let mut queued_node_ids: HashSet<String> = HashSet::new();

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

                // L19: Use combined key to avoid cloning both ID strings per check
                let edge_key = format!("{id}\0{neighbor_id}");
                if existing_edges.contains(&edge_key) {
                    continue;
                }

                let similarity = *sim as f64;

                // Cross-type links at 0.35 threshold, same-type links at 0.5.
                // Lower thresholds improve graph connectivity for conversational memories
                // which tend to have moderate but meaningful cosine similarities.
                let threshold = if my_type == neighbor_type { 0.5 } else { 0.35 };
                if similarity < threshold {
                    continue;
                }

                // M10: Ensure both nodes exist, using memory content as label when available
                for nid in [id, neighbor_id] {
                    if queued_node_ids.contains(nid) {
                        continue; // Already queued for batch insert
                    }
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
                    pending_nodes.push(node);
                    queued_node_ids.insert(nid.clone());
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

                pending_edges.push(edge);
                new_connections += 1;
            }
        }

        // Batch-insert collected nodes and edges into storage
        if !pending_nodes.is_empty() {
            if let Err(e) = self.storage.insert_graph_nodes_batch(&pending_nodes) {
                tracing::warn!(
                    "Failed to batch-insert {} graph nodes during creative consolidation: {e}",
                    pending_nodes.len()
                );
            }
        }
        if !pending_edges.is_empty() {
            if let Err(e) = self.storage.insert_graph_edges_batch(&pending_edges) {
                tracing::warn!(
                    "Failed to batch-insert {} graph edges during creative consolidation: {e}",
                    pending_edges.len()
                );
            }
        }

        // Add to in-memory graph
        for node in pending_nodes {
            let _ = graph.add_node(node);
        }
        for edge in pending_edges {
            let _ = graph.add_edge(edge);
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
}
