use crate::CodememEngine;
use codemem_core::{
    CodememError, Edge, GraphBackend, MemoryNode, NodeKind, NodeMemoryResult, RelationshipType,
};
use std::collections::HashSet;

#[cfg(test)]
#[path = "tests/graph_linking_tests.rs"]
mod tests;

impl CodememEngine {
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

        let existing_set: HashSet<&str> = existing_links.iter().map(|s| s.as_str()).collect();

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
        let mut seen = HashSet::new();

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

    // ── Tag-based Auto-linking ──────────────────────────────────────────

    /// Create edges between this memory and other memories that share tags.
    /// - `session:*` tags → PRECEDED_BY edges (temporal ordering within a session)
    /// - Other shared tags → SHARES_THEME edges (topical overlap)
    ///
    /// This runs during `persist_memory` so the graph builds connectivity at
    /// ingestion time, rather than relying solely on creative consolidation.
    pub fn auto_link_by_tags(&self, memory: &MemoryNode) {
        if memory.tags.is_empty() {
            return;
        }

        // Phase 1: Collect sibling IDs and build edges WITHOUT holding the graph lock.
        let now = chrono::Utc::now();
        let mut linked = HashSet::new();
        let mut edges_to_add = Vec::new();

        for tag in &memory.tags {
            let is_session_tag = tag.starts_with("session:");

            let sibling_ids = match self.storage.find_memory_ids_by_tag(
                tag,
                memory.namespace.as_deref(),
                &memory.id,
            ) {
                Ok(ids) => ids,
                Err(_) => continue,
            };

            for sibling_id in sibling_ids {
                if !linked.insert(sibling_id.clone()) {
                    continue;
                }

                let (relationship, edge_label) = if is_session_tag {
                    (RelationshipType::PrecededBy, "PRECEDED_BY")
                } else {
                    (RelationshipType::SharesTheme, "SHARES_THEME")
                };

                let edge_id = format!("{}-{edge_label}-{sibling_id}", memory.id);
                edges_to_add.push(Edge {
                    id: edge_id,
                    src: sibling_id,
                    dst: memory.id.clone(),
                    relationship,
                    weight: if is_session_tag { 0.8 } else { 0.5 },
                    properties: std::collections::HashMap::from([(
                        "auto_linked".to_string(),
                        serde_json::json!(true),
                    )]),
                    created_at: now,
                    valid_from: Some(now),
                    valid_to: None,
                });
            }
        }

        if edges_to_add.is_empty() {
            return;
        }

        // Phase 2: Acquire graph lock only for mutations.
        let mut graph = match self.lock_graph() {
            Ok(g) => g,
            Err(_) => return,
        };

        for edge in edges_to_add {
            if self.storage.insert_graph_edge(&edge).is_ok() {
                let _ = graph.add_edge(edge);
            }
        }
    }

    // ── Node Memory Queries ──────────────────────────────────────────────

    /// Retrieve all memories connected to a graph node via BFS traversal.
    ///
    /// Performs level-by-level BFS to track actual hop distance. For each
    /// Memory node found, reports the relationship type from the edge that
    /// connected it (or the edge leading into the path toward it).
    pub fn get_node_memories(
        &self,
        node_id: &str,
        max_depth: usize,
        include_relationships: Option<&[RelationshipType]>,
    ) -> Result<Vec<NodeMemoryResult>, CodememError> {
        let graph = self.lock_graph()?;

        // Manual BFS tracking (node_id, depth, relationship_from_parent_edge)
        let mut results: Vec<NodeMemoryResult> = Vec::new();
        let mut seen_memory_ids = HashSet::new();
        let mut visited = HashSet::new();
        let mut queue: std::collections::VecDeque<(String, usize, String)> =
            std::collections::VecDeque::new();

        visited.insert(node_id.to_string());
        queue.push_back((node_id.to_string(), 0, String::new()));

        while let Some((current_id, depth, parent_rel)) = queue.pop_front() {
            // Collect Memory nodes (skip the start node itself)
            if current_id != node_id {
                if let Some(node) = graph.get_node_ref(&current_id) {
                    if node.kind == NodeKind::Memory {
                        let memory_id = node.memory_id.as_deref().unwrap_or(&node.id);
                        if seen_memory_ids.insert(memory_id.to_string()) {
                            if let Ok(Some(memory)) = self.storage.get_memory_no_touch(memory_id) {
                                results.push(NodeMemoryResult {
                                    memory,
                                    relationship: parent_rel.clone(),
                                    depth,
                                });
                            }
                        }
                    }
                }
            }

            if depth >= max_depth {
                continue;
            }

            // Expand neighbors via edges, skipping Chunk nodes
            for edge in graph.get_edges_ref(&current_id) {
                let neighbor_id = if edge.src == current_id {
                    &edge.dst
                } else {
                    &edge.src
                };

                if visited.contains(neighbor_id.as_str()) {
                    continue;
                }

                // Apply relationship filter
                if let Some(allowed) = include_relationships {
                    if !allowed.contains(&edge.relationship) {
                        continue;
                    }
                }

                // Skip Chunk nodes (noisy, low-value for memory discovery)
                if let Some(neighbor) = graph.get_node_ref(neighbor_id) {
                    if neighbor.kind == NodeKind::Chunk {
                        continue;
                    }
                }

                visited.insert(neighbor_id.clone());
                queue.push_back((
                    neighbor_id.clone(),
                    depth + 1,
                    edge.relationship.to_string(),
                ));
            }
        }

        Ok(results)
    }
}
