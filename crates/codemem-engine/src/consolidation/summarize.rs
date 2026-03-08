use super::ConsolidationResult;
use crate::CodememEngine;
use codemem_core::{CodememError, Edge, GraphBackend, MemoryNode, MemoryType, RelationshipType};
use serde_json::json;
use std::collections::HashMap;

impl CodememEngine {
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
                session_id: None,
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
}
