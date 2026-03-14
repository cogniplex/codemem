use super::ConsolidationResult;
use crate::CodememEngine;
use codemem_core::CodememError;
use serde_json::json;

impl CodememEngine {
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

        // H2: Batch deletes in groups of 100, releasing all locks between batches.
        // SQLite cascade (memory + graph nodes/edges + embeddings) is batched into
        // a single transaction per chunk; in-memory indices are updated afterwards.
        for batch in ids.chunks(100) {
            let batch_refs: Vec<&str> = batch.iter().map(|s| s.as_str()).collect();
            if let Err(e) = self.storage.delete_memories_batch_cascade(&batch_refs) {
                tracing::warn!(
                    "Failed to batch-delete {} memories during forget consolidation: {e}",
                    batch.len()
                );
            }

            // C1: Lock ordering: graph first, then vector, then bm25
            let mut graph = self.lock_graph()?;
            let mut vector = self.lock_vector()?;
            let mut bm25 = self.lock_bm25()?;
            for id in batch {
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
            self.rebuild_vector_index_internal(&mut **vector);
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
}
