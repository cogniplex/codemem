//! Consolidation logic for the Codemem memory engine.
//!
//! Contains all 5 consolidation cycles (decay, creative, cluster, forget, summarize),
//! helper data structures (UnionFind), and consolidation status queries.
//!
//! Lock ordering: always graph -> vector -> bm25 (to prevent deadlocks).

mod cluster;
mod creative;
mod decay;
mod forget;
mod summarize;
pub mod union_find;

pub use union_find::UnionFind;

use crate::CodememEngine;
use codemem_core::{CodememError, VectorBackend};

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
    pub fn rebuild_vector_index_internal(&self, vector: &mut dyn VectorBackend) {
        let embeddings = match self.storage.list_all_embeddings() {
            Ok(e) => e,
            Err(e) => {
                tracing::warn!("Failed to rebuild vector index: {e}");
                return;
            }
        };

        let _ = vector.rebuild_from_entries(&embeddings);
    }
}
