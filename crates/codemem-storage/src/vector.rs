//! codemem-vector: HNSW vector index for Codemem using usearch.
//!
//! Provides persistent, incremental, SIMD-accelerated vector search
//! with M=16, efConstruction=200, efSearch=100, cosine distance, 768 dimensions.

use codemem_core::{CodememError, VectorBackend, VectorConfig, VectorStats};
use std::collections::HashMap;
use std::path::Path;
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

/// HNSW vector index backed by usearch.
pub struct HnswIndex {
    index: Index,
    config: VectorConfig,
    /// Map from string IDs to usearch u64 keys.
    id_to_key: HashMap<String, u64>,
    /// Reverse map from u64 keys to string IDs.
    key_to_id: HashMap<u64, String>,
    /// Next available key.
    next_key: u64,
    /// Number of ghost entries (removed from usearch but memory not freed).
    /// When this exceeds 20% of live entries, a rebuild is recommended.
    ghost_count: usize,
}

impl HnswIndex {
    /// Create a new HNSW index with the given configuration.
    pub fn new(config: VectorConfig) -> Result<Self, CodememError> {
        let metric = match config.metric {
            codemem_core::DistanceMetric::Cosine => MetricKind::Cos,
            codemem_core::DistanceMetric::L2 => MetricKind::L2sq,
            codemem_core::DistanceMetric::InnerProduct => MetricKind::IP,
        };

        let options = IndexOptions {
            dimensions: config.dimensions,
            metric,
            quantization: ScalarKind::F32,
            connectivity: config.m,
            expansion_add: config.ef_construction,
            expansion_search: config.ef_search,
            multi: false,
        };

        let index = Index::new(&options).map_err(|e| CodememError::Vector(e.to_string()))?;

        // Reserve initial capacity
        index
            .reserve(10_000)
            .map_err(|e| CodememError::Vector(e.to_string()))?;

        Ok(Self {
            index,
            config,
            id_to_key: HashMap::new(),
            key_to_id: HashMap::new(),
            next_key: 0,
            ghost_count: 0,
        })
    }

    /// Create a new index with default configuration (768-dim, cosine).
    pub fn with_defaults() -> Result<Self, CodememError> {
        Self::new(VectorConfig::default())
    }

    /// Get the number of vectors in the index.
    pub fn len(&self) -> usize {
        self.index.size()
    }

    /// Check if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn allocate_key(&mut self) -> u64 {
        let key = self.next_key;
        self.next_key += 1;
        key
    }

    /// Rebuild the index from scratch using the provided entries.
    ///
    /// This eliminates ghost entries left behind by `remove()` (usearch marks
    /// removed keys but does not free their memory). The caller should collect
    /// all live (id, embedding) pairs — typically from SQLite — and pass them
    /// here.
    pub fn rebuild_from_entries(
        &mut self,
        entries: &[(String, Vec<f32>)],
    ) -> Result<(), CodememError> {
        let new_index = Index::new(&IndexOptions {
            dimensions: self.config.dimensions,
            metric: match self.config.metric {
                codemem_core::DistanceMetric::Cosine => MetricKind::Cos,
                codemem_core::DistanceMetric::L2 => MetricKind::L2sq,
                codemem_core::DistanceMetric::InnerProduct => MetricKind::IP,
            },
            quantization: ScalarKind::F32,
            connectivity: self.config.m,
            expansion_add: self.config.ef_construction,
            expansion_search: self.config.ef_search,
            multi: false,
        })
        .map_err(|e| CodememError::Vector(e.to_string()))?;

        let capacity = entries.len().max(1024);
        new_index
            .reserve(capacity)
            .map_err(|e| CodememError::Vector(e.to_string()))?;

        self.index = new_index;
        self.id_to_key.clear();
        self.key_to_id.clear();
        self.next_key = 0;
        self.ghost_count = 0;

        for (id, embedding) in entries {
            self.insert(id, embedding)?;
        }

        Ok(())
    }

    /// Returns true if ghost entries exceed 20% of live entries, suggesting a rebuild.
    pub fn needs_compaction(&self) -> bool {
        let live = self.id_to_key.len();
        live > 0 && self.ghost_count > live / 5
    }

    /// Returns the number of ghost entries in the index.
    pub fn ghost_count(&self) -> usize {
        self.ghost_count
    }

    /// Returns the dimension actually allocated by the underlying usearch index.
    ///
    /// This may differ from `self.config.dimensions` after `load()` if the
    /// persisted index file was created with a different dimension. Callers
    /// performing dimension reconciliation should compare this against the
    /// expected dimension and rebuild on mismatch.
    pub fn actual_dimensions(&self) -> usize {
        self.index.dimensions()
    }
}

impl VectorBackend for HnswIndex {
    fn insert(&mut self, id: &str, embedding: &[f32]) -> Result<(), CodememError> {
        if embedding.len() != self.config.dimensions {
            return Err(CodememError::Vector(format!(
                "Expected {} dimensions, got {}",
                self.config.dimensions,
                embedding.len()
            )));
        }

        // If ID already exists, remove old entry first (creates a ghost in usearch)
        if let Some(&old_key) = self.id_to_key.get(id) {
            self.index
                .remove(old_key)
                .map_err(|e| CodememError::Vector(e.to_string()))?;
            self.key_to_id.remove(&old_key);
            self.ghost_count += 1;
        }

        let key = self.allocate_key();

        // Grow capacity if needed (gradual growth to avoid waste)
        if self.index.size() >= self.index.capacity() {
            let cap = self.index.capacity();
            let new_cap = cap + 1024.max(cap / 4);
            self.index
                .reserve(new_cap)
                .map_err(|e| CodememError::Vector(e.to_string()))?;
        }

        self.index
            .add(key, embedding)
            .map_err(|e| CodememError::Vector(e.to_string()))?;

        self.id_to_key.insert(id.to_string(), key);
        self.key_to_id.insert(key, id.to_string());

        Ok(())
    }

    fn insert_batch(&mut self, items: &[(String, Vec<f32>)]) -> Result<(), CodememError> {
        // Pre-allocate capacity for the entire batch
        let needed = self.index.size() + items.len();
        if needed > self.index.capacity() {
            self.index
                .reserve(needed)
                .map_err(|e| CodememError::Vector(e.to_string()))?;
        }
        for (id, embedding) in items {
            self.insert(id, embedding)?;
        }
        Ok(())
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<(String, f32)>, CodememError> {
        if self.is_empty() {
            return Ok(vec![]);
        }

        let results = self
            .index
            .search(query, k)
            .map_err(|e| CodememError::Vector(e.to_string()))?;

        let mut output = Vec::with_capacity(results.keys.len());
        for (key, distance) in results.keys.iter().zip(results.distances.iter()) {
            if let Some(id) = self.key_to_id.get(key) {
                // Convert cosine distance to similarity: similarity = 1 - distance
                let similarity = 1.0 - distance;
                output.push((id.clone(), similarity));
            }
        }

        Ok(output)
    }

    fn remove(&mut self, id: &str) -> Result<bool, CodememError> {
        if let Some(key) = self.id_to_key.remove(id) {
            self.index
                .remove(key)
                .map_err(|e| CodememError::Vector(e.to_string()))?;
            self.key_to_id.remove(&key);
            self.ghost_count += 1;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn save(&self, path: &Path) -> Result<(), CodememError> {
        let path_str = path
            .to_str()
            .ok_or_else(|| CodememError::Vector("Path contains non-UTF-8 characters".into()))?;

        let idmap_path = path.with_extension("idmap");

        // Serialize ID mappings
        let map_data = serde_json::to_string(&IdMapping {
            id_to_key: &self.id_to_key,
            next_key: self.next_key,
        })
        .map_err(|e| CodememError::Vector(e.to_string()))?;

        // Write both to temp files first, then rename atomically
        let tmp_idmap = path.with_extension("idmap.tmp");
        std::fs::write(&tmp_idmap, map_data)?;

        let tmp_idx = path.with_extension("idx.tmp");
        let tmp_idx_str = tmp_idx.to_str().ok_or_else(|| {
            CodememError::Vector("Temp path contains non-UTF-8 characters".into())
        })?;
        self.index
            .save(tmp_idx_str)
            .map_err(|e| CodememError::Vector(e.to_string()))?;

        // Atomic renames
        std::fs::rename(&tmp_idmap, &idmap_path)?;
        std::fs::rename(&tmp_idx, path_str)?;

        Ok(())
    }

    fn load(&mut self, path: &Path) -> Result<(), CodememError> {
        let path_str = path
            .to_str()
            .ok_or_else(|| CodememError::Vector("Path contains non-UTF-8 characters".into()))?;
        self.index
            .load(path_str)
            .map_err(|e| CodememError::Vector(e.to_string()))?;

        // Load ID mappings
        let map_path = path.with_extension("idmap");
        if map_path.exists() {
            let map_data = std::fs::read_to_string(map_path)?;
            let mapping: IdMappingOwned =
                serde_json::from_str(&map_data).map_err(|e| CodememError::Vector(e.to_string()))?;

            self.id_to_key = mapping.id_to_key;
            self.key_to_id = self
                .id_to_key
                .iter()
                .map(|(id, key)| (*key, id.clone()))
                .collect();
            self.next_key = mapping.next_key;
            self.ghost_count = 0; // Fresh load has no ghosts
        }

        Ok(())
    }

    fn stats(&self) -> VectorStats {
        VectorStats {
            count: self.len(),
            dimensions: self.config.dimensions,
            metric: format!("{:?}", self.config.metric),
            memory_bytes: self.index.memory_usage(),
        }
    }

    fn needs_compaction(&self) -> bool {
        // Ghost entries exceed 20% of live entries
        let live = self.id_to_key.len();
        live > 0 && self.ghost_count > live / 5
    }

    fn ghost_count(&self) -> usize {
        self.ghost_count
    }

    fn rebuild_from_entries(&mut self, entries: &[(String, Vec<f32>)]) -> Result<(), CodememError> {
        HnswIndex::rebuild_from_entries(self, entries)
    }
}

use serde::{Deserialize, Serialize};

#[derive(Serialize)]
struct IdMapping<'a> {
    id_to_key: &'a HashMap<String, u64>,
    next_key: u64,
}

#[derive(Deserialize)]
struct IdMappingOwned {
    id_to_key: HashMap<String, u64>,
    next_key: u64,
}

/// Cosine similarity between two embedding vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let x = *x as f64;
        let y = *y as f64;
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < 1e-12 {
        0.0
    } else {
        dot / denom
    }
}

#[cfg(test)]
#[path = "tests/vector_tests.rs"]
mod tests;
