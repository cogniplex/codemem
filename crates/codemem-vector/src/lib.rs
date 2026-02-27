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

        // If ID already exists, remove old entry first
        if let Some(&old_key) = self.id_to_key.get(id) {
            self.index
                .remove(old_key)
                .map_err(|e| CodememError::Vector(e.to_string()))?;
            self.key_to_id.remove(&old_key);
        }

        let key = self.allocate_key();

        // Grow capacity if needed
        if self.index.size() >= self.index.capacity() {
            let new_cap = self.index.capacity() * 2;
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
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn save(&self, path: &Path) -> Result<(), CodememError> {
        self.index
            .save(path.to_str().unwrap_or("hnsw.index"))
            .map_err(|e| CodememError::Vector(e.to_string()))?;

        // Save ID mappings alongside
        let map_path = path.with_extension("idmap");
        let map_data = serde_json::to_string(&IdMapping {
            id_to_key: &self.id_to_key,
            next_key: self.next_key,
        })
        .map_err(|e| CodememError::Vector(e.to_string()))?;

        std::fs::write(map_path, map_data)?;
        Ok(())
    }

    fn load(&mut self, path: &Path) -> Result<(), CodememError> {
        self.index
            .load(path.to_str().unwrap_or("hnsw.index"))
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

#[cfg(test)]
mod tests {
    use super::*;

    fn random_vector(dim: usize) -> Vec<f32> {
        (0..dim)
            .map(|i| ((i * 7 + 3) % 100) as f32 / 100.0)
            .collect()
    }

    #[test]
    fn insert_and_search() {
        let mut index = HnswIndex::with_defaults().unwrap();
        let v1 = random_vector(768);
        index.insert("mem-1", &v1).unwrap();

        let results = index.search(&v1, 1).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "mem-1");
        assert!(results[0].1 > 0.99); // self-similarity should be ~1.0
    }

    #[test]
    fn remove_vector() {
        let mut index = HnswIndex::with_defaults().unwrap();
        let v1 = random_vector(768);
        index.insert("mem-1", &v1).unwrap();
        assert!(index.remove("mem-1").unwrap());
        assert!(!index.remove("mem-1").unwrap()); // already removed
    }

    #[test]
    fn dimension_mismatch() {
        let mut index = HnswIndex::with_defaults().unwrap();
        let bad = vec![1.0f32; 128]; // wrong dimensions
        assert!(index.insert("bad", &bad).is_err());
    }

    #[test]
    fn stats() {
        let index = HnswIndex::with_defaults().unwrap();
        let stats = index.stats();
        assert_eq!(stats.count, 0);
        assert_eq!(stats.dimensions, 768);
    }
}
