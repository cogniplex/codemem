//! SHA-256 based change detection for incremental indexing.
//!
//! Tracks content hashes of previously indexed files so unchanged files
//! can be skipped on subsequent indexing runs.

use sha2::{Digest, Sha256};
use std::collections::HashMap;

/// Tracks file content hashes for incremental change detection.
pub struct ChangeDetector {
    /// Map of file_path -> SHA-256 hex hash from last index run.
    known_hashes: HashMap<String, String>,
}

impl ChangeDetector {
    /// Create a new empty ChangeDetector.
    pub fn new() -> Self {
        Self {
            known_hashes: HashMap::new(),
        }
    }

    /// Load previously known hashes from storage.
    ///
    /// This reads from the `file_hashes` table if it exists. If the table
    /// doesn't exist or the query fails, starts fresh with no known hashes.
    pub fn load_from_storage(&mut self, storage: &dyn codemem_core::StorageBackend) {
        match storage.load_file_hashes() {
            Ok(hashes) => {
                tracing::debug!("Loaded {} known file hashes", hashes.len());
                self.known_hashes = hashes;
            }
            Err(e) => {
                tracing::warn!("Failed to load file hashes, starting fresh: {e}");
            }
        }
    }

    /// Save current hashes to storage.
    pub fn save_to_storage(
        &self,
        storage: &dyn codemem_core::StorageBackend,
    ) -> Result<(), codemem_core::CodememError> {
        storage.save_file_hashes(&self.known_hashes)
    }

    /// Check if a file has changed since the last index.
    /// Returns `true` if the file is new or its content hash differs.
    pub fn is_changed(&self, path: &str, content: &[u8]) -> bool {
        let hash = Self::hash_content(content);
        self.known_hashes.get(path) != Some(&hash)
    }

    /// Check if a file has changed and return (changed, hash) to avoid double-hashing.
    /// Callers can pass the returned hash to `record_hash` to skip recomputation.
    pub fn check_changed(&self, path: &str, content: &[u8]) -> (bool, String) {
        let hash = Self::hash_content(content);
        let changed = self.known_hashes.get(path) != Some(&hash);
        (changed, hash)
    }

    /// Update the stored hash for a file after successful indexing.
    pub fn update_hash(&mut self, path: &str, content: &[u8]) {
        let hash = Self::hash_content(content);
        self.known_hashes.insert(path.to_string(), hash);
    }

    /// Record a pre-computed hash for a file (avoids re-hashing when used with `check_changed`).
    pub fn record_hash(&mut self, path: &str, hash: String) {
        self.known_hashes.insert(path.to_string(), hash);
    }

    /// Remove the hash for a file (e.g., when it's deleted).
    pub fn remove_hash(&mut self, path: &str) {
        self.known_hashes.remove(path);
    }

    /// Get the number of tracked files.
    pub fn tracked_count(&self) -> usize {
        self.known_hashes.len()
    }

    /// Compute SHA-256 hash of content bytes.
    fn hash_content(content: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(content);
        format!("{:x}", hasher.finalize())
    }
}

impl Default for ChangeDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[path = "tests/incremental_tests.rs"]
mod tests;
