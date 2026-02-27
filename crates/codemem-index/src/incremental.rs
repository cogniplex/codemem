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
    pub fn load_from_storage(&mut self, storage: &codemem_storage::Storage) {
        // Try to read from a file_hashes table. If it doesn't exist, that's fine.
        let conn = storage.connection();
        let result = conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS file_hashes (
                file_path TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL,
                indexed_at INTEGER NOT NULL
            )",
        );
        if result.is_err() {
            tracing::warn!("Failed to create file_hashes table, starting fresh");
            return;
        }

        let mut stmt = match conn.prepare("SELECT file_path, content_hash FROM file_hashes") {
            Ok(s) => s,
            Err(_) => return,
        };

        let rows = match stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        }) {
            Ok(r) => r,
            Err(_) => return,
        };

        for row in rows.flatten() {
            self.known_hashes.insert(row.0, row.1);
        }

        tracing::debug!("Loaded {} known file hashes", self.known_hashes.len());
    }

    /// Save current hashes to storage.
    pub fn save_to_storage(
        &self,
        storage: &codemem_storage::Storage,
    ) -> Result<(), codemem_core::CodememError> {
        let conn = storage.connection();
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS file_hashes (
                file_path TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL,
                indexed_at INTEGER NOT NULL
            )",
        )
        .map_err(|e| codemem_core::CodememError::Storage(e.to_string()))?;

        let now = chrono::Utc::now().timestamp();
        let mut stmt = conn
            .prepare("INSERT OR REPLACE INTO file_hashes (file_path, content_hash, indexed_at) VALUES (?1, ?2, ?3)")
            .map_err(|e| codemem_core::CodememError::Storage(e.to_string()))?;

        for (path, hash) in &self.known_hashes {
            stmt.execute(rusqlite::params![path, hash, now])
                .map_err(|e| codemem_core::CodememError::Storage(e.to_string()))?;
        }

        Ok(())
    }

    /// Check if a file has changed since the last index.
    /// Returns `true` if the file is new or its content hash differs.
    pub fn is_changed(&self, path: &str, content: &[u8]) -> bool {
        let hash = Self::hash_content(content);
        self.known_hashes.get(path) != Some(&hash)
    }

    /// Update the stored hash for a file after successful indexing.
    pub fn update_hash(&mut self, path: &str, content: &[u8]) {
        let hash = Self::hash_content(content);
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
mod tests {
    use super::*;

    #[test]
    fn new_file_is_changed() {
        let detector = ChangeDetector::new();
        assert!(detector.is_changed("foo.rs", b"fn main() {}"));
    }

    #[test]
    fn same_content_not_changed() {
        let mut detector = ChangeDetector::new();
        let content = b"fn main() {}";
        detector.update_hash("foo.rs", content);
        assert!(!detector.is_changed("foo.rs", content));
    }

    #[test]
    fn different_content_is_changed() {
        let mut detector = ChangeDetector::new();
        detector.update_hash("foo.rs", b"fn main() {}");
        assert!(detector.is_changed("foo.rs", b"fn main() { println!(\"hi\"); }"));
    }

    #[test]
    fn remove_hash_makes_changed() {
        let mut detector = ChangeDetector::new();
        detector.update_hash("foo.rs", b"content");
        assert!(!detector.is_changed("foo.rs", b"content"));
        detector.remove_hash("foo.rs");
        assert!(detector.is_changed("foo.rs", b"content"));
    }

    #[test]
    fn tracked_count() {
        let mut detector = ChangeDetector::new();
        assert_eq!(detector.tracked_count(), 0);
        detector.update_hash("a.rs", b"a");
        detector.update_hash("b.rs", b"b");
        assert_eq!(detector.tracked_count(), 2);
    }
}
