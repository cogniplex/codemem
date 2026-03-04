//! BM25 scoring module for code-aware text ranking.
//!
//! Implements Okapi BM25 with a code-aware tokenizer that handles camelCase,
//! snake_case, and other programming conventions. Replaces the naive
//! split+intersect token overlap in hybrid scoring.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

// ── Code-Aware Tokenizer ────────────────────────────────────────────────────

/// Tokenize text with code-awareness: splits camelCase, snake_case,
/// punctuation boundaries, lowercases, and filters short tokens.
pub fn tokenize(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();

    // First split on whitespace, then process each word
    for word in text.split_whitespace() {
        // Split on punctuation and non-alphanumeric chars (but keep segments)
        let segments = split_on_punctuation(word);
        for segment in segments {
            // Split camelCase and PascalCase
            let sub_tokens = split_camel_case(&segment);
            for token in sub_tokens {
                let lower = token.to_lowercase();
                if lower.len() >= 2 {
                    tokens.push(lower);
                }
            }
        }
    }

    tokens
}

/// Split a string on punctuation and non-alphanumeric characters.
/// Keeps alphanumeric segments, discards separators.
fn split_on_punctuation(s: &str) -> Vec<String> {
    let mut segments = Vec::new();
    let mut current = String::new();

    for ch in s.chars() {
        if ch.is_alphanumeric() {
            current.push(ch);
        } else {
            // Separator character (underscore, dot, dash, etc.)
            if !current.is_empty() {
                segments.push(std::mem::take(&mut current));
            }
        }
    }
    if !current.is_empty() {
        segments.push(current);
    }

    segments
}

/// Split a camelCase or PascalCase string into its components.
/// "processRequest" -> ["process", "Request"]
/// "HTMLParser" -> ["HTML", "Parser"]
/// "getHTTPResponse" -> ["get", "HTTP", "Response"]
fn split_camel_case(s: &str) -> Vec<String> {
    if s.is_empty() {
        return vec![];
    }

    let chars: Vec<char> = s.chars().collect();
    let mut parts = Vec::new();
    let mut start = 0;

    for i in 1..chars.len() {
        let prev = chars[i - 1];
        let curr = chars[i];

        // Split at lowercase -> uppercase boundary: "processRequest"
        let lower_to_upper = prev.is_lowercase() && curr.is_uppercase();

        // Split at uppercase run -> uppercase+lowercase boundary: "HTMLParser" -> "HTML" | "Parser"
        let upper_run_end =
            i >= 2 && chars[i - 2].is_uppercase() && prev.is_uppercase() && curr.is_lowercase();

        // Split at digit boundaries: "item2count" -> "item", "2", "count"
        let digit_boundary = (prev.is_alphabetic() && curr.is_ascii_digit())
            || (prev.is_ascii_digit() && curr.is_alphabetic());

        if lower_to_upper || upper_run_end || digit_boundary {
            let split_at = if upper_run_end { i - 1 } else { i };
            if split_at > start {
                let part: String = chars[start..split_at].iter().collect();
                parts.push(part);
                start = split_at;
            }
        }
    }

    // Push remaining
    if start < chars.len() {
        let part: String = chars[start..].iter().collect();
        parts.push(part);
    }

    parts
}

// ── BM25 Index ──────────────────────────────────────────────────────────────

/// BM25 index for scoring query-document relevance.
///
/// Maintains document frequency statistics incrementally. Documents are
/// identified by string IDs and can be added/removed dynamically.
///
/// Supports serialization via `serialize()`/`deserialize()` for persistence
/// across restarts, avoiding the need to rebuild from all stored memories.
#[derive(Serialize, Deserialize)]
pub struct Bm25Index {
    /// Number of documents containing each term: term -> count
    doc_freq: HashMap<String, usize>,
    /// Document lengths (in tokens): doc_id -> length
    doc_lengths: HashMap<String, usize>,
    /// Per-document term frequencies for removal support: doc_id -> (term -> count)
    doc_terms: HashMap<String, HashMap<String, usize>>,
    /// Total number of documents
    pub doc_count: usize,
    /// Average document length
    avg_doc_len: f64,
    /// Running total of all document lengths for incremental avg_doc_len tracking
    #[serde(default)]
    total_doc_len: usize,
    /// BM25 k1 parameter (term frequency saturation)
    k1: f64,
    /// BM25 b parameter (length normalization)
    b: f64,
    /// Maximum number of documents before eviction kicks in
    #[serde(default = "default_max_documents")]
    max_documents: usize,
    /// Insertion order for FIFO eviction when max_documents is exceeded
    #[serde(default)]
    insertion_order: VecDeque<String>,
}

fn default_max_documents() -> usize {
    100_000
}

impl Bm25Index {
    /// Create a new empty BM25 index with default parameters (k1=1.2, b=0.75).
    pub fn new() -> Self {
        Self {
            doc_freq: HashMap::new(),
            doc_lengths: HashMap::new(),
            doc_terms: HashMap::new(),
            doc_count: 0,
            avg_doc_len: 0.0,
            total_doc_len: 0,
            k1: 1.2,
            b: 0.75,
            max_documents: default_max_documents(),
            insertion_order: VecDeque::new(),
        }
    }

    /// Add a document to the index, updating term frequencies and statistics.
    /// If a document with the same ID already exists, it is replaced.
    /// When the index exceeds `max_documents`, the oldest document is evicted.
    pub fn add_document(&mut self, id: &str, content: &str) {
        // Remove old version if exists (to avoid double-counting)
        if self.doc_terms.contains_key(id) {
            self.remove_document(id);
        }

        // Evict oldest document if at capacity
        if self.doc_count >= self.max_documents {
            if let Some(oldest_id) = self.insertion_order.pop_front() {
                self.remove_document(&oldest_id);
            }
        }

        let tokens = tokenize(content);
        let doc_len = tokens.len();

        // Count term frequencies for this document
        let mut term_freqs: HashMap<String, usize> = HashMap::new();
        for token in &tokens {
            *term_freqs.entry(token.clone()).or_insert(0) += 1;
        }

        // Update global document frequency (each unique term in this doc)
        for term in term_freqs.keys() {
            *self.doc_freq.entry(term.clone()).or_insert(0) += 1;
        }

        // Store document info
        self.doc_lengths.insert(id.to_string(), doc_len);
        self.doc_terms.insert(id.to_string(), term_freqs);
        self.doc_count += 1;

        // Track insertion order for FIFO eviction
        self.insertion_order.push_back(id.to_string());

        // Incrementally update average document length
        self.total_doc_len += doc_len;
        self.avg_doc_len = self.total_doc_len as f64 / self.doc_count as f64;
    }

    /// Remove a document from the index, updating all statistics.
    pub fn remove_document(&mut self, id: &str) {
        if let Some(term_freqs) = self.doc_terms.remove(id) {
            // Decrement document frequency for each term
            for term in term_freqs.keys() {
                if let Some(df) = self.doc_freq.get_mut(term) {
                    *df = df.saturating_sub(1);
                    if *df == 0 {
                        self.doc_freq.remove(term);
                    }
                }
            }

            let removed_len = self.doc_lengths.remove(id).unwrap_or(0);
            self.doc_count = self.doc_count.saturating_sub(1);

            // Incrementally update average document length
            self.total_doc_len = self.total_doc_len.saturating_sub(removed_len);
            if self.doc_count == 0 {
                self.avg_doc_len = 0.0;
            } else {
                self.avg_doc_len = self.total_doc_len as f64 / self.doc_count as f64;
            }

            // Remove from insertion order
            self.insertion_order.retain(|x| x != id);
        }
    }

    /// Score a query against a specific document using BM25.
    ///
    /// The score is computed as:
    /// ```text
    /// score(q, d) = Σ IDF(qi) * (f(qi,d) * (k1+1)) / (f(qi,d) + k1*(1 - b + b*|d|/avgdl))
    /// IDF(qi) = ln((N - n(qi) + 0.5) / (n(qi) + 0.5) + 1)
    /// ```
    ///
    /// The returned score is normalized to [0, 1] by dividing by the maximum
    /// possible score (perfect self-match with all query terms).
    pub fn score(&self, query: &str, doc_id: &str) -> f64 {
        if self.doc_count == 0 {
            return 0.0;
        }

        let query_tokens = tokenize(query);
        if query_tokens.is_empty() {
            return 0.0;
        }

        let doc_len = match self.doc_lengths.get(doc_id) {
            Some(&len) => len,
            None => return 0.0,
        };

        let doc_term_freqs = match self.doc_terms.get(doc_id) {
            Some(tf) => tf,
            None => return 0.0,
        };

        let raw = self.raw_bm25_score(&query_tokens, doc_term_freqs, doc_len);

        // Normalize: compute max possible score (if every query term appeared
        // with high frequency in a short document).
        let max_score = self.max_possible_score(&query_tokens);
        if max_score <= 0.0 {
            return 0.0;
        }

        (raw / max_score).min(1.0)
    }

    /// Score pre-tokenized query tokens (as `&str` slices) against a specific indexed document.
    /// Use this when scoring multiple documents against the same query to avoid
    /// re-tokenizing the query each time. Zero-allocation: passes slices directly
    /// to the generic internal helpers.
    pub fn score_with_tokens_str(&self, query_tokens: &[&str], doc_id: &str) -> f64 {
        if self.doc_count == 0 || query_tokens.is_empty() {
            return 0.0;
        }

        let doc_len = match self.doc_lengths.get(doc_id) {
            Some(&len) => len,
            None => return 0.0,
        };

        let doc_term_freqs = match self.doc_terms.get(doc_id) {
            Some(tf) => tf,
            None => return 0.0,
        };

        let raw = self.raw_bm25_score(query_tokens, doc_term_freqs, doc_len);

        let max_score = self.max_possible_score(query_tokens);
        if max_score <= 0.0 {
            return 0.0;
        }

        (raw / max_score).min(1.0)
    }

    /// Score pre-tokenized query tokens against arbitrary text (not necessarily in the index).
    /// Use this when scoring multiple documents against the same query to avoid
    /// re-tokenizing the query each time.
    pub fn score_text_with_tokens(&self, query_tokens: &[String], document: &str) -> f64 {
        if self.doc_count == 0 || query_tokens.is_empty() {
            return 0.0;
        }

        let doc_tokens = tokenize(document);
        if doc_tokens.is_empty() {
            return 0.0;
        }

        let doc_len = doc_tokens.len();

        // Build term frequencies for this document
        let mut doc_term_freqs: HashMap<String, usize> = HashMap::new();
        for token in &doc_tokens {
            *doc_term_freqs.entry(token.clone()).or_insert(0) += 1;
        }

        let raw = self.raw_bm25_score(query_tokens, &doc_term_freqs, doc_len);

        let max_score = self.max_possible_score(query_tokens);
        if max_score <= 0.0 {
            return 0.0;
        }

        (raw / max_score).min(1.0)
    }

    /// Score pre-tokenized query tokens (as `&str` slices) against arbitrary text.
    /// Use this when scoring multiple documents against the same query to avoid
    /// re-tokenizing the query each time. Zero-allocation: generic helpers accept `&[&str]`.
    pub fn score_text_with_tokens_str(&self, query_tokens: &[&str], document: &str) -> f64 {
        if self.doc_count == 0 || query_tokens.is_empty() {
            return 0.0;
        }

        let doc_tokens = tokenize(document);
        if doc_tokens.is_empty() {
            return 0.0;
        }

        let doc_len = doc_tokens.len();
        let mut doc_term_freqs: HashMap<String, usize> = HashMap::new();
        for token in &doc_tokens {
            *doc_term_freqs.entry(token.clone()).or_insert(0) += 1;
        }

        let raw = self.raw_bm25_score(query_tokens, &doc_term_freqs, doc_len);
        let max_score = self.max_possible_score(query_tokens);
        if max_score <= 0.0 {
            return 0.0;
        }

        (raw / max_score).min(1.0)
    }

    /// Score a query against arbitrary text (not necessarily in the index).
    /// Useful for scoring documents that haven't been indexed yet.
    pub fn score_text(&self, query: &str, document: &str) -> f64 {
        let query_tokens = tokenize(query);
        self.score_text_with_tokens(&query_tokens, document)
    }

    /// Build a BM25 index from a slice of (id, content) pairs.
    pub fn build(documents: &[(String, String)]) -> Self {
        let mut index = Self::new();
        for (id, content) in documents {
            index.add_document(id, content);
        }
        index
    }

    // ── Internal helpers ────────────────────────────────────────────────

    /// Compute the raw (unnormalized) BM25 score.
    /// Generic over `AsRef<str>` so both `&[String]` and `&[&str]` work
    /// without per-call allocation.
    fn raw_bm25_score<S: AsRef<str>>(
        &self,
        query_tokens: &[S],
        doc_term_freqs: &HashMap<String, usize>,
        doc_len: usize,
    ) -> f64 {
        let n = self.doc_count as f64;
        let avgdl = if self.avg_doc_len > 0.0 {
            self.avg_doc_len
        } else {
            1.0
        };
        let dl = doc_len as f64;

        let mut score = 0.0;

        // De-duplicate query tokens for scoring (each unique term scored once)
        let mut seen_query_terms: HashSet<&str> = HashSet::new();

        for qt in query_tokens {
            let qt_str = qt.as_ref();
            if !seen_query_terms.insert(qt_str) {
                continue;
            }

            // Term frequency in document
            let tf = *doc_term_freqs.get(qt_str).unwrap_or(&0) as f64;
            if tf == 0.0 {
                continue;
            }

            // Document frequency (number of docs containing this term)
            let df = *self.doc_freq.get(qt_str).unwrap_or(&0) as f64;

            // IDF: ln((N - n(qi) + 0.5) / (n(qi) + 0.5) + 1)
            let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();

            // BM25 term score
            let numerator = tf * (self.k1 + 1.0);
            let denominator = tf + self.k1 * (1.0 - self.b + self.b * dl / avgdl);

            score += idf * numerator / denominator;
        }

        score
    }

    /// Compute the maximum possible BM25 score for a query (for normalization).
    /// Assumes a perfect document that contains every query term with high tf
    /// and has average length.
    fn max_possible_score<S: AsRef<str>>(&self, query_tokens: &[S]) -> f64 {
        let n = self.doc_count as f64;

        let mut max_score = 0.0;
        let mut seen: HashSet<&str> = HashSet::new();

        for qt in query_tokens {
            let qt_str = qt.as_ref();
            if !seen.insert(qt_str) {
                continue;
            }

            let df = *self.doc_freq.get(qt_str).unwrap_or(&0) as f64;
            let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();

            // Best case: high tf in average-length document
            // With tf large, the term score approaches idf * (k1 + 1) / 1 = idf * (k1 + 1)
            // But for normalization we use tf=10 in avg-length doc for a realistic ceiling.
            let tf = 10.0_f64;
            let numerator = tf * (self.k1 + 1.0);
            let denominator = tf + self.k1; // b * (dl/avgdl) = b * 1.0 when dl == avgdl
            max_score += idf * numerator / denominator;
        }

        max_score
    }

    /// Recompute average document length and total_doc_len from stored lengths.
    /// Public for use after deserialization to reconstruct the running total.
    pub fn recompute_avg_doc_len(&mut self) {
        let total: usize = self.doc_lengths.values().sum();
        self.total_doc_len = total;
        if self.doc_count == 0 {
            self.avg_doc_len = 0.0;
        } else {
            self.avg_doc_len = total as f64 / self.doc_count as f64;
        }
    }
}

impl Bm25Index {
    /// Serialize the BM25 index to a byte vector for persistence.
    ///
    /// Uses bincode for compact binary representation. The serialized format
    /// includes all document frequency statistics, term frequencies, and
    /// parameters, enabling fast startup without re-indexing all memories.
    pub fn serialize(&self) -> Vec<u8> {
        // Use JSON for reliable serialization (bincode not in deps)
        serde_json::to_vec(self).unwrap_or_default()
    }

    /// Deserialize a BM25 index from bytes previously produced by `serialize()`.
    ///
    /// Returns `Err` if the data is corrupt or from an incompatible version.
    /// Reconstructs `total_doc_len` and `insertion_order` if they were missing
    /// from older serialized data (via `#[serde(default)]`).
    pub fn deserialize(data: &[u8]) -> Result<Self, String> {
        let mut index: Self = serde_json::from_slice(data)
            .map_err(|e| format!("BM25 deserialization failed: {e}"))?;
        // Reconstruct total_doc_len from stored doc_lengths (handles old data without the field)
        index.recompute_avg_doc_len();
        // Rebuild insertion_order if it was empty (old data without the field)
        if index.insertion_order.is_empty() && index.doc_count > 0 {
            index.insertion_order = index.doc_lengths.keys().cloned().collect();
        }
        Ok(index)
    }

    /// Whether the index contains any documents and may need saving.
    ///
    /// Useful for batch operations: call `persist_memory_no_save()` in a loop,
    /// then check `needs_save()` before writing the vector index to disk once
    /// at the end. This avoids O(N) disk writes during bulk inserts.
    pub fn needs_save(&self) -> bool {
        self.doc_count > 0
    }
}

impl Default for Bm25Index {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[path = "tests/bm25_tests.rs"]
mod tests;
