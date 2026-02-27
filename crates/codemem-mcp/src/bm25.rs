//! BM25 scoring module for code-aware text ranking.
//!
//! Implements Okapi BM25 with a code-aware tokenizer that handles camelCase,
//! snake_case, and other programming conventions. Replaces the naive
//! split+intersect token overlap in hybrid scoring.

use std::collections::HashMap;

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
pub struct Bm25Index {
    /// Number of documents containing each term: term -> count
    doc_freq: HashMap<String, usize>,
    /// Document lengths (in tokens): doc_id -> length
    doc_lengths: HashMap<String, usize>,
    /// Per-document term frequencies for removal support: doc_id -> (term -> count)
    doc_terms: HashMap<String, HashMap<String, usize>>,
    /// Total number of documents
    doc_count: usize,
    /// Average document length
    avg_doc_len: f64,
    /// BM25 k1 parameter (term frequency saturation)
    k1: f64,
    /// BM25 b parameter (length normalization)
    b: f64,
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
            k1: 1.2,
            b: 0.75,
        }
    }

    /// Add a document to the index, updating term frequencies and statistics.
    /// If a document with the same ID already exists, it is replaced.
    pub fn add_document(&mut self, id: &str, content: &str) {
        // Remove old version if exists (to avoid double-counting)
        if self.doc_terms.contains_key(id) {
            self.remove_document(id);
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

        // Recompute average document length
        self.recompute_avg_doc_len();
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

            self.doc_lengths.remove(id);
            self.doc_count = self.doc_count.saturating_sub(1);
            self.recompute_avg_doc_len();
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

    /// Score a query against arbitrary text (not necessarily in the index).
    /// Useful for scoring documents that haven't been indexed yet.
    pub fn score_text(&self, query: &str, document: &str) -> f64 {
        if self.doc_count == 0 {
            return 0.0;
        }

        let query_tokens = tokenize(query);
        if query_tokens.is_empty() {
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

        let raw = self.raw_bm25_score(&query_tokens, &doc_term_freqs, doc_len);

        let max_score = self.max_possible_score(&query_tokens);
        if max_score <= 0.0 {
            return 0.0;
        }

        (raw / max_score).min(1.0)
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
    fn raw_bm25_score(
        &self,
        query_tokens: &[String],
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
        let mut seen_query_terms: HashMap<&str, bool> = HashMap::new();

        for qt in query_tokens {
            if seen_query_terms.contains_key(qt.as_str()) {
                continue;
            }
            seen_query_terms.insert(qt.as_str(), true);

            // Term frequency in document
            let tf = *doc_term_freqs.get(qt).unwrap_or(&0) as f64;
            if tf == 0.0 {
                continue;
            }

            // Document frequency (number of docs containing this term)
            let df = *self.doc_freq.get(qt).unwrap_or(&0) as f64;

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
    fn max_possible_score(&self, query_tokens: &[String]) -> f64 {
        let n = self.doc_count as f64;

        let mut max_score = 0.0;
        let mut seen: HashMap<&str, bool> = HashMap::new();

        for qt in query_tokens {
            if seen.contains_key(qt.as_str()) {
                continue;
            }
            seen.insert(qt.as_str(), true);

            let df = *self.doc_freq.get(qt).unwrap_or(&0) as f64;
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

    /// Recompute average document length from stored lengths.
    fn recompute_avg_doc_len(&mut self) {
        if self.doc_count == 0 {
            self.avg_doc_len = 0.0;
        } else {
            let total: usize = self.doc_lengths.values().sum();
            self.avg_doc_len = total as f64 / self.doc_count as f64;
        }
    }
}

impl Default for Bm25Index {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Tokenizer tests ─────────────────────────────────────────────────

    #[test]
    fn tokenize_camel_case() {
        let tokens = tokenize("processRequest");
        assert!(tokens.contains(&"process".to_string()));
        assert!(tokens.contains(&"request".to_string()));
    }

    #[test]
    fn tokenize_pascal_case() {
        let tokens = tokenize("ProcessRequest");
        assert!(tokens.contains(&"process".to_string()));
        assert!(tokens.contains(&"request".to_string()));
    }

    #[test]
    fn tokenize_snake_case() {
        let tokens = tokenize("process_request");
        assert!(tokens.contains(&"process".to_string()));
        assert!(tokens.contains(&"request".to_string()));
    }

    #[test]
    fn tokenize_mixed_case_acronym() {
        let tokens = tokenize("getHTTPResponse");
        assert!(tokens.contains(&"get".to_string()));
        assert!(tokens.contains(&"http".to_string()));
        assert!(tokens.contains(&"response".to_string()));
    }

    #[test]
    fn tokenize_filters_short_tokens() {
        let tokens = tokenize("a b cd ef");
        // "a" and "b" should be filtered (< 2 chars)
        assert!(!tokens.contains(&"a".to_string()));
        assert!(!tokens.contains(&"b".to_string()));
        assert!(tokens.contains(&"cd".to_string()));
        assert!(tokens.contains(&"ef".to_string()));
    }

    #[test]
    fn tokenize_lowercases() {
        let tokens = tokenize("HELLO World");
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
    }

    #[test]
    fn tokenize_punctuation_splitting() {
        let tokens = tokenize("foo.bar::baz-qux");
        assert!(tokens.contains(&"foo".to_string()));
        assert!(tokens.contains(&"bar".to_string()));
        assert!(tokens.contains(&"baz".to_string()));
        assert!(tokens.contains(&"qux".to_string()));
    }

    #[test]
    fn tokenize_digit_boundaries() {
        let tokens = tokenize("item2count");
        assert!(tokens.contains(&"item".to_string()));
        assert!(tokens.contains(&"count".to_string()));
    }

    #[test]
    fn tokenize_empty_string() {
        let tokens = tokenize("");
        assert!(tokens.is_empty());
    }

    #[test]
    fn tokenize_code_content() {
        let tokens = tokenize("fn computeScore(memory: &MemoryNode) -> f64");
        assert!(tokens.contains(&"fn".to_string()));
        assert!(tokens.contains(&"compute".to_string()));
        assert!(tokens.contains(&"score".to_string()));
        assert!(tokens.contains(&"memory".to_string()));
        assert!(tokens.contains(&"node".to_string()));
        // "f64" splits at digit boundary into "f" (filtered: <2 chars) and "64"
        assert!(tokens.contains(&"64".to_string()));
    }

    // ── BM25 scoring tests ──────────────────────────────────────────────

    #[test]
    fn bm25_relevant_doc_scores_higher() {
        let docs = vec![
            (
                "d1".to_string(),
                "rust ownership and borrowing rules".to_string(),
            ),
            (
                "d2".to_string(),
                "python garbage collection internals".to_string(),
            ),
        ];
        let index = Bm25Index::build(&docs);

        let score_d1 = index.score("rust ownership", "d1");
        let score_d2 = index.score("rust ownership", "d2");

        assert!(
            score_d1 > score_d2,
            "relevant doc should score higher: d1={score_d1}, d2={score_d2}"
        );
    }

    #[test]
    fn bm25_idf_rare_terms_score_higher() {
        // "quantum" appears in 1 doc, "the" appears in all 3
        let docs = vec![
            ("d1".to_string(), "the quick brown fox".to_string()),
            ("d2".to_string(), "the lazy dog jumps".to_string()),
            (
                "d3".to_string(),
                "the quantum computing revolution".to_string(),
            ),
        ];
        let index = Bm25Index::build(&docs);

        let score_common = index.score("the", "d1");
        let score_rare = index.score("quantum", "d3");

        assert!(
            score_rare > score_common,
            "rare term should score higher: rare={score_rare}, common={score_common}"
        );
    }

    #[test]
    fn bm25_empty_query_returns_zero() {
        let docs = vec![("d1".to_string(), "some content here".to_string())];
        let index = Bm25Index::build(&docs);

        assert_eq!(index.score("", "d1"), 0.0);
    }

    #[test]
    fn bm25_empty_index_returns_zero() {
        let index = Bm25Index::new();
        assert_eq!(index.score("test query", "nonexistent"), 0.0);
    }

    #[test]
    fn bm25_unknown_doc_returns_zero() {
        let docs = vec![("d1".to_string(), "some content".to_string())];
        let index = Bm25Index::build(&docs);
        assert_eq!(index.score("content", "nonexistent"), 0.0);
    }

    #[test]
    fn bm25_no_matching_terms_returns_zero() {
        let docs = vec![("d1".to_string(), "alpha beta gamma".to_string())];
        let index = Bm25Index::build(&docs);
        assert_eq!(index.score("delta epsilon", "d1"), 0.0);
    }

    #[test]
    fn bm25_score_in_zero_one_range() {
        let docs = vec![
            (
                "d1".to_string(),
                "rust memory safety and ownership".to_string(),
            ),
            ("d2".to_string(), "python dynamic typing system".to_string()),
        ];
        let index = Bm25Index::build(&docs);

        let score = index.score("rust memory", "d1");
        assert!(score >= 0.0, "score should be >= 0: {score}");
        assert!(score <= 1.0, "score should be <= 1: {score}");
    }

    #[test]
    fn bm25_incremental_add() {
        let mut index = Bm25Index::new();
        index.add_document("d1", "rust programming language");
        assert_eq!(index.doc_count, 1);

        index.add_document("d2", "python programming language");
        assert_eq!(index.doc_count, 2);

        // "rust" is in 1 of 2 docs => should have decent IDF
        let score = index.score("rust", "d1");
        assert!(score > 0.0);
    }

    #[test]
    fn bm25_incremental_remove() {
        let mut index = Bm25Index::new();
        index.add_document("d1", "rust programming");
        index.add_document("d2", "python programming");
        assert_eq!(index.doc_count, 2);

        index.remove_document("d1");
        assert_eq!(index.doc_count, 1);

        // d1 no longer exists
        assert_eq!(index.score("rust", "d1"), 0.0);

        // d2 still works
        let score = index.score("python", "d2");
        assert!(score > 0.0);
    }

    #[test]
    fn bm25_remove_nonexistent_is_noop() {
        let mut index = Bm25Index::new();
        index.add_document("d1", "test content");
        index.remove_document("nonexistent");
        assert_eq!(index.doc_count, 1);
    }

    #[test]
    fn bm25_replace_document() {
        let mut index = Bm25Index::new();
        index.add_document("d1", "old content about rust");
        index.add_document("d1", "new content about python");
        assert_eq!(index.doc_count, 1);

        // Should match the new content, not old
        let score_python = index.score("python", "d1");
        let score_rust = index.score("rust", "d1");
        assert!(score_python > 0.0);
        assert_eq!(score_rust, 0.0);
    }

    #[test]
    fn bm25_build_from_documents() {
        let docs = vec![
            ("d1".to_string(), "hello world".to_string()),
            ("d2".to_string(), "goodbye world".to_string()),
        ];
        let index = Bm25Index::build(&docs);
        assert_eq!(index.doc_count, 2);
    }

    #[test]
    fn bm25_score_text_works_without_indexing_document() {
        let docs = vec![
            ("d1".to_string(), "rust safety".to_string()),
            ("d2".to_string(), "python typing".to_string()),
        ];
        let index = Bm25Index::build(&docs);

        // Score arbitrary text that's not in the index
        let score = index.score_text("rust safety", "rust ownership and safety features");
        assert!(score > 0.0);
    }

    #[test]
    fn bm25_code_aware_scoring() {
        // camelCase and snake_case should split and match
        let docs = vec![
            (
                "d1".to_string(),
                "processRequest handles incoming data".to_string(),
            ),
            (
                "d2".to_string(),
                "unrelated database migration code".to_string(),
            ),
        ];
        let index = Bm25Index::build(&docs);

        // Query with snake_case should match camelCase doc
        let score_d1 = index.score("process_request", "d1");
        let score_d2 = index.score("process_request", "d2");
        assert!(
            score_d1 > score_d2,
            "code-aware match should work across naming conventions: d1={score_d1}, d2={score_d2}"
        );
    }

    #[test]
    fn bm25_term_frequency_matters() {
        let docs = vec![
            ("d1".to_string(), "rust rust rust is great".to_string()),
            ("d2".to_string(), "rust is a language".to_string()),
        ];
        let index = Bm25Index::build(&docs);

        let score_d1 = index.score("rust", "d1");
        let score_d2 = index.score("rust", "d2");

        // d1 mentions "rust" 3 times, should score higher (though BM25 saturates)
        assert!(
            score_d1 > score_d2,
            "higher tf should give higher score: d1={score_d1}, d2={score_d2}"
        );
    }

    #[test]
    fn bm25_multiple_query_terms() {
        let docs = vec![
            ("d1".to_string(), "rust ownership borrowing".to_string()),
            ("d2".to_string(), "rust generic types".to_string()),
            ("d3".to_string(), "python duck typing".to_string()),
        ];
        let index = Bm25Index::build(&docs);

        // d1 matches both "rust" and "ownership"
        let score_d1 = index.score("rust ownership", "d1");
        // d2 matches only "rust"
        let score_d2 = index.score("rust ownership", "d2");
        // d3 matches neither
        let score_d3 = index.score("rust ownership", "d3");

        assert!(
            score_d1 > score_d2,
            "more matching terms should score higher"
        );
        assert!(score_d2 > score_d3, "some match better than no match");
    }

    // ── Split function unit tests ───────────────────────────────────────

    #[test]
    fn split_camel_case_basic() {
        let parts = split_camel_case("processRequest");
        assert_eq!(parts, vec!["process", "Request"]);
    }

    #[test]
    fn split_camel_case_pascal() {
        let parts = split_camel_case("ProcessRequest");
        assert_eq!(parts, vec!["Process", "Request"]);
    }

    #[test]
    fn split_camel_case_acronym() {
        let parts = split_camel_case("HTMLParser");
        assert_eq!(parts, vec!["HTML", "Parser"]);
    }

    #[test]
    fn split_camel_case_mid_acronym() {
        let parts = split_camel_case("getHTTPResponse");
        assert_eq!(parts, vec!["get", "HTTP", "Response"]);
    }

    #[test]
    fn split_camel_case_all_lower() {
        let parts = split_camel_case("lowercase");
        assert_eq!(parts, vec!["lowercase"]);
    }

    #[test]
    fn split_camel_case_all_upper() {
        let parts = split_camel_case("ALLCAPS");
        assert_eq!(parts, vec!["ALLCAPS"]);
    }

    #[test]
    fn split_camel_case_empty() {
        let parts = split_camel_case("");
        assert!(parts.is_empty());
    }

    #[test]
    fn split_on_punctuation_basic() {
        let parts = split_on_punctuation("foo.bar");
        assert_eq!(parts, vec!["foo", "bar"]);
    }

    #[test]
    fn split_on_punctuation_multiple() {
        let parts = split_on_punctuation("a::b->c.d");
        assert_eq!(parts, vec!["a", "b", "c", "d"]);
    }
}
