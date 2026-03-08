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

// ── Test #1: BM25 tokenization consistency ──────────────────────────
// score() tokenizes internally; score_with_tokens_str() uses pre-tokenized input.
// They must produce identical results.

#[test]
fn score_with_tokens_str_matches_score() {
    let mut index = Bm25Index::new();
    index.add_document("d1", "processRequest handles incoming data");
    index.add_document("d2", "unrelated database migration code");

    let query = "processRequest";
    let tokens = tokenize(query);
    let token_refs: Vec<&str> = tokens.iter().map(|s| s.as_str()).collect();

    let score_direct = index.score(query, "d1");
    let score_tokens = index.score_with_tokens_str(&token_refs, "d1");

    assert!(
        (score_direct - score_tokens).abs() < 1e-10,
        "score() and score_with_tokens_str() must match: direct={score_direct}, tokens={score_tokens}"
    );
}

#[test]
fn score_text_with_tokens_str_matches_score_text() {
    let mut index = Bm25Index::new();
    // Need at least one doc for IDF stats
    index.add_document("d1", "some background document for statistics");

    let query = "parseFunction";
    let text = "parseFunction extracts AST nodes from source code";

    let tokens = tokenize(query);
    let token_refs: Vec<&str> = tokens.iter().map(|s| s.as_str()).collect();

    let score_direct = index.score_text(query, text);
    let score_tokens = index.score_text_with_tokens_str(&token_refs, text);

    assert!(
        (score_direct - score_tokens).abs() < 1e-10,
        "score_text() and score_text_with_tokens_str() must match: direct={score_direct}, tokens={score_tokens}"
    );
}

// ── BM25 serialization round-trip tests ─────────────────────────────

#[test]
fn bm25_serialize_roundtrip_scores_match() {
    let docs: Vec<(String, String)> = (0..12)
        .map(|i| {
            (
                format!("doc{i}"),
                format!(
                    "document number {i} about rust programming language features and ownership"
                ),
            )
        })
        .collect();
    let original = Bm25Index::build(&docs);

    // Collect scores before serialization
    let query = "rust ownership";
    let mut original_scores: Vec<(String, f64)> = docs
        .iter()
        .map(|(id, _)| (id.clone(), original.score(query, id)))
        .collect();
    original_scores.sort_by(|a, b| a.0.cmp(&b.0));

    // Serialize and deserialize
    let bytes = original.serialize();
    assert!(!bytes.is_empty(), "serialized bytes should not be empty");

    let restored = Bm25Index::deserialize(&bytes).expect("deserialization should succeed");

    // Verify all scores match
    let mut restored_scores: Vec<(String, f64)> = docs
        .iter()
        .map(|(id, _)| (id.clone(), restored.score(query, id)))
        .collect();
    restored_scores.sort_by(|a, b| a.0.cmp(&b.0));

    assert_eq!(original_scores.len(), restored_scores.len());
    for (orig, rest) in original_scores.iter().zip(restored_scores.iter()) {
        assert_eq!(orig.0, rest.0, "doc IDs should match");
        assert!(
            (orig.1 - rest.1).abs() < 1e-10,
            "scores should match after round-trip for {}: original={}, restored={}",
            orig.0,
            orig.1,
            rest.1
        );
    }
}

#[test]
fn bm25_serialize_roundtrip_preserves_doc_count() {
    let docs: Vec<(String, String)> = (0..10)
        .map(|i| (format!("d{i}"), format!("content for document {i} xyz")))
        .collect();
    let original = Bm25Index::build(&docs);
    assert_eq!(original.doc_count, 10);

    let bytes = original.serialize();
    let restored = Bm25Index::deserialize(&bytes).unwrap();
    assert_eq!(
        restored.doc_count, 10,
        "doc_count should be preserved through serialization"
    );
}

#[test]
fn bm25_empty_index_serialization_roundtrip() {
    let original = Bm25Index::new();
    assert_eq!(original.doc_count, 0);

    let bytes = original.serialize();
    let restored = Bm25Index::deserialize(&bytes).unwrap();
    assert_eq!(restored.doc_count, 0, "empty index should remain empty");
    assert_eq!(
        restored.score("test", "nonexistent"),
        0.0,
        "empty restored index should return 0 for any query"
    );
}

#[test]
fn bm25_roundtrip_with_removed_documents() {
    let mut index = Bm25Index::new();
    index.add_document("d1", "rust programming language features");
    index.add_document("d2", "python dynamic typing system");
    index.add_document("d3", "javascript async await promises");
    index.remove_document("d2");

    assert_eq!(index.doc_count, 2);

    let bytes = index.serialize();
    let restored = Bm25Index::deserialize(&bytes).unwrap();

    assert_eq!(restored.doc_count, 2, "doc_count should reflect removals");
    assert_eq!(
        restored.score("python", "d2"),
        0.0,
        "removed document should not be scoreable after round-trip"
    );
    assert!(
        restored.score("rust", "d1") > 0.0,
        "remaining document should still score after round-trip"
    );
    assert!(
        restored.score("javascript", "d3") > 0.0,
        "remaining document should still score after round-trip"
    );
}

#[test]
fn bm25_add_document_after_roundtrip_works() {
    let mut index = Bm25Index::new();
    index.add_document("d1", "original document about algorithms");
    index.add_document("d2", "second document about data structures");

    let bytes = index.serialize();
    let mut restored = Bm25Index::deserialize(&bytes).unwrap();

    // Add a new document to the restored index
    restored.add_document("d3", "new document about algorithms and optimization");
    assert_eq!(restored.doc_count, 3);

    // The new document should be scoreable
    let score = restored.score("algorithms optimization", "d3");
    assert!(
        score > 0.0,
        "newly added document after round-trip should be scoreable"
    );

    // Original documents should still work
    let score_d1 = restored.score("algorithms", "d1");
    assert!(
        score_d1 > 0.0,
        "original document should still score after adding new doc"
    );
}

#[test]
fn bm25_roundtrip_preserves_multi_term_scoring() {
    // Build an index with varied content to ensure multi-term scoring is preserved
    let mut index = Bm25Index::new();
    index.add_document("d1", "rust ownership borrowing lifetimes memory safety");
    index.add_document("d2", "python garbage collection reference counting cycles");
    index.add_document("d3", "javascript async await promises event loop");

    // Multi-term query so normalization doesn't flatten differences
    let query = "rust ownership memory";
    let score_d1_orig = index.score(query, "d1");
    let score_d2_orig = index.score(query, "d2");
    let score_d3_orig = index.score(query, "d3");

    let bytes = index.serialize();
    let restored = Bm25Index::deserialize(&bytes).unwrap();

    let score_d1_rest = restored.score(query, "d1");
    let score_d2_rest = restored.score(query, "d2");
    let score_d3_rest = restored.score(query, "d3");

    // Scores should be identical after round-trip
    assert!(
        (score_d1_orig - score_d1_rest).abs() < 1e-10,
        "d1 score should be preserved: {} vs {}",
        score_d1_orig,
        score_d1_rest
    );
    assert!(
        (score_d2_orig - score_d2_rest).abs() < 1e-10,
        "d2 score should be preserved"
    );
    assert!(
        (score_d3_orig - score_d3_rest).abs() < 1e-10,
        "d3 score should be preserved"
    );

    // Ranking should be preserved: d1 (matching) > d2 (no match) and d3 (no match)
    assert!(
        score_d1_rest > score_d2_rest,
        "ranking should be preserved: d1 ({}) > d2 ({})",
        score_d1_rest,
        score_d2_rest
    );
}

#[test]
fn bm25_deserialize_corrupt_data_returns_error() {
    let result = Bm25Index::deserialize(b"not valid json at all");
    assert!(result.is_err(), "corrupt data should fail deserialization");
    match result {
        Err(err) => {
            assert!(
                err.contains("deserialization failed"),
                "error should mention deserialization: {err}"
            );
        }
        Ok(_) => panic!("should have returned error"),
    }
}

#[test]
fn bm25_needs_save_after_roundtrip() {
    let mut index = Bm25Index::new();
    assert!(!index.needs_save(), "empty index should not need save");

    index.add_document("d1", "some content here");
    assert!(index.needs_save(), "index with docs should need save");

    let bytes = index.serialize();
    let restored = Bm25Index::deserialize(&bytes).unwrap();
    assert!(
        restored.needs_save(),
        "restored index with docs should still report needs_save"
    );
}

// ── BM25 eviction tests ──────────────────────────────────────────────

#[test]
fn bm25_eviction_at_capacity() {
    let mut index = Bm25Index::new();
    // Override max_documents to a small value for testing
    index.max_documents = 3;

    index.add_document("d1", "first document about rust");
    index.add_document("d2", "second document about python");
    index.add_document("d3", "third document about java");
    assert_eq!(index.doc_count, 3);

    // Adding a 4th doc should evict d1 (oldest)
    index.add_document("d4", "fourth document about golang");
    assert_eq!(index.doc_count, 3, "should stay at max capacity");

    // d1 was evicted
    assert_eq!(
        index.score("rust", "d1"),
        0.0,
        "evicted doc should return 0"
    );
    // d2, d3, d4 should still be scoreable
    assert!(index.score("python", "d2") > 0.0);
    assert!(index.score("java", "d3") > 0.0);
    assert!(index.score("golang", "d4") > 0.0);
}

#[test]
fn bm25_eviction_fifo_order() {
    let mut index = Bm25Index::new();
    index.max_documents = 2;

    index.add_document("d1", "alpha");
    index.add_document("d2", "beta");
    // Evicts d1
    index.add_document("d3", "gamma");
    assert_eq!(
        index.score("alpha", "d1"),
        0.0,
        "d1 should be evicted first"
    );
    assert!(index.score("beta", "d2") > 0.0);

    // Evicts d2
    index.add_document("d4", "delta");
    assert_eq!(
        index.score("beta", "d2"),
        0.0,
        "d2 should be evicted second"
    );
    assert!(index.score("gamma", "d3") > 0.0);
    assert!(index.score("delta", "d4") > 0.0);
}

#[test]
fn bm25_eviction_stats_remain_consistent() {
    let mut index = Bm25Index::new();
    index.max_documents = 2;

    index.add_document("d1", "word1 word2 word3");
    index.add_document("d2", "word4 word5");
    index.add_document("d3", "word6"); // evicts d1

    assert_eq!(index.doc_count, 2);
    // avg_doc_len should reflect only d2 (2 tokens) and d3 (1 token) = 1.5
    // (after tokenization: "word4"/"word5" = 2 tokens, "word6" = 1 token)
    assert!(
        (index.avg_doc_len - 1.5).abs() < 1e-10,
        "avg_doc_len should be recalculated after eviction: {}",
        index.avg_doc_len
    );
}

#[test]
fn bm25_eviction_with_replacement_does_not_double_evict() {
    let mut index = Bm25Index::new();
    index.max_documents = 3;

    index.add_document("d1", "alpha");
    index.add_document("d2", "beta");
    index.add_document("d3", "gamma");

    // Replacing d2 should NOT trigger eviction (same ID, removes then re-adds)
    index.add_document("d2", "beta updated content");
    assert_eq!(
        index.doc_count, 3,
        "replacement should not change doc_count"
    );

    // All docs should still be present
    assert!(
        index.score("alpha", "d1") > 0.0,
        "d1 should survive replacement of d2"
    );
    assert!(index.score("beta", "d2") > 0.0);
    assert!(index.score("gamma", "d3") > 0.0);
}

#[test]
fn bm25_score_text_works_after_roundtrip() {
    let mut index = Bm25Index::new();
    index.add_document("d1", "rust ownership and borrowing semantics");
    index.add_document("d2", "python garbage collection reference counting");

    let text = "rust ownership memory safety and lifetime rules";
    let score_orig = index.score_text("rust ownership", text);

    let bytes = index.serialize();
    let restored = Bm25Index::deserialize(&bytes).unwrap();

    let score_rest = restored.score_text("rust ownership", text);
    assert!(
        (score_orig - score_rest).abs() < 1e-10,
        "score_text should produce same results after round-trip: orig={}, rest={}",
        score_orig,
        score_rest
    );
}
