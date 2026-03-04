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
        (score_direct - score_tokens).abs() < f64::EPSILON,
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
        (score_direct - score_tokens).abs() < f64::EPSILON,
        "score_text() and score_text_with_tokens_str() must match: direct={score_direct}, tokens={score_tokens}"
    );
}
