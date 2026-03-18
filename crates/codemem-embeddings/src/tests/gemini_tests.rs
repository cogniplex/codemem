use super::GeminiProvider;
use codemem_core::EmbeddingProvider;

#[test]
fn provider_name_is_gemini() {
    let provider = GeminiProvider::new("test-key", "text-embedding-004", 768, None);
    assert_eq!(provider.name(), "gemini");
}

#[test]
fn dimensions_matches_config() {
    let provider = GeminiProvider::new("test-key", "text-embedding-004", 256, None);
    assert_eq!(provider.dimensions(), 256);
}

#[test]
fn custom_base_url() {
    let provider = GeminiProvider::new(
        "test-key",
        "text-embedding-004",
        768,
        Some("http://localhost:8080"),
    );
    assert_eq!(provider.base_url, "http://localhost:8080");
}

#[test]
fn default_base_url() {
    let provider = GeminiProvider::new("test-key", "text-embedding-004", 768, None);
    assert_eq!(
        provider.base_url,
        "https://generativelanguage.googleapis.com/v1beta"
    );
}

// ── Mockito-based HTTP tests ──────────────────────────────────────────

#[test]
fn embed_single_success() {
    let mut server = mockito::Server::new();
    let mock = server
        .mock("POST", "/models/text-embedding-004:embedContent")
        .match_header("x-goog-api-key", "test-key")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(r#"{"embedding": {"values": [0.1, 0.2, 0.3]}}"#)
        .create();

    let server_url = server.url();
    let provider = GeminiProvider::new("test-key", "text-embedding-004", 3, Some(&server_url));
    let result = provider.embed("hello world");
    mock.assert();

    let embedding = result.unwrap();
    assert_eq!(embedding.len(), 3);
    assert!((embedding[0] - 0.1).abs() < 1e-6);
    assert!((embedding[1] - 0.2).abs() < 1e-6);
    assert!((embedding[2] - 0.3).abs() < 1e-6);
}

#[test]
fn embed_single_401_unauthorized() {
    let mut server = mockito::Server::new();
    let mock = server
        .mock("POST", "/models/text-embedding-004:embedContent")
        .with_status(401)
        .with_body("Unauthorized")
        .create();

    let server_url = server.url();
    let provider = GeminiProvider::new("bad-key", "text-embedding-004", 768, Some(&server_url));
    let result = provider.embed("test");
    mock.assert();

    assert!(result.is_err());
    let err = result.err().unwrap().to_string();
    assert!(err.contains("401"), "Error should contain '401': {err}");
}

#[test]
fn embed_single_500_server_error() {
    let mut server = mockito::Server::new();
    let mock = server
        .mock("POST", "/models/text-embedding-004:embedContent")
        .with_status(500)
        .with_body("Internal Server Error")
        .create();

    let server_url = server.url();
    let provider = GeminiProvider::new("test-key", "text-embedding-004", 768, Some(&server_url));
    let result = provider.embed("test");
    mock.assert();

    assert!(result.is_err());
    let err = result.err().unwrap().to_string();
    assert!(err.contains("500"), "Error should contain '500': {err}");
}

#[test]
fn embed_single_429_rate_limited() {
    let mut server = mockito::Server::new();
    let mock = server
        .mock("POST", "/models/text-embedding-004:embedContent")
        .with_status(429)
        .with_body("Rate limit exceeded")
        .create();

    let server_url = server.url();
    let provider = GeminiProvider::new("test-key", "text-embedding-004", 768, Some(&server_url));
    let result = provider.embed("test");
    mock.assert();

    assert!(result.is_err());
    let err = result.err().unwrap().to_string();
    assert!(err.contains("429"), "Error should contain '429': {err}");
}

#[test]
fn embed_single_malformed_json() {
    let mut server = mockito::Server::new();
    let mock = server
        .mock("POST", "/models/text-embedding-004:embedContent")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body("not valid json at all")
        .create();

    let server_url = server.url();
    let provider = GeminiProvider::new("test-key", "text-embedding-004", 768, Some(&server_url));
    let result = provider.embed("test");
    mock.assert();

    assert!(result.is_err());
    let err = result.err().unwrap().to_string();
    assert!(
        err.contains("parse error"),
        "Should report parse error: {err}"
    );
}

#[test]
fn embed_single_missing_embedding_field() {
    let mut server = mockito::Server::new();
    let mock = server
        .mock("POST", "/models/text-embedding-004:embedContent")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(r#"{"result": "unexpected format"}"#)
        .create();

    let server_url = server.url();
    let provider = GeminiProvider::new("test-key", "text-embedding-004", 768, Some(&server_url));
    let result = provider.embed("test");
    mock.assert();

    assert!(result.is_err());
    let err = result.err().unwrap().to_string();
    assert!(
        err.contains("Missing embedding.values"),
        "Should report missing embedding field: {err}"
    );
}

#[test]
fn embed_single_dimension_mismatch() {
    let mut server = mockito::Server::new();
    let mock = server
        .mock("POST", "/models/text-embedding-004:embedContent")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(r#"{"embedding": {"values": [0.1, 0.2, 0.3]}}"#)
        .create();

    let server_url = server.url();
    let provider = GeminiProvider::new("test-key", "text-embedding-004", 768, Some(&server_url));
    let result = provider.embed("test");
    mock.assert();

    assert!(result.is_err());
    let err = result.err().unwrap().to_string();
    assert!(
        err.contains("3 dimensions, expected 768"),
        "Should report dimension mismatch: {err}"
    );
}

#[test]
fn embed_batch_success() {
    let mut server = mockito::Server::new();
    let mock = server
        .mock("POST", "/models/text-embedding-004:batchEmbedContents")
        .match_header("x-goog-api-key", "test-key")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            r#"{"embeddings": [
                {"values": [0.1, 0.2, 0.3]},
                {"values": [0.4, 0.5, 0.6]}
            ]}"#,
        )
        .create();

    let server_url = server.url();
    let provider = GeminiProvider::new("test-key", "text-embedding-004", 3, Some(&server_url));
    let result = provider.embed_batch(&["hello", "world"]);
    mock.assert();

    let embeddings = result.unwrap();
    assert_eq!(embeddings.len(), 2);
    assert_eq!(embeddings[0].len(), 3);
    assert_eq!(embeddings[1].len(), 3);
    assert!((embeddings[0][0] - 0.1).abs() < 1e-6);
    assert!((embeddings[1][0] - 0.4).abs() < 1e-6);
}

#[test]
fn embed_batch_empty_input() {
    let provider = GeminiProvider::new("test-key", "text-embedding-004", 768, None);
    let result = provider.embed_batch(&[]).unwrap();
    assert!(result.is_empty());
}

#[test]
fn embed_batch_count_mismatch() {
    let mut server = mockito::Server::new();
    let mock = server
        .mock("POST", "/models/text-embedding-004:batchEmbedContents")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(r#"{"embeddings": [{"values": [0.1, 0.2]}]}"#)
        .create();

    let server_url = server.url();
    let provider = GeminiProvider::new("test-key", "text-embedding-004", 2, Some(&server_url));
    let result = provider.embed_batch(&["a", "b", "c"]);
    mock.assert();

    assert!(result.is_err());
    let err = result.err().unwrap().to_string();
    assert!(
        err.contains("1 embeddings, expected 3"),
        "Should report count mismatch: {err}"
    );
}

#[test]
fn embed_batch_500_error() {
    let mut server = mockito::Server::new();
    let mock = server
        .mock("POST", "/models/text-embedding-004:batchEmbedContents")
        .with_status(500)
        .with_body("Internal Server Error")
        .create();

    let server_url = server.url();
    let provider = GeminiProvider::new("test-key", "text-embedding-004", 768, Some(&server_url));
    let result = provider.embed_batch(&["test"]);
    mock.assert();

    assert!(result.is_err());
    let err = result.err().unwrap().to_string();
    assert!(err.contains("500"), "Should report status code: {err}");
}

#[test]
fn embed_batch_malformed_response() {
    let mut server = mockito::Server::new();
    let mock = server
        .mock("POST", "/models/text-embedding-004:batchEmbedContents")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(r#"{"data": "wrong structure"}"#)
        .create();

    let server_url = server.url();
    let provider = GeminiProvider::new("test-key", "text-embedding-004", 768, Some(&server_url));
    let result = provider.embed_batch(&["test"]);
    mock.assert();

    assert!(result.is_err());
    let err = result.err().unwrap().to_string();
    assert!(
        err.contains("Missing 'embeddings'"),
        "Should report missing embeddings array: {err}"
    );
}

// Note: Actual API tests require a valid GEMINI_API_KEY and are not run in CI.
// Use `cargo test --ignored` with CODEMEM_EMBED_API_KEY set to run them.

#[test]
#[ignore]
fn live_embed_single() {
    let api_key = std::env::var("CODEMEM_EMBED_API_KEY")
        .or_else(|_| std::env::var("GEMINI_API_KEY"))
        .expect("Set CODEMEM_EMBED_API_KEY or GEMINI_API_KEY");
    let provider = GeminiProvider::new(&api_key, "text-embedding-004", 768, None);
    let embedding = provider.embed("Hello, world!").unwrap();
    assert_eq!(embedding.len(), 768);
    // Verify it's normalized (L2 norm ≈ 1.0)
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 0.1,
        "Expected normalized, got norm={norm}"
    );
}

#[test]
#[ignore]
fn live_embed_batch() {
    let api_key = std::env::var("CODEMEM_EMBED_API_KEY")
        .or_else(|_| std::env::var("GEMINI_API_KEY"))
        .expect("Set CODEMEM_EMBED_API_KEY or GEMINI_API_KEY");
    let provider = GeminiProvider::new(&api_key, "text-embedding-004", 768, None);
    let embeddings = provider.embed_batch(&["Hello", "World", "Test"]).unwrap();
    assert_eq!(embeddings.len(), 3);
    assert_eq!(embeddings[0].len(), 768);
}
