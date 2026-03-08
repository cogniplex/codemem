use super::*;

#[test]
fn openai_provider_construction() {
    let provider = OpenAIProvider::with_api_key("test-key");
    assert_eq!(provider.model, DEFAULT_MODEL);
    assert_eq!(provider.dimensions, 768);
    assert_eq!(provider.base_url, DEFAULT_BASE_URL);
}

#[test]
fn openai_provider_custom_base_url() {
    let provider = OpenAIProvider::new(
        "test-key",
        "custom-model",
        1536,
        Some("https://api.together.xyz/v1"),
    );
    assert_eq!(provider.base_url, "https://api.together.xyz/v1");
    assert_eq!(provider.model, "custom-model");
    assert_eq!(provider.dimensions, 1536);
}

#[test]
fn openai_name_returns_openai() {
    use crate::EmbeddingProvider;
    let provider = OpenAIProvider::with_api_key("test-key");
    assert_eq!(provider.name(), "openai");
}

#[test]
fn openai_dimensions_matches_constructor() {
    use crate::EmbeddingProvider;
    let provider = OpenAIProvider::new("key", "model", 1024, None);
    assert_eq!(provider.dimensions(), 1024);
}

#[test]
fn openai_embed_success_mock() {
    use crate::EmbeddingProvider;
    let mut server = mockito::Server::new();
    let mock = server
        .mock("POST", "/embeddings")
        .match_header("Authorization", "Bearer test-key")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(r#"{"data": [{"embedding": [0.1, 0.2, 0.3]}]}"#)
        .create();

    let server_url = server.url();
    let provider = OpenAIProvider::new("test-key", "custom-model", 3, Some(&server_url));
    let result = provider.embed("test");
    mock.assert();

    let embedding = result.unwrap();
    assert_eq!(embedding.len(), 3);
    assert!((embedding[0] - 0.1).abs() < 1e-6);
    assert!((embedding[1] - 0.2).abs() < 1e-6);
    assert!((embedding[2] - 0.3).abs() < 1e-6);
}

#[test]
fn openai_embed_unauthorized_mock() {
    use crate::EmbeddingProvider;
    let mut server = mockito::Server::new();
    let mock = server
        .mock("POST", "/embeddings")
        .with_status(401)
        .with_body("Unauthorized")
        .create();

    let server_url = server.url();
    let provider = OpenAIProvider::new("bad-key", "custom-model", 768, Some(&server_url));
    let result = provider.embed("test");
    mock.assert();

    assert!(result.is_err());
    let err = result.err().unwrap().to_string();
    assert!(err.contains("401"), "Error should contain '401': {err}");
}

#[test]
fn openai_embed_text_embedding_3_includes_dimensions() {
    use crate::EmbeddingProvider;
    let mut server = mockito::Server::new();
    let mock = server
        .mock("POST", "/embeddings")
        .match_body(mockito::Matcher::PartialJsonString(
            r#"{"dimensions": 3}"#.to_string(),
        ))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(r#"{"data": [{"embedding": [0.1, 0.2, 0.3]}]}"#)
        .create();

    let server_url = server.url();
    let provider = OpenAIProvider::new("test-key", "text-embedding-3-small", 3, Some(&server_url));
    let result = provider.embed("test");
    mock.assert();

    assert!(result.is_ok());
}

#[test]
fn openai_embed_non_v3_model_omits_dimensions() {
    use crate::EmbeddingProvider;
    let mut server = mockito::Server::new();
    // The body should NOT contain "dimensions" for non-v3 models
    let mock = server
        .mock("POST", "/embeddings")
        .match_body(mockito::Matcher::PartialJsonString(
            r#"{"model": "ada-002"}"#.to_string(),
        ))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(r#"{"data": [{"embedding": [0.5, 0.6]}]}"#)
        .create();

    let server_url = server.url();
    let provider = OpenAIProvider::new("test-key", "ada-002", 2, Some(&server_url));
    let result = provider.embed("test");
    mock.assert();

    let embedding = result.unwrap();
    assert_eq!(embedding.len(), 2);
}

#[test]
fn openai_embed_malformed_response() {
    use crate::EmbeddingProvider;
    let mut server = mockito::Server::new();
    let mock = server
        .mock("POST", "/embeddings")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(r#"{"result": "unexpected format"}"#)
        .create();

    let server_url = server.url();
    let provider = OpenAIProvider::new("test-key", "model", 768, Some(&server_url));
    let result = provider.embed("test");
    mock.assert();

    assert!(result.is_err());
    let err = result.err().unwrap().to_string();
    assert!(
        err.contains("Missing embedding"),
        "Should report missing embedding: {err}"
    );
}

#[test]
fn openai_embed_batch_empty() {
    use crate::EmbeddingProvider;
    let provider = OpenAIProvider::with_api_key("test-key");
    let result = provider.embed_batch(&[]).unwrap();
    assert!(result.is_empty());
}

#[test]
fn openai_embed_batch_success_mock() {
    use crate::EmbeddingProvider;
    let mut server = mockito::Server::new();
    let mock = server
        .mock("POST", "/embeddings")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            r#"{"data": [
                {"index": 0, "embedding": [0.1, 0.2]},
                {"index": 1, "embedding": [0.3, 0.4]}
            ]}"#,
        )
        .create();

    let server_url = server.url();
    let provider = OpenAIProvider::new("test-key", "model", 2, Some(&server_url));
    let result = provider.embed_batch(&["hello", "world"]);
    mock.assert();

    let embeddings = result.unwrap();
    assert_eq!(embeddings.len(), 2);
    assert_eq!(embeddings[0].len(), 2);
    assert_eq!(embeddings[1].len(), 2);
    assert!((embeddings[0][0] - 0.1).abs() < 1e-6);
    assert!((embeddings[1][0] - 0.3).abs() < 1e-6);
}

#[test]
fn openai_embed_batch_out_of_order_indices() {
    use crate::EmbeddingProvider;
    let mut server = mockito::Server::new();
    let mock = server
        .mock("POST", "/embeddings")
        .with_status(200)
        .with_header("content-type", "application/json")
        // Return indices out of order — should be sorted by the provider
        .with_body(
            r#"{"data": [
                {"index": 1, "embedding": [0.3, 0.4]},
                {"index": 0, "embedding": [0.1, 0.2]}
            ]}"#,
        )
        .create();

    let server_url = server.url();
    let provider = OpenAIProvider::new("test-key", "model", 2, Some(&server_url));
    let result = provider.embed_batch(&["hello", "world"]);
    mock.assert();

    let embeddings = result.unwrap();
    // After sorting by index, first embedding should be [0.1, 0.2]
    assert!((embeddings[0][0] - 0.1).abs() < 1e-6);
    assert!((embeddings[1][0] - 0.3).abs() < 1e-6);
}

#[test]
fn openai_embed_batch_count_mismatch() {
    use crate::EmbeddingProvider;
    let mut server = mockito::Server::new();
    let mock = server
        .mock("POST", "/embeddings")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(r#"{"data": [{"index": 0, "embedding": [0.1]}]}"#)
        .create();

    let server_url = server.url();
    let provider = OpenAIProvider::new("test-key", "model", 1, Some(&server_url));
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
fn openai_embed_batch_missing_data_field() {
    use crate::EmbeddingProvider;
    let mut server = mockito::Server::new();
    let mock = server
        .mock("POST", "/embeddings")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(r#"{"embeddings": [[0.1]]}"#)
        .create();

    let server_url = server.url();
    let provider = OpenAIProvider::new("test-key", "model", 1, Some(&server_url));
    let result = provider.embed_batch(&["test"]);
    mock.assert();

    assert!(result.is_err());
    let err = result.err().unwrap().to_string();
    assert!(
        err.contains("Missing 'data'"),
        "Should report missing data: {err}"
    );
}

#[test]
fn openai_embed_batch_server_error() {
    use crate::EmbeddingProvider;
    let mut server = mockito::Server::new();
    let mock = server
        .mock("POST", "/embeddings")
        .with_status(500)
        .with_body("Internal Server Error")
        .create();

    let server_url = server.url();
    let provider = OpenAIProvider::new("test-key", "model", 768, Some(&server_url));
    let result = provider.embed_batch(&["test"]);
    mock.assert();

    assert!(result.is_err());
    let err = result.err().unwrap().to_string();
    assert!(err.contains("500"), "Should report status code: {err}");
}

#[test]
fn openai_default_base_url_constant() {
    assert_eq!(DEFAULT_BASE_URL, "https://api.openai.com/v1");
}

#[test]
fn openai_default_model_constant() {
    assert_eq!(DEFAULT_MODEL, "text-embedding-3-small");
}

// ── Dimension mismatch detection ──────────────────────────────────────

#[test]
fn openai_embed_wrong_dimensions_no_validation() {
    // The OpenAI provider does NOT validate that the returned embedding
    // dimension matches self.dimensions. This test documents that behavior:
    // requesting 768 dims but receiving 512 dims succeeds silently.
    use crate::EmbeddingProvider;
    let mut server = mockito::Server::new();

    // Return a 3-dim embedding when provider expects 768
    let mock = server
        .mock("POST", "/embeddings")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(r#"{"data": [{"embedding": [0.1, 0.2, 0.3]}]}"#)
        .create();

    let server_url = server.url();
    let provider = OpenAIProvider::new("test-key", "custom-model", 768, Some(&server_url));
    let result = provider.embed("test");
    mock.assert();

    // FINDING: No dimension validation — provider silently returns wrong-size vector.
    // The caller would need to check embedding.len() == provider.dimensions().
    let embedding = result.unwrap();
    assert_eq!(
        embedding.len(),
        3,
        "Provider returns whatever the server sends, even if dimensions mismatch"
    );
    assert_ne!(
        embedding.len(),
        provider.dimensions(),
        "Returned dimensions do not match configured dimensions — no validation exists"
    );
}

#[test]
fn openai_embed_batch_wrong_dimensions_no_validation() {
    use crate::EmbeddingProvider;
    let mut server = mockito::Server::new();

    // Return 2-dim embeddings when provider expects 768
    let mock = server
        .mock("POST", "/embeddings")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            r#"{"data": [
                {"index": 0, "embedding": [0.1, 0.2]},
                {"index": 1, "embedding": [0.3, 0.4]}
            ]}"#,
        )
        .create();

    let server_url = server.url();
    let provider = OpenAIProvider::new("test-key", "custom-model", 768, Some(&server_url));
    let result = provider.embed_batch(&["hello", "world"]);
    mock.assert();

    // FINDING: No dimension validation on batch either
    let embeddings = result.unwrap();
    assert_eq!(embeddings.len(), 2);
    assert_eq!(embeddings[0].len(), 2, "Wrong dimensions accepted silently");
}

// ── Network failure scenarios ──────────────────────────────────────────

#[test]
fn openai_embed_429_too_many_requests() {
    use crate::EmbeddingProvider;
    let mut server = mockito::Server::new();
    let mock = server
        .mock("POST", "/embeddings")
        .with_status(429)
        .with_body("Rate limit exceeded")
        .create();

    let server_url = server.url();
    let provider = OpenAIProvider::new("test-key", "model", 768, Some(&server_url));
    let result = provider.embed("test");
    mock.assert();

    assert!(result.is_err());
    let err = result.err().unwrap().to_string();
    assert!(err.contains("429"), "Should report 429 status code: {err}");
}

#[test]
fn openai_embed_503_service_unavailable() {
    use crate::EmbeddingProvider;
    let mut server = mockito::Server::new();
    let mock = server
        .mock("POST", "/embeddings")
        .with_status(503)
        .with_body("Service Unavailable")
        .create();

    let server_url = server.url();
    let provider = OpenAIProvider::new("test-key", "model", 768, Some(&server_url));
    let result = provider.embed("test");
    mock.assert();

    assert!(result.is_err());
    let err = result.err().unwrap().to_string();
    assert!(err.contains("503"), "Should report 503 status code: {err}");
}

#[test]
fn openai_embed_empty_body_200() {
    use crate::EmbeddingProvider;
    let mut server = mockito::Server::new();
    let mock = server
        .mock("POST", "/embeddings")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body("")
        .create();

    let server_url = server.url();
    let provider = OpenAIProvider::new("test-key", "model", 768, Some(&server_url));
    let result = provider.embed("test");
    mock.assert();

    assert!(
        result.is_err(),
        "Empty body with 200 should be detected as error"
    );
    let err = result.err().unwrap().to_string();
    assert!(
        err.contains("parse error") || err.contains("Missing embedding"),
        "Should report parse error or missing data: {err}"
    );
}

#[test]
fn openai_embed_connection_refused() {
    use crate::EmbeddingProvider;

    // Use a port that nothing is listening on
    let provider = OpenAIProvider::new("test-key", "model", 768, Some("http://127.0.0.1:1"));
    let result = provider.embed("test");

    assert!(result.is_err());
    let err = result.err().unwrap().to_string();
    assert!(
        err.contains("request failed") || err.contains("error"),
        "Should report connection error: {err}"
    );
}

#[test]
fn openai_embed_batch_connection_refused() {
    use crate::EmbeddingProvider;

    let provider = OpenAIProvider::new("test-key", "model", 768, Some("http://127.0.0.1:1"));
    let result = provider.embed_batch(&["hello", "world"]);

    assert!(result.is_err());
    let err = result.err().unwrap().to_string();
    assert!(
        err.contains("request failed") || err.contains("error"),
        "Should report connection error: {err}"
    );
}

// ── Edge case inputs ──────────────────────────────────────────────────

#[test]
fn openai_embed_empty_string() {
    use crate::EmbeddingProvider;
    let mut server = mockito::Server::new();
    let mock = server
        .mock("POST", "/embeddings")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(r#"{"data": [{"embedding": [0.1, 0.2]}]}"#)
        .create();

    let server_url = server.url();
    let provider = OpenAIProvider::new("test-key", "model", 2, Some(&server_url));
    let result = provider.embed("");
    mock.assert();

    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 2);
}

#[test]
fn openai_embed_very_long_string() {
    use crate::EmbeddingProvider;
    let mut server = mockito::Server::new();
    let mock = server
        .mock("POST", "/embeddings")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(r#"{"data": [{"embedding": [0.1, 0.2]}]}"#)
        .create();

    let server_url = server.url();
    let provider = OpenAIProvider::new("test-key", "model", 2, Some(&server_url));
    let long_text = "x".repeat(15_000);
    let result = provider.embed(&long_text);
    mock.assert();

    assert!(result.is_ok());
}

// ── API key handling ──────────────────────────────────────────────────

#[test]
fn openai_embed_blank_api_key() {
    use crate::EmbeddingProvider;
    let mut server = mockito::Server::new();
    // Server still gets called — blank key is sent in Authorization header.
    // reqwest may trim the trailing space from "Bearer ", so we match loosely.
    let mock = server
        .mock("POST", "/embeddings")
        .with_status(401)
        .with_body("Invalid API key")
        .create();

    let server_url = server.url();
    let provider = OpenAIProvider::new("", "model", 768, Some(&server_url));
    let result = provider.embed("test");
    mock.assert();

    // FINDING: Blank API key is not rejected at construction time;
    // it proceeds to make the request and fails with a server-side auth error.
    assert!(result.is_err());
    let err = result.err().unwrap().to_string();
    assert!(
        err.contains("401"),
        "Blank API key should cause auth failure: {err}"
    );
}

#[test]
fn openai_embed_whitespace_only_api_key() {
    use crate::EmbeddingProvider;
    let mut server = mockito::Server::new();
    // reqwest trims header values, so whitespace-only key becomes effectively blank.
    let mock = server
        .mock("POST", "/embeddings")
        .with_status(401)
        .with_body("Invalid API key")
        .create();

    let server_url = server.url();
    let provider = OpenAIProvider::new("   ", "model", 768, Some(&server_url));
    let result = provider.embed("test");
    mock.assert();

    // FINDING: Whitespace-only API key is not rejected at construction time;
    // it proceeds to make the request and fails with a server-side auth error.
    assert!(result.is_err());
    let err = result.err().unwrap().to_string();
    assert!(
        err.contains("401"),
        "Whitespace-only API key should cause auth failure: {err}"
    );
}
