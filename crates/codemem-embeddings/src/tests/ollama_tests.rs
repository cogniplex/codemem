use super::*;

#[test]
fn ollama_provider_construction() {
    let provider = OllamaProvider::with_defaults();
    assert_eq!(provider.base_url, DEFAULT_BASE_URL);
    assert_eq!(provider.model, DEFAULT_MODEL);
    assert_eq!(provider.dimensions, 768);
}

#[test]
fn ollama_provider_custom() {
    let provider = OllamaProvider::new("http://myhost:11434", "mxbai-embed-large", 1024);
    assert_eq!(provider.base_url, "http://myhost:11434");
    assert_eq!(provider.model, "mxbai-embed-large");
    assert_eq!(provider.dimensions, 1024);
}

#[test]
fn ollama_name_returns_ollama() {
    use crate::EmbeddingProvider;
    let provider = OllamaProvider::with_defaults();
    assert_eq!(provider.name(), "ollama");
}

#[test]
fn ollama_dimensions_matches_constructor() {
    use crate::EmbeddingProvider;
    let provider = OllamaProvider::new("http://localhost:11434", "nomic-embed-text", 512);
    assert_eq!(EmbeddingProvider::dimensions(&provider), 512);
}

#[test]
fn ollama_embed_success_mock() {
    use crate::EmbeddingProvider;
    let mut server = mockito::Server::new();
    let mock = server
        .mock("POST", "/api/embeddings")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(r#"{"embedding": [0.1, 0.2, 0.3]}"#)
        .create();

    let provider = OllamaProvider::new(&server.url(), "nomic-embed-text", 3);
    let result = provider.embed("test");
    mock.assert();

    let embedding = result.unwrap();
    assert_eq!(embedding.len(), 3);
    assert!((embedding[0] - 0.1).abs() < 1e-6);
    assert!((embedding[1] - 0.2).abs() < 1e-6);
    assert!((embedding[2] - 0.3).abs() < 1e-6);
}

#[test]
fn ollama_embed_server_error_mock() {
    use crate::EmbeddingProvider;
    let mut server = mockito::Server::new();
    let mock = server
        .mock("POST", "/api/embeddings")
        .with_status(500)
        .with_body("Internal Server Error")
        .create();

    let provider = OllamaProvider::new(&server.url(), "nomic-embed-text", 768);
    let result = provider.embed("test");
    mock.assert();

    assert!(result.is_err());
}

#[test]
fn ollama_embed_malformed_response() {
    use crate::EmbeddingProvider;
    let mut server = mockito::Server::new();
    let mock = server
        .mock("POST", "/api/embeddings")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(r#"{"result": "no embedding field"}"#)
        .create();

    let provider = OllamaProvider::new(&server.url(), "nomic-embed-text", 768);
    let result = provider.embed("test");
    mock.assert();

    assert!(result.is_err());
    let err = result.err().unwrap().to_string();
    assert!(
        err.contains("Missing 'embedding'"),
        "Should report missing embedding: {err}"
    );
}

#[test]
fn ollama_embed_batch_empty() {
    use crate::EmbeddingProvider;
    let provider = OllamaProvider::with_defaults();
    let result = provider.embed_batch(&[]).unwrap();
    assert!(result.is_empty());
}

#[test]
fn ollama_embed_batch_success_mock() {
    use crate::EmbeddingProvider;
    let mut server = mockito::Server::new();
    let mock = server
        .mock("POST", "/api/embed")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(r#"{"embeddings": [[0.1, 0.2], [0.3, 0.4]]}"#)
        .create();

    let provider = OllamaProvider::new(&server.url(), "nomic-embed-text", 2);
    let result = provider.embed_batch(&["hello", "world"]);
    mock.assert();

    let embeddings = result.unwrap();
    assert_eq!(embeddings.len(), 2);
    assert!((embeddings[0][0] - 0.1).abs() < 1e-6);
    assert!((embeddings[1][0] - 0.3).abs() < 1e-6);
}

#[test]
fn ollama_embed_batch_count_mismatch() {
    use crate::EmbeddingProvider;
    let mut server = mockito::Server::new();
    let mock = server
        .mock("POST", "/api/embed")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(r#"{"embeddings": [[0.1]]}"#)
        .create();

    let provider = OllamaProvider::new(&server.url(), "nomic-embed-text", 1);
    let result = provider.embed_batch(&["a", "b"]);
    mock.assert();

    assert!(result.is_err());
    let err = result.err().unwrap().to_string();
    assert!(
        err.contains("1 embeddings, expected 2"),
        "Should report count mismatch: {err}"
    );
}

#[test]
fn ollama_embed_batch_fallback_on_error() {
    use crate::EmbeddingProvider;
    let mut server = mockito::Server::new();

    // Batch endpoint fails
    let batch_mock = server
        .mock("POST", "/api/embed")
        .with_status(500)
        .with_body("batch not supported")
        .create();

    // Individual embed calls succeed as fallback
    let single_mock = server
        .mock("POST", "/api/embeddings")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(r#"{"embedding": [0.5, 0.6]}"#)
        .expect(2)
        .create();

    let provider = OllamaProvider::new(&server.url(), "nomic-embed-text", 2);
    let result = provider.embed_batch(&["hello", "world"]);
    batch_mock.assert();
    single_mock.assert();

    let embeddings = result.unwrap();
    assert_eq!(embeddings.len(), 2);
    // Both should be the same since the mock returns the same response
    assert!((embeddings[0][0] - 0.5).abs() < 1e-6);
    assert!((embeddings[1][0] - 0.5).abs() < 1e-6);
}

#[test]
fn ollama_embed_batch_missing_embeddings_field() {
    use crate::EmbeddingProvider;
    let mut server = mockito::Server::new();
    let mock = server
        .mock("POST", "/api/embed")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(r#"{"data": [[0.1]]}"#)
        .create();

    let provider = OllamaProvider::new(&server.url(), "nomic-embed-text", 1);
    let result = provider.embed_batch(&["test"]);
    mock.assert();

    assert!(result.is_err());
    let err = result.err().unwrap().to_string();
    assert!(
        err.contains("Missing 'embeddings'"),
        "Should report missing embeddings: {err}"
    );
}

#[test]
fn ollama_default_base_url_constant() {
    assert_eq!(DEFAULT_BASE_URL, "http://localhost:11434");
}

#[test]
fn ollama_default_model_constant() {
    assert_eq!(DEFAULT_MODEL, "nomic-embed-text");
}

// ── Dimension mismatch detection ──────────────────────────────────────

#[test]
fn ollama_embed_wrong_dimensions_no_validation() {
    // The Ollama provider does NOT validate that the returned embedding
    // dimension matches self.dimensions. This test documents that behavior.
    use crate::EmbeddingProvider;
    let mut server = mockito::Server::new();

    // Return a 3-dim embedding when provider expects 768
    let mock = server
        .mock("POST", "/api/embeddings")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(r#"{"embedding": [0.1, 0.2, 0.3]}"#)
        .create();

    let provider = OllamaProvider::new(&server.url(), "nomic-embed-text", 768);
    let result = provider.embed("test");
    mock.assert();

    // FINDING: No dimension validation — provider silently returns wrong-size vector.
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
fn ollama_embed_batch_wrong_dimensions_no_validation() {
    use crate::EmbeddingProvider;
    let mut server = mockito::Server::new();

    // Return 2-dim embeddings when provider expects 768
    let mock = server
        .mock("POST", "/api/embed")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(r#"{"embeddings": [[0.1, 0.2], [0.3, 0.4]]}"#)
        .create();

    let provider = OllamaProvider::new(&server.url(), "nomic-embed-text", 768);
    let result = provider.embed_batch(&["hello", "world"]);
    mock.assert();

    // FINDING: No dimension validation on batch either
    let embeddings = result.unwrap();
    assert_eq!(embeddings.len(), 2);
    assert_eq!(embeddings[0].len(), 2, "Wrong dimensions accepted silently");
}

// ── Network failure scenarios ──────────────────────────────────────────

#[test]
fn ollama_embed_429_too_many_requests() {
    use crate::EmbeddingProvider;
    let mut server = mockito::Server::new();
    let mock = server
        .mock("POST", "/api/embeddings")
        .with_status(429)
        .with_body("Rate limit exceeded")
        .create();

    let provider = OllamaProvider::new(&server.url(), "nomic-embed-text", 768);
    let result = provider.embed("test");
    mock.assert();

    assert!(result.is_err());
    let err = result.err().unwrap().to_string();
    assert!(err.contains("429"), "Should report 429 status code: {err}");
}

#[test]
fn ollama_embed_503_service_unavailable() {
    use crate::EmbeddingProvider;
    let mut server = mockito::Server::new();
    let mock = server
        .mock("POST", "/api/embeddings")
        .with_status(503)
        .with_body("Service Unavailable")
        .create();

    let provider = OllamaProvider::new(&server.url(), "nomic-embed-text", 768);
    let result = provider.embed("test");
    mock.assert();

    assert!(result.is_err());
    let err = result.err().unwrap().to_string();
    assert!(err.contains("503"), "Should report 503 status code: {err}");
}

#[test]
fn ollama_embed_empty_body_200() {
    use crate::EmbeddingProvider;
    let mut server = mockito::Server::new();
    let mock = server
        .mock("POST", "/api/embeddings")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body("")
        .create();

    let provider = OllamaProvider::new(&server.url(), "nomic-embed-text", 768);
    let result = provider.embed("test");
    mock.assert();

    assert!(
        result.is_err(),
        "Empty body with 200 should be detected as error"
    );
    let err = result.err().unwrap().to_string();
    assert!(
        err.contains("parse error") || err.contains("Missing"),
        "Should report parse error or missing data: {err}"
    );
}

#[test]
fn ollama_embed_connection_refused() {
    use crate::EmbeddingProvider;

    // Use a port that nothing is listening on
    let provider = OllamaProvider::new("http://127.0.0.1:1", "nomic-embed-text", 768);
    let result = provider.embed("test");

    assert!(result.is_err());
    let err = result.err().unwrap().to_string();
    assert!(
        err.contains("request failed") || err.contains("error"),
        "Should report connection error: {err}"
    );
}

#[test]
fn ollama_embed_batch_connection_refused() {
    use crate::EmbeddingProvider;

    let provider = OllamaProvider::new("http://127.0.0.1:1", "nomic-embed-text", 768);
    let result = provider.embed_batch(&["hello", "world"]);

    // Batch endpoint fails, then fallback single calls also fail
    assert!(result.is_err());
    let err = result.err().unwrap().to_string();
    assert!(
        err.contains("request failed") || err.contains("error"),
        "Should report connection error: {err}"
    );
}

// ── Edge case inputs ──────────────────────────────────────────────────

#[test]
fn ollama_embed_empty_string() {
    use crate::EmbeddingProvider;
    let mut server = mockito::Server::new();
    let mock = server
        .mock("POST", "/api/embeddings")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(r#"{"embedding": [0.1, 0.2]}"#)
        .create();

    let provider = OllamaProvider::new(&server.url(), "nomic-embed-text", 2);
    let result = provider.embed("");
    mock.assert();

    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 2);
}

#[test]
fn ollama_embed_very_long_string() {
    use crate::EmbeddingProvider;
    let mut server = mockito::Server::new();
    let mock = server
        .mock("POST", "/api/embeddings")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(r#"{"embedding": [0.1, 0.2]}"#)
        .create();

    let provider = OllamaProvider::new(&server.url(), "nomic-embed-text", 2);
    let long_text = "x".repeat(15_000);
    let result = provider.embed(&long_text);
    mock.assert();

    assert!(result.is_ok());
}
