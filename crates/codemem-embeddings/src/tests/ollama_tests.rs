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
