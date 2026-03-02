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
    let err = result.unwrap_err().to_string();
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
