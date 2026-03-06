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
