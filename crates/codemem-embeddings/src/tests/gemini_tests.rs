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
