use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Mutex to serialize tests that manipulate environment variables.
static ENV_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// A mock embedding provider for testing CachedProvider behavior.
struct MockProvider {
    dims: usize,
    call_count: AtomicUsize,
}

impl MockProvider {
    fn new(dims: usize) -> Self {
        Self {
            dims,
            call_count: AtomicUsize::new(0),
        }
    }
}

impl EmbeddingProvider for MockProvider {
    fn dimensions(&self) -> usize {
        self.dims
    }

    fn embed(&self, _text: &str) -> Result<Vec<f32>, CodememError> {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        Ok(vec![0.1; self.dims])
    }

    fn name(&self) -> &str {
        "mock"
    }
}

#[test]
fn cached_provider_cache_hit() {
    let mock = MockProvider::new(4);
    let provider = CachedProvider::new(Box::new(mock), 100);

    // First call: cache miss
    let v1 = provider.embed("hello").unwrap();
    assert_eq!(v1.len(), 4);

    // Second call: cache hit (inner should only be called once)
    let v2 = provider.embed("hello").unwrap();
    assert_eq!(v1, v2);

    // Access inner mock through the provider trait -- call_count should be 1
    // We can check cache_stats instead
    let (size, cap) = provider.cache_stats();
    assert_eq!(size, 1);
    assert_eq!(cap, 100);
}

#[test]
fn cached_provider_cache_miss() {
    let mock = MockProvider::new(4);
    let provider = CachedProvider::new(Box::new(mock), 100);

    provider.embed("hello").unwrap();
    provider.embed("world").unwrap();

    let (size, _) = provider.cache_stats();
    assert_eq!(size, 2);
}

#[test]
fn cached_provider_batch_empty() {
    let mock = MockProvider::new(4);
    let provider = CachedProvider::new(Box::new(mock), 100);

    let result = provider.embed_batch(&[]).unwrap();
    assert!(result.is_empty());
}

#[test]
fn cached_provider_batch_single() {
    let mock = MockProvider::new(4);
    let provider = CachedProvider::new(Box::new(mock), 100);

    let result = provider.embed_batch(&["hello"]).unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].len(), 4);

    let (size, _) = provider.cache_stats();
    assert_eq!(size, 1);
}

#[test]
fn cached_provider_batch_mixed_cache() {
    let mock = MockProvider::new(4);
    let provider = CachedProvider::new(Box::new(mock), 100);

    // Pre-populate cache
    provider.embed("hello").unwrap();

    // Batch with one cached and one uncached
    let result = provider.embed_batch(&["hello", "world"]).unwrap();
    assert_eq!(result.len(), 2);

    let (size, _) = provider.cache_stats();
    assert_eq!(size, 2);
}

#[test]
fn cached_provider_zero_capacity() {
    // Capacity of 0 should default to 1
    let mock = MockProvider::new(4);
    let provider = CachedProvider::new(Box::new(mock), 0);

    provider.embed("a").unwrap();
    provider.embed("b").unwrap();

    let (size, cap) = provider.cache_stats();
    // Cap should be 1 (the fallback), so only 1 entry retained
    assert_eq!(cap, 1);
    assert_eq!(size, 1);
}

#[test]
fn cached_provider_name_delegates() {
    let mock = MockProvider::new(4);
    let provider = CachedProvider::new(Box::new(mock), 100);
    assert_eq!(provider.name(), "mock");
}

#[test]
fn cached_provider_dimensions_delegates() {
    let mock = MockProvider::new(768);
    let provider = CachedProvider::new(Box::new(mock), 100);
    assert_eq!(provider.dimensions(), 768);
}

#[test]
fn from_env_unknown_provider() {
    let _lock = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    // Set env var to trigger the error path
    unsafe { std::env::set_var("CODEMEM_EMBED_PROVIDER", "nonexistent_provider_xyz") };
    let result = from_env(None);
    unsafe { std::env::remove_var("CODEMEM_EMBED_PROVIDER") };

    match result {
        Err(e) => {
            let err = e.to_string();
            assert!(
                err.contains("Unknown embedding provider"),
                "Error should mention unknown provider: {err}"
            );
        }
        Ok(_) => panic!("Expected error for unknown provider"),
    }
}

#[test]
fn embedding_service_missing_model() {
    match EmbeddingService::new(Path::new("/nonexistent/path")) {
        Err(e) => {
            let err = e.to_string();
            assert!(
                err.contains("Model not found"),
                "Error should mention missing model: {err}"
            );
        }
        Ok(_) => panic!("Expected error for missing model"),
    }
}

#[test]
fn default_model_dir_path() {
    let dir = EmbeddingService::default_model_dir();
    assert!(dir.to_string_lossy().contains(MODEL_NAME));
    assert!(dir.to_string_lossy().contains(".codemem"));
}

#[test]
fn constants_are_sensible() {
    assert_eq!(DIMENSIONS, 768);
    assert_eq!(CACHE_CAPACITY, 10_000);
    assert_eq!(MODEL_NAME, "bge-base-en-v1.5");
}

#[test]
fn from_env_ollama_provider() {
    let _lock = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    unsafe { std::env::set_var("CODEMEM_EMBED_PROVIDER", "ollama") };
    let result = from_env(None);
    unsafe { std::env::remove_var("CODEMEM_EMBED_PROVIDER") };

    let provider = result.expect("from_env should succeed for ollama");
    assert_eq!(provider.name(), "ollama");
}

#[test]
fn from_env_openai_provider() {
    let _lock = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    unsafe { std::env::set_var("CODEMEM_EMBED_PROVIDER", "openai") };
    unsafe { std::env::set_var("OPENAI_API_KEY", "test-key-123") };
    let result = from_env(None);
    unsafe { std::env::remove_var("CODEMEM_EMBED_PROVIDER") };
    unsafe { std::env::remove_var("OPENAI_API_KEY") };

    let provider = result.expect("from_env should succeed for openai");
    assert_eq!(provider.name(), "openai");
}
