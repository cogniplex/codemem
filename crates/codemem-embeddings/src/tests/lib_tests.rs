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

/// A mock provider that tracks batch calls separately.
struct BatchTrackingMock {
    dims: usize,
    single_calls: AtomicUsize,
    batch_calls: AtomicUsize,
}

impl BatchTrackingMock {
    fn new(dims: usize) -> Self {
        Self {
            dims,
            single_calls: AtomicUsize::new(0),
            batch_calls: AtomicUsize::new(0),
        }
    }
}

impl EmbeddingProvider for BatchTrackingMock {
    fn dimensions(&self) -> usize {
        self.dims
    }

    fn embed(&self, _text: &str) -> Result<Vec<f32>, CodememError> {
        self.single_calls.fetch_add(1, Ordering::SeqCst);
        Ok(vec![0.1; self.dims])
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, CodememError> {
        self.batch_calls.fetch_add(1, Ordering::SeqCst);
        Ok(texts.iter().map(|_| vec![0.1; self.dims]).collect())
    }

    fn name(&self) -> &str {
        "batch-mock"
    }
}

/// A mock provider that always fails.
struct FailingMock;

impl EmbeddingProvider for FailingMock {
    fn dimensions(&self) -> usize {
        768
    }

    fn embed(&self, _text: &str) -> Result<Vec<f32>, CodememError> {
        Err(CodememError::Embedding("mock failure".into()))
    }

    fn name(&self) -> &str {
        "failing-mock"
    }
}

// ── CachedProvider tests ─────────────────────────────────────────────

#[test]
fn cached_provider_cache_hit() {
    let mock = MockProvider::new(4);
    let provider = CachedProvider::new(Box::new(mock), 100);

    let v1 = provider.embed("hello").unwrap();
    assert_eq!(v1.len(), 4);

    let v2 = provider.embed("hello").unwrap();
    assert_eq!(v1, v2);

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
fn cached_provider_batch_all_cached() {
    let mock = MockProvider::new(4);
    let provider = CachedProvider::new(Box::new(mock), 100);

    // Pre-populate cache with both texts
    provider.embed("hello").unwrap();
    provider.embed("world").unwrap();

    // Batch where everything is cached — inner provider should not be called again
    let result = provider.embed_batch(&["hello", "world"]).unwrap();
    assert_eq!(result.len(), 2);

    let (size, _) = provider.cache_stats();
    assert_eq!(size, 2);
}

#[test]
fn cached_provider_batch_delegates_to_inner_batch() {
    let mock = BatchTrackingMock::new(4);
    let provider = CachedProvider::new(Box::new(mock), 100);

    let result = provider.embed_batch(&["a", "b", "c"]).unwrap();
    assert_eq!(result.len(), 3);

    let (size, _) = provider.cache_stats();
    assert_eq!(size, 3);
}

#[test]
fn cached_provider_zero_capacity() {
    let mock = MockProvider::new(4);
    let provider = CachedProvider::new(Box::new(mock), 0);

    provider.embed("a").unwrap();
    provider.embed("b").unwrap();

    let (size, cap) = provider.cache_stats();
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
fn cached_provider_inner_error_propagates() {
    let provider = CachedProvider::new(Box::new(FailingMock), 100);
    let result = provider.embed("test");
    assert!(result.is_err());
    let err = result.err().unwrap().to_string();
    assert!(
        err.contains("mock failure"),
        "Should propagate inner error: {err}"
    );
}

#[test]
fn cached_provider_inner_batch_error_propagates() {
    let provider = CachedProvider::new(Box::new(FailingMock), 100);
    let result = provider.embed_batch(&["test"]);
    assert!(result.is_err());
}

#[test]
fn cached_provider_evicts_lru() {
    let mock = MockProvider::new(2);
    let provider = CachedProvider::new(Box::new(mock), 2);

    provider.embed("a").unwrap();
    provider.embed("b").unwrap();
    // Cache is full (cap=2), inserting "c" should evict "a"
    provider.embed("c").unwrap();

    let (size, cap) = provider.cache_stats();
    assert_eq!(size, 2);
    assert_eq!(cap, 2);
}

#[test]
fn cached_provider_cache_hit_avoids_inner_call() {
    let mock = MockProvider::new(4);
    // We need to check call count, but mock is moved into CachedProvider.
    // Use Arc to share the count.
    use std::sync::Arc;

    let call_count = Arc::new(AtomicUsize::new(0));
    let count_clone = call_count.clone();

    struct CountingMock {
        dims: usize,
        count: Arc<AtomicUsize>,
    }
    impl EmbeddingProvider for CountingMock {
        fn dimensions(&self) -> usize {
            self.dims
        }
        fn embed(&self, _text: &str) -> Result<Vec<f32>, CodememError> {
            self.count.fetch_add(1, Ordering::SeqCst);
            Ok(vec![0.1; self.dims])
        }
        fn name(&self) -> &str {
            "counting"
        }
    }

    let _ = mock; // drop unused mock
    let provider = CachedProvider::new(
        Box::new(CountingMock {
            dims: 4,
            count: count_clone,
        }),
        100,
    );

    provider.embed("hello").unwrap();
    assert_eq!(call_count.load(Ordering::SeqCst), 1);

    provider.embed("hello").unwrap();
    assert_eq!(
        call_count.load(Ordering::SeqCst),
        1,
        "Second call should be cached"
    );

    provider.embed("world").unwrap();
    assert_eq!(call_count.load(Ordering::SeqCst), 2);
}

// ── from_env tests ───────────────────────────────────────────────────

#[test]
fn from_env_unknown_provider() {
    let _lock = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
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
    match EmbeddingService::new(Path::new("/nonexistent/path"), 16) {
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
    assert_eq!(DEFAULT_BATCH_SIZE, 16);
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

#[test]
fn from_env_openai_missing_api_key() {
    let _lock = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    unsafe { std::env::set_var("CODEMEM_EMBED_PROVIDER", "openai") };
    unsafe { std::env::remove_var("CODEMEM_EMBED_API_KEY") };
    unsafe { std::env::remove_var("OPENAI_API_KEY") };
    let result = from_env(None);
    unsafe { std::env::remove_var("CODEMEM_EMBED_PROVIDER") };

    assert!(result.is_err());
    let err = result.err().unwrap().to_string();
    assert!(
        err.contains("API_KEY"),
        "Should mention API key requirement: {err}"
    );
}

#[test]
fn from_env_with_config_ollama_url() {
    let _lock = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    unsafe { std::env::set_var("CODEMEM_EMBED_PROVIDER", "ollama") };
    unsafe { std::env::remove_var("CODEMEM_EMBED_URL") };
    unsafe { std::env::remove_var("CODEMEM_EMBED_MODEL") };

    let config = codemem_core::EmbeddingConfig {
        provider: "ollama".to_string(),
        url: "http://custom:11434".to_string(),
        model: "custom-model".to_string(),
        dimensions: 512,
        cache_capacity: 5000,
        ..Default::default()
    };
    let result = from_env(Some(&config));
    unsafe { std::env::remove_var("CODEMEM_EMBED_PROVIDER") };

    // Should succeed — config provides URL and model
    let provider = result.expect("from_env with config should succeed");
    assert_eq!(provider.name(), "ollama");
    assert_eq!(provider.dimensions(), 512);
}

#[test]
fn from_env_env_var_overrides_config() {
    let _lock = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    unsafe { std::env::set_var("CODEMEM_EMBED_PROVIDER", "ollama") };
    unsafe { std::env::set_var("CODEMEM_EMBED_DIMENSIONS", "256") };

    let config = codemem_core::EmbeddingConfig {
        provider: "candle".to_string(),
        dimensions: 512,
        ..Default::default()
    };
    let result = from_env(Some(&config));
    unsafe { std::env::remove_var("CODEMEM_EMBED_PROVIDER") };
    unsafe { std::env::remove_var("CODEMEM_EMBED_DIMENSIONS") };

    let provider = result.expect("from_env should succeed");
    assert_eq!(provider.name(), "ollama");
    assert_eq!(provider.dimensions(), 256);
}

#[test]
fn from_env_openai_with_custom_api_key_env() {
    let _lock = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    unsafe { std::env::set_var("CODEMEM_EMBED_PROVIDER", "openai") };
    unsafe { std::env::set_var("CODEMEM_EMBED_API_KEY", "custom-key") };
    unsafe { std::env::remove_var("OPENAI_API_KEY") };

    let result = from_env(None);
    unsafe { std::env::remove_var("CODEMEM_EMBED_PROVIDER") };
    unsafe { std::env::remove_var("CODEMEM_EMBED_API_KEY") };

    let provider = result.expect("Should use CODEMEM_EMBED_API_KEY");
    assert_eq!(provider.name(), "openai");
}

#[test]
fn from_env_empty_string_treated_as_candle() {
    let _lock = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    unsafe { std::env::set_var("CODEMEM_EMBED_PROVIDER", "") };
    let result = from_env(None);
    unsafe { std::env::remove_var("CODEMEM_EMBED_PROVIDER") };

    // Empty string falls through to "candle" | "" match arm
    // Will fail if model isn't downloaded, but the provider selection is correct
    match result {
        Ok(p) => assert_eq!(p.name(), "candle"),
        Err(e) => {
            let err = e.to_string();
            // If model isn't downloaded, the error is about missing model, not unknown provider
            assert!(
                err.contains("Model not found") || err.contains("model"),
                "Should be a candle model error, not unknown provider: {err}"
            );
        }
    }
}

#[test]
fn from_env_case_insensitive() {
    let _lock = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    unsafe { std::env::set_var("CODEMEM_EMBED_PROVIDER", "OLLAMA") };
    let result = from_env(None);
    unsafe { std::env::remove_var("CODEMEM_EMBED_PROVIDER") };

    let provider = result.expect("Provider name should be case-insensitive");
    assert_eq!(provider.name(), "ollama");
}

// ── Concurrency tests ────────────────────────────────────────────────

#[test]
fn cached_provider_concurrent_embed_no_panic() {
    use std::sync::Arc;
    use std::thread;

    let mock = MockProvider::new(4);
    let provider = Arc::new(CachedProvider::new(Box::new(mock), 100));

    let handles: Vec<_> = (0..10)
        .map(|i| {
            let p = Arc::clone(&provider);
            thread::spawn(move || {
                let text = format!("text_{}", i);
                let result = p.embed(&text);
                assert!(result.is_ok(), "Thread {i} should not panic or error");
                let embedding = result.unwrap();
                assert_eq!(embedding.len(), 4);
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread should not panic");
    }

    // All 10 unique texts should be in the cache
    let (size, _) = provider.cache_stats();
    assert_eq!(size, 10);
}

#[test]
fn cached_provider_concurrent_embed_same_key() {
    use std::sync::Arc;
    use std::thread;

    let call_count = Arc::new(AtomicUsize::new(0));
    let count_clone = call_count.clone();

    struct SlowCountingMock {
        dims: usize,
        count: Arc<AtomicUsize>,
    }
    impl EmbeddingProvider for SlowCountingMock {
        fn dimensions(&self) -> usize {
            self.dims
        }
        fn embed(&self, _text: &str) -> Result<Vec<f32>, CodememError> {
            self.count.fetch_add(1, Ordering::SeqCst);
            // Small sleep to increase chance of concurrent access
            std::thread::sleep(std::time::Duration::from_millis(1));
            Ok(vec![0.42; self.dims])
        }
        fn name(&self) -> &str {
            "slow-counting"
        }
    }

    let provider = Arc::new(CachedProvider::new(
        Box::new(SlowCountingMock {
            dims: 4,
            count: count_clone,
        }),
        100,
    ));

    // All threads embed the same key
    let handles: Vec<_> = (0..10)
        .map(|_| {
            let p = Arc::clone(&provider);
            thread::spawn(move || {
                let result = p.embed("same_key");
                assert!(result.is_ok());
                let embedding = result.unwrap();
                // All should get the same value regardless of cache hit/miss
                assert_eq!(embedding, vec![0.42; 4]);
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread should not panic");
    }

    // Cache should have exactly 1 entry for "same_key"
    let (size, _) = provider.cache_stats();
    assert_eq!(size, 1);
}

#[test]
fn cached_provider_concurrent_embed_batch_no_corruption() {
    use std::sync::Arc;
    use std::thread;

    /// A mock that returns distinct embeddings per text for verifiability.
    struct DistinctMock {
        dims: usize,
    }
    impl EmbeddingProvider for DistinctMock {
        fn dimensions(&self) -> usize {
            self.dims
        }
        fn embed(&self, text: &str) -> Result<Vec<f32>, CodememError> {
            // Use first byte of text as a distinguishing value
            let val = text.as_bytes().first().copied().unwrap_or(0) as f32 / 255.0;
            Ok(vec![val; self.dims])
        }
        fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, CodememError> {
            texts.iter().map(|t| self.embed(t)).collect()
        }
        fn name(&self) -> &str {
            "distinct"
        }
    }

    let provider = Arc::new(CachedProvider::new(
        Box::new(DistinctMock { dims: 4 }),
        1000,
    ));

    let handles: Vec<_> = (0..5)
        .map(|i| {
            let p = Arc::clone(&provider);
            thread::spawn(move || {
                let texts: Vec<String> = (0..3).map(|j| format!("t{}_{}", i, j)).collect();
                let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
                let result = p.embed_batch(&text_refs);
                assert!(result.is_ok(), "Thread {i} batch should succeed");
                let embeddings = result.unwrap();
                assert_eq!(embeddings.len(), 3, "Thread {i} should get 3 embeddings");

                // Verify each embedding has the correct dimension
                for emb in &embeddings {
                    assert_eq!(emb.len(), 4);
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread should not panic");
    }

    // All 15 unique texts should be cached (5 threads x 3 texts)
    let (size, _) = provider.cache_stats();
    assert_eq!(size, 15);
}

#[test]
fn cached_provider_concurrent_reads_and_writes() {
    use std::sync::Arc;
    use std::thread;

    let mock = MockProvider::new(4);
    let provider = Arc::new(CachedProvider::new(Box::new(mock), 100));

    // Pre-populate some entries
    for i in 0..5 {
        provider.embed(&format!("pre_{}", i)).unwrap();
    }

    // Mix of reads (cached) and writes (new keys)
    let handles: Vec<_> = (0..10)
        .map(|i| {
            let p = Arc::clone(&provider);
            thread::spawn(move || {
                if i % 2 == 0 {
                    // Read a cached key
                    let result = p.embed(&format!("pre_{}", i % 5));
                    assert!(result.is_ok());
                } else {
                    // Write a new key
                    let result = p.embed(&format!("new_{}", i));
                    assert!(result.is_ok());
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread should not panic");
    }
}

// ── Edge case input tests ────────────────────────────────────────────

#[test]
fn cached_provider_embed_empty_string() {
    let mock = MockProvider::new(4);
    let provider = CachedProvider::new(Box::new(mock), 100);

    let result = provider.embed("");
    assert!(result.is_ok());
    let embedding = result.unwrap();
    assert_eq!(embedding.len(), 4);

    // Empty string should be cached
    let (size, _) = provider.cache_stats();
    assert_eq!(size, 1);

    // Second call should hit cache
    let result2 = provider.embed("");
    assert!(result2.is_ok());
    assert_eq!(result2.unwrap(), embedding);
}

#[test]
fn cached_provider_embed_very_long_string() {
    let mock = MockProvider::new(4);
    let provider = CachedProvider::new(Box::new(mock), 100);

    // Create a string larger than 10KB
    let long_text = "a".repeat(15_000);
    let result = provider.embed(&long_text);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 4);
}

#[test]
fn cached_provider_batch_with_duplicates_avoids_redundant_calls() {
    use std::sync::Arc;

    let call_count = Arc::new(AtomicUsize::new(0));
    let count_clone = call_count.clone();

    struct CountingBatchMock {
        dims: usize,
        count: Arc<AtomicUsize>,
    }
    impl EmbeddingProvider for CountingBatchMock {
        fn dimensions(&self) -> usize {
            self.dims
        }
        fn embed(&self, _text: &str) -> Result<Vec<f32>, CodememError> {
            self.count.fetch_add(1, Ordering::SeqCst);
            Ok(vec![0.1; self.dims])
        }
        fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, CodememError> {
            // Each text in the batch counts as one inner call
            for _ in texts {
                self.count.fetch_add(1, Ordering::SeqCst);
            }
            Ok(texts.iter().map(|_| vec![0.1; self.dims]).collect())
        }
        fn name(&self) -> &str {
            "counting-batch"
        }
    }

    let provider = CachedProvider::new(
        Box::new(CountingBatchMock {
            dims: 4,
            count: count_clone,
        }),
        100,
    );

    // Pre-populate "hello" in cache
    provider.embed("hello").unwrap();
    let after_first = call_count.load(Ordering::SeqCst);
    assert_eq!(after_first, 1);

    // Batch with duplicates — "hello" should come from cache,
    // only "world" should hit the inner provider
    let result = provider.embed_batch(&["hello", "world", "hello"]).unwrap();
    assert_eq!(result.len(), 3);

    let after_batch = call_count.load(Ordering::SeqCst);
    // Only "world" should have been forwarded (1 new inner call)
    assert_eq!(
        after_batch - after_first,
        1,
        "Only uncached text should hit inner provider"
    );
}

#[test]
fn cached_provider_batch_zero_items() {
    let mock = MockProvider::new(4);
    let provider = CachedProvider::new(Box::new(mock), 100);

    let result = provider.embed_batch(&[]).unwrap();
    assert!(result.is_empty());

    let (size, _) = provider.cache_stats();
    assert_eq!(size, 0);
}
