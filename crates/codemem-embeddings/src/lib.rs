//! codemem-embeddings: Pluggable embedding providers for Codemem.
//!
//! Supports multiple backends:
//! - **Candle** (default): Local BAAI/bge-base-en-v1.5 via pure Rust ML
//! - **Ollama**: Local Ollama server with any embedding model
//! - **OpenAI**: OpenAI API or any compatible endpoint (Together, Azure, etc.)

pub mod ollama;
pub mod openai;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use codemem_core::CodememError;
use lru::LruCache;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use tokenizers::{PaddingParams, PaddingStrategy, TruncationParams};

/// Default model name.
pub const MODEL_NAME: &str = "bge-base-en-v1.5";

/// HuggingFace model repo ID.
const HF_MODEL_REPO: &str = "BAAI/bge-base-en-v1.5";

/// Default embedding dimensions.
pub const DIMENSIONS: usize = 768;

/// Max sequence length for bge-base-en-v1.5.
const MAX_SEQ_LENGTH: usize = 512;

/// Default LRU cache capacity.
pub const CACHE_CAPACITY: usize = 10_000;

// ── Embedding Provider Trait ────────────────────────────────────────────────

/// Trait for pluggable embedding providers.
pub trait EmbeddingProvider: Send + Sync {
    /// Embedding vector dimensions.
    fn dimensions(&self) -> usize;

    /// Embed a single text string.
    fn embed(&self, text: &str) -> Result<Vec<f32>, CodememError>;

    /// Embed a batch of texts (default: sequential).
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, CodememError> {
        texts.iter().map(|t| self.embed(t)).collect()
    }

    /// Provider name for display.
    fn name(&self) -> &str;

    /// Cache statistics: (current_size, capacity). Returns (0, 0) if no cache.
    fn cache_stats(&self) -> (usize, usize) {
        (0, 0)
    }
}

// ── Candle Embedding Service ────────────────────────────────────────────────

/// Maximum batch size for batched embedding forward passes.
const BATCH_SIZE: usize = 32;

/// Select the best available compute device.
///
/// Tries Metal (macOS GPU) first, then CUDA (NVIDIA GPU), then falls back to CPU.
/// GPU backends are only available when the corresponding feature flag is enabled.
fn select_device() -> Device {
    #[cfg(feature = "metal")]
    {
        if let Ok(device) = Device::new_metal(0) {
            tracing::info!("Using Metal GPU for embeddings");
            return device;
        }
        tracing::warn!("Metal feature enabled but device creation failed, falling back");
    }
    #[cfg(feature = "cuda")]
    {
        if let Ok(device) = Device::new_cuda(0) {
            tracing::info!("Using CUDA GPU for embeddings");
            return device;
        }
        tracing::warn!("CUDA feature enabled but device creation failed, falling back");
    }
    tracing::info!("Using CPU for embeddings");
    Device::Cpu
}

/// Embedding service with Candle inference and LRU caching.
pub struct EmbeddingService {
    model: Mutex<BertModel>,
    tokenizer: tokenizers::Tokenizer,
    device: Device,
    cache: Mutex<LruCache<String, Vec<f32>>>,
}

impl EmbeddingService {
    /// Create a new embedding service, loading model from the given directory.
    /// Expects `model.safetensors`, `config.json`, and `tokenizer.json` in the directory.
    pub fn new(model_dir: &Path) -> Result<Self, CodememError> {
        let model_path = model_dir.join("model.safetensors");
        let config_path = model_dir.join("config.json");
        let tokenizer_path = model_dir.join("tokenizer.json");

        if !model_path.exists() {
            return Err(CodememError::Embedding(format!(
                "Model not found at {}. Run `codemem init` to download it.",
                model_path.display()
            )));
        }

        let device = select_device();

        // Load BERT config
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| CodememError::Embedding(format!("Failed to read config: {e}")))?;
        let config: BertConfig = serde_json::from_str(&config_str)
            .map_err(|e| CodememError::Embedding(format!("Failed to parse config: {e}")))?;

        // Load model weights from safetensors via memory-mapped IO
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&model_path], DType::F32, &device)
                .map_err(|e| CodememError::Embedding(format!("Failed to load weights: {e}")))?
        };

        // Try with "bert." prefix first (standard HF BERT models), then without
        let model = BertModel::load(vb.pp("bert"), &config)
            .or_else(|_| {
                let vb2 = unsafe {
                    VarBuilder::from_mmaped_safetensors(&[&model_path], DType::F32, &device)
                        .map_err(|e| {
                            candle_core::Error::Msg(format!("Failed to load weights: {e}"))
                        })
                }?;
                BertModel::load(vb2, &config)
            })
            .map_err(|e| CodememError::Embedding(format!("Failed to load BERT model: {e}")))?;

        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| CodememError::Embedding(e.to_string()))?;

        let cache = Mutex::new(LruCache::new(NonZeroUsize::new(CACHE_CAPACITY).unwrap()));

        Ok(Self {
            model: Mutex::new(model),
            tokenizer,
            device,
            cache,
        })
    }

    /// Get the default model directory path (~/.codemem/models/{MODEL_NAME}).
    pub fn default_model_dir() -> PathBuf {
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".codemem")
            .join("models")
            .join(MODEL_NAME)
    }

    /// Download the model from HuggingFace Hub to the given directory.
    /// Returns the directory path. No-ops if model already exists.
    pub fn download_model(dest_dir: &Path) -> Result<PathBuf, CodememError> {
        let model_dest = dest_dir.join("model.safetensors");
        let config_dest = dest_dir.join("config.json");
        let tokenizer_dest = dest_dir.join("tokenizer.json");

        if model_dest.exists() && config_dest.exists() && tokenizer_dest.exists() {
            tracing::info!("Model already downloaded at {}", dest_dir.display());
            return Ok(dest_dir.to_path_buf());
        }

        std::fs::create_dir_all(dest_dir)
            .map_err(|e| CodememError::Embedding(format!("Failed to create dir: {e}")))?;

        tracing::info!("Downloading {} from HuggingFace...", HF_MODEL_REPO);

        let api = hf_hub::api::sync::Api::new()
            .map_err(|e| CodememError::Embedding(format!("HuggingFace API error: {e}")))?;
        let repo = api.model(HF_MODEL_REPO.to_string());

        let cached_model = repo
            .get("model.safetensors")
            .map_err(|e| CodememError::Embedding(format!("Failed to download model: {e}")))?;

        let cached_config = repo
            .get("config.json")
            .map_err(|e| CodememError::Embedding(format!("Failed to download config: {e}")))?;

        let cached_tokenizer = repo
            .get("tokenizer.json")
            .map_err(|e| CodememError::Embedding(format!("Failed to download tokenizer: {e}")))?;

        std::fs::copy(&cached_model, &model_dest)
            .map_err(|e| CodememError::Embedding(format!("Failed to copy model: {e}")))?;
        std::fs::copy(&cached_config, &config_dest)
            .map_err(|e| CodememError::Embedding(format!("Failed to copy config: {e}")))?;
        std::fs::copy(&cached_tokenizer, &tokenizer_dest)
            .map_err(|e| CodememError::Embedding(format!("Failed to copy tokenizer: {e}")))?;

        tracing::info!("Model downloaded to {}", dest_dir.display());
        Ok(dest_dir.to_path_buf())
    }

    /// Embed a single text string. Returns a 768-dim L2-normalized vector.
    /// Uses LRU cache for repeated queries.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>, CodememError> {
        // Check cache
        {
            let mut cache = self.cache.lock().unwrap();
            if let Some(cached) = cache.get(text) {
                return Ok(cached.clone());
            }
        }

        let embedding = self.embed_uncached(text)?;

        // Store in cache
        {
            let mut cache = self.cache.lock().unwrap();
            cache.put(text.to_string(), embedding.clone());
        }

        Ok(embedding)
    }

    /// Embed without caching. Uses mean pooling with attention mask.
    fn embed_uncached(&self, text: &str) -> Result<Vec<f32>, CodememError> {
        // Tokenize with truncation
        let mut tokenizer = self.tokenizer.clone();

        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: MAX_SEQ_LENGTH,
                ..Default::default()
            }))
            .map_err(|e| CodememError::Embedding(format!("Truncation error: {e}")))?;

        let encoding = tokenizer
            .encode(text, true)
            .map_err(|e| CodememError::Embedding(e.to_string()))?;

        let input_ids: Vec<u32> = encoding.get_ids().to_vec();
        let attention_mask: Vec<u32> = encoding.get_attention_mask().to_vec();

        // Build candle tensors with shape [1, seq_len]
        let input_ids_tensor = Tensor::new(&input_ids[..], &self.device)
            .and_then(|t| t.unsqueeze(0))
            .map_err(|e| CodememError::Embedding(format!("Tensor error: {e}")))?;

        let token_type_ids = input_ids_tensor
            .zeros_like()
            .map_err(|e| CodememError::Embedding(format!("Tensor error: {e}")))?;

        let attention_mask_tensor = Tensor::new(&attention_mask[..], &self.device)
            .and_then(|t| t.unsqueeze(0))
            .map_err(|e| CodememError::Embedding(format!("Tensor error: {e}")))?;

        // Forward pass -> [1, seq_len, hidden_size]
        let model = self.model.lock().unwrap();
        let hidden_states = model
            .forward(
                &input_ids_tensor,
                &token_type_ids,
                Some(&attention_mask_tensor),
            )
            .map_err(|e| CodememError::Embedding(format!("Model forward error: {e}")))?;
        drop(model);

        // Mean pooling weighted by attention mask
        // attention_mask: [1, seq_len] -> [1, seq_len, 1] for broadcasting
        let mask = attention_mask_tensor
            .to_dtype(DType::F32)
            .and_then(|t| t.unsqueeze(2))
            .map_err(|e| CodememError::Embedding(format!("Mask error: {e}")))?;

        let sum_mask = mask
            .sum(1)
            .map_err(|e| CodememError::Embedding(format!("Sum error: {e}")))?;

        let pooled = hidden_states
            .broadcast_mul(&mask)
            .and_then(|t| t.sum(1))
            .and_then(|t| t.broadcast_div(&sum_mask))
            .map_err(|e| CodememError::Embedding(format!("Pooling error: {e}")))?;

        // L2 normalize
        let normalized = pooled
            .sqr()
            .and_then(|t| t.sum_keepdim(1))
            .and_then(|t| t.sqrt())
            .and_then(|norm| pooled.broadcast_div(&norm))
            .map_err(|e| CodememError::Embedding(format!("Normalize error: {e}")))?;

        // Extract as Vec<f32> — shape is [1, hidden_size], squeeze to [hidden_size]
        let embedding: Vec<f32> = normalized
            .squeeze(0)
            .and_then(|t| t.to_vec1())
            .map_err(|e| CodememError::Embedding(format!("Extract error: {e}")))?;

        Ok(embedding)
    }

    /// Embed a batch of texts with cache-aware batching.
    ///
    /// Checks the LRU cache first and only runs the model on uncached texts,
    /// using a true batched forward pass (single GPU/CPU kernel launch per chunk).
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, CodememError> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        // Partition into cached and uncached
        let mut results: Vec<Option<Vec<f32>>> = vec![None; texts.len()];
        let mut uncached_indices = Vec::new();
        let mut uncached_texts = Vec::new();

        {
            let mut cache = self.cache.lock().unwrap();
            for (i, text) in texts.iter().enumerate() {
                if let Some(cached) = cache.get(*text) {
                    results[i] = Some(cached.clone());
                } else {
                    uncached_indices.push(i);
                    uncached_texts.push(*text);
                }
            }
        }

        if !uncached_texts.is_empty() {
            let new_embeddings = self.embed_batch_uncached(&uncached_texts)?;

            let mut cache = self.cache.lock().unwrap();
            for (idx, embedding) in uncached_indices.into_iter().zip(new_embeddings) {
                cache.put(texts[idx].to_string(), embedding.clone());
                results[idx] = Some(embedding);
            }
        }

        Ok(results.into_iter().map(|r| r.unwrap()).collect())
    }

    /// Embed a batch of texts without caching, using a true batched forward pass.
    ///
    /// Tokenizes all texts, pads to the longest sequence in each chunk, runs a
    /// single forward pass per chunk of up to `BATCH_SIZE` texts, then performs
    /// mean pooling and L2 normalization on the batched output.
    fn embed_batch_uncached(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, CodememError> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let mut all_embeddings = Vec::with_capacity(texts.len());

        for chunk in texts.chunks(BATCH_SIZE) {
            let mut tokenizer = self.tokenizer.clone();

            tokenizer
                .with_truncation(Some(TruncationParams {
                    max_length: MAX_SEQ_LENGTH,
                    ..Default::default()
                }))
                .map_err(|e| CodememError::Embedding(format!("Truncation error: {e}")))?;

            // Pad all sequences in this chunk to the length of the longest
            tokenizer.with_padding(Some(PaddingParams {
                strategy: PaddingStrategy::BatchLongest,
                ..Default::default()
            }));

            let encodings = tokenizer
                .encode_batch(chunk.to_vec(), true)
                .map_err(|e| CodememError::Embedding(format!("Batch encode error: {e}")))?;

            let batch_len = encodings.len();
            let seq_len = encodings[0].get_ids().len();

            // Flatten token IDs and attention masks into contiguous arrays
            let all_ids: Vec<u32> = encodings
                .iter()
                .flat_map(|e| e.get_ids())
                .copied()
                .collect();
            let all_masks: Vec<u32> = encodings
                .iter()
                .flat_map(|e| e.get_attention_mask())
                .copied()
                .collect();

            // Build tensors with shape [batch_size, seq_len]
            let input_ids = Tensor::new(all_ids.as_slice(), &self.device)
                .and_then(|t| t.reshape((batch_len, seq_len)))
                .map_err(|e| CodememError::Embedding(format!("Tensor error: {e}")))?;

            let token_type_ids = input_ids
                .zeros_like()
                .map_err(|e| CodememError::Embedding(format!("Tensor error: {e}")))?;

            let attention_mask = Tensor::new(all_masks.as_slice(), &self.device)
                .and_then(|t| t.reshape((batch_len, seq_len)))
                .map_err(|e| CodememError::Embedding(format!("Tensor error: {e}")))?;

            // Single forward pass -> [batch_size, seq_len, hidden_size]
            let model = self.model.lock().unwrap();
            let hidden_states = model
                .forward(&input_ids, &token_type_ids, Some(&attention_mask))
                .map_err(|e| CodememError::Embedding(format!("Forward error: {e}")))?;
            drop(model);

            // Mean pooling: mask [batch, seq] -> [batch, seq, 1] for broadcast
            let mask = attention_mask
                .to_dtype(DType::F32)
                .and_then(|t| t.unsqueeze(2))
                .map_err(|e| CodememError::Embedding(format!("Mask error: {e}")))?;

            let sum_mask = mask
                .sum(1)
                .map_err(|e| CodememError::Embedding(format!("Sum error: {e}")))?;

            let pooled = hidden_states
                .broadcast_mul(&mask)
                .and_then(|t| t.sum(1))
                .and_then(|t| t.broadcast_div(&sum_mask))
                .map_err(|e| CodememError::Embedding(format!("Pooling error: {e}")))?;

            // L2 normalize: [batch, hidden]
            let norm = pooled
                .sqr()
                .and_then(|t| t.sum_keepdim(1))
                .and_then(|t| t.sqrt())
                .map_err(|e| CodememError::Embedding(format!("Norm error: {e}")))?;

            let normalized = pooled
                .broadcast_div(&norm)
                .map_err(|e| CodememError::Embedding(format!("Normalize error: {e}")))?;

            // Extract each row as Vec<f32>
            for i in 0..batch_len {
                let row: Vec<f32> = normalized
                    .get(i)
                    .and_then(|t| t.to_vec1())
                    .map_err(|e| CodememError::Embedding(format!("Extract error: {e}")))?;
                all_embeddings.push(row);
            }
        }

        Ok(all_embeddings)
    }

    /// Get cache statistics: (current_size, capacity).
    pub fn cache_stats(&self) -> (usize, usize) {
        let cache = self.cache.lock().unwrap();
        (cache.len(), CACHE_CAPACITY)
    }
}

impl EmbeddingProvider for EmbeddingService {
    fn dimensions(&self) -> usize {
        DIMENSIONS
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>, CodememError> {
        self.embed(text)
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, CodememError> {
        self.embed_batch(texts)
    }

    fn name(&self) -> &str {
        "candle"
    }

    fn cache_stats(&self) -> (usize, usize) {
        self.cache_stats()
    }
}

// ── Cached Provider Wrapper ───────────────────────────────────────────────

/// Wraps any `EmbeddingProvider` with an LRU cache.
pub struct CachedProvider {
    inner: Box<dyn EmbeddingProvider>,
    cache: Mutex<LruCache<String, Vec<f32>>>,
}

impl CachedProvider {
    pub fn new(inner: Box<dyn EmbeddingProvider>, capacity: usize) -> Self {
        Self {
            inner,
            cache: Mutex::new(LruCache::new(
                NonZeroUsize::new(capacity).unwrap_or(NonZeroUsize::new(1).unwrap()),
            )),
        }
    }
}

impl EmbeddingProvider for CachedProvider {
    fn dimensions(&self) -> usize {
        self.inner.dimensions()
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>, CodememError> {
        {
            let mut cache = self.cache.lock().unwrap();
            if let Some(cached) = cache.get(text) {
                return Ok(cached.clone());
            }
        }
        let embedding = self.inner.embed(text)?;
        {
            let mut cache = self.cache.lock().unwrap();
            cache.put(text.to_string(), embedding.clone());
        }
        Ok(embedding)
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, CodememError> {
        // Check cache, only forward uncached texts
        let mut results: Vec<Option<Vec<f32>>> = vec![None; texts.len()];
        let mut uncached = Vec::new();
        let mut uncached_idx = Vec::new();

        {
            let mut cache = self.cache.lock().unwrap();
            for (i, text) in texts.iter().enumerate() {
                if let Some(cached) = cache.get(*text) {
                    results[i] = Some(cached.clone());
                } else {
                    uncached_idx.push(i);
                    uncached.push(*text);
                }
            }
        }

        if !uncached.is_empty() {
            let new_embeddings = self.inner.embed_batch(&uncached)?;
            let mut cache = self.cache.lock().unwrap();
            for (idx, embedding) in uncached_idx.into_iter().zip(new_embeddings) {
                cache.put(texts[idx].to_string(), embedding.clone());
                results[idx] = Some(embedding);
            }
        }

        Ok(results.into_iter().map(|r| r.unwrap()).collect())
    }

    fn name(&self) -> &str {
        self.inner.name()
    }

    fn cache_stats(&self) -> (usize, usize) {
        let cache = self.cache.lock().unwrap();
        (cache.len(), cache.cap().into())
    }
}

// ── Factory ───────────────────────────────────────────────────────────────

/// Create an embedding provider from environment variables.
///
/// | Variable | Values | Default |
/// |----------|--------|---------|
/// | `CODEMEM_EMBED_PROVIDER` | `candle`, `ollama`, `openai` | `candle` |
/// | `CODEMEM_EMBED_MODEL` | model name | provider default |
/// | `CODEMEM_EMBED_URL` | base URL | provider default |
/// | `CODEMEM_EMBED_API_KEY` | API key | also reads `OPENAI_API_KEY` |
/// | `CODEMEM_EMBED_DIMENSIONS` | integer | `768` |
pub fn from_env() -> Result<Box<dyn EmbeddingProvider>, CodememError> {
    let provider = std::env::var("CODEMEM_EMBED_PROVIDER")
        .unwrap_or_else(|_| "candle".to_string())
        .to_lowercase();
    let dimensions: usize = std::env::var("CODEMEM_EMBED_DIMENSIONS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DIMENSIONS);

    match provider.as_str() {
        "ollama" => {
            let base_url = std::env::var("CODEMEM_EMBED_URL")
                .unwrap_or_else(|_| ollama::DEFAULT_BASE_URL.to_string());
            let model = std::env::var("CODEMEM_EMBED_MODEL")
                .unwrap_or_else(|_| ollama::DEFAULT_MODEL.to_string());
            let inner = Box::new(ollama::OllamaProvider::new(&base_url, &model, dimensions));
            Ok(Box::new(CachedProvider::new(inner, CACHE_CAPACITY)))
        }
        "openai" => {
            let api_key = std::env::var("CODEMEM_EMBED_API_KEY")
                .or_else(|_| std::env::var("OPENAI_API_KEY"))
                .map_err(|_| {
                    CodememError::Embedding(
                        "CODEMEM_EMBED_API_KEY or OPENAI_API_KEY required for OpenAI embeddings"
                            .into(),
                    )
                })?;
            let model = std::env::var("CODEMEM_EMBED_MODEL")
                .unwrap_or_else(|_| openai::DEFAULT_MODEL.to_string());
            let base_url = std::env::var("CODEMEM_EMBED_URL").ok();
            let inner = Box::new(openai::OpenAIProvider::new(
                &api_key,
                &model,
                dimensions,
                base_url.as_deref(),
            ));
            Ok(Box::new(CachedProvider::new(inner, CACHE_CAPACITY)))
        }
        "candle" | "" => {
            let model_dir = EmbeddingService::default_model_dir();
            let service = EmbeddingService::new(&model_dir)?;
            Ok(Box::new(service))
        }
        other => Err(CodememError::Embedding(format!(
            "Unknown embedding provider: '{}'. Use 'candle', 'ollama', or 'openai'.",
            other
        ))),
    }
}
