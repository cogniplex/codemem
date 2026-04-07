//! codemem-embeddings: Pluggable embedding providers for Codemem.
//!
//! Supports multiple backends:
//! - **Candle** (default): Local BERT models via pure Rust ML (any HF BERT model)
//! - **Ollama**: Local Ollama server with any embedding model
//! - **OpenAI**: OpenAI API or any compatible endpoint (Together, Azure, etc.)
//! - **Gemini**: Google Generative Language API (text-embedding-004)

pub mod gemini;
pub mod ollama;
pub mod openai;

use candle_core::{DType, Device, Tensor};
use candle_nn::{Module, VarBuilder};
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use candle_transformers::models::jina_bert::{
    BertModel as JinaBertModel, Config as JinaBertConfig,
};
use codemem_core::CodememError;
use lru::LruCache;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use tokenizers::{PaddingParams, PaddingStrategy};

/// Default model name (short form used for directory naming).
pub const MODEL_NAME: &str = "bge-base-en-v1.5";

/// Default HuggingFace model repo ID.
/// Used internally and by `commands_init` for the default model download.
pub const DEFAULT_HF_REPO: &str = "BAAI/bge-base-en-v1.5";

/// Default embedding dimensions for remote providers (Ollama/OpenAI).
/// Candle reads `hidden_size` from the model's config.json instead.
pub const DEFAULT_REMOTE_DIMENSIONS: usize = 768;

/// Default max sequence length for standard BERT models (used when config doesn't specify).
const DEFAULT_MAX_SEQ_LENGTH: usize = 512;

/// Default LRU cache capacity.
pub const CACHE_CAPACITY: usize = 10_000;

// Re-export EmbeddingProvider trait from core
pub use codemem_core::EmbeddingProvider;

// ── Candle Embedding Service ────────────────────────────────────────────────

/// Default batch size for batched embedding forward passes.
/// Configurable via `EmbeddingConfig.batch_size` or `CODEMEM_EMBED_BATCH_SIZE`.
pub const DEFAULT_BATCH_SIZE: usize = 16;

/// Select the best available compute device.
///
/// Tries Metal (macOS GPU) first, then CUDA (NVIDIA GPU), then falls back to CPU.
/// GPU backends are only available when the corresponding feature flag is enabled.
fn select_device() -> Device {
    #[cfg(feature = "metal")]
    {
        // Use catch_unwind to handle SIGBUS/panics on CI runners without GPU access.
        match std::panic::catch_unwind(|| Device::new_metal(0)) {
            Ok(Ok(device)) => {
                tracing::info!("Using Metal GPU for embeddings");
                return device;
            }
            Ok(Err(e)) => {
                tracing::warn!("Metal device creation failed: {e}, falling back to CPU");
            }
            Err(_) => {
                tracing::warn!("Metal device creation panicked, falling back to CPU");
            }
        }
    }
    #[cfg(feature = "cuda")]
    {
        match std::panic::catch_unwind(|| Device::new_cuda(0)) {
            Ok(Ok(device)) => {
                tracing::info!("Using CUDA GPU for embeddings");
                return device;
            }
            Ok(Err(e)) => {
                tracing::warn!("CUDA device creation failed: {e}, falling back to CPU");
            }
            Err(_) => {
                tracing::warn!("CUDA device creation panicked, falling back to CPU");
            }
        }
    }
    tracing::info!("Using CPU for embeddings");
    Device::Cpu
}

/// Model backend enum — dispatches forward passes to the correct architecture.
enum ModelBackend {
    /// Standard BERT (absolute positional embeddings). Used by BGE, MiniLM, etc.
    Bert(BertModel),
    /// JinaBERT (ALiBi positional embeddings). Used by Jina embeddings v2.
    JinaBert(JinaBertModel),
}

/// Embedding service with Candle inference (no internal cache — use `CachedProvider` wrapper).
pub struct EmbeddingService {
    model: Mutex<ModelBackend>,
    /// Tokenizer pre-configured with truncation (no padding).
    /// Used directly for single embeds; cloned and augmented with padding for batch.
    tokenizer: tokenizers::Tokenizer,
    device: Device,
    /// Maximum texts per forward pass (GPU memory trade-off).
    batch_size: usize,
    /// Hidden size read from model config (e.g. 768 for bge-base, 384 for bge-small).
    hidden_size: usize,
    /// Max sequence length (512 for BERT, up to 8192 for JinaBERT).
    max_seq_length: usize,
}

/// Minimal struct for sniffing model architecture from config.json before full parsing.
#[derive(serde::Deserialize)]
struct ConfigProbe {
    #[serde(default)]
    position_embedding_type: Option<String>,
    hidden_size: usize,
    #[serde(default = "default_max_position_embeddings")]
    max_position_embeddings: usize,
}

fn default_max_position_embeddings() -> usize {
    DEFAULT_MAX_SEQ_LENGTH
}

impl EmbeddingService {
    /// Create a new embedding service, loading model from the given directory.
    /// Expects `model.safetensors`, `config.json`, and `tokenizer.json` in the directory.
    ///
    /// Auto-detects model architecture (BERT vs JinaBERT) from config.json.
    /// `dtype` controls precision: `DType::F32` (default) or `DType::F16` (half memory, faster on Metal).
    pub fn new(model_dir: &Path, batch_size: usize, dtype: DType) -> Result<Self, CodememError> {
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

        tracing::info!(
            "Loading model from {} (dtype: {:?}, device: {:?})",
            model_dir.display(),
            dtype,
            device
        );

        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| CodememError::Embedding(format!("Failed to read config: {e}")))?;

        // Probe config to detect architecture before full parsing
        let probe: ConfigProbe = serde_json::from_str(&config_str)
            .map_err(|e| CodememError::Embedding(format!("Failed to probe config: {e}")))?;
        let hidden_size = probe.hidden_size;
        let is_alibi = probe
            .position_embedding_type
            .as_deref()
            .is_some_and(|t| t == "alibi");
        // Cap at 8192 to avoid excessive memory usage even if model claims more
        let max_seq_length = probe.max_position_embeddings.min(8192);

        let (model, arch_name) = if is_alibi {
            // JinaBERT (ALiBi positional embeddings)
            let config: JinaBertConfig = serde_json::from_str(&config_str)
                .map_err(|e| CodememError::Embedding(format!("Failed to parse JinaBERT config: {e}")))?;
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(&[&model_path], dtype, &device)
                    .map_err(|e| CodememError::Embedding(format!("Failed to load weights: {e}")))?
            };
            // JinaBERT weights use "bert." prefix
            let jina_model = JinaBertModel::new(vb.pp("bert"), &config).map_err(|e| {
                CodememError::Embedding(format!("Failed to load JinaBERT model: {e}"))
            })?;
            (ModelBackend::JinaBert(jina_model), "JinaBERT (ALiBi)")
        } else {
            // Standard BERT (absolute positional embeddings)
            let config: BertConfig = serde_json::from_str(&config_str)
                .map_err(|e| CodememError::Embedding(format!("Failed to parse BERT config: {e}")))?;
            // Load model weights from safetensors via memory-mapped IO.
            // Scope vb so it drops before a potential retry, avoiding two VarBuilders
            // holding materialized Metal tensors simultaneously.
            let bert_model = {
                let vb = unsafe {
                    VarBuilder::from_mmaped_safetensors(&[&model_path], dtype, &device)
                        .map_err(|e| CodememError::Embedding(format!("Failed to load weights: {e}")))?
                };
                BertModel::load(vb.pp("bert"), &config)
            };
            // Try with "bert." prefix first (standard HF BERT models), then without
            let bert_model = match bert_model {
                Ok(m) => m,
                Err(_) => {
                    let vb2 = unsafe {
                        VarBuilder::from_mmaped_safetensors(&[&model_path], dtype, &device).map_err(
                            |e| CodememError::Embedding(format!("Failed to load weights: {e}")),
                        )?
                    };
                    BertModel::load(vb2, &config).map_err(|e| {
                        CodememError::Embedding(format!("Failed to load BERT model: {e}"))
                    })?
                }
            };
            (ModelBackend::Bert(bert_model), "BERT (absolute)")
        };

        tracing::info!(
            "Loaded {} model (hidden_size={}, max_seq_length={})",
            arch_name,
            hidden_size,
            max_seq_length
        );

        let mut tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| CodememError::Embedding(e.to_string()))?;

        // Pre-configure truncation once so we don't need to clone on every embed call.
        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: max_seq_length,
                ..Default::default()
            }))
            .map_err(|e| CodememError::Embedding(format!("Truncation error: {e}")))?;

        Ok(Self {
            model: Mutex::new(model),
            tokenizer,
            device,
            batch_size,
            hidden_size,
            max_seq_length,
        })
    }

    /// Maximum sequence length this model supports.
    pub fn max_seq_length(&self) -> usize {
        self.max_seq_length
    }

    /// Get the model directory path for a given model name.
    /// Falls back to `~/.codemem/models/{model_name}`.
    pub fn model_dir_for(model_name: &str) -> PathBuf {
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".codemem")
            .join("models")
            .join(model_name)
    }

    /// Get the default model directory path (~/.codemem/models/{MODEL_NAME}).
    pub fn default_model_dir() -> PathBuf {
        Self::model_dir_for(MODEL_NAME)
    }

    /// Download a model from HuggingFace Hub to the given directory.
    /// `hf_repo` is the full repo ID (e.g. "BAAI/bge-base-en-v1.5").
    /// Returns the directory path. No-ops if model already exists.
    pub fn download_model(dest_dir: &Path, hf_repo: &str) -> Result<PathBuf, CodememError> {
        let model_dest = dest_dir.join("model.safetensors");
        let config_dest = dest_dir.join("config.json");
        let tokenizer_dest = dest_dir.join("tokenizer.json");

        if model_dest.exists() && config_dest.exists() && tokenizer_dest.exists() {
            tracing::info!("Model already downloaded at {}", dest_dir.display());
            return Ok(dest_dir.to_path_buf());
        }

        std::fs::create_dir_all(dest_dir)
            .map_err(|e| CodememError::Embedding(format!("Failed to create dir: {e}")))?;

        tracing::info!("Downloading {} from HuggingFace...", hf_repo);

        let api = hf_hub::api::sync::Api::new()
            .map_err(|e| CodememError::Embedding(format!("HuggingFace API error: {e}")))?;
        let repo = api.model(hf_repo.to_string());

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

    /// Download the default model (BAAI/bge-base-en-v1.5) to the default directory.
    /// Convenience wrapper for `download_model(&default_model_dir(), DEFAULT_HF_REPO)`.
    pub fn download_default_model() -> Result<PathBuf, CodememError> {
        Self::download_model(&Self::default_model_dir(), DEFAULT_HF_REPO)
    }

    /// Embed a single text string. Returns an L2-normalized vector (dimension = model's hidden_size).
    pub fn embed(&self, text: &str) -> Result<Vec<f32>, CodememError> {
        // Tokenize using pre-configured tokenizer (truncation already set in constructor)
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| CodememError::Embedding(e.to_string()))?;

        let input_ids: Vec<u32> = encoding.get_ids().to_vec();
        let attention_mask: Vec<u32> = encoding.get_attention_mask().to_vec();

        // Build candle tensors with shape [1, seq_len]
        let input_ids_tensor = Tensor::new(&input_ids[..], &self.device)
            .and_then(|t| t.unsqueeze(0))
            .map_err(|e| CodememError::Embedding(format!("Tensor error: {e}")))?;

        let attention_mask_tensor = Tensor::new(&attention_mask[..], &self.device)
            .and_then(|t| t.unsqueeze(0))
            .map_err(|e| CodememError::Embedding(format!("Tensor error: {e}")))?;

        // Forward pass -> [1, seq_len, hidden_size]
        let model = self
            .model
            .lock()
            .map_err(|e| CodememError::LockPoisoned(format!("embedding model: {e}")))?;
        let hidden_states = match &*model {
            ModelBackend::Bert(bert) => {
                let token_type_ids = input_ids_tensor
                    .zeros_like()
                    .map_err(|e| CodememError::Embedding(format!("Tensor error: {e}")))?;
                let result = bert
                    .forward(&input_ids_tensor, &token_type_ids, Some(&attention_mask_tensor))
                    .map_err(|e| CodememError::Embedding(format!("Model forward error: {e}")))?;
                drop(token_type_ids);
                result
            }
            ModelBackend::JinaBert(jina) => jina
                .forward(&input_ids_tensor)
                .map_err(|e| CodememError::Embedding(format!("Model forward error: {e}")))?,
        };
        drop(model);
        drop(input_ids_tensor);

        // Cast hidden states to F32 for pooling math (model may output F16/BF16)
        let hidden_states = hidden_states
            .to_dtype(DType::F32)
            .map_err(|e| CodememError::Embedding(format!("Cast error: {e}")))?;

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

    /// Embed a batch of texts using a true batched forward pass.
    ///
    /// Tokenizes all texts, pads to the longest sequence in each chunk, runs a
    /// single forward pass per chunk of up to `batch_size` texts, then performs
    /// mean pooling and L2 normalization on the batched output.
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, CodememError> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let mut all_embeddings = Vec::with_capacity(texts.len());

        for chunk in texts.chunks(self.batch_size) {
            // Clone tokenizer only for batch path — needs per-chunk padding config.
            // Truncation is already configured on self.tokenizer.
            let mut tokenizer = self.tokenizer.clone();
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

            let attention_mask = Tensor::new(all_masks.as_slice(), &self.device)
                .and_then(|t| t.reshape((batch_len, seq_len)))
                .map_err(|e| CodememError::Embedding(format!("Tensor error: {e}")))?;

            // Single forward pass -> [batch_size, seq_len, hidden_size]
            let model = self
                .model
                .lock()
                .map_err(|e| CodememError::LockPoisoned(format!("embedding model: {e}")))?;
            let hidden_states = match &*model {
                ModelBackend::Bert(bert) => {
                    let token_type_ids = input_ids
                        .zeros_like()
                        .map_err(|e| CodememError::Embedding(format!("Tensor error: {e}")))?;
                    let result = bert
                        .forward(&input_ids, &token_type_ids, Some(&attention_mask))
                        .map_err(|e| CodememError::Embedding(format!("Forward error: {e}")))?;
                    drop(token_type_ids);
                    result
                }
                ModelBackend::JinaBert(jina) => jina
                    .forward(&input_ids)
                    .map_err(|e| CodememError::Embedding(format!("Forward error: {e}")))?,
            };
            drop(model);
            drop(input_ids);

            // Cast hidden states to F32 for pooling math (model may output F16/BF16)
            let hidden_states = hidden_states
                .to_dtype(DType::F32)
                .map_err(|e| CodememError::Embedding(format!("Cast error: {e}")))?;

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

            // Single GPU→CPU blit: flatten all rows, then slice on CPU.
            // to_vec1() implicitly syncs the GPU pipeline (data must be ready to read).
            let flat: Vec<f32> = normalized
                .flatten_all()
                .and_then(|t| t.to_vec1())
                .map_err(|e| CodememError::Embedding(format!("Extract error: {e}")))?;
            for i in 0..batch_len {
                let start = i * self.hidden_size;
                all_embeddings.push(flat[start..start + self.hidden_size].to_vec());
            }
        }

        Ok(all_embeddings)
    }
}

impl EmbeddingProvider for EmbeddingService {
    fn dimensions(&self) -> usize {
        self.hidden_size
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
}

// ── Cached Provider Wrapper ───────────────────────────────────────────────

/// Wraps any `EmbeddingProvider` with an LRU cache.
pub struct CachedProvider {
    inner: Box<dyn EmbeddingProvider>,
    cache: Mutex<LruCache<String, Vec<f32>>>,
}

impl CachedProvider {
    pub fn new(inner: Box<dyn EmbeddingProvider>, capacity: usize) -> Self {
        // SAFETY: 1 is non-zero, so the inner expect is infallible
        let cap =
            NonZeroUsize::new(capacity).unwrap_or(NonZeroUsize::new(1).expect("1 is non-zero"));
        Self {
            inner,
            cache: Mutex::new(LruCache::new(cap)),
        }
    }
}

impl EmbeddingProvider for CachedProvider {
    fn dimensions(&self) -> usize {
        self.inner.dimensions()
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>, CodememError> {
        {
            let mut cache = self
                .cache
                .lock()
                .map_err(|e| CodememError::LockPoisoned(format!("cached provider: {e}")))?;
            if let Some(cached) = cache.get(text) {
                return Ok(cached.clone());
            }
        }
        let embedding = self.inner.embed(text)?;
        {
            let mut cache = self
                .cache
                .lock()
                .map_err(|e| CodememError::LockPoisoned(format!("cached provider: {e}")))?;
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
            let mut cache = self
                .cache
                .lock()
                .map_err(|e| CodememError::LockPoisoned(format!("cached provider: {e}")))?;
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
            let mut cache = self
                .cache
                .lock()
                .map_err(|e| CodememError::LockPoisoned(format!("cached provider: {e}")))?;
            for (idx, embedding) in uncached_idx.into_iter().zip(new_embeddings) {
                cache.put(texts[idx].to_string(), embedding.clone());
                results[idx] = Some(embedding);
            }
        }

        // Verify all texts got embeddings — flatten() would silently drop Nones
        let expected = texts.len();
        let output: Vec<Vec<f32>> = results
            .into_iter()
            .enumerate()
            .map(|(i, opt)| {
                opt.ok_or_else(|| {
                    CodememError::Embedding(format!(
                        "Missing embedding for text at index {i} (batch size {expected})"
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(output)
    }

    fn name(&self) -> &str {
        self.inner.name()
    }

    fn cache_stats(&self) -> (usize, usize) {
        match self.cache.lock() {
            Ok(cache) => (cache.len(), cache.cap().into()),
            Err(_) => (0, 0),
        }
    }
}

// ── Factory ───────────────────────────────────────────────────────────────

/// Parse a dtype string into a Candle DType.
///
/// Supported values: "f16" (default, half precision — less memory, faster on Metal), "f32", "bf16".
pub fn parse_dtype(s: &str) -> Result<DType, CodememError> {
    match s.to_lowercase().as_str() {
        "f16" | "float16" | "half" | "" => Ok(DType::F16),
        "f32" | "float32" => Ok(DType::F32),
        "bf16" | "bfloat16" => Ok(DType::BF16),
        other => Err(CodememError::Embedding(format!(
            "Unknown dtype: '{}'. Use 'f16', 'f32', or 'bf16'.",
            other
        ))),
    }
}

/// Resolve the HuggingFace repo ID and local directory name from a model identifier.
///
/// Accepts:
/// - Full HF repo: `"BAAI/bge-base-en-v1.5"` → repo=`"BAAI/bge-base-en-v1.5"`, dir=`"bge-base-en-v1.5"`
/// - Short name: `"bge-small-en-v1.5"` → repo=`"BAAI/bge-small-en-v1.5"`, dir=`"bge-small-en-v1.5"`
///
/// Returns `Err` if the model identifier is a bare name without an org prefix and isn't
/// a recognized `bge-*` shorthand — HuggingFace requires `org/repo` format.
pub fn resolve_model_id(model: &str) -> Result<(String, String), CodememError> {
    if model.contains('/') {
        // Full repo ID — directory name is the part after the slash
        let dir_name = model.rsplit('/').next().unwrap_or(model);
        Ok((model.to_string(), dir_name.to_string()))
    } else if model.starts_with("bge-") {
        // Short name — assume BAAI namespace for bge-* models
        Ok((format!("BAAI/{model}"), model.to_string()))
    } else {
        Err(CodememError::Embedding(format!(
            "Model identifier '{}' must be a full HuggingFace repo ID (e.g., 'BAAI/bge-base-en-v1.5' \
             or 'sentence-transformers/all-MiniLM-L6-v2'). Short names are only supported for 'bge-*' models.",
            model
        )))
    }
}

/// Create an embedding provider from environment variables.
///
/// When `config` is provided, its fields serve as defaults; env vars override them.
///
/// | Variable | Values | Default |
/// |----------|--------|---------|
/// | `CODEMEM_EMBED_PROVIDER` | `candle`, `ollama`, `openai`, `gemini` | `candle` |
/// | `CODEMEM_EMBED_MODEL` | model name or HF repo | `BAAI/bge-base-en-v1.5` |
/// | `CODEMEM_EMBED_URL` | base URL | provider default |
/// | `CODEMEM_EMBED_API_KEY` | API key | also reads `OPENAI_API_KEY` / `GEMINI_API_KEY` / `GOOGLE_API_KEY` |
/// | `CODEMEM_EMBED_DIMENSIONS` | integer | read from model config |
/// | `CODEMEM_EMBED_BATCH_SIZE` | integer | `16` |
/// | `CODEMEM_EMBED_DTYPE` | `f16`, `f32`, `bf16` | `f16` |
pub fn from_env(
    config: Option<&codemem_core::EmbeddingConfig>,
) -> Result<Box<dyn EmbeddingProvider>, CodememError> {
    let provider = std::env::var("CODEMEM_EMBED_PROVIDER")
        .unwrap_or_else(|_| {
            config
                .map(|c| c.provider.clone())
                .unwrap_or_else(|| "candle".to_string())
        })
        .to_lowercase();
    // For Ollama/OpenAI, dimensions must be specified explicitly (remote APIs need it).
    // For Candle, this value is ignored — hidden_size is read from the model's config.json.
    let dimensions: usize = std::env::var("CODEMEM_EMBED_DIMENSIONS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| config.map_or(DEFAULT_REMOTE_DIMENSIONS, |c| c.dimensions));
    let cache_capacity = config.map_or(CACHE_CAPACITY, |c| c.cache_capacity);
    let batch_size: usize = std::env::var("CODEMEM_EMBED_BATCH_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| config.map_or(DEFAULT_BATCH_SIZE, |c| c.batch_size));

    match provider.as_str() {
        "ollama" => {
            let base_url = std::env::var("CODEMEM_EMBED_URL").unwrap_or_else(|_| {
                config
                    .filter(|c| !c.url.is_empty())
                    .map(|c| c.url.clone())
                    .unwrap_or_else(|| ollama::DEFAULT_BASE_URL.to_string())
            });
            let model = std::env::var("CODEMEM_EMBED_MODEL").unwrap_or_else(|_| {
                config
                    .filter(|c| !c.model.is_empty())
                    .map(|c| c.model.clone())
                    .unwrap_or_else(|| ollama::DEFAULT_MODEL.to_string())
            });
            let inner = Box::new(ollama::OllamaProvider::new(&base_url, &model, dimensions));
            Ok(Box::new(CachedProvider::new(inner, cache_capacity)))
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
            let model = std::env::var("CODEMEM_EMBED_MODEL").unwrap_or_else(|_| {
                config
                    .filter(|c| !c.model.is_empty())
                    .map(|c| c.model.clone())
                    .unwrap_or_else(|| openai::DEFAULT_MODEL.to_string())
            });
            let base_url = std::env::var("CODEMEM_EMBED_URL")
                .ok()
                .or_else(|| config.filter(|c| !c.url.is_empty()).map(|c| c.url.clone()));
            let inner = Box::new(openai::OpenAIProvider::new(
                &api_key,
                &model,
                dimensions,
                base_url.as_deref(),
            ));
            Ok(Box::new(CachedProvider::new(inner, cache_capacity)))
        }
        "gemini" | "google" => {
            let api_key = std::env::var("CODEMEM_EMBED_API_KEY")
                .or_else(|_| std::env::var("GEMINI_API_KEY"))
                .or_else(|_| std::env::var("GOOGLE_API_KEY"))
                .map_err(|_| {
                    CodememError::Embedding(
                        "CODEMEM_EMBED_API_KEY, GEMINI_API_KEY, or GOOGLE_API_KEY required for Gemini embeddings"
                            .into(),
                    )
                })?;
            let model = std::env::var("CODEMEM_EMBED_MODEL").unwrap_or_else(|_| {
                config
                    .filter(|c| !c.model.is_empty())
                    .map(|c| c.model.clone())
                    .unwrap_or_else(|| gemini::DEFAULT_MODEL.to_string())
            });
            let base_url = std::env::var("CODEMEM_EMBED_URL")
                .ok()
                .or_else(|| config.filter(|c| !c.url.is_empty()).map(|c| c.url.clone()));
            let inner = Box::new(gemini::GeminiProvider::new(
                &api_key,
                &model,
                dimensions,
                base_url.as_deref(),
            ));
            Ok(Box::new(CachedProvider::new(inner, cache_capacity)))
        }
        "candle" | "" => {
            let model_id = std::env::var("CODEMEM_EMBED_MODEL").unwrap_or_else(|_| {
                config
                    .filter(|c| !c.model.is_empty())
                    .map(|c| c.model.clone())
                    .unwrap_or_else(|| DEFAULT_HF_REPO.to_string())
            });
            let (hf_repo, dir_name) = resolve_model_id(&model_id)?;
            let model_dir = EmbeddingService::model_dir_for(&dir_name);

            let dtype_str = std::env::var("CODEMEM_EMBED_DTYPE").unwrap_or_else(|_| {
                config
                    .filter(|c| !c.dtype.is_empty())
                    .map(|c| c.dtype.clone())
                    .unwrap_or_else(|| "f16".to_string())
            });
            let dtype = parse_dtype(&dtype_str)?;

            let service = EmbeddingService::new(&model_dir, batch_size, dtype).map_err(|e| {
                // Enhance error message with download hint for non-default models
                if e.to_string().contains("Model not found") && hf_repo != DEFAULT_HF_REPO {
                    CodememError::Embedding(format!(
                        "Model '{}' not found at {}. Download it with:\n  \
                         CODEMEM_EMBED_MODEL={} codemem init",
                        hf_repo,
                        model_dir.display(),
                        hf_repo
                    ))
                } else {
                    e
                }
            })?;
            Ok(Box::new(CachedProvider::new(
                Box::new(service),
                cache_capacity,
            )))
        }
        other => Err(CodememError::Embedding(format!(
            "Unknown embedding provider: '{}'. Use 'candle', 'ollama', 'openai', or 'gemini'.",
            other
        ))),
    }
}

#[cfg(test)]
#[path = "tests/lib_tests.rs"]
mod tests;
