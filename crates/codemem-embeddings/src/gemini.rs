//! Google Gemini embedding provider for Codemem.
//!
//! Uses the Generative Language API (`generativelanguage.googleapis.com`).
//! Default model: `text-embedding-004` (768 dimensions).
//!
//! ```bash
//! export CODEMEM_EMBED_PROVIDER=gemini
//! export CODEMEM_EMBED_API_KEY=AIza...
//! # Optional:
//! export CODEMEM_EMBED_MODEL=text-embedding-004
//! export CODEMEM_EMBED_DIMENSIONS=768
//! ```

use codemem_core::CodememError;

/// Default Gemini API base URL.
pub const DEFAULT_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta";

/// Default embedding model.
pub const DEFAULT_MODEL: &str = "text-embedding-004";

/// Gemini embedding provider.
pub struct GeminiProvider {
    api_key: String,
    model: String,
    dimensions: usize,
    pub(crate) base_url: String,
    client: reqwest::blocking::Client,
}

impl GeminiProvider {
    /// Create a new Gemini provider.
    pub fn new(api_key: &str, model: &str, dimensions: usize, base_url: Option<&str>) -> Self {
        Self {
            api_key: api_key.to_string(),
            model: model.to_string(),
            dimensions,
            base_url: base_url.unwrap_or(DEFAULT_BASE_URL).to_string(),
            client: reqwest::blocking::Client::new(),
        }
    }
}

impl super::EmbeddingProvider for GeminiProvider {
    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>, CodememError> {
        let url = format!("{}/models/{}:embedContent", self.base_url, self.model);

        let mut body = serde_json::json!({
            "model": format!("models/{}", self.model),
            "content": {
                "parts": [{"text": text}]
            },
            "taskType": "RETRIEVAL_DOCUMENT",
        });

        if self.dimensions > 0 {
            body["outputDimensionality"] = serde_json::json!(self.dimensions);
        }

        let response = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .header("x-goog-api-key", &self.api_key)
            .json(&body)
            .send()
            .map_err(|e| CodememError::Embedding(format!("Gemini request failed: {e}")))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().unwrap_or_default();
            return Err(CodememError::Embedding(format!(
                "Gemini returned status {status}: {body}",
            )));
        }

        let json: serde_json::Value = response
            .json()
            .map_err(|e| CodememError::Embedding(format!("Gemini response parse error: {e}")))?;

        let embedding: Vec<f32> = json
            .get("embedding")
            .and_then(|v| v.get("values"))
            .and_then(|v| v.as_array())
            .ok_or_else(|| {
                CodememError::Embedding("Missing embedding.values in Gemini response".into())
            })?
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0) as f32)
            .collect();

        if self.dimensions > 0 && embedding.len() != self.dimensions {
            return Err(CodememError::Embedding(format!(
                "Gemini returned {} dimensions, expected {}",
                embedding.len(),
                self.dimensions
            )));
        }

        Ok(embedding)
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, CodememError> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let url = format!("{}/models/{}:batchEmbedContents", self.base_url, self.model);

        // Gemini batchEmbedContents accepts max 100 requests per call.
        const MAX_BATCH: usize = 100;
        let mut all_embeddings = Vec::with_capacity(texts.len());

        for chunk in texts.chunks(MAX_BATCH) {
            let requests: Vec<serde_json::Value> = chunk
                .iter()
                .map(|text| {
                    let mut req = serde_json::json!({
                        "model": format!("models/{}", self.model),
                        "content": {
                            "parts": [{"text": text}]
                        },
                        "taskType": "RETRIEVAL_DOCUMENT",
                    });
                    if self.dimensions > 0 {
                        req["outputDimensionality"] = serde_json::json!(self.dimensions);
                    }
                    req
                })
                .collect();

            let body = serde_json::json!({ "requests": requests });

            let response = self
                .client
                .post(&url)
                .header("Content-Type", "application/json")
                .header("x-goog-api-key", &self.api_key)
                .json(&body)
                .send()
                .map_err(|e| {
                    CodememError::Embedding(format!("Gemini batch request failed: {e}"))
                })?;

            if !response.status().is_success() {
                let status = response.status();
                let body = response.text().unwrap_or_default();
                return Err(CodememError::Embedding(format!(
                    "Gemini returned status {status}: {body}",
                )));
            }

            let json: serde_json::Value = response.json().map_err(|e| {
                CodememError::Embedding(format!("Gemini response parse error: {e}"))
            })?;

            let embeddings = json
                .get("embeddings")
                .and_then(|v| v.as_array())
                .ok_or_else(|| {
                    CodememError::Embedding("Missing 'embeddings' array in Gemini response".into())
                })?;

            if embeddings.len() != chunk.len() {
                return Err(CodememError::Embedding(format!(
                    "Gemini returned {} embeddings, expected {}",
                    embeddings.len(),
                    chunk.len()
                )));
            }

            for (i, item) in embeddings.iter().enumerate() {
                let embedding: Vec<f32> = item
                    .get("values")
                    .and_then(|v| v.as_array())
                    .ok_or_else(|| {
                        CodememError::Embedding(format!(
                            "Missing values in Gemini embedding at index {i}"
                        ))
                    })?
                    .iter()
                    .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                    .collect();

                if self.dimensions > 0 && embedding.len() != self.dimensions {
                    return Err(CodememError::Embedding(format!(
                        "Gemini returned {} dimensions at index {i}, expected {}",
                        embedding.len(),
                        self.dimensions
                    )));
                }
                all_embeddings.push(embedding);
            }
        }

        Ok(all_embeddings)
    }

    fn name(&self) -> &str {
        "gemini"
    }
}

#[cfg(test)]
#[path = "tests/gemini_tests.rs"]
mod tests;
