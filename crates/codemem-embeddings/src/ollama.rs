//! Ollama embedding provider for Codemem.
//!
//! Uses Ollama's local API to generate embeddings.
//! Default model: nomic-embed-text (768 dimensions).

use codemem_core::CodememError;

/// Default Ollama base URL.
pub const DEFAULT_BASE_URL: &str = "http://localhost:11434";

/// Default Ollama embedding model.
pub const DEFAULT_MODEL: &str = "nomic-embed-text";

/// Ollama embedding provider.
pub struct OllamaProvider {
    base_url: String,
    model: String,
    dimensions: usize,
    client: reqwest::blocking::Client,
}

impl OllamaProvider {
    /// Create a new Ollama provider.
    pub fn new(base_url: &str, model: &str, dimensions: usize) -> Self {
        Self {
            base_url: base_url.to_string(),
            model: model.to_string(),
            dimensions,
            client: reqwest::blocking::Client::new(),
        }
    }

    /// Create with default settings (localhost:11434, nomic-embed-text).
    pub fn with_defaults() -> Self {
        Self::new(DEFAULT_BASE_URL, DEFAULT_MODEL, 768)
    }
}

impl super::EmbeddingProvider for OllamaProvider {
    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>, CodememError> {
        let url = format!("{}/api/embeddings", self.base_url);
        let body = serde_json::json!({
            "model": self.model,
            "prompt": text,
        });

        let response = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .map_err(|e| CodememError::Embedding(format!("Ollama request failed: {e}")))?;

        if !response.status().is_success() {
            return Err(CodememError::Embedding(format!(
                "Ollama returned status {}",
                response.status()
            )));
        }

        let json: serde_json::Value = response
            .json()
            .map_err(|e| CodememError::Embedding(format!("Ollama response parse error: {e}")))?;

        let embedding = json
            .get("embedding")
            .and_then(|v| v.as_array())
            .ok_or_else(|| CodememError::Embedding("Missing 'embedding' field in response".into()))?
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0) as f32)
            .collect();

        Ok(embedding)
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, CodememError> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        // Ollama /api/embed supports batch via "input" array (Ollama >= 0.3)
        let url = format!("{}/api/embed", self.base_url);
        let body = serde_json::json!({
            "model": self.model,
            "input": texts,
        });

        let response =
            self.client.post(&url).json(&body).send().map_err(|e| {
                CodememError::Embedding(format!("Ollama batch request failed: {e}"))
            })?;

        if !response.status().is_success() {
            // Fall back to sequential calls if batch endpoint unavailable
            let mut results = Vec::with_capacity(texts.len());
            for text in texts {
                results.push(self.embed(text)?);
            }
            return Ok(results);
        }

        let json: serde_json::Value = response
            .json()
            .map_err(|e| CodememError::Embedding(format!("Ollama response parse error: {e}")))?;

        let embeddings_arr = json
            .get("embeddings")
            .and_then(|v| v.as_array())
            .ok_or_else(|| {
                CodememError::Embedding("Missing 'embeddings' array in Ollama response".into())
            })?;

        if embeddings_arr.len() != texts.len() {
            return Err(CodememError::Embedding(format!(
                "Ollama returned {} embeddings, expected {}",
                embeddings_arr.len(),
                texts.len()
            )));
        }

        let results: Vec<Vec<f32>> = embeddings_arr
            .iter()
            .map(|arr| {
                arr.as_array()
                    .unwrap_or(&vec![])
                    .iter()
                    .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                    .collect()
            })
            .collect();

        Ok(results)
    }

    fn name(&self) -> &str {
        "ollama"
    }
}

#[cfg(test)]
#[path = "tests/ollama_tests.rs"]
mod tests;
