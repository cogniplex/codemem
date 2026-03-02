//! OpenAI-compatible embedding provider for Codemem.
//!
//! Works with OpenAI, Azure OpenAI, Together.ai, and any OpenAI-compatible API.
//! Default model: text-embedding-3-small.

use codemem_core::CodememError;

/// Default OpenAI API base URL.
pub const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

/// Default model.
pub const DEFAULT_MODEL: &str = "text-embedding-3-small";

/// OpenAI embedding provider.
pub struct OpenAIProvider {
    api_key: String,
    model: String,
    dimensions: usize,
    base_url: String,
    client: reqwest::blocking::Client,
}

impl OpenAIProvider {
    /// Create a new OpenAI provider.
    pub fn new(api_key: &str, model: &str, dimensions: usize, base_url: Option<&str>) -> Self {
        Self {
            api_key: api_key.to_string(),
            model: model.to_string(),
            dimensions,
            base_url: base_url.unwrap_or(DEFAULT_BASE_URL).to_string(),
            client: reqwest::blocking::Client::new(),
        }
    }

    /// Create with default model (text-embedding-3-small, 768 dims).
    pub fn with_api_key(api_key: &str) -> Self {
        Self::new(api_key, DEFAULT_MODEL, 768, None)
    }
}

impl super::EmbeddingProvider for OpenAIProvider {
    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>, CodememError> {
        let url = format!("{}/embeddings", self.base_url);

        let mut body = serde_json::json!({
            "model": self.model,
            "input": text,
        });

        // text-embedding-3-* supports custom dimensions
        if self.model.starts_with("text-embedding-3") {
            body["dimensions"] = serde_json::json!(self.dimensions);
        }

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .map_err(|e| CodememError::Embedding(format!("OpenAI request failed: {e}")))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().unwrap_or_default();
            return Err(CodememError::Embedding(format!(
                "OpenAI returned status {}: {}",
                status, body
            )));
        }

        let json: serde_json::Value = response
            .json()
            .map_err(|e| CodememError::Embedding(format!("OpenAI response parse error: {e}")))?;

        let embedding = json
            .get("data")
            .and_then(|v| v.as_array())
            .and_then(|arr| arr.first())
            .and_then(|item| item.get("embedding"))
            .and_then(|v| v.as_array())
            .ok_or_else(|| CodememError::Embedding("Missing embedding in OpenAI response".into()))?
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0) as f32)
            .collect();

        Ok(embedding)
    }

    fn name(&self) -> &str {
        "openai"
    }
}

#[cfg(test)]
#[path = "tests/openai_tests.rs"]
mod tests;
