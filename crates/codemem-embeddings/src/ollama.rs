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

    fn name(&self) -> &str {
        "ollama"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ollama_provider_construction() {
        let provider = OllamaProvider::with_defaults();
        assert_eq!(provider.base_url, DEFAULT_BASE_URL);
        assert_eq!(provider.model, DEFAULT_MODEL);
        assert_eq!(provider.dimensions, 768);
    }

    #[test]
    fn ollama_provider_custom() {
        let provider = OllamaProvider::new("http://myhost:11434", "mxbai-embed-large", 1024);
        assert_eq!(provider.base_url, "http://myhost:11434");
        assert_eq!(provider.model, "mxbai-embed-large");
        assert_eq!(provider.dimensions, 1024);
    }

    #[test]
    fn ollama_name_returns_ollama() {
        use crate::EmbeddingProvider;
        let provider = OllamaProvider::with_defaults();
        assert_eq!(provider.name(), "ollama");
    }

    #[test]
    fn ollama_dimensions_matches_constructor() {
        use crate::EmbeddingProvider;
        let provider = OllamaProvider::new("http://localhost:11434", "nomic-embed-text", 512);
        assert_eq!(EmbeddingProvider::dimensions(&provider), 512);
    }

    #[test]
    fn ollama_embed_success_mock() {
        use crate::EmbeddingProvider;
        let mut server = mockito::Server::new();
        let mock = server
            .mock("POST", "/api/embeddings")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{"embedding": [0.1, 0.2, 0.3]}"#)
            .create();

        let provider = OllamaProvider::new(&server.url(), "nomic-embed-text", 3);
        let result = provider.embed("test");
        mock.assert();

        let embedding = result.unwrap();
        assert_eq!(embedding.len(), 3);
        assert!((embedding[0] - 0.1).abs() < 1e-6);
        assert!((embedding[1] - 0.2).abs() < 1e-6);
        assert!((embedding[2] - 0.3).abs() < 1e-6);
    }

    #[test]
    fn ollama_embed_server_error_mock() {
        use crate::EmbeddingProvider;
        let mut server = mockito::Server::new();
        let mock = server
            .mock("POST", "/api/embeddings")
            .with_status(500)
            .with_body("Internal Server Error")
            .create();

        let provider = OllamaProvider::new(&server.url(), "nomic-embed-text", 768);
        let result = provider.embed("test");
        mock.assert();

        assert!(result.is_err());
    }
}
