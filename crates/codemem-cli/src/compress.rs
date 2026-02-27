//! LLM-powered observation compression for Codemem.
//!
//! Compresses raw tool observations into concise structural summaries
//! using a configured LLM provider (Ollama, OpenAI-compatible, or Anthropic).
//! Falls back to raw content on failure or when not configured.
//!
//! # Configuration (environment variables)
//!
//! - `CODEMEM_COMPRESS_PROVIDER`: `ollama` | `openai` | `anthropic` (default: disabled)
//! - `CODEMEM_COMPRESS_MODEL`: model name (defaults: `llama3.2`, `gpt-4o-mini`, `claude-haiku-4-5-20251001`)
//! - `CODEMEM_COMPRESS_URL`: base URL override (defaults: `http://localhost:11434`, `https://api.openai.com/v1`)
//! - `CODEMEM_API_KEY` / `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`: API key for cloud providers

use std::time::Duration;

const COMPRESS_TIMEOUT: Duration = Duration::from_secs(30);

/// Minimum content length worth compressing. Shorter observations are already concise.
const MIN_COMPRESS_LEN: usize = 200;

const SYSTEM_PROMPT: &str = "\
You are a code observation compressor for a memory engine. \
Given a raw tool observation from an AI coding session, produce a concise summary (under 200 words) that captures:\n\
1. What: the key structures, functions, types, and patterns observed\n\
2. Why it matters: dependencies, relationships, design decisions, purpose\n\
3. Details worth remembering: important names, signatures, constants\n\n\
Rules:\n\
- Be specific — use actual function/type/file names\n\
- Skip boilerplate and obvious information\n\
- Focus on structural and behavioral insights\n\
- For file reads: what is this file's role and key exports?\n\
- For edits: what changed and why does it matter?\n\
- For searches: what patterns were found and where?\n\
- Output plain text, no markdown formatting";

pub enum CompressProvider {
    Ollama {
        base_url: String,
        model: String,
        client: reqwest::blocking::Client,
    },
    OpenAi {
        base_url: String,
        model: String,
        api_key: String,
        client: reqwest::blocking::Client,
    },
    Anthropic {
        api_key: String,
        model: String,
        client: reqwest::blocking::Client,
    },
    None,
}

impl CompressProvider {
    /// Create a provider from environment variables.
    pub fn from_env() -> Self {
        let provider = std::env::var("CODEMEM_COMPRESS_PROVIDER").unwrap_or_default();

        let client = || {
            reqwest::blocking::Client::builder()
                .timeout(COMPRESS_TIMEOUT)
                .build()
                .unwrap_or_default()
        };

        match provider.to_lowercase().as_str() {
            "ollama" => {
                let base_url = std::env::var("CODEMEM_COMPRESS_URL")
                    .unwrap_or_else(|_| "http://localhost:11434".to_string());
                let model = std::env::var("CODEMEM_COMPRESS_MODEL")
                    .unwrap_or_else(|_| "llama3.2".to_string());
                CompressProvider::Ollama {
                    base_url,
                    model,
                    client: client(),
                }
            }
            "openai" => {
                let base_url = std::env::var("CODEMEM_COMPRESS_URL")
                    .unwrap_or_else(|_| "https://api.openai.com/v1".to_string());
                let model = std::env::var("CODEMEM_COMPRESS_MODEL")
                    .unwrap_or_else(|_| "gpt-4o-mini".to_string());
                let api_key = std::env::var("CODEMEM_API_KEY")
                    .or_else(|_| std::env::var("OPENAI_API_KEY"))
                    .unwrap_or_default();
                CompressProvider::OpenAi {
                    base_url,
                    model,
                    api_key,
                    client: client(),
                }
            }
            "anthropic" => {
                let api_key = std::env::var("CODEMEM_API_KEY")
                    .or_else(|_| std::env::var("ANTHROPIC_API_KEY"))
                    .unwrap_or_default();
                let model = std::env::var("CODEMEM_COMPRESS_MODEL")
                    .unwrap_or_else(|_| "claude-haiku-4-5-20251001".to_string());
                CompressProvider::Anthropic {
                    api_key,
                    model,
                    client: client(),
                }
            }
            _ => CompressProvider::None,
        }
    }

    /// Whether compression is enabled.
    pub fn is_enabled(&self) -> bool {
        !matches!(self, CompressProvider::None)
    }

    /// Compress a tool observation into a concise summary.
    ///
    /// Returns `None` if compression is disabled, content is too short,
    /// or the LLM call fails (caller should use raw content as fallback).
    pub fn compress(&self, content: &str, tool: &str, file_path: Option<&str>) -> Option<String> {
        if !self.is_enabled() || content.len() < MIN_COMPRESS_LEN {
            return None;
        }

        let user_prompt = build_user_prompt(content, tool, file_path);

        match self.call_llm(&user_prompt) {
            Ok(compressed) if compressed.trim().is_empty() => {
                tracing::warn!("Compression returned empty output, using raw content");
                None
            }
            Ok(compressed) => {
                tracing::info!(
                    "Compressed observation: {} → {} chars ({:.0}% reduction)",
                    content.len(),
                    compressed.len(),
                    (1.0 - compressed.len() as f64 / content.len() as f64) * 100.0
                );
                Some(compressed)
            }
            Err(e) => {
                tracing::warn!("Compression failed, using raw content: {e}");
                None
            }
        }
    }

    fn call_llm(&self, user_prompt: &str) -> anyhow::Result<String> {
        match self {
            CompressProvider::Ollama {
                base_url,
                model,
                client,
            } => {
                let url = format!("{}/api/chat", base_url);
                let body = serde_json::json!({
                    "model": model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    "stream": false,
                });
                let response = client.post(&url).json(&body).send()?;
                if !response.status().is_success() {
                    anyhow::bail!("Ollama returned {}", response.status());
                }
                let json: serde_json::Value = response.json()?;
                json.get("message")
                    .and_then(|m| m.get("content"))
                    .and_then(|c| c.as_str())
                    .map(|s| s.trim().to_string())
                    .ok_or_else(|| anyhow::anyhow!("Unexpected Ollama response format"))
            }
            CompressProvider::OpenAi {
                base_url,
                model,
                api_key,
                client,
            } => {
                let url = format!("{}/chat/completions", base_url);
                let body = serde_json::json!({
                    "model": model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    "max_tokens": 512,
                    "temperature": 0.3,
                });
                let response = client
                    .post(&url)
                    .header("Authorization", format!("Bearer {}", api_key))
                    .json(&body)
                    .send()?;
                if !response.status().is_success() {
                    let status = response.status();
                    let text = response.text().unwrap_or_default();
                    anyhow::bail!("OpenAI returned {}: {}", status, text);
                }
                let json: serde_json::Value = response.json()?;
                json.get("choices")
                    .and_then(|c| c.as_array())
                    .and_then(|arr| arr.first())
                    .and_then(|choice| choice.get("message"))
                    .and_then(|m| m.get("content"))
                    .and_then(|c| c.as_str())
                    .map(|s| s.trim().to_string())
                    .ok_or_else(|| anyhow::anyhow!("Unexpected OpenAI response format"))
            }
            CompressProvider::Anthropic {
                api_key,
                model,
                client,
            } => {
                let body = serde_json::json!({
                    "model": model,
                    "max_tokens": 512,
                    "system": SYSTEM_PROMPT,
                    "messages": [
                        {"role": "user", "content": user_prompt}
                    ],
                });
                let response = client
                    .post("https://api.anthropic.com/v1/messages")
                    .header("x-api-key", api_key.as_str())
                    .header("anthropic-version", "2023-06-01")
                    .header("content-type", "application/json")
                    .json(&body)
                    .send()?;
                if !response.status().is_success() {
                    let status = response.status();
                    let text = response.text().unwrap_or_default();
                    anyhow::bail!("Anthropic returned {}: {}", status, text);
                }
                let json: serde_json::Value = response.json()?;
                json.get("content")
                    .and_then(|c| c.as_array())
                    .and_then(|arr| arr.first())
                    .and_then(|block| block.get("text"))
                    .and_then(|t| t.as_str())
                    .map(|s| s.trim().to_string())
                    .ok_or_else(|| anyhow::anyhow!("Unexpected Anthropic response format"))
            }
            CompressProvider::None => unreachable!(),
        }
    }
}

fn build_user_prompt(content: &str, tool: &str, file_path: Option<&str>) -> String {
    let file_info = file_path
        .map(|p| format!("File: {p}\n"))
        .unwrap_or_default();
    // Cap at 8KB to avoid excessive LLM input costs
    let truncated = if content.len() > 8000 {
        &content[..8000]
    } else {
        content
    };
    format!("Tool: {tool}\n{file_info}\nObservation:\n{truncated}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn none_provider_returns_none() {
        let provider = CompressProvider::None;
        assert!(!provider.is_enabled());
        assert!(provider.compress("some content here that is long enough to compress blah blah blah blah blah blah blah blah blah blah blah blah blah blah blah blah blah blah blah blah", "Read", Some("src/main.rs")).is_none());
    }

    #[test]
    fn short_content_skips_compression() {
        // Even with a "real" provider variant, short content should skip
        let provider = CompressProvider::Ollama {
            base_url: "http://localhost:99999".to_string(),
            model: "test".to_string(),
            client: reqwest::blocking::Client::new(),
        };
        assert!(provider.compress("short", "Read", None).is_none());
    }

    #[test]
    fn build_user_prompt_with_file() {
        let prompt = build_user_prompt("content here", "Read", Some("src/lib.rs"));
        assert!(prompt.contains("Tool: Read"));
        assert!(prompt.contains("File: src/lib.rs"));
        assert!(prompt.contains("content here"));
    }

    #[test]
    fn build_user_prompt_without_file() {
        let prompt = build_user_prompt("content here", "Grep", None);
        assert!(prompt.contains("Tool: Grep"));
        assert!(!prompt.contains("File:"));
    }

    #[test]
    fn build_user_prompt_truncates_long_content() {
        let long = "x".repeat(10000);
        let prompt = build_user_prompt(&long, "Read", None);
        // 8000 chars of content + header
        assert!(prompt.len() < 8200);
    }

    #[test]
    fn from_env_defaults_to_none() {
        // Without env vars set, should default to None
        let provider = CompressProvider::from_env();
        assert!(!provider.is_enabled());
    }
}
