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
