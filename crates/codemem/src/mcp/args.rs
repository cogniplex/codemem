//! Parameter parsing helpers for MCP tool arguments.

use codemem_core::MemoryType;
use serde_json::Value;

/// Parse a JSON array of strings from tool arguments.
/// Returns empty Vec if the key is missing or not an array.
pub(crate) fn parse_string_array(args: &Value, key: &str) -> Vec<String> {
    args.get(key)
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default()
}

/// Parse a memory type from tool arguments, with a default fallback.
pub(crate) fn parse_memory_type(args: &Value, default: MemoryType) -> MemoryType {
    args.get("memory_type")
        .and_then(|v| v.as_str())
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// Parse an optional string from tool arguments.
pub(crate) fn parse_opt_string(args: &Value, key: &str) -> Option<String> {
    args.get(key).and_then(|v| v.as_str()).map(String::from)
}
