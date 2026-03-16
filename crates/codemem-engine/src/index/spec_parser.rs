//! OpenAPI and AsyncAPI spec file parsing.
//!
//! Parses OpenAPI (2.0/3.x) and AsyncAPI (2.x/3.0) specification files to extract
//! API endpoints, channels, schemas, and metadata. Supports both JSON and YAML formats.
//! Discovered endpoints are normalized via `api_surface::normalize_path_pattern()`.

use serde::{Deserialize, Serialize};
use std::path::Path;

// ── OpenAPI Types ────────────────────────────────────────────────────────

/// A single parsed endpoint from an OpenAPI/Swagger spec.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecEndpoint {
    /// HTTP method (uppercase: GET, POST, etc.).
    pub method: String,
    /// URL path pattern (normalized).
    pub path: String,
    /// The `operationId` if present.
    pub operation_id: Option<String>,
    /// Operation description or summary.
    pub description: Option<String>,
    /// Stringified JSON of the request body schema.
    pub request_schema: Option<String>,
    /// Stringified JSON of the primary success response schema.
    pub response_schema: Option<String>,
    /// Path to the spec file this was extracted from.
    pub spec_file: String,
}

/// Result of parsing an OpenAPI/Swagger spec file.
#[derive(Debug, Clone)]
pub struct SpecParseResult {
    /// All endpoints discovered in the spec.
    pub endpoints: Vec<SpecEndpoint>,
    /// API title from `info.title`.
    pub title: Option<String>,
    /// API version from `info.version`.
    pub version: Option<String>,
}

// ── AsyncAPI Types ───────────────────────────────────────────────────────

/// A single parsed channel operation from an AsyncAPI spec.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecChannel {
    /// Channel name/path.
    pub channel: String,
    /// Direction: `"publish"` or `"subscribe"`.
    pub direction: String,
    /// Protocol (e.g., `"kafka"`, `"amqp"`, `"mqtt"`).
    pub protocol: Option<String>,
    /// Stringified JSON of the message payload schema.
    pub message_schema: Option<String>,
    /// Operation description.
    pub description: Option<String>,
    /// The `operationId` if present.
    pub operation_id: Option<String>,
    /// Path to the spec file this was extracted from.
    pub spec_file: String,
}

/// Result of parsing an AsyncAPI spec file.
#[derive(Debug, Clone)]
pub struct AsyncApiParseResult {
    /// All channel operations discovered in the spec.
    pub channels: Vec<SpecChannel>,
    /// API title from `info.title`.
    pub title: Option<String>,
    /// API version from `info.version`.
    pub version: Option<String>,
}

// ── Unified Result ───────────────────────────────────────────────────────

/// A parsed spec file: either OpenAPI or AsyncAPI.
#[derive(Debug, Clone)]
pub enum SpecFileResult {
    OpenApi(SpecParseResult),
    AsyncApi(AsyncApiParseResult),
}

// ── Helpers ──────────────────────────────────────────────────────────────

/// HTTP methods recognized in OpenAPI path items.
const HTTP_METHODS: &[&str] = &["get", "post", "put", "delete", "patch", "options", "head"];

/// Read a file and parse it into `serde_json::Value`, detecting JSON vs YAML by extension.
fn read_spec_file(path: &Path) -> Option<serde_json::Value> {
    let content = std::fs::read_to_string(path).ok()?;
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    match ext.as_str() {
        "json" => serde_json::from_str(&content).ok(),
        "yaml" | "yml" => {
            let yaml_val: serde_yaml::Value = serde_yaml::from_str(&content).ok()?;
            // Convert serde_yaml::Value → serde_json::Value via serialization round-trip.
            let json_str = serde_json::to_string(&yaml_val).ok()?;
            serde_json::from_str(&json_str).ok()
        }
        _ => {
            // Unknown extension: try JSON first, then YAML.
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&content) {
                return Some(v);
            }
            let yaml_val: serde_yaml::Value = serde_yaml::from_str(&content).ok()?;
            let json_str = serde_json::to_string(&yaml_val).ok()?;
            serde_json::from_str(&json_str).ok()
        }
    }
}

/// Stringify a `serde_json::Value` for schema storage.
/// Returns `None` for `Value::Null`.
fn stringify_schema(value: &serde_json::Value) -> Option<String> {
    if value.is_null() {
        return None;
    }
    Some(serde_json::to_string(value).unwrap_or_default())
}

/// Extract `info.title` from a parsed spec.
fn extract_info_title(root: &serde_json::Value) -> Option<String> {
    root.get("info")?
        .get("title")?
        .as_str()
        .map(|s| s.to_string())
}

/// Extract `info.version` from a parsed spec.
fn extract_info_version(root: &serde_json::Value) -> Option<String> {
    root.get("info")?
        .get("version")?
        .as_str()
        .map(|s| s.to_string())
}

// ── OpenAPI Parsing ──────────────────────────────────────────────────────

/// Parse an OpenAPI (2.0 Swagger / 3.x) spec file.
///
/// Returns `None` if the file cannot be read, is not valid JSON/YAML,
/// or lacks an `openapi` or `swagger` top-level key.
pub fn parse_openapi(path: &Path) -> Option<SpecParseResult> {
    let root = read_spec_file(path)?;
    let obj = root.as_object()?;

    // Confirm this is an OpenAPI/Swagger file.
    let is_openapi = obj.contains_key("openapi");
    let is_swagger = obj.contains_key("swagger");
    if !is_openapi && !is_swagger {
        return None;
    }

    let spec_file = path.to_string_lossy().to_string();
    let title = extract_info_title(&root);
    let version = extract_info_version(&root);

    let mut endpoints = Vec::new();

    let paths = match obj.get("paths").and_then(|v| v.as_object()) {
        Some(p) => p,
        None => {
            return Some(SpecParseResult {
                endpoints,
                title,
                version,
            })
        }
    };

    for (url_path, path_item) in paths {
        let path_obj = match path_item.as_object() {
            Some(o) => o,
            None => continue,
        };

        let normalized = super::api_surface::normalize_path_pattern(url_path);

        for method in HTTP_METHODS {
            let operation = match path_obj.get(*method).and_then(|v| v.as_object()) {
                Some(op) => op,
                None => continue,
            };

            let operation_id = operation
                .get("operationId")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());

            // Prefer summary, fall back to description.
            let description = operation
                .get("summary")
                .and_then(|v| v.as_str())
                .or_else(|| operation.get("description").and_then(|v| v.as_str()))
                .map(|s| s.to_string());

            let request_schema = if is_swagger {
                extract_swagger_request_schema(operation)
            } else {
                extract_openapi3_request_schema(operation)
            };

            let response_schema = if is_swagger {
                extract_swagger_response_schema(operation)
            } else {
                extract_openapi3_response_schema(operation)
            };

            endpoints.push(SpecEndpoint {
                method: method.to_uppercase(),
                path: normalized.clone(),
                operation_id,
                description,
                request_schema,
                response_schema,
                spec_file: spec_file.clone(),
            });
        }
    }

    Some(SpecParseResult {
        endpoints,
        title,
        version,
    })
}

/// Extract request body schema from an OpenAPI 3.x operation.
/// Looks for `requestBody.content.application/json.schema`.
fn extract_openapi3_request_schema(
    operation: &serde_json::Map<String, serde_json::Value>,
) -> Option<String> {
    let schema = operation
        .get("requestBody")?
        .get("content")?
        .get("application/json")?
        .get("schema")?;
    stringify_schema(schema)
}

/// Extract response schema from an OpenAPI 3.x operation.
/// Checks `responses.200` then `responses.201` for `content.application/json.schema`.
fn extract_openapi3_response_schema(
    operation: &serde_json::Map<String, serde_json::Value>,
) -> Option<String> {
    let responses = operation.get("responses")?.as_object()?;

    for status in &["200", "201"] {
        if let Some(schema) = responses
            .get(*status)
            .and_then(|r| r.get("content"))
            .and_then(|c| c.get("application/json"))
            .and_then(|j| j.get("schema"))
        {
            return stringify_schema(schema);
        }
    }
    None
}

/// Extract request body schema from a Swagger 2.0 operation.
/// Looks for a parameter with `in: body` and extracts its `schema`.
fn extract_swagger_request_schema(
    operation: &serde_json::Map<String, serde_json::Value>,
) -> Option<String> {
    let parameters = operation.get("parameters")?.as_array()?;
    for param in parameters {
        if param.get("in").and_then(|v| v.as_str()) == Some("body") {
            if let Some(schema) = param.get("schema") {
                return stringify_schema(schema);
            }
        }
    }
    None
}

/// Extract response schema from a Swagger 2.0 operation.
/// Checks `responses.200.schema` then `responses.201.schema`.
fn extract_swagger_response_schema(
    operation: &serde_json::Map<String, serde_json::Value>,
) -> Option<String> {
    let responses = operation.get("responses")?.as_object()?;

    for status in &["200", "201"] {
        if let Some(schema) = responses.get(*status).and_then(|r| r.get("schema")) {
            return stringify_schema(schema);
        }
    }
    None
}

// ── AsyncAPI Parsing ─────────────────────────────────────────────────────

/// Parse an AsyncAPI (2.x / 3.0) spec file.
///
/// Returns `None` if the file cannot be read, is not valid JSON/YAML,
/// or lacks an `asyncapi` top-level key.
pub fn parse_asyncapi(path: &Path) -> Option<AsyncApiParseResult> {
    let root = read_spec_file(path)?;
    let obj = root.as_object()?;

    if !obj.contains_key("asyncapi") {
        return None;
    }

    let spec_file = path.to_string_lossy().to_string();
    let title = extract_info_title(&root);
    let version = extract_info_version(&root);

    // Detect protocol from servers object.
    let protocol = detect_asyncapi_protocol(obj);

    let asyncapi_version = obj.get("asyncapi").and_then(|v| v.as_str()).unwrap_or("");

    let channels = if asyncapi_version.starts_with("3.") {
        parse_asyncapi_v3(obj, &spec_file, &protocol)
    } else {
        // 2.x (default)
        parse_asyncapi_v2(obj, &spec_file, &protocol)
    };

    Some(AsyncApiParseResult {
        channels,
        title,
        version,
    })
}

/// Detect the protocol from the `servers` object (first server's `protocol` field).
fn detect_asyncapi_protocol(obj: &serde_json::Map<String, serde_json::Value>) -> Option<String> {
    let servers = obj.get("servers")?.as_object()?;
    // Take the first server entry.
    let (_name, server) = servers.iter().next()?;
    server
        .get("protocol")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

/// Parse AsyncAPI 2.x channels.
///
/// Structure: `channels.<name>.publish` / `channels.<name>.subscribe`, each with
/// `operationId`, `description`, `message.payload`.
fn parse_asyncapi_v2(
    obj: &serde_json::Map<String, serde_json::Value>,
    spec_file: &str,
    protocol: &Option<String>,
) -> Vec<SpecChannel> {
    let mut result = Vec::new();

    let channels = match obj.get("channels").and_then(|v| v.as_object()) {
        Some(c) => c,
        None => return result,
    };

    for (channel_name, channel_value) in channels {
        let channel_obj = match channel_value.as_object() {
            Some(o) => o,
            None => continue,
        };

        for direction in &["publish", "subscribe"] {
            let operation = match channel_obj.get(*direction).and_then(|v| v.as_object()) {
                Some(op) => op,
                None => continue,
            };

            let operation_id = operation
                .get("operationId")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());

            let description = operation
                .get("description")
                .and_then(|v| v.as_str())
                .or_else(|| operation.get("summary").and_then(|v| v.as_str()))
                .map(|s| s.to_string());

            let message_schema = operation
                .get("message")
                .and_then(|m| m.get("payload"))
                .and_then(stringify_schema);

            result.push(SpecChannel {
                channel: channel_name.clone(),
                direction: direction.to_string(),
                protocol: protocol.clone(),
                message_schema,
                description,
                operation_id,
                spec_file: spec_file.to_string(),
            });
        }
    }

    result
}

/// Parse AsyncAPI 3.0 channels and operations.
///
/// In v3, channels are under `channels` and operations are under `operations`.
/// Each operation has a `channel.$ref` pointing to a channel, plus `action` (send/receive).
fn parse_asyncapi_v3(
    obj: &serde_json::Map<String, serde_json::Value>,
    spec_file: &str,
    protocol: &Option<String>,
) -> Vec<SpecChannel> {
    let mut result = Vec::new();

    let operations = match obj.get("operations").and_then(|v| v.as_object()) {
        Some(o) => o,
        None => return result,
    };

    for (_op_name, op_value) in operations {
        let operation = match op_value.as_object() {
            Some(o) => o,
            None => continue,
        };

        // Resolve channel name from $ref: "#/channels/channelName"
        let channel_name = operation
            .get("channel")
            .and_then(|c| {
                // Could be a $ref object or a direct reference.
                if let Some(ref_str) = c.get("$ref").and_then(|v| v.as_str()) {
                    // Extract last segment: "#/channels/myChannel" → "myChannel"
                    ref_str.rsplit('/').next().map(|s| s.to_string())
                } else {
                    c.as_str().map(|s| s.to_string())
                }
            })
            .unwrap_or_default();

        // action: "send" or "receive" → map to "publish"/"subscribe"
        let direction = match operation.get("action").and_then(|v| v.as_str()) {
            Some("send") => "publish".to_string(),
            Some("receive") => "subscribe".to_string(),
            Some(other) => other.to_string(),
            None => continue,
        };

        let operation_id = operation
            .get("operationId")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let description = operation
            .get("description")
            .and_then(|v| v.as_str())
            .or_else(|| operation.get("summary").and_then(|v| v.as_str()))
            .map(|s| s.to_string());

        // In v3, messages can be on the operation or on the channel.
        let message_schema = extract_v3_message_schema(operation, obj, &channel_name);

        result.push(SpecChannel {
            channel: channel_name,
            direction,
            protocol: protocol.clone(),
            message_schema,
            description,
            operation_id,
            spec_file: spec_file.to_string(),
        });
    }

    result
}

/// Extract message payload schema for an AsyncAPI 3.0 operation.
///
/// Checks the operation's `messages` first, then falls back to the channel's
/// `messages` in the root `channels` object.
fn extract_v3_message_schema(
    operation: &serde_json::Map<String, serde_json::Value>,
    root: &serde_json::Map<String, serde_json::Value>,
    channel_name: &str,
) -> Option<String> {
    // Try operation-level messages first.
    if let Some(messages) = operation.get("messages") {
        if let Some(schema) = first_message_payload(messages) {
            return Some(schema);
        }
    }

    // Fall back to channel-level messages.
    let channel = root.get("channels")?.get(channel_name)?;
    let messages = channel.get("messages")?;
    first_message_payload(messages)
}

/// Extract the `payload` schema from the first message in a messages value.
/// Messages can be an object (keyed by name) or an array.
fn first_message_payload(messages: &serde_json::Value) -> Option<String> {
    if let Some(obj) = messages.as_object() {
        for (_name, msg) in obj {
            if let Some(payload) = msg.get("payload") {
                return stringify_schema(payload);
            }
        }
    } else if let Some(arr) = messages.as_array() {
        for msg in arr {
            if let Some(payload) = msg.get("payload") {
                return stringify_schema(payload);
            }
        }
    }
    None
}

// ── Directory Scanning ───────────────────────────────────────────────────

/// Well-known spec file names to match by filename alone.
const SPEC_FILE_NAMES: &[&str] = &[
    "openapi.yaml",
    "openapi.yml",
    "openapi.json",
    "swagger.yaml",
    "swagger.yml",
    "swagger.json",
    "asyncapi.yaml",
    "asyncapi.yml",
    "asyncapi.json",
];

/// Scan a directory for API spec files (OpenAPI / AsyncAPI) and parse them.
///
/// Uses `ignore::WalkBuilder` to walk the directory tree, respecting `.gitignore`.
/// Detects spec files by well-known filenames and by peeking at file contents
/// for top-level `openapi`, `swagger`, or `asyncapi` keys.
pub fn scan_api_specs(root: &Path) -> Vec<SpecFileResult> {
    let mut results = Vec::new();

    let walker = ignore::WalkBuilder::new(root)
        .hidden(true)
        .git_ignore(true)
        .git_global(true)
        .git_exclude(true)
        .build();

    for entry in walker {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };

        if !entry.file_type().is_some_and(|ft| ft.is_file()) {
            continue;
        }

        let path = entry.path();
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        // Only consider JSON/YAML files.
        if !matches!(ext.as_str(), "json" | "yaml" | "yml") {
            continue;
        }

        let file_name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_lowercase();

        let is_well_known = SPEC_FILE_NAMES.contains(&file_name.as_str());

        if !is_well_known {
            // Peek at the file to check for spec-identifying keys.
            if !peek_is_spec_file(path) {
                continue;
            }
        }

        // Try OpenAPI first, then AsyncAPI.
        if let Some(openapi) = parse_openapi(path) {
            results.push(SpecFileResult::OpenApi(openapi));
        } else if let Some(asyncapi) = parse_asyncapi(path) {
            results.push(SpecFileResult::AsyncApi(asyncapi));
        }
    }

    results
}

/// Quick check: read the first 200 bytes and look for spec-identifying keys.
fn peek_is_spec_file(path: &Path) -> bool {
    let mut buf = [0u8; 200];
    let file = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return false,
    };

    use std::io::Read;
    let mut reader = std::io::BufReader::new(file);
    let n = match reader.read(&mut buf) {
        Ok(n) => n,
        Err(_) => return false,
    };

    let snippet = String::from_utf8_lossy(&buf[..n]).to_lowercase();

    // Check for top-level keys that identify spec files.
    // In YAML: `openapi:` or `swagger:` or `asyncapi:` at start of line.
    // In JSON: `"openapi"` or `"swagger"` or `"asyncapi"` near the start.
    snippet.contains("\"openapi\"")
        || snippet.contains("\"swagger\"")
        || snippet.contains("\"asyncapi\"")
        || snippet.contains("openapi:")
        || snippet.contains("swagger:")
        || snippet.contains("asyncapi:")
}

#[cfg(test)]
#[path = "tests/spec_parser_tests.rs"]
mod tests;
