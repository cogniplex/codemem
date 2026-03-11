//! API surface detection: endpoint definitions and HTTP client calls.
//!
//! Post-processing pass on extracted symbols to detect REST/HTTP endpoints
//! and client calls for cross-service linking.

use crate::index::symbol::{Reference, ReferenceKind, Symbol, SymbolKind};
use std::sync::LazyLock;

// ── Precompiled regexes ─────────────────────────────────────────────────────

static RE_QUOTED_STRING: LazyLock<regex::Regex> =
    LazyLock::new(|| regex::Regex::new(r#"["']([^"']+)["']"#).unwrap());

static RE_METHODS_PARAM: LazyLock<regex::Regex> =
    LazyLock::new(|| regex::Regex::new(r#"methods\s*=\s*\[([^\]]+)\]"#).unwrap());

static RE_NESTJS_METHOD: LazyLock<regex::Regex> =
    LazyLock::new(|| regex::Regex::new(r"^@(Get|Post|Put|Delete|Patch|Head|Options)\b").unwrap());

static RE_FLASK_PARAM: LazyLock<regex::Regex> =
    LazyLock::new(|| regex::Regex::new(r"<(?:\w+:)?(\w+)>").unwrap());

static RE_EXPRESS_PARAM: LazyLock<regex::Regex> =
    LazyLock::new(|| regex::Regex::new(r":(\w+)").unwrap());

/// A detected API endpoint.
#[derive(Debug, Clone, PartialEq)]
pub struct DetectedEndpoint {
    /// Endpoint ID: "ep:{namespace}:{method}:{path}"
    pub id: String,
    /// HTTP method (GET, POST, PUT, DELETE, PATCH, etc.) or None for catch-all.
    pub method: Option<String>,
    /// URL path pattern, normalized (e.g., "/api/users/{id}").
    pub path: String,
    /// Handler symbol qualified name.
    pub handler: String,
    /// File path of the handler.
    pub file_path: String,
    /// Line number.
    pub line: usize,
}

/// A detected HTTP client call.
#[derive(Debug, Clone, PartialEq)]
pub struct DetectedClientCall {
    /// Symbol making the HTTP call.
    pub caller: String,
    /// HTTP method if detectable.
    pub method: Option<String>,
    /// URL pattern extracted from the call (may be partial/relative).
    pub url_pattern: Option<String>,
    /// The HTTP client library being used.
    pub client_library: String,
    /// File path.
    pub file_path: String,
    /// Line number.
    pub line: usize,
}

/// Result of API surface detection.
#[derive(Debug, Default)]
pub struct ApiSurfaceResult {
    pub endpoints: Vec<DetectedEndpoint>,
    pub client_calls: Vec<DetectedClientCall>,
}

/// Detect API endpoints from extracted symbols.
///
/// Scans symbol attributes/decorators for framework-specific route patterns:
/// - Python: `@app.route`, `@router.get`, `@api_view`, `@GetMapping` (for Django views, Flask, FastAPI)
/// - TypeScript: `@Get`, `@Post` (NestJS), `app.get` (Express) — detected from call references
/// - Java: `@GetMapping`, `@PostMapping`, `@RequestMapping`
/// - Go: detected via call patterns (`http.HandleFunc`, `router.GET`, etc.)
pub fn detect_endpoints(symbols: &[Symbol], namespace: &str) -> Vec<DetectedEndpoint> {
    let mut endpoints = Vec::new();

    for sym in symbols {
        // Check attributes/decorators for route patterns
        for attr in &sym.attributes {
            if let Some(ep) = parse_route_decorator(attr, sym, namespace) {
                endpoints.push(ep);
            }
        }

        // Check for Django URL pattern-style views (class-based views)
        if is_django_view_class(sym) {
            // Django CBVs: methods like get(), post() on View subclasses
            // The URL pattern linking happens elsewhere; here we just mark the handler
            for method in &["get", "post", "put", "patch", "delete"] {
                if sym.kind == SymbolKind::Method && sym.name == *method {
                    if let Some(parent) = &sym.parent {
                        endpoints.push(DetectedEndpoint {
                            id: format!("ep:{namespace}:{}:view:{parent}", method.to_uppercase()),
                            method: Some(method.to_uppercase()),
                            path: format!("view:{parent}"), // placeholder until URL conf resolved
                            handler: sym.qualified_name.clone(),
                            file_path: sym.file_path.clone(),
                            line: sym.line_start,
                        });
                    }
                }
            }
        }
    }

    endpoints
}

/// Parse a route decorator/annotation string into an endpoint.
///
/// Handles patterns like:
/// - `@app.route("/users")` or `@app.route("/users", methods=["GET", "POST"])`
/// - `@router.get("/users/{id}")` or `@app.get("/users/<int:id>")`
/// - `@GetMapping("/users")` or `@RequestMapping(value="/users", method=RequestMethod.GET)`
/// - `@Get("/users")` (NestJS)
/// - `@api_view(["GET"])` (DRF — path comes from urls.py, not decorator)
fn parse_route_decorator(attr: &str, sym: &Symbol, namespace: &str) -> Option<DetectedEndpoint> {
    let attr_lower = attr.to_lowercase();

    // Flask/FastAPI style: @app.route("/path") or @router.get("/path")
    if attr_lower.contains("route(")
        || attr_lower.contains(".get(")
        || attr_lower.contains(".post(")
        || attr_lower.contains(".put(")
        || attr_lower.contains(".delete(")
        || attr_lower.contains(".patch(")
    {
        let method = extract_http_method_from_decorator(attr);
        let path = extract_path_from_decorator(attr)?;
        let normalized_path = normalize_path_pattern(&path);

        return Some(DetectedEndpoint {
            id: format!(
                "ep:{namespace}:{}:{normalized_path}",
                method.as_deref().unwrap_or("ANY")
            ),
            method,
            path: normalized_path,
            handler: sym.qualified_name.clone(),
            file_path: sym.file_path.clone(),
            line: sym.line_start,
        });
    }

    // Spring style: @GetMapping("/path"), @PostMapping("/path"), @RequestMapping(...)
    if attr_lower.contains("mapping(") || attr_lower.contains("mapping\"") {
        let method = extract_spring_method(attr);
        let path = extract_path_from_decorator(attr)?;
        let normalized_path = normalize_path_pattern(&path);

        return Some(DetectedEndpoint {
            id: format!(
                "ep:{namespace}:{}:{normalized_path}",
                method.as_deref().unwrap_or("ANY")
            ),
            method,
            path: normalized_path,
            handler: sym.qualified_name.clone(),
            file_path: sym.file_path.clone(),
            line: sym.line_start,
        });
    }

    // NestJS style: @Get("/path"), @Post("/path")
    if let Some(method) = extract_nestjs_method(attr) {
        let path = extract_path_from_decorator(attr).unwrap_or_else(|| "/".to_string());
        let normalized_path = normalize_path_pattern(&path);

        return Some(DetectedEndpoint {
            id: format!("ep:{namespace}:{method}:{normalized_path}"),
            method: Some(method),
            path: normalized_path,
            handler: sym.qualified_name.clone(),
            file_path: sym.file_path.clone(),
            line: sym.line_start,
        });
    }

    None
}

/// Detect HTTP client calls from extracted references.
///
/// Scans call references for known HTTP client patterns:
/// - Python: `requests.get`/`post`/..., `httpx.get`/`post`/..., `aiohttp`
/// - TS/JS: `fetch`, `axios.get`/`post`/..., `got`
/// - Java: `RestTemplate`, `WebClient`, `HttpClient`
/// - Go: `http.Get`, `http.Post`, `http.NewRequest`
pub fn detect_client_calls(references: &[Reference]) -> Vec<DetectedClientCall> {
    let mut calls = Vec::new();

    for r in references {
        if r.kind != ReferenceKind::Call {
            continue;
        }

        if let Some(call) = parse_client_call(&r.target_name, r) {
            calls.push(call);
        }
    }

    calls
}

fn parse_client_call(target: &str, reference: &Reference) -> Option<DetectedClientCall> {
    let target_lower = target.to_lowercase();

    // Python: requests.get, requests.post, httpx.get, etc.
    if target_lower.starts_with("requests.") || target_lower.starts_with("httpx.") {
        let parts: Vec<&str> = target.splitn(2, '.').collect();
        let library = parts[0].to_string();
        let method = parts.get(1).and_then(|m| http_method_from_name(m));

        return Some(DetectedClientCall {
            caller: reference.source_qualified_name.clone(),
            method,
            url_pattern: None, // would need string literal analysis
            client_library: library,
            file_path: reference.file_path.clone(),
            line: reference.line,
        });
    }

    // TS/JS: fetch (global function)
    if target_lower == "fetch" {
        return Some(DetectedClientCall {
            caller: reference.source_qualified_name.clone(),
            method: None, // determined by options argument
            url_pattern: None,
            client_library: "fetch".to_string(),
            file_path: reference.file_path.clone(),
            line: reference.line,
        });
    }

    // TS/JS: axios.get, axios.post, etc.
    if target_lower.starts_with("axios.") {
        let method = target.split('.').nth(1).and_then(http_method_from_name);
        return Some(DetectedClientCall {
            caller: reference.source_qualified_name.clone(),
            method,
            url_pattern: None,
            client_library: "axios".to_string(),
            file_path: reference.file_path.clone(),
            line: reference.line,
        });
    }

    // Go: http.Get, http.Post, http.NewRequest
    if target_lower.starts_with("http.")
        && (target.contains("Get")
            || target.contains("Post")
            || target.contains("NewRequest")
            || target.contains("Do"))
    {
        let method = if target.contains("Get") {
            Some("GET".to_string())
        } else if target.contains("Post") {
            Some("POST".to_string())
        } else {
            None
        };
        return Some(DetectedClientCall {
            caller: reference.source_qualified_name.clone(),
            method,
            url_pattern: None,
            client_library: "net/http".to_string(),
            file_path: reference.file_path.clone(),
            line: reference.line,
        });
    }

    // Java: RestTemplate, WebClient
    if target_lower.contains("resttemplate")
        || target_lower.contains("webclient")
        || target_lower.contains("httpclient")
    {
        return Some(DetectedClientCall {
            caller: reference.source_qualified_name.clone(),
            method: None,
            url_pattern: None,
            client_library: target.split('.').next().unwrap_or(target).to_string(),
            file_path: reference.file_path.clone(),
            line: reference.line,
        });
    }

    None
}

// ── Helper functions ──

/// Extract the first quoted string from a decorator (the path argument).
fn extract_path_from_decorator(attr: &str) -> Option<String> {
    RE_QUOTED_STRING.captures(attr).map(|c| c[1].to_string())
}

/// Extract HTTP method from a decorator like `@app.get(...)` or `@router.post(...)`
fn extract_http_method_from_decorator(attr: &str) -> Option<String> {
    let attr_lower = attr.to_lowercase();
    for method in &["get", "post", "put", "delete", "patch", "head", "options"] {
        // Match .get( or .post( etc
        if attr_lower.contains(&format!(".{method}(")) {
            return Some(method.to_uppercase());
        }
    }
    // @app.route with methods= parameter
    if attr_lower.contains("route(") {
        if let Some(methods) = extract_methods_param(attr) {
            return methods.first().cloned();
        }
    }
    None
}

/// Extract `methods=["GET", "POST"]` from a route decorator.
fn extract_methods_param(attr: &str) -> Option<Vec<String>> {
    let caps = RE_METHODS_PARAM.captures(attr)?;
    let methods_str = &caps[1];
    let methods: Vec<String> = methods_str
        .split(',')
        .map(|m| {
            m.trim()
                .trim_matches(|c| c == '"' || c == '\'')
                .to_uppercase()
        })
        .filter(|m| !m.is_empty())
        .collect();
    if methods.is_empty() {
        None
    } else {
        Some(methods)
    }
}

/// Extract HTTP method from Spring annotations.
fn extract_spring_method(attr: &str) -> Option<String> {
    let attr_lower = attr.to_lowercase();
    if attr_lower.contains("getmapping") {
        return Some("GET".to_string());
    }
    if attr_lower.contains("postmapping") {
        return Some("POST".to_string());
    }
    if attr_lower.contains("putmapping") {
        return Some("PUT".to_string());
    }
    if attr_lower.contains("deletemapping") {
        return Some("DELETE".to_string());
    }
    if attr_lower.contains("patchmapping") {
        return Some("PATCH".to_string());
    }
    // @RequestMapping with method= parameter
    if attr_lower.contains("requestmapping") {
        if attr_lower.contains("get") {
            return Some("GET".to_string());
        }
        if attr_lower.contains("post") {
            return Some("POST".to_string());
        }
    }
    None
}

/// Extract HTTP method from NestJS decorators.
fn extract_nestjs_method(attr: &str) -> Option<String> {
    // NestJS: @Get, @Post, @Put, @Delete, @Patch
    // These are standalone decorators (not method calls on an object)
    RE_NESTJS_METHOD.captures(attr).map(|c| c[1].to_uppercase())
}

/// Normalize a URL path pattern:
/// - Flask: `/users/<int:id>` -> `/users/{id}`
/// - Express: `/users/:id` -> `/users/{id}`
/// - Spring: `/users/{id}` -> already normalized
/// - Go: `/users/{id}` -> already normalized
pub fn normalize_path_pattern(path: &str) -> String {
    let mut result = path.to_string();

    // Flask: <type:name> or <name> → {name}
    result = RE_FLASK_PARAM.replace_all(&result, "{$1}").to_string();

    // Express: :name → {name}
    let express_re = &*RE_EXPRESS_PARAM;
    result = express_re.replace_all(&result, "{$1}").to_string();

    // Ensure leading slash
    if !result.starts_with('/') {
        result = format!("/{result}");
    }

    // Remove trailing slash (unless it's just "/")
    if result.len() > 1 && result.ends_with('/') {
        result.pop();
    }

    result
}

/// Check if a symbol looks like a Django class-based view.
fn is_django_view_class(sym: &Symbol) -> bool {
    if sym.kind != SymbolKind::Method {
        return false;
    }
    // Check if parent class has View-like attributes
    sym.parent
        .as_ref()
        .is_some_and(|p| p.ends_with("View") || p.ends_with("ViewSet") || p.ends_with("APIView"))
}

/// Map a method name to HTTP method string.
fn http_method_from_name(name: &str) -> Option<String> {
    match name.to_lowercase().as_str() {
        "get" => Some("GET".to_string()),
        "post" => Some("POST".to_string()),
        "put" => Some("PUT".to_string()),
        "delete" => Some("DELETE".to_string()),
        "patch" => Some("PATCH".to_string()),
        "head" => Some("HEAD".to_string()),
        "options" => Some("OPTIONS".to_string()),
        _ => None,
    }
}

/// Match a client call URL against registered endpoints.
///
/// Returns the best matching endpoint with confidence.
pub fn match_endpoint<'a>(
    url_path: &str,
    method: Option<&str>,
    endpoints: &'a [DetectedEndpoint],
) -> Option<(&'a DetectedEndpoint, f64)> {
    let normalized = normalize_path_pattern(url_path);
    let mut best: Option<(&DetectedEndpoint, f64)> = None;

    for ep in endpoints {
        // Base confidence from path matching
        let base_confidence: f64 = if ep.path == normalized {
            1.0
        } else if paths_match_with_params(&normalized, &ep.path) {
            0.9
        } else if normalized.starts_with(&ep.path) || ep.path.starts_with(&normalized) {
            0.7
        } else {
            continue;
        };

        let mut confidence = base_confidence;

        // Method match bonus
        if let (Some(call_method), Some(ep_method)) = (method, ep.method.as_deref()) {
            if call_method.eq_ignore_ascii_case(ep_method) {
                confidence += 0.05;
            } else {
                confidence -= 0.1;
            }
        }

        confidence = confidence.clamp(0.0, 1.0);

        if best.is_none() || confidence > best.unwrap().1 {
            best = Some((ep, confidence));
        }
    }

    // Only return matches above threshold
    best.filter(|(_, c)| *c >= 0.5)
}

/// Check if two paths match allowing parameter substitution.
/// e.g., "/users/123" matches "/users/{id}"
fn paths_match_with_params(actual: &str, pattern: &str) -> bool {
    let actual_parts: Vec<&str> = actual.split('/').collect();
    let pattern_parts: Vec<&str> = pattern.split('/').collect();

    if actual_parts.len() != pattern_parts.len() {
        return false;
    }

    actual_parts
        .iter()
        .zip(pattern_parts.iter())
        .all(|(a, p)| a == p || (p.starts_with('{') && p.ends_with('}')))
}

#[cfg(test)]
#[path = "tests/api_surface_tests.rs"]
mod tests;
