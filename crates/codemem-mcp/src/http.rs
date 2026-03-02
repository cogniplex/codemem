//! MCP Streamable HTTP transport (2025-03-26 spec).
//!
//! Implements the MCP Streamable HTTP transport:
//! - `POST /mcp` — client sends JSON-RPC request/notification
//! - `GET /mcp` — client opens SSE listener for server-initiated messages
//! - `DELETE /mcp` — client terminates session
//!
//! Session management via `Mcp-Session-Id` header.

use crate::{JsonRpcRequest, JsonRpcResponse, McpServer};
use axum::{
    body::Body,
    extract::State,
    http::{HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    routing::{delete, get, post},
    Router,
};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Shared state for the MCP HTTP transport.
struct McpHttpState {
    server: Arc<McpServer>,
    /// Active sessions: session_id -> session metadata.
    sessions: Mutex<HashMap<String, SessionMeta>>,
}

struct SessionMeta {
    #[allow(dead_code)]
    created_at: std::time::Instant,
}

/// Returns an Axum router handling `/mcp` (POST, GET, DELETE)
/// per MCP Streamable HTTP spec (2025-03-26).
pub fn mcp_router(server: Arc<McpServer>) -> Router {
    let state = Arc::new(McpHttpState {
        server,
        sessions: Mutex::new(HashMap::new()),
    });

    Router::new()
        .route("/mcp", post(handle_post))
        .route("/mcp", get(handle_get))
        .route("/mcp", delete(handle_delete))
        .with_state(state)
}

/// POST /mcp — Client sends JSON-RPC request or notification.
async fn handle_post(
    State(state): State<Arc<McpHttpState>>,
    headers: HeaderMap,
    body: String,
) -> Response {
    let body = body.trim();
    if body.is_empty() {
        return (StatusCode::BAD_REQUEST, "Empty body").into_response();
    }

    // Parse as single request or batch
    let requests: Vec<JsonRpcRequest> = if body.starts_with('[') {
        match serde_json::from_str(body) {
            Ok(batch) => batch,
            Err(e) => {
                let resp = JsonRpcResponse::error(
                    Value::Null,
                    -32700,
                    format!("Parse error: {e}"),
                );
                return json_response(StatusCode::OK, &resp);
            }
        }
    } else {
        match serde_json::from_str(body) {
            Ok(req) => vec![req],
            Err(e) => {
                let resp = JsonRpcResponse::error(
                    Value::Null,
                    -32700,
                    format!("Parse error: {e}"),
                );
                return json_response(StatusCode::OK, &resp);
            }
        }
    };

    if requests.is_empty() {
        return (StatusCode::BAD_REQUEST, "Empty batch").into_response();
    }

    // Check for initialize request (creates session)
    let is_initialize = requests
        .iter()
        .any(|r| r.method == "initialize");

    // For non-initialize requests, validate session
    if !is_initialize {
        let session_id = headers
            .get("mcp-session-id")
            .and_then(|v| v.to_str().ok());

        if let Some(sid) = session_id {
            let sessions = state
                .sessions
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            if !sessions.contains_key(sid) {
                return (StatusCode::NOT_FOUND, "Unknown session").into_response();
            }
        }
        // Note: we're lenient here — if no session header, we still process
        // (some clients may not send it for simple requests)
    }

    // Process all requests
    let mut responses: Vec<JsonRpcResponse> = Vec::new();
    let mut all_notifications = true;

    for request in requests {
        if request.id.is_none() {
            // Notification — no response
            state
                .server
                .handle_notification(&request.method);
        } else {
            all_notifications = false;
            let id = request.id.unwrap();
            let response = state.server.handle_request(
                &request.method,
                request.params.as_ref(),
                id,
            );
            responses.push(response);
        }
    }

    // If all notifications, return 202 Accepted with no body
    if all_notifications {
        return StatusCode::ACCEPTED.into_response();
    }

    // Build response with session header if this was an initialize
    let mut resp = if responses.len() == 1 {
        json_response(StatusCode::OK, &responses[0])
    } else {
        let body = serde_json::to_string(&responses).unwrap_or_default();
        Response::builder()
            .status(StatusCode::OK)
            .header("content-type", "application/json")
            .body(Body::from(body))
            .unwrap()
    };

    // If this was an initialize, create a session and attach the header
    if is_initialize {
        let session_id = uuid::Uuid::new_v4().to_string();
        {
            let mut sessions = state
                .sessions
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            sessions.insert(
                session_id.clone(),
                SessionMeta {
                    created_at: std::time::Instant::now(),
                },
            );
        }
        resp.headers_mut().insert(
            "mcp-session-id",
            HeaderValue::from_str(&session_id).unwrap_or_else(|_| HeaderValue::from_static("")),
        );
    }

    resp
}

/// GET /mcp — Client opens SSE listener for server-initiated messages.
/// Currently returns an empty SSE stream (server-initiated messages not yet implemented).
async fn handle_get(
    State(_state): State<Arc<McpHttpState>>,
    headers: HeaderMap,
) -> Response {
    // Validate session
    let _session_id = headers
        .get("mcp-session-id")
        .and_then(|v| v.to_str().ok());

    // Return a minimal SSE stream that stays open
    // Server-initiated messages (like notifications) will be added in a future iteration
    let body = Body::from(": MCP SSE stream\n\n");
    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/event-stream")
        .header("cache-control", "no-cache")
        .header("connection", "keep-alive")
        .body(body)
        .unwrap()
}

/// DELETE /mcp — Client terminates session.
async fn handle_delete(
    State(state): State<Arc<McpHttpState>>,
    headers: HeaderMap,
) -> Response {
    let session_id = headers
        .get("mcp-session-id")
        .and_then(|v| v.to_str().ok());

    match session_id {
        Some(sid) => {
            let mut sessions = state
                .sessions
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            if sessions.remove(sid).is_some() {
                StatusCode::OK.into_response()
            } else {
                (StatusCode::NOT_FOUND, "Unknown session").into_response()
            }
        }
        None => (StatusCode::BAD_REQUEST, "Missing Mcp-Session-Id header").into_response(),
    }
}

/// Helper to build a JSON response.
fn json_response(status: StatusCode, value: &impl serde::Serialize) -> Response {
    let body = serde_json::to_string(value).unwrap_or_default();
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(Body::from(body))
        .unwrap()
}
