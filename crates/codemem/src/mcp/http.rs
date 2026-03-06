//! MCP Streamable HTTP transport (2025-03-26 spec).
//!
//! Implements the MCP Streamable HTTP transport:
//! - `POST /mcp` — client sends JSON-RPC request/notification
//! - `GET /mcp` — client opens SSE listener for server-initiated messages
//! - `DELETE /mcp` — client terminates session
//!
//! Session management via `Mcp-Session-Id` header.

use super::{JsonRpcRequest, JsonRpcResponse, McpServer};
use axum::{
    body::Body,
    extract::State,
    http::{HeaderMap, HeaderValue, StatusCode},
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    routing::{delete, get, post},
    Router,
};
use serde_json::Value;
use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::broadcast;
use tokio_stream::wrappers::BroadcastStream;
use tokio_stream::StreamExt;

/// Capacity of the per-session broadcast channel.
const SSE_CHANNEL_CAPACITY: usize = 64;

/// Keepalive interval for SSE streams.
const SSE_KEEPALIVE_SECS: u64 = 30;

/// Shared state for the MCP HTTP transport.
pub(crate) struct McpHttpState {
    server: Arc<McpServer>,
    /// Active sessions: session_id -> session metadata.
    sessions: Mutex<HashMap<String, SessionMeta>>,
    /// Global broadcast channel for server-initiated messages.
    /// Each message is (session_id, json_rpc_notification).
    sse_tx: broadcast::Sender<SseMessage>,
}

/// A message broadcast over the SSE channel.
#[derive(Clone, Debug)]
pub(crate) struct SseMessage {
    /// The session this message belongs to (empty string for global).
    pub session_id: String,
    /// The JSON-RPC notification payload.
    pub payload: String,
}

/// Per-session state for MCP HTTP transport sessions.
struct SessionMeta {
    /// When this session was created.
    created_at: Instant,
    /// When the last request was received for this session.
    last_active_at: Instant,
    /// Number of requests processed in this session.
    request_count: u64,
}

/// Returns an Axum router handling `/mcp` (POST, GET, DELETE)
/// per MCP Streamable HTTP spec (2025-03-26).
pub fn mcp_router(server: Arc<McpServer>) -> Router {
    let (sse_tx, _) = broadcast::channel(SSE_CHANNEL_CAPACITY);
    let state = Arc::new(McpHttpState {
        server,
        sessions: Mutex::new(HashMap::new()),
        sse_tx,
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
                let resp = JsonRpcResponse::error(Value::Null, -32700, format!("Parse error: {e}"));
                return json_response(StatusCode::OK, &resp);
            }
        }
    } else {
        match serde_json::from_str(body) {
            Ok(req) => vec![req],
            Err(e) => {
                let resp = JsonRpcResponse::error(Value::Null, -32700, format!("Parse error: {e}"));
                return json_response(StatusCode::OK, &resp);
            }
        }
    };

    if requests.is_empty() {
        return (StatusCode::BAD_REQUEST, "Empty batch").into_response();
    }

    // Check for initialize request (creates session)
    let is_initialize = requests.iter().any(|r| r.method == "initialize");

    // For non-initialize requests, validate session
    if !is_initialize {
        let session_id = headers.get("mcp-session-id").and_then(|v| v.to_str().ok());

        if let Some(sid) = session_id {
            let mut sessions = state.sessions.lock().unwrap_or_else(|e| e.into_inner());
            match sessions.get_mut(sid) {
                Some(meta) => {
                    meta.last_active_at = Instant::now();
                    meta.request_count += 1;
                }
                None => {
                    return (StatusCode::NOT_FOUND, "Unknown session").into_response();
                }
            }
        }
        // Note: we're lenient here — if no session header, we still process
        // (some clients may not send it for simple requests)
    }

    // Process all requests
    let mut responses: Vec<JsonRpcResponse> = Vec::new();
    let mut all_notifications = true;

    for request in requests {
        if let Some(id) = request.id {
            all_notifications = false;
            let response =
                state
                    .server
                    .handle_request(&request.method, request.params.as_ref(), id);
            responses.push(response);
        } else {
            // Notification — no response
            state.server.handle_notification(&request.method);
        }
    }

    // If all notifications, return 202 Accepted with no body
    if all_notifications {
        return StatusCode::ACCEPTED.into_response();
    }

    // Determine the session_id for broadcasting
    let current_session_id = if is_initialize {
        // Will be set below after creating the session
        None
    } else {
        headers
            .get("mcp-session-id")
            .and_then(|v| v.to_str().ok())
            .map(String::from)
    };

    // Broadcast responses to SSE listeners for this session
    if let Some(ref sid) = current_session_id {
        for resp in &responses {
            if let Ok(payload) = serde_json::to_string(resp) {
                let _ = state.sse_tx.send(SseMessage {
                    session_id: sid.clone(),
                    payload,
                });
            }
        }
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
            let mut sessions = state.sessions.lock().unwrap_or_else(|e| e.into_inner());
            let now = Instant::now();
            sessions.insert(
                session_id.clone(),
                SessionMeta {
                    created_at: now,
                    last_active_at: now,
                    request_count: 1,
                },
            );
        }
        resp.headers_mut().insert(
            "mcp-session-id",
            HeaderValue::from_str(&session_id).unwrap_or_else(|_| HeaderValue::from_static("")),
        );

        // Broadcast the initialize response to SSE listeners for the new session
        for r in &responses {
            if let Ok(payload) = serde_json::to_string(r) {
                let _ = state.sse_tx.send(SseMessage {
                    session_id: session_id.clone(),
                    payload,
                });
            }
        }
    }

    resp
}

/// GET /mcp — Client opens SSE listener for server-initiated messages.
///
/// Returns a streaming SSE response filtered by session ID. The stream
/// delivers JSON-RPC notifications broadcast from POST /mcp handlers and
/// sends periodic keepalive comments to prevent connection timeouts.
async fn handle_get(
    State(state): State<Arc<McpHttpState>>,
    headers: HeaderMap,
) -> impl IntoResponse {
    let session_id = headers
        .get("mcp-session-id")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();

    let rx = state.sse_tx.subscribe();
    let stream = BroadcastStream::new(rx).filter_map(move |result| {
        let sid = session_id.clone();
        match result {
            Ok(msg) if msg.session_id == sid || sid.is_empty() => Some(Ok::<_, Infallible>(
                Event::default().event("message").data(msg.payload),
            )),
            _ => None,
        }
    });

    Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(Duration::from_secs(SSE_KEEPALIVE_SECS))
            .text("keepalive"),
    )
}

/// DELETE /mcp — Client terminates session.
async fn handle_delete(State(state): State<Arc<McpHttpState>>, headers: HeaderMap) -> Response {
    let session_id = headers.get("mcp-session-id").and_then(|v| v.to_str().ok());

    match session_id {
        Some(sid) => {
            let mut sessions = state.sessions.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(meta) = sessions.remove(sid) {
                let duration = meta.created_at.elapsed();
                tracing::info!(
                    session_id = sid,
                    duration_secs = duration.as_secs(),
                    request_count = meta.request_count,
                    "MCP session terminated"
                );
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

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests/http_tests.rs"]
mod tests;
