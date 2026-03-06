use crate::mcp::http::mcp_router;
use crate::mcp::McpServer;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use std::sync::Arc;
use tower::ServiceExt; // for oneshot

fn test_router() -> axum::Router {
    let server = Arc::new(McpServer::for_testing());
    mcp_router(server)
}

// ── POST /mcp ───────────────────────────────────────────────────────

#[tokio::test]
async fn post_empty_body_returns_bad_request() {
    let app = test_router();
    let req = Request::builder()
        .method("POST")
        .uri("/mcp")
        .header("content-type", "application/json")
        .body(Body::from(""))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn post_invalid_json_returns_parse_error() {
    let app = test_router();
    let req = Request::builder()
        .method("POST")
        .uri("/mcp")
        .header("content-type", "application/json")
        .body(Body::from("{not valid json"))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let parsed: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(parsed["error"]["code"], -32700);
}

#[tokio::test]
async fn post_initialize_returns_session_header() {
    let app = test_router();
    let body = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {}
    });
    let req = Request::builder()
        .method("POST")
        .uri("/mcp")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&body).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert!(
        resp.headers().get("mcp-session-id").is_some(),
        "initialize response should contain mcp-session-id header"
    );
}

#[tokio::test]
async fn post_ping_returns_success() {
    let app = test_router();
    let body = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "ping"
    });
    let req = Request::builder()
        .method("POST")
        .uri("/mcp")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&body).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let parsed: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert!(parsed["result"].is_object());
}

#[tokio::test]
async fn post_notification_returns_accepted() {
    let app = test_router();
    // Notifications have no "id" field
    let body = serde_json::json!({
        "jsonrpc": "2.0",
        "method": "notifications/initialized"
    });
    let req = Request::builder()
        .method("POST")
        .uri("/mcp")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&body).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::ACCEPTED);
}

#[tokio::test]
async fn post_batch_request() {
    let app = test_router();
    let body = serde_json::json!([
        { "jsonrpc": "2.0", "id": 1, "method": "ping" },
        { "jsonrpc": "2.0", "id": 2, "method": "ping" }
    ]);
    let req = Request::builder()
        .method("POST")
        .uri("/mcp")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&body).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let parsed: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert!(parsed.is_array(), "batch response should be an array");
    assert_eq!(parsed.as_array().unwrap().len(), 2);
}

#[tokio::test]
async fn post_empty_batch_returns_bad_request() {
    let app = test_router();
    let req = Request::builder()
        .method("POST")
        .uri("/mcp")
        .header("content-type", "application/json")
        .body(Body::from("[]"))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn post_unknown_session_returns_not_found() {
    let app = test_router();
    let body = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "ping"
    });
    let req = Request::builder()
        .method("POST")
        .uri("/mcp")
        .header("content-type", "application/json")
        .header("mcp-session-id", "nonexistent-session-id")
        .body(Body::from(serde_json::to_string(&body).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

// ── GET /mcp ────────────────────────────────────────────────────────

#[tokio::test]
async fn get_mcp_returns_sse_content_type() {
    let app = test_router();
    let req = Request::builder()
        .method("GET")
        .uri("/mcp")
        .body(Body::empty())
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let ct = resp
        .headers()
        .get("content-type")
        .unwrap()
        .to_str()
        .unwrap();
    assert_eq!(ct, "text/event-stream");
}

// ── DELETE /mcp ─────────────────────────────────────────────────────

#[tokio::test]
async fn delete_without_session_header_returns_bad_request() {
    let app = test_router();
    let req = Request::builder()
        .method("DELETE")
        .uri("/mcp")
        .body(Body::empty())
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn delete_unknown_session_returns_not_found() {
    let app = test_router();
    let req = Request::builder()
        .method("DELETE")
        .uri("/mcp")
        .header("mcp-session-id", "no-such-session")
        .body(Body::empty())
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

// ── Session lifecycle ───────────────────────────────────────────────

#[tokio::test]
async fn session_create_and_delete() {
    let server = Arc::new(McpServer::for_testing());
    let app = mcp_router(Arc::clone(&server));

    // Step 1: Initialize to get a session
    let init_body = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {}
    });
    let req = Request::builder()
        .method("POST")
        .uri("/mcp")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&init_body).unwrap()))
        .unwrap();

    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let session_id = resp
        .headers()
        .get("mcp-session-id")
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();

    // Step 2: Use the session for a request
    let ping_body = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "ping"
    });
    let req = Request::builder()
        .method("POST")
        .uri("/mcp")
        .header("content-type", "application/json")
        .header("mcp-session-id", &session_id)
        .body(Body::from(serde_json::to_string(&ping_body).unwrap()))
        .unwrap();

    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // Step 3: Delete the session
    let req = Request::builder()
        .method("DELETE")
        .uri("/mcp")
        .header("mcp-session-id", &session_id)
        .body(Body::empty())
        .unwrap();

    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // Step 4: Deleting again should return NOT_FOUND
    let req = Request::builder()
        .method("DELETE")
        .uri("/mcp")
        .header("mcp-session-id", &session_id)
        .body(Body::empty())
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}
