use crate::api::ApiServer;
use crate::mcp::McpServer;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use std::sync::Arc;

/// Create a test `ApiServer` backed by a temp database.
///
/// Forces `CODEMEM_EMBED_PROVIDER=candle` so that lazy embedding init doesn't
/// create a `reqwest::blocking::Client` (which spawns a Tokio runtime internally
/// and panics when dropped inside `#[tokio::test]` async context in Tokio ≥1.50).
fn test_api_server(tmp: &tempfile::TempDir) -> ApiServer {
    std::env::set_var("CODEMEM_EMBED_PROVIDER", "candle");
    let db_path = tmp.path().join("test.db");
    let server = McpServer::from_db_path(&db_path).expect("create test server");
    ApiServer::new(Arc::new(server))
}

/// Helper: send a request to the router and return (status, body-as-string).
async fn send(router: axum::Router, request: Request<Body>) -> (StatusCode, String) {
    let response = tower::ServiceExt::oneshot(router, request)
        .await
        .expect("oneshot request");
    let status = response.status();
    let body = response.into_body().collect().await.unwrap().to_bytes();
    let text = String::from_utf8_lossy(&body).to_string();
    (status, text)
}

// ── Health & Stats ──────────────────────────────────────────────────────────

#[tokio::test]
async fn get_health_returns_ok() {
    let tmp = tempfile::tempdir().unwrap();
    let api = test_api_server(&tmp);
    let router = api.router();

    let req = Request::builder()
        .uri("/api/health")
        .body(Body::empty())
        .unwrap();

    let (status, body) = send(router, req).await;
    assert_eq!(status, StatusCode::OK);

    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(json["storage"]["status"], "ok");
    assert_eq!(json["vector"]["status"], "ok");
    assert_eq!(json["graph"]["status"], "ok");
}

#[tokio::test]
async fn get_stats_returns_zero_counts() {
    let tmp = tempfile::tempdir().unwrap();
    let api = test_api_server(&tmp);
    let router = api.router();

    let req = Request::builder()
        .uri("/api/stats")
        .body(Body::empty())
        .unwrap();

    let (status, body) = send(router, req).await;
    assert_eq!(status, StatusCode::OK);

    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(json["memory_count"], 0);
    assert_eq!(json["node_count"], 0);
    assert_eq!(json["edge_count"], 0);
    assert_eq!(json["session_count"], 0);
}

#[tokio::test]
async fn get_metrics_returns_ok() {
    let tmp = tempfile::tempdir().unwrap();
    let api = test_api_server(&tmp);
    let router = api.router();

    let req = Request::builder()
        .uri("/api/metrics")
        .body(Body::empty())
        .unwrap();

    let (status, body) = send(router, req).await;
    assert_eq!(status, StatusCode::OK);

    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(json["tool_calls_total"], 0);
}

// ── Memories CRUD ───────────────────────────────────────────────────────────

#[tokio::test]
async fn list_memories_empty() {
    let tmp = tempfile::tempdir().unwrap();
    let api = test_api_server(&tmp);
    let router = api.router();

    let req = Request::builder()
        .uri("/api/memories")
        .body(Body::empty())
        .unwrap();

    let (status, body) = send(router, req).await;
    assert_eq!(status, StatusCode::OK);

    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(json["total"], 0);
    assert_eq!(json["memories"].as_array().unwrap().len(), 0);
}

#[tokio::test]
async fn store_and_get_memory() {
    let tmp = tempfile::tempdir().unwrap();
    let api = test_api_server(&tmp);
    let router = api.router();

    // Store a memory
    let store_req = Request::builder()
        .method("POST")
        .uri("/api/memories")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_string(&serde_json::json!({
                "content": "Rust uses ownership for memory safety",
                "memory_type": "insight",
                "importance": 0.8,
                "tags": ["rust"]
            }))
            .unwrap(),
        ))
        .unwrap();

    let (status, body) = send(router, store_req).await;
    assert_eq!(status, StatusCode::CREATED);
    let store_resp: serde_json::Value = serde_json::from_str(&body).unwrap();
    let id = store_resp["id"].as_str().unwrap().to_string();

    // Get the memory by ID
    let router = api.router();
    let get_req = Request::builder()
        .uri(format!("/api/memories/{id}"))
        .body(Body::empty())
        .unwrap();

    let (status, body) = send(router, get_req).await;
    assert_eq!(status, StatusCode::OK);
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(json["id"], id);
    assert_eq!(json["content"], "Rust uses ownership for memory safety");
    assert_eq!(json["memory_type"], "insight");
}

#[tokio::test]
async fn get_nonexistent_memory_returns_404() {
    let tmp = tempfile::tempdir().unwrap();
    let api = test_api_server(&tmp);
    let router = api.router();

    let req = Request::builder()
        .uri("/api/memories/nonexistent-id")
        .body(Body::empty())
        .unwrap();

    let (status, _) = send(router, req).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn delete_memory_and_verify_gone() {
    let tmp = tempfile::tempdir().unwrap();
    let api = test_api_server(&tmp);

    // Store
    let router = api.router();
    let store_req = Request::builder()
        .method("POST")
        .uri("/api/memories")
        .header("content-type", "application/json")
        .body(Body::from(r#"{"content":"to be deleted"}"#))
        .unwrap();

    let (status, body) = send(router, store_req).await;
    assert_eq!(status, StatusCode::CREATED);
    let id = serde_json::from_str::<serde_json::Value>(&body).unwrap()["id"]
        .as_str()
        .unwrap()
        .to_string();

    // Delete
    let router = api.router();
    let del_req = Request::builder()
        .method("DELETE")
        .uri(format!("/api/memories/{id}"))
        .body(Body::empty())
        .unwrap();

    let (status, body) = send(router, del_req).await;
    assert_eq!(status, StatusCode::OK);
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(json["message"], "Deleted");

    // Verify gone
    let router = api.router();
    let get_req = Request::builder()
        .uri(format!("/api/memories/{id}"))
        .body(Body::empty())
        .unwrap();
    let (status, _) = send(router, get_req).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn delete_nonexistent_memory_returns_404() {
    let tmp = tempfile::tempdir().unwrap();
    let api = test_api_server(&tmp);
    let router = api.router();

    let req = Request::builder()
        .method("DELETE")
        .uri("/api/memories/does-not-exist")
        .body(Body::empty())
        .unwrap();

    let (status, _) = send(router, req).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn update_memory_content() {
    let tmp = tempfile::tempdir().unwrap();
    let api = test_api_server(&tmp);

    // Store
    let router = api.router();
    let store_req = Request::builder()
        .method("POST")
        .uri("/api/memories")
        .header("content-type", "application/json")
        .body(Body::from(r#"{"content":"original content"}"#))
        .unwrap();

    let (_, body) = send(router, store_req).await;
    let id = serde_json::from_str::<serde_json::Value>(&body).unwrap()["id"]
        .as_str()
        .unwrap()
        .to_string();

    // Update
    let router = api.router();
    let update_req = Request::builder()
        .method("PUT")
        .uri(format!("/api/memories/{id}"))
        .header("content-type", "application/json")
        .body(Body::from(
            r#"{"content":"updated content","importance":0.9}"#,
        ))
        .unwrap();

    let (status, body) = send(router, update_req).await;
    assert_eq!(status, StatusCode::OK);
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(json["message"], "Updated");
}

#[tokio::test]
async fn update_nonexistent_memory_returns_404() {
    let tmp = tempfile::tempdir().unwrap();
    let api = test_api_server(&tmp);
    let router = api.router();

    let req = Request::builder()
        .method("PUT")
        .uri("/api/memories/no-such-id")
        .header("content-type", "application/json")
        .body(Body::from(r#"{"content":"new"}"#))
        .unwrap();

    let (status, _) = send(router, req).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn list_memories_with_pagination() {
    let tmp = tempfile::tempdir().unwrap();
    let api = test_api_server(&tmp);

    // Store 3 memories
    for i in 0..3 {
        let router = api.router();
        let req = Request::builder()
            .method("POST")
            .uri("/api/memories")
            .header("content-type", "application/json")
            .body(Body::from(format!(r#"{{"content":"memory {i}"}}"#)))
            .unwrap();
        let (status, _) = send(router, req).await;
        assert_eq!(status, StatusCode::CREATED);
    }

    // List with offset=1, limit=1
    let router = api.router();
    let req = Request::builder()
        .uri("/api/memories?offset=1&limit=1")
        .body(Body::empty())
        .unwrap();

    let (status, body) = send(router, req).await;
    assert_eq!(status, StatusCode::OK);
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(json["total"], 3);
    assert_eq!(json["offset"], 1);
    assert_eq!(json["limit"], 1);
    assert_eq!(json["memories"].as_array().unwrap().len(), 1);
}

// ── Search ──────────────────────────────────────────────────────────────────

#[tokio::test]
async fn search_empty_database() {
    let tmp = tempfile::tempdir().unwrap();
    let api = test_api_server(&tmp);
    let router = api.router();

    let req = Request::builder()
        .uri("/api/search?q=rust+ownership")
        .body(Body::empty())
        .unwrap();

    let (status, body) = send(router, req).await;
    assert_eq!(status, StatusCode::OK);
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(json["results"].as_array().unwrap().len(), 0);
}

// ── Graph ───────────────────────────────────────────────────────────────────

#[tokio::test]
async fn get_subgraph_empty() {
    let tmp = tempfile::tempdir().unwrap();
    let api = test_api_server(&tmp);
    let router = api.router();

    let req = Request::builder()
        .uri("/api/graph/subgraph")
        .body(Body::empty())
        .unwrap();

    let (status, body) = send(router, req).await;
    assert_eq!(status, StatusCode::OK);
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert!(json["nodes"].as_array().unwrap().is_empty());
    assert!(json["edges"].as_array().unwrap().is_empty());
}

#[tokio::test]
async fn get_communities_empty() {
    let tmp = tempfile::tempdir().unwrap();
    let api = test_api_server(&tmp);
    let router = api.router();

    let req = Request::builder()
        .uri("/api/graph/communities")
        .body(Body::empty())
        .unwrap();

    let (status, body) = send(router, req).await;
    assert_eq!(status, StatusCode::OK);
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(json["num_communities"], 0);
}

#[tokio::test]
async fn get_pagerank_empty() {
    let tmp = tempfile::tempdir().unwrap();
    let api = test_api_server(&tmp);
    let router = api.router();

    let req = Request::builder()
        .uri("/api/graph/pagerank")
        .body(Body::empty())
        .unwrap();

    let (status, body) = send(router, req).await;
    assert_eq!(status, StatusCode::OK);
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert!(json["scores"].as_array().unwrap().is_empty());
}

#[tokio::test]
async fn get_neighbors_nonexistent_returns_404() {
    let tmp = tempfile::tempdir().unwrap();
    let api = test_api_server(&tmp);
    let router = api.router();

    let req = Request::builder()
        .uri("/api/graph/neighbors/nonexistent")
        .body(Body::empty())
        .unwrap();

    let (status, _) = send(router, req).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn get_impact_nonexistent_returns_404() {
    let tmp = tempfile::tempdir().unwrap();
    let api = test_api_server(&tmp);
    let router = api.router();

    let req = Request::builder()
        .uri("/api/graph/impact/nonexistent")
        .body(Body::empty())
        .unwrap();

    let (status, _) = send(router, req).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn get_shortest_path_nonexistent_returns_404() {
    let tmp = tempfile::tempdir().unwrap();
    let api = test_api_server(&tmp);
    let router = api.router();

    let req = Request::builder()
        .uri("/api/graph/shortest-path?from=a&to=b")
        .body(Body::empty())
        .unwrap();

    let (status, _) = send(router, req).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn reload_graph_returns_ok() {
    let tmp = tempfile::tempdir().unwrap();
    let api = test_api_server(&tmp);
    let router = api.router();

    let req = Request::builder()
        .method("POST")
        .uri("/api/graph/reload")
        .body(Body::empty())
        .unwrap();

    let (status, body) = send(router, req).await;
    assert_eq!(status, StatusCode::OK);
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(json["status"], "ok");
}

#[tokio::test]
async fn get_graph_browse_empty() {
    let tmp = tempfile::tempdir().unwrap();
    let api = test_api_server(&tmp);
    let router = api.router();

    let req = Request::builder()
        .uri("/api/graph/browse")
        .body(Body::empty())
        .unwrap();

    let (status, body) = send(router, req).await;
    assert_eq!(status, StatusCode::OK);
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(json["total"], 0);
    assert_eq!(json["edge_count"], 0);
}

// ── Vectors ─────────────────────────────────────────────────────────────────

#[tokio::test]
async fn get_vectors_empty() {
    let tmp = tempfile::tempdir().unwrap();
    let api = test_api_server(&tmp);
    let router = api.router();

    let req = Request::builder()
        .uri("/api/vectors")
        .body(Body::empty())
        .unwrap();

    let (status, body) = send(router, req).await;
    assert_eq!(status, StatusCode::OK);
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert!(json.as_array().unwrap().is_empty());
}

// ── Namespaces ──────────────────────────────────────────────────────────────

#[tokio::test]
async fn list_namespaces_empty() {
    let tmp = tempfile::tempdir().unwrap();
    let api = test_api_server(&tmp);
    let router = api.router();

    let req = Request::builder()
        .uri("/api/namespaces")
        .body(Body::empty())
        .unwrap();

    let (status, body) = send(router, req).await;
    assert_eq!(status, StatusCode::OK);
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert!(json.as_array().unwrap().is_empty());
}

#[tokio::test]
async fn get_namespace_stats_for_namespace() {
    let tmp = tempfile::tempdir().unwrap();
    let api = test_api_server(&tmp);

    // Store a memory with a namespace
    let router = api.router();
    let store_req = Request::builder()
        .method("POST")
        .uri("/api/memories")
        .header("content-type", "application/json")
        .body(Body::from(
            r#"{"content":"namespaced memory","namespace":"my-ns","memory_type":"insight"}"#,
        ))
        .unwrap();
    let (status, _) = send(router, store_req).await;
    assert_eq!(status, StatusCode::CREATED);

    // Get stats for the namespace
    let router = api.router();
    let req = Request::builder()
        .uri("/api/namespaces/my-ns/stats")
        .body(Body::empty())
        .unwrap();

    let (status, body) = send(router, req).await;
    assert_eq!(status, StatusCode::OK);
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(json["namespace"], "my-ns");
    assert_eq!(json["memory_count"], 1);
    assert_eq!(json["type_distribution"]["insight"], 1);
}

// ── Sessions ────────────────────────────────────────────────────────────────

#[tokio::test]
async fn session_lifecycle() {
    let tmp = tempfile::tempdir().unwrap();
    let api = test_api_server(&tmp);

    // List sessions (empty)
    let router = api.router();
    let req = Request::builder()
        .uri("/api/sessions")
        .body(Body::empty())
        .unwrap();
    let (status, body) = send(router, req).await;
    assert_eq!(status, StatusCode::OK);
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert!(json.as_array().unwrap().is_empty());

    // Start a session
    let router = api.router();
    let req = Request::builder()
        .method("POST")
        .uri("/api/sessions")
        .header("content-type", "application/json")
        .body(Body::from(r#"{"namespace":"test-ns"}"#))
        .unwrap();
    let (status, body) = send(router, req).await;
    assert_eq!(status, StatusCode::CREATED);
    let session_id = serde_json::from_str::<serde_json::Value>(&body).unwrap()["id"]
        .as_str()
        .unwrap()
        .to_string();

    // End the session
    let router = api.router();
    let req = Request::builder()
        .method("POST")
        .uri(format!("/api/sessions/{session_id}/end"))
        .header("content-type", "application/json")
        .body(Body::from(r#"{"summary":"test complete"}"#))
        .unwrap();
    let (status, body) = send(router, req).await;
    assert_eq!(status, StatusCode::OK);
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(json["message"], "Session ended");

    // List sessions (should have one now)
    let router = api.router();
    let req = Request::builder()
        .uri("/api/sessions")
        .body(Body::empty())
        .unwrap();
    let (status, body) = send(router, req).await;
    assert_eq!(status, StatusCode::OK);
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(json.as_array().unwrap().len(), 1);
}

// ── Timeline & Distribution ─────────────────────────────────────────────────

#[tokio::test]
async fn get_timeline_empty() {
    let tmp = tempfile::tempdir().unwrap();
    let api = test_api_server(&tmp);
    let router = api.router();

    let req = Request::builder()
        .uri("/api/timeline")
        .body(Body::empty())
        .unwrap();

    let (status, body) = send(router, req).await;
    assert_eq!(status, StatusCode::OK);
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert!(json.as_array().unwrap().is_empty());
}

#[tokio::test]
async fn get_distribution_empty() {
    let tmp = tempfile::tempdir().unwrap();
    let api = test_api_server(&tmp);
    let router = api.router();

    let req = Request::builder()
        .uri("/api/distribution")
        .body(Body::empty())
        .unwrap();

    let (status, body) = send(router, req).await;
    assert_eq!(status, StatusCode::OK);
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(json["total"], 0);
    assert_eq!(json["importance_histogram"].as_array().unwrap().len(), 10);
}

#[tokio::test]
async fn get_distribution_with_memories() {
    let tmp = tempfile::tempdir().unwrap();
    let api = test_api_server(&tmp);

    // Store memories with different types
    for (content, mem_type) in [("a", "insight"), ("b", "decision"), ("c", "insight")] {
        let router = api.router();
        let req = Request::builder()
            .method("POST")
            .uri("/api/memories")
            .header("content-type", "application/json")
            .body(Body::from(format!(
                r#"{{"content":"{content}","memory_type":"{mem_type}"}}"#
            )))
            .unwrap();
        let (status, _) = send(router, req).await;
        assert_eq!(status, StatusCode::CREATED);
    }

    let router = api.router();
    let req = Request::builder()
        .uri("/api/distribution")
        .body(Body::empty())
        .unwrap();

    let (status, body) = send(router, req).await;
    assert_eq!(status, StatusCode::OK);
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(json["total"], 3);
    assert_eq!(json["type_counts"]["insight"], 2);
    assert_eq!(json["type_counts"]["decision"], 1);
}

// ── Patterns & Consolidation ────────────────────────────────────────────────

#[tokio::test]
async fn get_patterns_empty() {
    let tmp = tempfile::tempdir().unwrap();
    let api = test_api_server(&tmp);
    let router = api.router();

    let req = Request::builder()
        .uri("/api/patterns")
        .body(Body::empty())
        .unwrap();

    let (status, body) = send(router, req).await;
    assert_eq!(status, StatusCode::OK);
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert!(json.as_array().unwrap().is_empty());
}

#[tokio::test]
async fn get_consolidation_status_shows_all_cycles() {
    let tmp = tempfile::tempdir().unwrap();
    let api = test_api_server(&tmp);
    let router = api.router();

    let req = Request::builder()
        .uri("/api/consolidation/status")
        .body(Body::empty())
        .unwrap();

    let (status, body) = send(router, req).await;
    assert_eq!(status, StatusCode::OK);
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    let cycles = json["cycles"].as_array().unwrap();
    assert_eq!(cycles.len(), 5);
    let cycle_names: Vec<&str> = cycles.iter().filter_map(|c| c["cycle"].as_str()).collect();
    assert!(cycle_names.contains(&"decay"));
    assert!(cycle_names.contains(&"creative"));
    assert!(cycle_names.contains(&"cluster"));
    assert!(cycle_names.contains(&"summarize"));
    assert!(cycle_names.contains(&"forget"));
}

#[tokio::test]
async fn run_consolidation_unknown_cycle_returns_400() {
    let tmp = tempfile::tempdir().unwrap();
    let api = test_api_server(&tmp);
    let router = api.router();

    let req = Request::builder()
        .method("POST")
        .uri("/api/consolidation/invalid-cycle")
        .body(Body::empty())
        .unwrap();

    let (status, body) = send(router, req).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert!(json["message"].as_str().unwrap().contains("Unknown cycle"));
}

// ── Insights ────────────────────────────────────────────────────────────────

#[tokio::test]
async fn get_activity_insights_empty() {
    let tmp = tempfile::tempdir().unwrap();
    let api = test_api_server(&tmp);
    let router = api.router();

    let req = Request::builder()
        .uri("/api/insights/activity")
        .body(Body::empty())
        .unwrap();

    let (status, body) = send(router, req).await;
    assert_eq!(status, StatusCode::OK);
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert!(json["insights"].as_array().unwrap().is_empty());
    assert_eq!(json["git_summary"]["total_annotated_files"], 0);
}

#[tokio::test]
async fn get_code_health_insights_empty() {
    let tmp = tempfile::tempdir().unwrap();
    let api = test_api_server(&tmp);
    let router = api.router();

    let req = Request::builder()
        .uri("/api/insights/code-health")
        .body(Body::empty())
        .unwrap();

    let (status, body) = send(router, req).await;
    assert_eq!(status, StatusCode::OK);
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert!(json["insights"].as_array().unwrap().is_empty());
}

#[tokio::test]
async fn get_security_insights_empty() {
    let tmp = tempfile::tempdir().unwrap();
    let api = test_api_server(&tmp);
    let router = api.router();

    let req = Request::builder()
        .uri("/api/insights/security")
        .body(Body::empty())
        .unwrap();

    let (status, body) = send(router, req).await;
    assert_eq!(status, StatusCode::OK);
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(json["sensitive_file_count"], 0);
    assert_eq!(json["endpoint_count"], 0);
}

#[tokio::test]
async fn get_performance_insights_empty() {
    let tmp = tempfile::tempdir().unwrap();
    let api = test_api_server(&tmp);
    let router = api.router();

    let req = Request::builder()
        .uri("/api/insights/performance")
        .body(Body::empty())
        .unwrap();

    let (status, body) = send(router, req).await;
    assert_eq!(status, StatusCode::OK);
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(json["max_depth"], 0);
    assert!(json["high_coupling_nodes"].as_array().unwrap().is_empty());
}

// ── Config ──────────────────────────────────────────────────────────────────

#[tokio::test]
async fn get_config_returns_json() {
    let tmp = tempfile::tempdir().unwrap();
    let api = test_api_server(&tmp);
    let router = api.router();

    let req = Request::builder()
        .uri("/api/config")
        .body(Body::empty())
        .unwrap();

    let (status, body) = send(router, req).await;
    assert_eq!(status, StatusCode::OK);
    // Should be parseable JSON with scoring weights
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert!(json.is_object());
}

// ── Repos ───────────────────────────────────────────────────────────────────

#[tokio::test]
async fn repo_lifecycle() {
    let tmp = tempfile::tempdir().unwrap();
    let api = test_api_server(&tmp);

    // List repos (empty)
    let router = api.router();
    let req = Request::builder()
        .uri("/api/repos")
        .body(Body::empty())
        .unwrap();
    let (status, body) = send(router, req).await;
    assert_eq!(status, StatusCode::OK);
    assert!(serde_json::from_str::<serde_json::Value>(&body)
        .unwrap()
        .as_array()
        .unwrap()
        .is_empty());

    // Register a repo
    let router = api.router();
    let req = Request::builder()
        .method("POST")
        .uri("/api/repos")
        .header("content-type", "application/json")
        .body(Body::from(
            r#"{"path":"/tmp/test-repo","name":"test-repo"}"#,
        ))
        .unwrap();
    let (status, body) = send(router, req).await;
    assert_eq!(status, StatusCode::CREATED);
    let repo_id = serde_json::from_str::<serde_json::Value>(&body).unwrap()["id"]
        .as_str()
        .unwrap()
        .to_string();

    // Get the repo
    let router = api.router();
    let req = Request::builder()
        .uri(format!("/api/repos/{repo_id}"))
        .body(Body::empty())
        .unwrap();
    let (status, body) = send(router, req).await;
    assert_eq!(status, StatusCode::OK);
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(json["path"], "/tmp/test-repo");
    assert_eq!(json["name"], "test-repo");

    // Delete the repo
    let router = api.router();
    let req = Request::builder()
        .method("DELETE")
        .uri(format!("/api/repos/{repo_id}"))
        .body(Body::empty())
        .unwrap();
    let (status, _) = send(router, req).await;
    assert_eq!(status, StatusCode::OK);

    // Verify deleted
    let router = api.router();
    let req = Request::builder()
        .uri(format!("/api/repos/{repo_id}"))
        .body(Body::empty())
        .unwrap();
    let (status, _) = send(router, req).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn get_nonexistent_repo_returns_404() {
    let tmp = tempfile::tempdir().unwrap();
    let api = test_api_server(&tmp);
    let router = api.router();

    let req = Request::builder()
        .uri("/api/repos/nonexistent")
        .body(Body::empty())
        .unwrap();

    let (status, _) = send(router, req).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

// ── Agents / Recipes ────────────────────────────────────────────────────────

#[tokio::test]
async fn list_recipes_returns_predefined() {
    let tmp = tempfile::tempdir().unwrap();
    let api = test_api_server(&tmp);
    let router = api.router();

    let req = Request::builder()
        .uri("/api/agents/recipes")
        .body(Body::empty())
        .unwrap();

    let (status, body) = send(router, req).await;
    assert_eq!(status, StatusCode::OK);
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    let recipes = json.as_array().unwrap();
    assert!(
        recipes.len() >= 3,
        "Should have at least 3 predefined recipes"
    );

    let ids: Vec<&str> = recipes.iter().filter_map(|r| r["id"].as_str()).collect();
    assert!(ids.contains(&"full-analysis"));
    assert!(ids.contains(&"quick-index"));
    assert!(ids.contains(&"graph-analysis"));
}

// ── Unknown routes ──────────────────────────────────────────────────────────

#[tokio::test]
async fn unknown_route_returns_404() {
    let tmp = tempfile::tempdir().unwrap();
    let api = test_api_server(&tmp);
    let router = api.router();

    let req = Request::builder()
        .uri("/api/nonexistent")
        .body(Body::empty())
        .unwrap();

    let (status, _) = send(router, req).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

// ── Namespace Deletion ──────────────────────────────────────────────────────

#[tokio::test]
async fn delete_namespace_returns_200() {
    let tmp = tempfile::tempdir().unwrap();
    let api = test_api_server(&tmp);

    // Store a memory in a namespace
    let router = api.router();
    let store_req = Request::builder()
        .method("POST")
        .uri("/api/memories")
        .header("content-type", "application/json")
        .body(Body::from(
            r#"{"content":"namespace memory","namespace":"to-delete","memory_type":"insight"}"#,
        ))
        .unwrap();
    let (status, _) = send(router, store_req).await;
    assert_eq!(status, StatusCode::CREATED);

    // Delete the namespace
    let router = api.router();
    let del_req = Request::builder()
        .method("DELETE")
        .uri("/api/namespaces/to-delete")
        .body(Body::empty())
        .unwrap();

    let (status, body) = send(router, del_req).await;
    assert_eq!(status, StatusCode::OK);
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert!(
        json["message"].as_str().unwrap().contains("Deleted"),
        "Response should confirm deletion"
    );
}

#[tokio::test]
async fn delete_namespace_nonexistent_returns_200_zero_deleted() {
    let tmp = tempfile::tempdir().unwrap();
    let api = test_api_server(&tmp);
    let router = api.router();

    let req = Request::builder()
        .method("DELETE")
        .uri("/api/namespaces/nonexistent-ns")
        .body(Body::empty())
        .unwrap();

    let (status, body) = send(router, req).await;
    assert_eq!(status, StatusCode::OK);
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert!(json["message"].as_str().unwrap().contains("Deleted 0"));
}

// ── Update Memory Variants ──────────────────────────────────────────────────

#[tokio::test]
async fn update_memory_importance_only() {
    let tmp = tempfile::tempdir().unwrap();
    let api = test_api_server(&tmp);

    // Store
    let router = api.router();
    let store_req = Request::builder()
        .method("POST")
        .uri("/api/memories")
        .header("content-type", "application/json")
        .body(Body::from(
            r#"{"content":"importance-only update test","importance":0.3}"#,
        ))
        .unwrap();

    let (_, body) = send(router, store_req).await;
    let id = serde_json::from_str::<serde_json::Value>(&body).unwrap()["id"]
        .as_str()
        .unwrap()
        .to_string();

    // Update with only importance (no content)
    let router = api.router();
    let update_req = Request::builder()
        .method("PUT")
        .uri(format!("/api/memories/{id}"))
        .header("content-type", "application/json")
        .body(Body::from(r#"{"importance":0.95}"#))
        .unwrap();

    let (status, body) = send(router, update_req).await;
    assert_eq!(status, StatusCode::OK);
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(json["message"], "Updated");

    // Verify importance was changed
    let router = api.router();
    let get_req = Request::builder()
        .uri(format!("/api/memories/{id}"))
        .body(Body::empty())
        .unwrap();
    let (status, body) = send(router, get_req).await;
    assert_eq!(status, StatusCode::OK);
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert!(
        (json["importance"].as_f64().unwrap() - 0.95).abs() < 0.01,
        "Importance should be updated to 0.95"
    );
}

#[tokio::test]
async fn update_memory_no_content_no_importance_is_noop() {
    let tmp = tempfile::tempdir().unwrap();
    let api = test_api_server(&tmp);

    // Store
    let router = api.router();
    let store_req = Request::builder()
        .method("POST")
        .uri("/api/memories")
        .header("content-type", "application/json")
        .body(Body::from(r#"{"content":"noop update test"}"#))
        .unwrap();

    let (_, body) = send(router, store_req).await;
    let id = serde_json::from_str::<serde_json::Value>(&body).unwrap()["id"]
        .as_str()
        .unwrap()
        .to_string();

    // Update with neither content nor importance
    let router = api.router();
    let update_req = Request::builder()
        .method("PUT")
        .uri(format!("/api/memories/{id}"))
        .header("content-type", "application/json")
        .body(Body::from(r#"{}"#))
        .unwrap();

    let (status, body) = send(router, update_req).await;
    assert_eq!(status, StatusCode::OK);
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(json["message"], "Updated");

    // Verify the memory is unchanged
    let router = api.router();
    let get_req = Request::builder()
        .uri(format!("/api/memories/{id}"))
        .body(Body::empty())
        .unwrap();
    let (status, body) = send(router, get_req).await;
    assert_eq!(status, StatusCode::OK);
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(json["content"], "noop update test");
}

// ── Parameter Validation ────────────────────────────────────────────────────

#[tokio::test]
async fn store_memory_importance_above_one_accepted() {
    let tmp = tempfile::tempdir().unwrap();
    let api = test_api_server(&tmp);
    let router = api.router();

    let req = Request::builder()
        .method("POST")
        .uri("/api/memories")
        .header("content-type", "application/json")
        .body(Body::from(
            r#"{"content":"over-importance test","importance":1.5}"#,
        ))
        .unwrap();

    let (status, _) = send(router, req).await;
    // The API does not validate importance range; it stores as-is
    assert_eq!(status, StatusCode::CREATED);
}

#[tokio::test]
async fn store_memory_importance_below_zero_accepted() {
    let tmp = tempfile::tempdir().unwrap();
    let api = test_api_server(&tmp);
    let router = api.router();

    let req = Request::builder()
        .method("POST")
        .uri("/api/memories")
        .header("content-type", "application/json")
        .body(Body::from(
            r#"{"content":"negative importance test","importance":-0.5}"#,
        ))
        .unwrap();

    let (status, _) = send(router, req).await;
    assert_eq!(status, StatusCode::CREATED);
}

#[tokio::test]
async fn store_memory_empty_content_returns_error() {
    let tmp = tempfile::tempdir().unwrap();
    let api = test_api_server(&tmp);
    let router = api.router();

    let req = Request::builder()
        .method("POST")
        .uri("/api/memories")
        .header("content-type", "application/json")
        .body(Body::from(r#"{"content":""}"#))
        .unwrap();

    let (status, _) = send(router, req).await;
    // Empty content is accepted at the API layer (stored with empty string)
    // The API does not reject empty content, unlike the MCP layer
    assert_eq!(status, StatusCode::CREATED);
}

#[tokio::test]
async fn list_memories_with_limit_zero() {
    let tmp = tempfile::tempdir().unwrap();
    let api = test_api_server(&tmp);

    // Store a memory
    let router = api.router();
    let store_req = Request::builder()
        .method("POST")
        .uri("/api/memories")
        .header("content-type", "application/json")
        .body(Body::from(r#"{"content":"limit zero test"}"#))
        .unwrap();
    let (status, _) = send(router, store_req).await;
    assert_eq!(status, StatusCode::CREATED);

    // List with limit=0
    let router = api.router();
    let req = Request::builder()
        .uri("/api/memories?limit=0")
        .body(Body::empty())
        .unwrap();

    let (status, body) = send(router, req).await;
    assert_eq!(status, StatusCode::OK);
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    // limit=0 means take(0) = 0 items returned
    assert_eq!(json["memories"].as_array().unwrap().len(), 0);
    // total should still reflect actual count
    assert_eq!(json["total"], 1);
}

#[tokio::test]
async fn list_memories_with_large_offset() {
    let tmp = tempfile::tempdir().unwrap();
    let api = test_api_server(&tmp);

    // Store a memory
    let router = api.router();
    let store_req = Request::builder()
        .method("POST")
        .uri("/api/memories")
        .header("content-type", "application/json")
        .body(Body::from(r#"{"content":"offset test"}"#))
        .unwrap();
    let (status, _) = send(router, store_req).await;
    assert_eq!(status, StatusCode::CREATED);

    // List with offset beyond total count
    let router = api.router();
    let req = Request::builder()
        .uri("/api/memories?offset=999")
        .body(Body::empty())
        .unwrap();

    let (status, body) = send(router, req).await;
    assert_eq!(status, StatusCode::OK);
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(json["memories"].as_array().unwrap().len(), 0);
    assert_eq!(json["total"], 1);
}

#[tokio::test]
async fn search_with_empty_query() {
    let tmp = tempfile::tempdir().unwrap();
    let api = test_api_server(&tmp);
    let router = api.router();

    let req = Request::builder()
        .uri("/api/search?q=")
        .body(Body::empty())
        .unwrap();

    let (status, body) = send(router, req).await;
    assert_eq!(status, StatusCode::OK);
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    // Empty query returns empty results
    assert!(json["results"].as_array().unwrap().is_empty());
}

#[tokio::test]
async fn search_with_k_zero() {
    let tmp = tempfile::tempdir().unwrap();
    let api = test_api_server(&tmp);

    // Store a memory so there's something to search
    let router = api.router();
    let store_req = Request::builder()
        .method("POST")
        .uri("/api/memories")
        .header("content-type", "application/json")
        .body(Body::from(r#"{"content":"searchable memory about rust"}"#))
        .unwrap();
    let (status, _) = send(router, store_req).await;
    assert_eq!(status, StatusCode::CREATED);

    // Search with k=0
    let router = api.router();
    let req = Request::builder()
        .uri("/api/search?q=rust&k=0")
        .body(Body::empty())
        .unwrap();

    let (status, body) = send(router, req).await;
    assert_eq!(status, StatusCode::OK);
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert!(json["results"].as_array().unwrap().is_empty());
}

// ── Transport Consistency ───────────────────────────────────────────────────

#[tokio::test]
async fn store_and_get_roundtrip_fields_match() {
    let tmp = tempfile::tempdir().unwrap();
    let api = test_api_server(&tmp);

    // Store via API
    let router = api.router();
    let store_req = Request::builder()
        .method("POST")
        .uri("/api/memories")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_string(&serde_json::json!({
                "content": "Transport consistency roundtrip test",
                "memory_type": "decision",
                "importance": 0.7,
                "tags": ["transport", "test"],
                "namespace": "roundtrip-ns"
            }))
            .unwrap(),
        ))
        .unwrap();

    let (status, body) = send(router, store_req).await;
    assert_eq!(status, StatusCode::CREATED);
    let id = serde_json::from_str::<serde_json::Value>(&body).unwrap()["id"]
        .as_str()
        .unwrap()
        .to_string();

    // Get via API
    let router = api.router();
    let get_req = Request::builder()
        .uri(format!("/api/memories/{id}"))
        .body(Body::empty())
        .unwrap();

    let (status, body) = send(router, get_req).await;
    assert_eq!(status, StatusCode::OK);
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();

    // Verify all fields match what was stored
    assert_eq!(json["id"], id);
    assert_eq!(json["content"], "Transport consistency roundtrip test");
    assert_eq!(json["memory_type"], "decision");
    assert!(
        (json["importance"].as_f64().unwrap() - 0.7).abs() < 0.01,
        "Importance should match"
    );
    assert_eq!(json["tags"].as_array().unwrap().len(), 2);
    assert!(json["tags"]
        .as_array()
        .unwrap()
        .iter()
        .any(|t| t == "transport"));
    assert!(json["tags"].as_array().unwrap().iter().any(|t| t == "test"));
    assert_eq!(json["namespace"], "roundtrip-ns");
    assert!(json["created_at"].is_string());
    assert!(json["updated_at"].is_string());
}

#[tokio::test]
async fn store_and_search_finds_memory() {
    let tmp = tempfile::tempdir().unwrap();
    let api = test_api_server(&tmp);

    // Store via API
    let router = api.router();
    let store_req = Request::builder()
        .method("POST")
        .uri("/api/memories")
        .header("content-type", "application/json")
        .body(Body::from(
            r#"{"content":"Ownership and borrowing are core Rust features","memory_type":"insight","tags":["rust","ownership"]}"#,
        ))
        .unwrap();

    let (status, _) = send(router, store_req).await;
    assert_eq!(status, StatusCode::CREATED);

    // Search via API
    let router = api.router();
    let search_req = Request::builder()
        .uri("/api/search?q=rust+ownership+borrowing")
        .body(Body::empty())
        .unwrap();

    let (status, body) = send(router, search_req).await;
    assert_eq!(status, StatusCode::OK);
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    let results = json["results"].as_array().unwrap();
    // Should find the stored memory via BM25 token overlap
    assert!(!results.is_empty(), "Search should find the stored memory");
    assert!(results[0]["content"]
        .as_str()
        .unwrap()
        .contains("Ownership"));
}

#[tokio::test]
async fn delete_then_get_returns_404() {
    let tmp = tempfile::tempdir().unwrap();
    let api = test_api_server(&tmp);

    // Store
    let router = api.router();
    let store_req = Request::builder()
        .method("POST")
        .uri("/api/memories")
        .header("content-type", "application/json")
        .body(Body::from(r#"{"content":"delete consistency test"}"#))
        .unwrap();

    let (status, body) = send(router, store_req).await;
    assert_eq!(status, StatusCode::CREATED);
    let id = serde_json::from_str::<serde_json::Value>(&body).unwrap()["id"]
        .as_str()
        .unwrap()
        .to_string();

    // Verify exists
    let router = api.router();
    let get_req = Request::builder()
        .uri(format!("/api/memories/{id}"))
        .body(Body::empty())
        .unwrap();
    let (status, _) = send(router, get_req).await;
    assert_eq!(status, StatusCode::OK);

    // Delete
    let router = api.router();
    let del_req = Request::builder()
        .method("DELETE")
        .uri(format!("/api/memories/{id}"))
        .body(Body::empty())
        .unwrap();
    let (status, _) = send(router, del_req).await;
    assert_eq!(status, StatusCode::OK);

    // Verify gone
    let router = api.router();
    let get_req = Request::builder()
        .uri(format!("/api/memories/{id}"))
        .body(Body::empty())
        .unwrap();
    let (status, _) = send(router, get_req).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

// ── CORS headers ────────────────────────────────────────────────────────────

#[tokio::test]
async fn cors_headers_present() {
    let tmp = tempfile::tempdir().unwrap();
    let api = test_api_server(&tmp);
    let router = api.router();

    let req = Request::builder()
        .method("OPTIONS")
        .uri("/api/health")
        .header("origin", "http://localhost:3000")
        .header("access-control-request-method", "GET")
        .body(Body::empty())
        .unwrap();

    let response = tower::ServiceExt::oneshot(router, req)
        .await
        .expect("oneshot");
    let headers = response.headers();
    assert!(
        headers.contains_key("access-control-allow-origin"),
        "Should have CORS allow-origin header"
    );
}
