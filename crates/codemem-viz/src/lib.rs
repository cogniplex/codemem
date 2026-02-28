//! codemem-viz: Interactive visualization dashboard for Codemem.
//!
//! Provides REST API endpoints and an embedded HTML frontend for exploring
//! memories, embeddings (projected to 3D via PCA), and the knowledge graph.

mod pca;
mod routes;
mod types;

use axum::{response::Html, routing::get, Router};
use codemem_storage::Storage;
use std::sync::{Arc, Mutex};
use tower_http::cors::CorsLayer;
use tracing::info;

async fn index() -> Html<&'static str> {
    Html(include_str!("frontend.html"))
}

/// Start the Codemem visualization server.
pub async fn serve(
    db_path: std::path::PathBuf,
    port: u16,
    open_browser: bool,
) -> anyhow::Result<()> {
    let storage = Storage::open(&db_path)?;
    let state: routes::AppState = Arc::new(Mutex::new(storage));

    let app = Router::new()
        .route("/", get(index))
        .route("/api/stats", get(routes::api_stats))
        .route("/api/namespaces", get(routes::api_namespaces))
        .route("/api/memories", get(routes::api_memories))
        .route("/api/memories/{id}", get(routes::api_memory_detail))
        .route("/api/vectors", get(routes::api_vectors))
        .route("/api/graph/nodes", get(routes::api_graph_nodes))
        .route("/api/graph/edges", get(routes::api_graph_edges))
        .route(
            "/api/graph/neighbors/{id}",
            get(routes::api_graph_neighbors),
        )
        .route("/api/graph/browse", get(routes::api_graph_browse))
        .route("/api/graph", get(routes::api_graph_d3))
        .route("/api/timeline", get(routes::api_timeline))
        .route("/api/distribution", get(routes::api_distribution))
        .route("/api/search", get(routes::api_search))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let url = format!("http://localhost:{}", port);
    info!("Codemem Viz serving at {}", url);
    println!("Codemem Viz: {}", url);

    if open_browser {
        let _ = open::that(&url);
    }

    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", port)).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
