//! codemem-api: REST API and web server for the Codemem control plane.
//!
//! Provides an Axum HTTP server with REST endpoints, SSE event streams,
//! and optional embedded frontend. Can optionally mount the MCP HTTP
//! transport for remote MCP clients.

mod pca;
pub mod routes;
pub mod sse;
pub mod types;

use crate::mcp::McpServer;
use axum::{
    routing::{get, post},
    Router,
};
use codemem_engine::IndexProgress;
use std::sync::Arc;
use tokio::sync::broadcast;
use tower_http::cors::{Any, CorsLayer};

/// Shared application state for all routes.
pub struct AppState {
    /// The shared MCP server instance (holds storage, graph, vector, embeddings).
    pub server: Arc<McpServer>,
    /// Broadcast channel for indexing progress events (SSE).
    pub indexing_events: broadcast::Sender<IndexProgress>,
    /// Broadcast channel for file watcher events (SSE).
    pub watch_events: broadcast::Sender<codemem_engine::watch::WatchEvent>,
    /// Direct storage handle for repo operations (needs Arc<Storage> for async tasks).
    storage_direct: Arc<codemem_storage::Storage>,
}

impl AppState {
    /// Access the direct storage handle for repo operations.
    pub fn storage_direct(&self) -> &codemem_storage::Storage {
        &self.storage_direct
    }

    /// Get an Arc clone of the storage for async tasks.
    pub fn storage_direct_arc(&self) -> Arc<codemem_storage::Storage> {
        Arc::clone(&self.storage_direct)
    }
}

/// The API server that wraps the MCP server and provides REST endpoints.
pub struct ApiServer {
    state: Arc<AppState>,
}

impl ApiServer {
    /// Create a new API server wrapping the given MCP server.
    ///
    /// Opens a separate storage connection for repo operations that need
    /// to be sent across async task boundaries.
    pub fn new(server: Arc<McpServer>, storage: codemem_storage::Storage) -> Self {
        let (indexing_tx, _) = broadcast::channel::<IndexProgress>(256);
        let (watch_tx, _) = broadcast::channel::<codemem_engine::watch::WatchEvent>(256);

        let state = Arc::new(AppState {
            server,
            indexing_events: indexing_tx,
            watch_events: watch_tx,
            storage_direct: Arc::new(storage),
        });

        Self { state }
    }

    /// Build the REST API router (no MCP HTTP transport).
    pub fn router(&self) -> Router {
        let cors = CorsLayer::new()
            .allow_origin(Any)
            .allow_methods(Any)
            .allow_headers(Any);

        let api = Router::new()
            // Stats & Health
            .route("/api/stats", get(routes::stats::get_stats))
            .route("/api/health", get(routes::stats::get_health))
            .route("/api/metrics", get(routes::stats::get_metrics))
            // Memories
            .route(
                "/api/memories",
                get(routes::memories::list_memories).post(routes::memories::store_memory),
            )
            .route(
                "/api/memories/{id}",
                get(routes::memories::get_memory)
                    .put(routes::memories::update_memory)
                    .delete(routes::memories::delete_memory),
            )
            // Search
            .route("/api/search", get(routes::memories::search_memories))
            // Graph
            .route("/api/graph/subgraph", get(routes::graph::get_subgraph))
            .route(
                "/api/graph/neighbors/{id}",
                get(routes::graph::get_neighbors),
            )
            .route(
                "/api/graph/communities",
                get(routes::graph::get_communities),
            )
            .route("/api/graph/pagerank", get(routes::graph::get_pagerank))
            .route(
                "/api/graph/shortest-path",
                get(routes::graph::get_shortest_path),
            )
            .route("/api/graph/impact/{id}", get(routes::graph::get_impact))
            .route("/api/graph/reload", post(routes::graph::reload_graph))
            .route("/api/graph/browse", get(routes::graph::get_graph_browse))
            // Vectors
            .route("/api/vectors", get(routes::vectors::get_vectors))
            // Namespaces
            .route("/api/namespaces", get(routes::namespaces::list_namespaces))
            .route(
                "/api/namespaces/{ns}/stats",
                get(routes::namespaces::get_namespace_stats),
            )
            // Repos
            .route(
                "/api/repos",
                get(routes::repos::list_repos).post(routes::repos::register_repo),
            )
            .route(
                "/api/repos/{id}",
                get(routes::repos::get_repo).delete(routes::repos::delete_repo),
            )
            .route(
                "/api/repos/{id}/index",
                axum::routing::post(routes::repos::index_repo),
            )
            // Sessions
            .route(
                "/api/sessions",
                get(routes::sessions::list_sessions).post(routes::sessions::start_session),
            )
            .route(
                "/api/sessions/{id}/end",
                axum::routing::post(routes::sessions::end_session),
            )
            // Timeline & Distribution
            .route("/api/timeline", get(routes::timeline::get_timeline))
            .route("/api/distribution", get(routes::timeline::get_distribution))
            // Patterns & Consolidation
            .route("/api/patterns", get(routes::patterns::get_patterns))
            .route(
                "/api/patterns/insights",
                get(routes::patterns::get_pattern_insights),
            )
            .route(
                "/api/consolidation/status",
                get(routes::patterns::get_consolidation_status),
            )
            .route(
                "/api/consolidation/{cycle}",
                axum::routing::post(routes::patterns::run_consolidation),
            )
            // Insights
            .route(
                "/api/insights/activity",
                get(routes::insights::get_activity_insights),
            )
            .route(
                "/api/insights/code-health",
                get(routes::insights::get_code_health_insights),
            )
            .route(
                "/api/insights/security",
                get(routes::insights::get_security_insights),
            )
            .route(
                "/api/insights/performance",
                get(routes::insights::get_performance_insights),
            )
            // Config
            .route(
                "/api/config",
                get(routes::config::get_config).put(routes::config::update_config),
            )
            .route(
                "/api/config/scoring",
                axum::routing::put(routes::config::update_scoring_weights),
            )
            // Agent Recipes
            .route("/api/agents/recipes", get(routes::agents::list_recipes))
            .route(
                "/api/agents/run",
                axum::routing::post(routes::agents::run_recipe),
            )
            // SSE Event Streams
            .route("/api/events/indexing", get(sse::indexing_events))
            .route("/api/events/watch", get(sse::watch_events))
            .with_state(Arc::clone(&self.state))
            .layer(cors);

        // Optionally serve embedded frontend
        #[cfg(feature = "ui")]
        let api = api.fallback(serve_embedded_ui);

        api
    }

    /// Build router with MCP HTTP endpoint mounted at /mcp.
    pub fn router_with_mcp(&self) -> Router {
        let mcp = crate::mcp::http::mcp_router(Arc::clone(&self.state.server));
        self.router().merge(mcp)
    }

    /// Start the HTTP server on the given port.
    pub async fn serve(
        &self,
        port: u16,
        include_mcp: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let router = if include_mcp {
            self.router_with_mcp()
        } else {
            self.router()
        };

        let addr = format!("0.0.0.0:{port}");
        tracing::info!("Codemem API server listening on http://localhost:{port}");

        let listener = tokio::net::TcpListener::bind(&addr).await?;
        axum::serve(listener, router).await?;

        Ok(())
    }

    /// Get a reference to the shared state (for external access).
    pub fn state(&self) -> &Arc<AppState> {
        &self.state
    }
}

/// Serve embedded frontend files (when compiled with `ui` feature).
#[cfg(feature = "ui")]
async fn serve_embedded_ui(uri: axum::http::Uri) -> axum::response::Response {
    use axum::http::{header, StatusCode};
    use axum::response::IntoResponse;
    use include_dir::{include_dir, Dir};

    static UI_DIR: Dir<'_> = include_dir!("$CARGO_MANIFEST_DIR/ui-dist");

    let path = uri.path().trim_start_matches('/');
    let path = if path.is_empty() { "index.html" } else { path };

    match UI_DIR.get_file(path) {
        Some(file) => {
            let mime = mime_guess::from_path(path)
                .first_or_octet_stream()
                .to_string();
            (
                StatusCode::OK,
                [(header::CONTENT_TYPE, mime)],
                file.contents().to_vec(),
            )
                .into_response()
        }
        None => {
            // SPA fallback: serve index.html for all unknown routes
            match UI_DIR.get_file("index.html") {
                Some(file) => (
                    StatusCode::OK,
                    [(header::CONTENT_TYPE, "text/html".to_string())],
                    file.contents().to_vec(),
                )
                    .into_response(),
                None => (StatusCode::NOT_FOUND, "Not found").into_response(),
            }
        }
    }
}
