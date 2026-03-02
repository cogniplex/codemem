//! Stats and health routes.

use crate::types::{ComponentHealth, HealthResponse, MetricsResponse, StatsResponse};
use crate::AppState;
use axum::{extract::State, Json};
use codemem_core::VectorBackend;
use std::collections::HashMap;
use std::sync::Arc;

pub async fn get_stats(State(state): State<Arc<AppState>>) -> Json<StatsResponse> {
    let storage = state.server.storage();
    let memory_count = storage.memory_count().unwrap_or(0);
    let embedding_count = storage
        .list_all_embeddings()
        .map(|e| e.len())
        .unwrap_or(0);

    let (node_count, edge_count) = {
        let graph = state.server.graph().lock().unwrap_or_else(|e| e.into_inner());
        (graph.node_count(), graph.edge_count())
    };

    let session_count = storage.session_count(None).unwrap_or(0);
    let namespace_count = storage.list_namespaces().map(|n| n.len()).unwrap_or(0);

    Json(StatsResponse {
        memory_count,
        embedding_count,
        node_count,
        edge_count,
        session_count,
        namespace_count,
    })
}

pub async fn get_health(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    let storage_health = match state.server.storage().stats() {
        Ok(_) => ComponentHealth {
            status: "ok".to_string(),
            detail: None,
        },
        Err(e) => ComponentHealth {
            status: "error".to_string(),
            detail: Some(e.to_string()),
        },
    };

    let vector_health = match state.server.vector().lock() {
        Ok(v) => {
            let stats = v.stats();
            ComponentHealth {
                status: "ok".to_string(),
                detail: Some(format!("{} vectors", stats.count)),
            }
        }
        Err(e) => ComponentHealth {
            status: "error".to_string(),
            detail: Some(e.to_string()),
        },
    };

    let graph_health = match state.server.graph().lock() {
        Ok(g) => ComponentHealth {
            status: "ok".to_string(),
            detail: Some(format!("{} nodes, {} edges", g.node_count(), g.edge_count())),
        },
        Err(e) => ComponentHealth {
            status: "error".to_string(),
            detail: Some(e.to_string()),
        },
    };

    let embeddings_health = match state.server.embeddings() {
        Some(emb) => match emb.lock() {
            Ok(_) => ComponentHealth {
                status: "ok".to_string(),
                detail: None,
            },
            Err(e) => ComponentHealth {
                status: "error".to_string(),
                detail: Some(e.to_string()),
            },
        },
        None => ComponentHealth {
            status: "unavailable".to_string(),
            detail: Some("No embedding provider configured".to_string()),
        },
    };

    Json(HealthResponse {
        storage: storage_health,
        vector: vector_health,
        graph: graph_health,
        embeddings: embeddings_health,
    })
}

pub async fn get_metrics(State(state): State<Arc<AppState>>) -> Json<MetricsResponse> {
    let metrics = state.server.metrics_collector();
    let snapshot = metrics.snapshot();

    let mut latency_percentiles = HashMap::new();
    for (name, stats) in &snapshot.latencies {
        latency_percentiles.insert(format!("{name}_p50"), stats.p50_ms);
        latency_percentiles.insert(format!("{name}_p95"), stats.p95_ms);
        latency_percentiles.insert(format!("{name}_p99"), stats.p99_ms);
    }

    Json(MetricsResponse {
        tool_calls_total: snapshot
            .counters
            .get("tool_calls_total")
            .copied()
            .unwrap_or(0),
        latency_percentiles,
    })
}
