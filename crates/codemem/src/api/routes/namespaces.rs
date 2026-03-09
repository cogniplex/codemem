//! Namespace routes.

use crate::api::types::{MessageResponse, NamespaceItem, NamespaceStatsResponse};
use crate::api::AppState;
use axum::{
    extract::{Path, State},
    http::StatusCode,
    Json,
};
use std::sync::Arc;

pub async fn list_namespaces(State(state): State<Arc<AppState>>) -> Json<Vec<NamespaceItem>> {
    let storage = state.server.storage();
    let namespaces = storage.list_namespaces().unwrap_or_default();

    let items: Vec<NamespaceItem> = namespaces
        .into_iter()
        .map(|name| {
            let count = storage
                .list_memory_ids_for_namespace(&name)
                .map(|ids| ids.len())
                .unwrap_or(0);
            NamespaceItem {
                name,
                memory_count: count,
            }
        })
        .collect();

    Json(items)
}

pub async fn get_namespace_stats(
    State(state): State<Arc<AppState>>,
    Path(ns): Path<String>,
) -> Result<Json<NamespaceStatsResponse>, StatusCode> {
    let stats = state
        .server
        .engine
        .namespace_stats(&ns)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    Ok(Json(NamespaceStatsResponse {
        namespace: stats.namespace,
        memory_count: stats.count,
        avg_importance: stats.avg_importance,
        avg_confidence: stats.avg_confidence,
        type_distribution: stats.type_distribution,
        tag_frequency: stats.tag_frequency,
        oldest: stats.oldest.map(|d| d.to_rfc3339()),
        newest: stats.newest.map(|d| d.to_rfc3339()),
    }))
}

pub async fn delete_namespace(
    State(state): State<Arc<AppState>>,
    Path(ns): Path<String>,
) -> Result<Json<MessageResponse>, StatusCode> {
    let deleted = state
        .server
        .engine
        .delete_namespace(&ns)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    Ok(Json(MessageResponse {
        message: format!("Deleted {deleted} memories from namespace '{ns}'"),
    }))
}
