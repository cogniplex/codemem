//! Namespace routes.

use crate::types::{NamespaceItem, NamespaceStatsResponse};
use crate::AppState;
use axum::{
    extract::{Path, State},
    http::StatusCode,
    Json,
};
use std::collections::HashMap;
use std::sync::Arc;

pub async fn list_namespaces(
    State(state): State<Arc<AppState>>,
) -> Json<Vec<NamespaceItem>> {
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
    let storage = state.server.storage();

    let memories = storage
        .list_memories_filtered(Some(&ns), None)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let memory_count = memories.len();
    let mut type_distribution: HashMap<String, usize> = HashMap::new();
    for m in &memories {
        *type_distribution
            .entry(m.memory_type.to_string())
            .or_insert(0) += 1;
    }

    Ok(Json(NamespaceStatsResponse {
        namespace: ns,
        memory_count,
        type_distribution,
    }))
}
