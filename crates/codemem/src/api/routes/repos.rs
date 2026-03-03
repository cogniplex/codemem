//! Repository management routes.

use crate::api::types::{IdResponse, MessageResponse, RegisterRepoRequest};
use crate::api::AppState;
use axum::{
    extract::{Path, State},
    http::StatusCode,
    Json,
};
use codemem_core::Repository;
use std::sync::Arc;

pub async fn list_repos(
    State(state): State<Arc<AppState>>,
) -> Result<Json<Vec<Repository>>, StatusCode> {
    let storage = state.storage_direct();
    match storage.list_repos() {
        Ok(repos) => Ok(Json(repos)),
        Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
    }
}

pub async fn register_repo(
    State(state): State<Arc<AppState>>,
    Json(req): Json<RegisterRepoRequest>,
) -> Result<(StatusCode, Json<IdResponse>), (StatusCode, Json<MessageResponse>)> {
    let id = uuid::Uuid::new_v4().to_string();
    let now = chrono::Utc::now().to_rfc3339();

    // Derive namespace from path
    let namespace = Some(req.path.clone());

    let repo = Repository {
        id: id.clone(),
        path: req.path,
        name: req.name,
        namespace,
        created_at: now,
        last_indexed_at: None,
        status: "idle".to_string(),
    };

    let storage = state.storage_direct();
    match storage.add_repo(&repo) {
        Ok(()) => Ok((StatusCode::CREATED, Json(IdResponse { id }))),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(MessageResponse {
                message: e.to_string(),
            }),
        )),
    }
}

pub async fn get_repo(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<Repository>, StatusCode> {
    let storage = state.storage_direct();
    match storage.get_repo(&id) {
        Ok(Some(repo)) => Ok(Json(repo)),
        Ok(None) => Err(StatusCode::NOT_FOUND),
        Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
    }
}

pub async fn delete_repo(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<MessageResponse>, StatusCode> {
    let storage = state.storage_direct();
    match storage.remove_repo(&id) {
        Ok(true) => Ok(Json(MessageResponse {
            message: "Deleted".to_string(),
        })),
        Ok(false) => Err(StatusCode::NOT_FOUND),
        Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
    }
}

pub async fn index_repo(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<MessageResponse>, (StatusCode, Json<MessageResponse>)> {
    let storage = state.storage_direct();
    let repo = match storage.get_repo(&id) {
        Ok(Some(r)) => r,
        Ok(None) => {
            return Err((
                StatusCode::NOT_FOUND,
                Json(MessageResponse {
                    message: "Repository not found".to_string(),
                }),
            ))
        }
        Err(e) => {
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(MessageResponse {
                    message: e.to_string(),
                }),
            ))
        }
    };

    // Update status to indexing
    let _ = storage.update_repo_status(&id, "indexing", None);

    // Trigger indexing in background
    let path = repo.path.clone();
    let repo_id = id.clone();
    let indexing_tx = state.indexing_events.clone();
    let storage_for_task = state.storage_direct_arc();

    tokio::spawn(async move {
        let mut indexer = codemem_engine::Indexer::new();
        let root = std::path::Path::new(&path);

        match indexer.index_directory_with_progress(root, Some(&indexing_tx)) {
            Ok(_result) => {
                let now = chrono::Utc::now().to_rfc3339();
                let _ = storage_for_task.update_repo_status(&repo_id, "idle", Some(&now));
            }
            Err(_) => {
                let _ = storage_for_task.update_repo_status(&repo_id, "error", None);
            }
        }
    });

    Ok(Json(MessageResponse {
        message: "Indexing started".to_string(),
    }))
}
