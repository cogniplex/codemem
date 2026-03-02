//! Session management routes.

use crate::types::{
    EndSessionRequest, IdResponse, MessageResponse, SessionResponse, SessionsQuery,
    StartSessionRequest,
};
use crate::AppState;
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    Json,
};
use std::sync::Arc;

pub async fn list_sessions(
    State(state): State<Arc<AppState>>,
    Query(query): Query<SessionsQuery>,
) -> Json<Vec<SessionResponse>> {
    let storage = state.server.storage();
    let limit = query.limit.unwrap_or(50);

    let sessions = storage
        .list_sessions(query.namespace.as_deref(), limit)
        .unwrap_or_default();

    let responses: Vec<SessionResponse> = sessions
        .into_iter()
        .map(|s| SessionResponse {
            id: s.id,
            namespace: s.namespace,
            started_at: s.started_at.to_rfc3339(),
            ended_at: s.ended_at.map(|t| t.to_rfc3339()),
            memory_count: s.memory_count,
            summary: s.summary,
        })
        .collect();

    Json(responses)
}

pub async fn start_session(
    State(state): State<Arc<AppState>>,
    Json(req): Json<StartSessionRequest>,
) -> Result<(StatusCode, Json<IdResponse>), (StatusCode, Json<MessageResponse>)> {
    let id = uuid::Uuid::new_v4().to_string();
    let storage = state.server.storage();

    match storage.start_session(&id, req.namespace.as_deref()) {
        Ok(()) => Ok((StatusCode::CREATED, Json(IdResponse { id }))),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(MessageResponse {
                message: e.to_string(),
            }),
        )),
    }
}

pub async fn end_session(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Json(req): Json<EndSessionRequest>,
) -> Result<Json<MessageResponse>, (StatusCode, Json<MessageResponse>)> {
    let storage = state.server.storage();

    match storage.end_session(&id, req.summary.as_deref()) {
        Ok(()) => Ok(Json(MessageResponse {
            message: "Session ended".to_string(),
        })),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(MessageResponse {
                message: e.to_string(),
            }),
        )),
    }
}
