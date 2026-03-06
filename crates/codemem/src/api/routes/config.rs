//! Configuration routes.
//!
//! Only scoring weights are hot-reloadable. Other config fields (chunking,
//! enrichment, vector/graph settings) are saved to disk but only take effect
//! after a restart.

use crate::api::types::{MessageResponse, ScoringWeightsUpdate};
use crate::api::AppState;
use axum::{extract::State, http::StatusCode, Json};
use std::sync::Arc;

pub async fn get_config(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    // Read live config from the running engine, not from disk
    let config = state.server.engine.config().clone();
    Json(serde_json::to_value(config).unwrap_or_default())
}

pub async fn update_config(
    State(state): State<Arc<AppState>>,
    Json(partial): Json<serde_json::Value>,
) -> Result<Json<MessageResponse>, (StatusCode, Json<MessageResponse>)> {
    // Start from the engine's live config so we don't lose in-memory state
    let mut config = state.server.engine.config().clone();

    // Merge partial updates
    let new_scoring = partial
        .get("scoring")
        .and_then(|s| serde_json::from_value::<codemem_core::ScoringWeights>(s.clone()).ok());

    if let Some(weights) = &new_scoring {
        config.scoring = weights.clone();
    }

    let config_path = codemem_core::CodememConfig::default_path();
    config.save(&config_path).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(MessageResponse {
                message: e.to_string(),
            }),
        )
    })?;

    // Propagate scoring weights to the running engine
    if let Some(weights) = new_scoring {
        state
            .server
            .engine
            .scoring_weights_mut()
            .map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(MessageResponse {
                        message: format!("Config saved to disk but failed to apply in-memory: {e}"),
                    }),
                )
            })?
            .clone_from(&weights);
    }

    Ok(Json(MessageResponse {
        message: "Config updated".to_string(),
    }))
}

pub async fn update_scoring_weights(
    State(state): State<Arc<AppState>>,
    Json(update): Json<ScoringWeightsUpdate>,
) -> Result<Json<MessageResponse>, (StatusCode, Json<MessageResponse>)> {
    // Start from the engine's live config
    let mut config = state.server.engine.config().clone();

    if let Some(v) = update.vector_similarity {
        config.scoring.vector_similarity = v;
    }
    if let Some(v) = update.graph_strength {
        config.scoring.graph_strength = v;
    }
    if let Some(v) = update.token_overlap {
        config.scoring.token_overlap = v;
    }
    if let Some(v) = update.temporal {
        config.scoring.temporal = v;
    }
    if let Some(v) = update.tag_matching {
        config.scoring.tag_matching = v;
    }
    if let Some(v) = update.importance {
        config.scoring.importance = v;
    }
    if let Some(v) = update.confidence {
        config.scoring.confidence = v;
    }
    if let Some(v) = update.recency {
        config.scoring.recency = v;
    }

    let config_path = codemem_core::CodememConfig::default_path();
    config.save(&config_path).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(MessageResponse {
                message: e.to_string(),
            }),
        )
    })?;

    // Propagate scoring weights to the running engine
    state
        .server
        .engine
        .scoring_weights_mut()
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(MessageResponse {
                    message: format!("Config saved to disk but failed to apply in-memory: {e}"),
                }),
            )
        })?
        .clone_from(&config.scoring);

    Ok(Json(MessageResponse {
        message: "Scoring weights updated".to_string(),
    }))
}
