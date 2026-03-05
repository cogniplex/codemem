//! Configuration routes.

use crate::api::types::{MessageResponse, ScoringWeightsUpdate};
use crate::api::AppState;
use axum::{extract::State, http::StatusCode, Json};
use std::sync::Arc;

pub async fn get_config(State(_state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let config = codemem_core::CodememConfig::load_or_default();
    Json(serde_json::to_value(config).unwrap_or_default())
}

pub async fn update_config(
    State(_state): State<Arc<AppState>>,
    Json(partial): Json<serde_json::Value>,
) -> Result<Json<MessageResponse>, (StatusCode, Json<MessageResponse>)> {
    // Load current config, merge, and save
    let mut config = codemem_core::CodememConfig::load_or_default();

    // Merge partial updates
    if let Some(scoring) = partial.get("scoring") {
        if let Ok(weights) = serde_json::from_value::<codemem_core::ScoringWeights>(scoring.clone())
        {
            config.scoring = weights;
        }
    }

    let config_path = codemem_core::CodememConfig::default_path();
    match config.save(&config_path) {
        Ok(()) => Ok(Json(MessageResponse {
            message: "Config updated".to_string(),
        })),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(MessageResponse {
                message: e.to_string(),
            }),
        )),
    }
}

pub async fn update_scoring_weights(
    State(_state): State<Arc<AppState>>,
    Json(update): Json<ScoringWeightsUpdate>,
) -> Result<Json<MessageResponse>, (StatusCode, Json<MessageResponse>)> {
    // Load current config, merge scoring weights, and save
    let mut config = codemem_core::CodememConfig::load_or_default();

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
    match config.save(&config_path) {
        Ok(()) => Ok(Json(MessageResponse {
            message: "Scoring weights updated".to_string(),
        })),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(MessageResponse {
                message: e.to_string(),
            }),
        )),
    }
}
