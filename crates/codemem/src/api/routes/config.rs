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
    State(state): State<Arc<AppState>>,
    Json(update): Json<ScoringWeightsUpdate>,
) -> Result<Json<MessageResponse>, (StatusCode, Json<MessageResponse>)> {
    // Use MCP server's scoring weight update via tool dispatch
    let mut args = serde_json::json!({});
    if let Some(v) = update.vector_similarity {
        args["vector_similarity"] = serde_json::json!(v);
    }
    if let Some(v) = update.graph_strength {
        args["graph_strength"] = serde_json::json!(v);
    }
    if let Some(v) = update.token_overlap {
        args["token_overlap"] = serde_json::json!(v);
    }
    if let Some(v) = update.temporal {
        args["temporal"] = serde_json::json!(v);
    }
    if let Some(v) = update.tag_matching {
        args["tag_matching"] = serde_json::json!(v);
    }
    if let Some(v) = update.importance {
        args["importance"] = serde_json::json!(v);
    }
    if let Some(v) = update.confidence {
        args["confidence"] = serde_json::json!(v);
    }
    if let Some(v) = update.recency {
        args["recency"] = serde_json::json!(v);
    }

    let params = serde_json::json!({
        "name": "set_scoring_weights",
        "arguments": args,
    });

    let response = state.server.handle_request(
        "tools/call",
        Some(&params),
        serde_json::Value::Number(0.into()),
    );

    let message = response
        .result
        .and_then(|r| {
            r.get("content")?
                .get(0)?
                .get("text")?
                .as_str()
                .map(String::from)
        })
        .unwrap_or_else(|| "Scoring weights updated".to_string());

    Ok(Json(MessageResponse { message }))
}
