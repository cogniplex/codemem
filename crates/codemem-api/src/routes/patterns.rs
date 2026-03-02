//! Pattern detection and consolidation routes.

use crate::types::{
    ConsolidationCycleStatus, ConsolidationStatusResponse, MessageResponse, PatternResponse,
};
use crate::AppState;
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    Json,
};
use std::sync::Arc;

#[derive(Debug, serde::Deserialize)]
pub struct PatternsQuery {
    pub namespace: Option<String>,
}

pub async fn get_patterns(
    State(state): State<Arc<AppState>>,
    Query(query): Query<PatternsQuery>,
) -> Json<Vec<PatternResponse>> {
    // Use MCP server to detect patterns
    let mut args = serde_json::json!({});
    if let Some(ref ns) = query.namespace {
        args["namespace"] = serde_json::Value::String(ns.clone());
    }

    let params = serde_json::json!({
        "name": "detect_patterns",
        "arguments": args,
    });

    let response = state.server.handle_request(
        "tools/call",
        Some(&params),
        serde_json::Value::Number(0.into()),
    );

    // Parse patterns from tool result
    let patterns = if let Some(result) = response.result {
        parse_patterns_result(&result)
    } else {
        Vec::new()
    };

    Json(patterns)
}

pub async fn get_pattern_insights(
    State(state): State<Arc<AppState>>,
    Query(query): Query<PatternsQuery>,
) -> Json<MessageResponse> {
    let mut args = serde_json::json!({});
    if let Some(ref ns) = query.namespace {
        args["namespace"] = serde_json::Value::String(ns.clone());
    }

    let params = serde_json::json!({
        "name": "pattern_insights",
        "arguments": args,
    });

    let response = state.server.handle_request(
        "tools/call",
        Some(&params),
        serde_json::Value::Number(0.into()),
    );

    let message = response
        .result
        .and_then(|r| r.get("content")?.get(0)?.get("text")?.as_str().map(String::from))
        .unwrap_or_else(|| "No patterns detected".to_string());

    Json(MessageResponse { message })
}

pub async fn get_consolidation_status(
    State(state): State<Arc<AppState>>,
) -> Json<ConsolidationStatusResponse> {
    let storage = state.server.storage();
    let logs = storage.last_consolidation_runs().unwrap_or_default();

    // Build a map of cycle -> status from logs
    let mut cycle_map: std::collections::HashMap<String, ConsolidationCycleStatus> = logs
        .into_iter()
        .map(|log| {
            (
                log.cycle_type.clone(),
                ConsolidationCycleStatus {
                    cycle: log.cycle_type,
                    last_run: Some(
                        chrono::DateTime::from_timestamp(log.run_at, 0)
                            .map(|dt| dt.to_rfc3339())
                            .unwrap_or_else(|| log.run_at.to_string()),
                    ),
                    affected_count: log.affected_count,
                },
            )
        })
        .collect();

    // Always include all 5 cycles with defaults for those never run
    let all_cycles = ["decay", "creative", "cluster", "summarize", "forget"];
    let cycles: Vec<ConsolidationCycleStatus> = all_cycles
        .into_iter()
        .map(|name| {
            cycle_map.remove(name).unwrap_or(ConsolidationCycleStatus {
                cycle: name.to_string(),
                last_run: None,
                affected_count: 0,
            })
        })
        .collect();

    Json(ConsolidationStatusResponse { cycles })
}

pub async fn run_consolidation(
    State(state): State<Arc<AppState>>,
    Path(cycle): Path<String>,
) -> Result<Json<MessageResponse>, (StatusCode, Json<MessageResponse>)> {
    let tool_name = match cycle.as_str() {
        "decay" => "consolidate_decay",
        "creative" => "consolidate_creative",
        "cluster" => "consolidate_cluster",
        "forget" => "consolidate_forget",
        "summarize" => "consolidate_summarize",
        _ => {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(MessageResponse {
                    message: format!("Unknown cycle: {cycle}"),
                }),
            ))
        }
    };

    let params = serde_json::json!({
        "name": tool_name,
        "arguments": {},
    });

    let response = state.server.handle_request(
        "tools/call",
        Some(&params),
        serde_json::Value::Number(0.into()),
    );

    let message = response
        .result
        .and_then(|r| r.get("content")?.get(0)?.get("text")?.as_str().map(String::from))
        .unwrap_or_else(|| "Consolidation completed".to_string());

    Ok(Json(MessageResponse { message }))
}

fn parse_patterns_result(result: &serde_json::Value) -> Vec<PatternResponse> {
    let text = result
        .get("content")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("text"))
        .and_then(|t| t.as_str())
        .unwrap_or("");

    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(text) {
        if let Some(arr) = parsed.as_array() {
            return arr
                .iter()
                .filter_map(|item| {
                    Some(PatternResponse {
                        pattern_type: item
                            .get("pattern_type")
                            .and_then(|t| t.as_str())
                            .unwrap_or("unknown")
                            .to_string(),
                        description: item
                            .get("description")
                            .and_then(|d| d.as_str())
                            .unwrap_or("")
                            .to_string(),
                        frequency: item
                            .get("frequency")
                            .and_then(|f| f.as_u64())
                            .unwrap_or(0) as usize,
                        confidence: item
                            .get("confidence")
                            .and_then(|c| c.as_f64())
                            .unwrap_or(0.0),
                        related_memories: item
                            .get("related_memories")
                            .and_then(|r| r.as_array())
                            .map(|arr| {
                                arr.iter()
                                    .filter_map(|v| v.as_str().map(String::from))
                                    .collect()
                            })
                            .unwrap_or_default(),
                    })
                })
                .collect();
        }
    }

    Vec::new()
}
