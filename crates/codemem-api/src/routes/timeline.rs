//! Timeline and distribution routes.

use crate::types::{DistributionResponse, TimelineBucket, TimelineQuery};
use crate::AppState;
use axum::{
    extract::{Query, State},
    Json,
};
use std::collections::HashMap;
use std::sync::Arc;

pub async fn get_timeline(
    State(state): State<Arc<AppState>>,
    Query(query): Query<TimelineQuery>,
) -> Json<Vec<TimelineBucket>> {
    let storage = state.server.storage();
    let memories = storage
        .list_memories_filtered(query.namespace.as_deref(), None)
        .unwrap_or_default();

    // Group by date
    let mut buckets: HashMap<String, HashMap<String, usize>> = HashMap::new();
    for m in &memories {
        let date = m.created_at.format("%Y-%m-%d").to_string();

        // Apply date range filter (inclusive on both ends)
        if let Some(ref from) = query.from {
            if date.as_str() < from.as_str() {
                continue;
            }
        }
        if let Some(ref to) = query.to {
            if date.as_str() > to.as_str() {
                continue;
            }
        }

        let type_name = m.memory_type.to_string();
        *buckets
            .entry(date)
            .or_default()
            .entry(type_name)
            .or_insert(0) += 1;
    }

    let mut result: Vec<TimelineBucket> = buckets
        .into_iter()
        .map(|(date, counts)| {
            let total = counts.values().sum();
            TimelineBucket {
                date,
                counts,
                total,
            }
        })
        .collect();

    result.sort_by(|a, b| a.date.cmp(&b.date));
    Json(result)
}

pub async fn get_distribution(
    State(state): State<Arc<AppState>>,
    Query(query): Query<crate::types::TimelineQuery>,
) -> Json<DistributionResponse> {
    let storage = state.server.storage();
    let memories = storage
        .list_memories_filtered(query.namespace.as_deref(), None)
        .unwrap_or_default();

    let total = memories.len();
    let mut type_counts: HashMap<String, usize> = HashMap::new();
    // 10-bucket histogram for importance [0.0-0.1, 0.1-0.2, ..., 0.9-1.0]
    let mut importance_histogram = vec![0usize; 10];

    for m in &memories {
        *type_counts.entry(m.memory_type.to_string()).or_insert(0) += 1;

        let bucket = ((m.importance * 10.0).floor() as usize).min(9);
        importance_histogram[bucket] += 1;
    }

    Json(DistributionResponse {
        type_counts,
        importance_histogram,
        total,
    })
}
