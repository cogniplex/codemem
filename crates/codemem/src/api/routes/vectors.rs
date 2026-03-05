//! Vector space visualization (PCA projection) routes.

use crate::api::pca::power_iteration_top_k;
use crate::api::types::{VectorPoint, VectorQuery};
use crate::api::AppState;
use axum::{
    extract::{Query, State},
    Json,
};
use ndarray::{Array1, Array2};
use std::collections::HashMap;
use std::sync::Arc;

use crate::truncate_str;

pub async fn get_vectors(
    State(state): State<Arc<AppState>>,
    Query(query): Query<VectorQuery>,
) -> Json<Vec<VectorPoint>> {
    let storage = state.server.storage();

    let all_embeddings = match storage.list_all_embeddings() {
        Ok(e) => e,
        Err(_) => return Json(vec![]),
    };

    if all_embeddings.is_empty() {
        return Json(vec![]);
    }

    // Batch lookup memories
    let ids: Vec<&str> = all_embeddings.iter().map(|(id, _)| id.as_str()).collect();
    let memories = storage.get_memories_batch(&ids).unwrap_or_default();
    let mem_map: HashMap<&str, _> = memories.iter().map(|m| (m.id.as_str(), m)).collect();

    // Graph node lookup
    let all_nodes = storage.all_graph_nodes().unwrap_or_default();
    let node_map: HashMap<&str, _> = all_nodes.iter().map(|n| (n.id.as_str(), n)).collect();

    struct EmbeddingRow {
        memory_id: String,
        embedding: Vec<f32>,
        memory_type: String,
        importance: f64,
        namespace: Option<String>,
        content: String,
    }

    let rows: Vec<EmbeddingRow> = all_embeddings
        .into_iter()
        .map(|(id, embedding)| {
            let mem = mem_map.get(id.as_str());
            let node = node_map.get(id.as_str());
            EmbeddingRow {
                memory_id: id,
                embedding,
                memory_type: mem
                    .map(|m| m.memory_type.to_string())
                    .or_else(|| node.map(|n| n.kind.to_string()))
                    .unwrap_or_else(|| "context".to_string()),
                importance: mem.map(|m| m.importance).unwrap_or(0.5),
                namespace: mem
                    .and_then(|m| m.namespace.clone())
                    .or_else(|| node.and_then(|n| n.namespace.clone())),
                content: mem
                    .map(|m| m.content.clone())
                    .or_else(|| node.map(|n| n.label.clone()))
                    .unwrap_or_default(),
            }
        })
        .collect();

    // Apply namespace filter
    let rows: Vec<EmbeddingRow> = if let Some(ref ns) = query.namespace {
        rows.into_iter()
            .filter(|r| r.namespace.as_deref() == Some(ns.as_str()))
            .collect()
    } else {
        rows
    };

    if rows.is_empty() {
        return Json(vec![]);
    }

    let n = rows.len();
    let dim = rows[0].embedding.len();

    // Compute mean
    let mut mean = Array1::<f64>::zeros(dim);
    for row in &rows {
        for (j, &v) in row.embedding.iter().enumerate() {
            mean[j] += v as f64;
        }
    }
    mean /= n as f64;

    // Center the data
    let mut centered = Array2::<f64>::zeros((n, dim));
    for (i, row) in rows.iter().enumerate() {
        for (j, &v) in row.embedding.iter().enumerate() {
            centered[[i, j]] = v as f64 - mean[j];
        }
    }

    // Covariance matrix + PCA
    let divisor = if n > 1 { (n - 1) as f64 } else { 1.0 };
    let cov = centered.t().dot(&centered) / divisor;
    let components = power_iteration_top_k(&cov, 3, 10);

    let points: Vec<VectorPoint> = rows
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let c = centered.row(i);
            let x = if !components.is_empty() {
                c.dot(&components[0])
            } else {
                0.0
            };
            let y = if components.len() > 1 {
                c.dot(&components[1])
            } else {
                0.0
            };
            let z = if components.len() > 2 {
                c.dot(&components[2])
            } else {
                0.0
            };
            VectorPoint {
                id: row.memory_id.clone(),
                x,
                y,
                z,
                memory_type: row.memory_type.clone(),
                importance: row.importance,
                namespace: row.namespace.clone(),
                label: truncate_str(&row.content, 80),
            }
        })
        .collect();

    Json(points)
}
