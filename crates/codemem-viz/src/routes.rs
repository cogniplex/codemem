use axum::{
    extract::{Path, Query, State},
    response::Json,
};
use codemem_core::{GraphNode, MemoryNode, NodeKind, StorageBackend};
use codemem_storage::Storage;
use ndarray::{Array1, Array2};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex};

use crate::pca::power_iteration_top_k;
use crate::types::*;

pub type AppState = Arc<Mutex<Storage>>;
type ApiResult<T> = Result<Json<T>, (axum::http::StatusCode, String)>;

fn lock_err(e: impl std::fmt::Display) -> (axum::http::StatusCode, String) {
    (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
}

fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        let mut end = max_len;
        while end > 0 && !s.is_char_boundary(end) {
            end -= 1;
        }
        format!("{}...", &s[..end])
    }
}

// -- Stats & Namespaces -----------------------------------------------------------

pub async fn api_stats(State(storage): State<AppState>) -> ApiResult<serde_json::Value> {
    let storage = storage.lock().map_err(lock_err)?;
    let stats = storage.stats().map_err(lock_err)?;
    Ok(Json(serde_json::to_value(stats).unwrap()))
}

pub async fn api_namespaces(State(storage): State<AppState>) -> ApiResult<Vec<String>> {
    let storage = storage.lock().map_err(lock_err)?;
    let namespaces = storage.list_namespaces().map_err(lock_err)?;
    Ok(Json(namespaces))
}

// -- Memories ---------------------------------------------------------------------

pub async fn api_memories(
    State(storage): State<AppState>,
    Query(query): Query<MemoryQuery>,
) -> ApiResult<Vec<MemoryListItem>> {
    let storage = storage.lock().map_err(lock_err)?;
    let memories = storage
        .list_memories_filtered(query.namespace.as_deref(), query.memory_type.as_deref())
        .map_err(lock_err)?;

    let items = memories
        .into_iter()
        .map(|m| MemoryListItem {
            id: m.id,
            content: truncate_str(&m.content, 120),
            memory_type: m.memory_type.to_string(),
            importance: m.importance,
            tags: m.tags,
            namespace: m.namespace,
        })
        .collect();

    Ok(Json(items))
}

pub async fn api_memory_detail(
    State(storage): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<serde_json::Value> {
    let storage = storage.lock().map_err(lock_err)?;

    // Try memory first
    let memory = storage.get_memory(&id).map_err(lock_err)?;
    if let Some(m) = memory {
        return Ok(Json(serde_json::to_value(m).unwrap()));
    }

    // Fall back to graph node -- prevents 404 when clicking graph-only nodes
    let node = storage.get_graph_node(&id).map_err(lock_err)?;
    match node {
        Some(n) => Ok(Json(serde_json::to_value(n).unwrap())),
        None => Err((
            axum::http::StatusCode::NOT_FOUND,
            format!("Memory or node {} not found", id),
        )),
    }
}

// -- Vectors (PCA) ----------------------------------------------------------------

pub async fn api_vectors(
    State(storage): State<AppState>,
    Query(query): Query<VectorQuery>,
) -> ApiResult<Vec<VectorPoint>> {
    let storage = storage.lock().map_err(lock_err)?;
    let all_embeddings = storage.list_all_embeddings().map_err(lock_err)?;

    if all_embeddings.is_empty() {
        return Ok(Json(vec![]));
    }

    // Collect all embedding IDs for batch lookup
    let ids: Vec<&str> = all_embeddings.iter().map(|(id, _)| id.as_str()).collect();
    let memories = storage.get_memories_batch(&ids).map_err(lock_err)?;
    let mem_map: HashMap<String, &MemoryNode> =
        memories.iter().map(|m| (m.id.clone(), m)).collect();

    // Build graph node lookup
    let all_nodes = storage.all_graph_nodes().map_err(lock_err)?;
    let node_map: HashMap<String, &GraphNode> =
        all_nodes.iter().map(|n| (n.id.clone(), n)).collect();

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
            let mem = mem_map.get(&id);
            let node = node_map.get(&id);
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
        return Ok(Json(vec![]));
    }

    let n = rows.len();
    let dim = rows[0].embedding.len();

    let mut mean = Array1::<f64>::zeros(dim);
    for row in &rows {
        for (j, &v) in row.embedding.iter().enumerate() {
            mean[j] += v as f64;
        }
    }
    mean /= n as f64;

    let mut centered = Array2::<f64>::zeros((n, dim));
    for (i, row) in rows.iter().enumerate() {
        for (j, &v) in row.embedding.iter().enumerate() {
            centered[[i, j]] = v as f64 - mean[j];
        }
    }

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

    Ok(Json(points))
}

// -- Graph ------------------------------------------------------------------------

pub async fn api_graph_nodes(
    State(storage): State<AppState>,
    Query(query): Query<GraphNodeQuery>,
) -> ApiResult<Vec<codemem_core::GraphNode>> {
    let storage = storage.lock().map_err(lock_err)?;
    let mut nodes = storage.all_graph_nodes().map_err(lock_err)?;

    if let Some(ref ns) = query.namespace {
        nodes.retain(|n| n.namespace.as_deref() == Some(ns.as_str()));
    }
    if let Some(ref kind_str) = query.kind {
        if let Ok(kind) = kind_str.parse::<NodeKind>() {
            nodes.retain(|n| n.kind == kind);
        }
    }

    Ok(Json(nodes))
}

pub async fn api_graph_edges(
    State(storage): State<AppState>,
    Query(query): Query<EdgeQuery>,
) -> ApiResult<Vec<EdgeResponse>> {
    let storage = storage.lock().map_err(lock_err)?;

    let edges = if let Some(ref ns) = query.namespace {
        storage.graph_edges_for_namespace(ns)
    } else {
        storage.all_graph_edges()
    }
    .map_err(lock_err)?;

    let response: Vec<EdgeResponse> = edges
        .into_iter()
        .map(|e| EdgeResponse {
            id: e.id,
            src: e.src,
            dst: e.dst,
            relationship: e.relationship.to_string(),
            weight: e.weight,
        })
        .collect();

    Ok(Json(response))
}

pub async fn api_graph_neighbors(
    State(storage): State<AppState>,
    Path(id): Path<String>,
    Query(query): Query<NeighborQuery>,
) -> ApiResult<NeighborResponse> {
    let depth = query.depth.unwrap_or(2).min(5);
    let storage = storage.lock().map_err(lock_err)?;

    let mut visited: HashSet<String> = HashSet::new();
    let mut queue: VecDeque<(String, usize)> = VecDeque::new();
    let mut all_nodes = Vec::new();
    let mut all_edges = Vec::new();
    let mut seen_edges: HashSet<String> = HashSet::new();

    visited.insert(id.clone());
    queue.push_back((id.clone(), 0));

    if let Ok(Some(start_node)) = storage.get_graph_node(&id) {
        all_nodes.push(start_node);
    }

    while let Some((node_id, current_depth)) = queue.pop_front() {
        if current_depth >= depth {
            continue;
        }

        let edges = storage.get_edges_for_node(&node_id).map_err(lock_err)?;

        for edge in edges {
            let neighbor_id = if edge.src == node_id {
                edge.dst.clone()
            } else {
                edge.src.clone()
            };
            let edge_id = edge.id.clone();

            if seen_edges.insert(edge_id) {
                all_edges.push(edge);
            }

            if visited.insert(neighbor_id.clone()) {
                if let Ok(Some(neighbor_node)) = storage.get_graph_node(&neighbor_id) {
                    all_nodes.push(neighbor_node);
                }
                queue.push_back((neighbor_id, current_depth + 1));
            }
        }
    }

    Ok(Json(NeighborResponse {
        nodes: all_nodes,
        edges: all_edges,
    }))
}

// -- Search -----------------------------------------------------------------------

pub async fn api_search(
    State(storage): State<AppState>,
    Query(query): Query<SearchQuery>,
) -> ApiResult<Vec<MemoryListItem>> {
    let q = match query.q {
        Some(ref q) if !q.is_empty() => q.clone(),
        _ => return Ok(Json(vec![])),
    };

    let storage = storage.lock().map_err(lock_err)?;
    let q_lower = q.to_lowercase();

    // Search memories
    let all_memories = storage
        .list_memories_filtered(query.namespace.as_deref(), None)
        .map_err(lock_err)?;

    let mut results: Vec<MemoryListItem> = all_memories
        .into_iter()
        .filter(|m| m.content.to_lowercase().contains(&q_lower))
        .take(100)
        .map(|m| MemoryListItem {
            id: m.id,
            content: truncate_str(&m.content, 120),
            memory_type: m.memory_type.to_string(),
            importance: m.importance,
            tags: m.tags,
            namespace: m.namespace,
        })
        .collect();

    // Search graph nodes
    let gn_results = storage
        .search_graph_nodes(&q, query.namespace.as_deref(), 100)
        .map_err(lock_err)?;

    let seen_ids: HashSet<String> = results.iter().map(|r| r.id.clone()).collect();

    let gn_items: Vec<MemoryListItem> = gn_results
        .into_iter()
        .filter(|n| !seen_ids.contains(&n.id))
        .map(|n| MemoryListItem {
            id: n.id,
            content: truncate_str(&n.label, 120),
            memory_type: n.kind.to_string(),
            importance: 0.5,
            tags: vec![],
            namespace: n.namespace,
        })
        .collect();

    results.extend(gn_items);
    Ok(Json(results))
}

// -- Browse (paginated graph nodes) -----------------------------------------------

pub async fn api_graph_browse(
    State(storage): State<AppState>,
    Query(query): Query<BrowseQuery>,
) -> ApiResult<BrowseResponse> {
    let storage = storage.lock().map_err(lock_err)?;

    let limit = query.limit.unwrap_or(50).min(200);
    let offset = query.offset.unwrap_or(0);

    let mut all_nodes = storage.all_graph_nodes().map_err(lock_err)?;
    let all_edges = storage.all_graph_edges().map_err(lock_err)?;

    // Apply filters
    if let Some(ref ns) = query.namespace {
        all_nodes.retain(|n| n.namespace.as_deref() == Some(ns.as_str()));
    }
    if let Some(ref kind) = query.kind {
        all_nodes.retain(|n| n.kind.to_string() == *kind);
    }
    if let Some(ref q) = query.q {
        if !q.is_empty() {
            let q_lower = q.to_lowercase();
            all_nodes.retain(|n| n.label.to_lowercase().contains(&q_lower));
        }
    }

    // Sort by centrality descending
    all_nodes.sort_by(|a, b| {
        b.centrality
            .partial_cmp(&a.centrality)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let total = all_nodes.len();

    // Kind counts (only filter by namespace + search, not kind)
    let mut kind_filtered_nodes = storage.all_graph_nodes().map_err(lock_err)?;
    if let Some(ref ns) = query.namespace {
        kind_filtered_nodes.retain(|n| n.namespace.as_deref() == Some(ns.as_str()));
    }
    if let Some(ref q) = query.q {
        if !q.is_empty() {
            let q_lower = q.to_lowercase();
            kind_filtered_nodes.retain(|n| n.label.to_lowercase().contains(&q_lower));
        }
    }
    let mut kinds: HashMap<String, usize> = HashMap::new();
    for node in &kind_filtered_nodes {
        *kinds.entry(node.kind.to_string()).or_insert(0) += 1;
    }

    // Compute degree for each node
    let mut degree_map: HashMap<String, usize> = HashMap::new();
    for edge in &all_edges {
        *degree_map.entry(edge.src.clone()).or_insert(0) += 1;
        *degree_map.entry(edge.dst.clone()).or_insert(0) += 1;
    }

    // Paginate
    let nodes: Vec<BrowseNodeItem> = all_nodes
        .iter()
        .skip(offset)
        .take(limit)
        .map(|n| BrowseNodeItem {
            id: n.id.clone(),
            kind: n.kind.to_string(),
            label: n.label.clone(),
            centrality: n.centrality,
            namespace: n.namespace.clone(),
            degree: *degree_map.get(&n.id).unwrap_or(&0),
        })
        .collect();

    // Edge count
    let edge_count = if query.namespace.is_some() {
        let ns_node_ids: HashSet<&str> =
            kind_filtered_nodes.iter().map(|n| n.id.as_str()).collect();
        all_edges
            .iter()
            .filter(|e| {
                ns_node_ids.contains(e.src.as_str()) && ns_node_ids.contains(e.dst.as_str())
            })
            .count()
    } else {
        all_edges.len()
    };

    Ok(Json(BrowseResponse {
        nodes,
        total,
        kinds,
        edge_count,
    }))
}
