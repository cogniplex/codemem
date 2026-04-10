//! Graph exploration routes.

use crate::api::types::{
    BrowseNodeItem, BrowseQuery, BrowseResponse, CommunitiesQuery, CommunitiesResponse,
    GraphEdgeResponse, GraphNodeResponse, NeighborsQuery, PagerankEntry, PagerankQuery,
    PagerankResponse, ShortestPathQuery, SubgraphQuery, SubgraphResponse,
};
use crate::api::AppState;
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    Json,
};
use codemem_core::{GraphBackend, NodeKind};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Collect edges whose both endpoints are in `node_ids`, deduplicating by edge ID.
fn collect_edges_between(
    graph: &dyn GraphBackend,
    nodes: &[codemem_core::GraphNode],
    node_ids: &std::collections::HashSet<&str>,
) -> Vec<GraphEdgeResponse> {
    let mut edges = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for node in nodes {
        let node_edges = match graph.get_edges(&node.id) {
            Ok(e) => e,
            Err(_) => continue,
        };
        for edge in node_edges {
            if !node_ids.contains(edge.src.as_str())
                || !node_ids.contains(edge.dst.as_str())
                || !seen.insert(edge.id.clone())
            {
                continue;
            }
            edges.push(GraphEdgeResponse {
                id: edge.id,
                src: edge.src,
                dst: edge.dst,
                relationship: edge.relationship.to_string(),
                weight: edge.weight,
            });
        }
    }
    edges
}

/// Collect a BFS subgraph (nodes + edges between them) from a start node.
/// Shared helper for `get_neighbors` and `get_impact`.
fn collect_subgraph(
    graph: &dyn GraphBackend,
    start_id: &str,
    depth: usize,
) -> Result<SubgraphResponse, codemem_core::CodememError> {
    let reachable = graph.bfs(start_id, depth)?;
    let node_ids: std::collections::HashSet<&str> =
        reachable.iter().map(|n| n.id.as_str()).collect();

    let edges = collect_edges_between(graph, &reachable, &node_ids);

    let nodes = reachable
        .into_iter()
        .map(|n| GraphNodeResponse {
            id: n.id,
            kind: n.kind.to_string(),
            label: n.label,
            centrality: n.centrality,
            memory_id: n.memory_id,
            namespace: n.namespace,
            payload: n.payload,
        })
        .collect();

    Ok(SubgraphResponse { nodes, edges })
}

/// Build a top-N subgraph response with optional centrality and kind filters.
fn build_subgraph_response(
    graph: &dyn GraphBackend,
    max_nodes: usize,
    namespace: Option<&str>,
    kinds: Option<&[NodeKind]>,
    min_centrality: Option<f64>,
) -> SubgraphResponse {
    let (nodes, edges) = graph.subgraph_top_n(max_nodes, namespace, kinds);

    let node_responses: Vec<GraphNodeResponse> = nodes
        .into_iter()
        .filter(|n| min_centrality.is_none_or(|min_c| n.centrality >= min_c))
        .map(|n| GraphNodeResponse {
            id: n.id,
            kind: n.kind.to_string(),
            label: n.label,
            centrality: n.centrality,
            memory_id: n.memory_id,
            namespace: n.namespace,
            payload: n.payload,
        })
        .collect();

    // Only include edges whose both endpoints survived the min_centrality filter.
    let node_ids: std::collections::HashSet<&str> =
        node_responses.iter().map(|n| n.id.as_str()).collect();
    let edge_responses: Vec<GraphEdgeResponse> = edges
        .into_iter()
        .filter(|e| node_ids.contains(e.src.as_str()) && node_ids.contains(e.dst.as_str()))
        .map(|e| GraphEdgeResponse {
            id: e.id,
            src: e.src,
            dst: e.dst,
            relationship: e.relationship.to_string(),
            weight: e.weight,
        })
        .collect();

    SubgraphResponse {
        nodes: node_responses,
        edges: edge_responses,
    }
}

pub async fn get_subgraph(
    State(state): State<Arc<AppState>>,
    Query(query): Query<SubgraphQuery>,
) -> Json<SubgraphResponse> {
    let max_nodes = query.max_nodes.unwrap_or(1000);

    let kinds: Option<Vec<NodeKind>> = query.kinds.as_deref().map(|k| {
        k.split(',')
            .filter_map(|s| s.trim().parse::<NodeKind>().ok())
            .collect()
    });

    let min_centrality = query.min_centrality;
    let namespace = query.namespace.clone();

    let result = state.server.engine.with_graph(|graph| {
        build_subgraph_response(
            graph,
            max_nodes,
            namespace.as_deref(),
            kinds.as_deref(),
            min_centrality,
        )
    });

    Json(result.unwrap_or_else(|_| SubgraphResponse {
        nodes: vec![],
        edges: vec![],
    }))
}

pub async fn get_neighbors(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Query(query): Query<NeighborsQuery>,
) -> Result<Json<SubgraphResponse>, StatusCode> {
    let depth = query.depth.unwrap_or(1);

    state
        .server
        .engine
        .with_graph(|graph| collect_subgraph(graph, &id, depth))
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .map(Json)
        .map_err(|_| StatusCode::NOT_FOUND)
}

pub async fn get_communities(
    State(state): State<Arc<AppState>>,
    Query(query): Query<CommunitiesQuery>,
) -> Json<CommunitiesResponse> {
    let resolution = query.resolution.unwrap_or(1.0);
    let namespace = query.namespace.clone();

    let result = state.server.engine.with_graph(|graph| {
        let mut assignment = graph.louvain_with_assignment(resolution);

        if let Some(ref ns) = namespace {
            let ns_node_ids: std::collections::HashSet<String> = assignment
                .keys()
                .filter(|id| {
                    graph
                        .get_node(id)
                        .ok()
                        .flatten()
                        .and_then(|n| n.namespace)
                        .as_deref()
                        == Some(ns.as_str())
                })
                .cloned()
                .collect();
            assignment.retain(|id, _| ns_node_ids.contains(id));
        }

        let num_communities = assignment
            .values()
            .collect::<std::collections::HashSet<_>>()
            .len();

        CommunitiesResponse {
            communities: assignment,
            num_communities,
        }
    });

    Json(result.unwrap_or_else(|_| CommunitiesResponse {
        communities: HashMap::new(),
        num_communities: 0,
    }))
}

pub async fn get_pagerank(
    State(state): State<Arc<AppState>>,
    Query(query): Query<PagerankQuery>,
) -> Json<PagerankResponse> {
    let top = query.top.unwrap_or(20);
    let namespace = query.namespace.clone();

    let result = state.server.engine.with_graph(|graph| {
        let scores = graph.pagerank(
            codemem_core::PAGERANK_DAMPING_DEFAULT,
            codemem_core::PAGERANK_ITERATIONS_DEFAULT,
            codemem_core::PAGERANK_TOLERANCE_DEFAULT,
        );

        let mut entries: Vec<PagerankEntry> = scores
            .into_iter()
            .filter(|(id, _)| {
                if let Some(ref ns) = namespace {
                    graph
                        .get_node(id)
                        .ok()
                        .flatten()
                        .and_then(|n| n.namespace)
                        .as_deref()
                        == Some(ns.as_str())
                } else {
                    true
                }
            })
            .map(|(id, score)| {
                let label = graph
                    .get_node(&id)
                    .ok()
                    .flatten()
                    .map(|n| n.label.clone())
                    .unwrap_or_else(|| id.clone());
                PagerankEntry {
                    node_id: id,
                    label,
                    score,
                }
            })
            .collect();

        entries.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        entries.truncate(top);

        PagerankResponse { scores: entries }
    });

    Json(result.unwrap_or_else(|_| PagerankResponse { scores: vec![] }))
}

pub async fn get_shortest_path(
    State(state): State<Arc<AppState>>,
    Query(query): Query<ShortestPathQuery>,
) -> Result<Json<Vec<String>>, StatusCode> {
    state
        .server
        .engine
        .with_graph(|graph| graph.shortest_path(&query.from, &query.to))
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .map(Json)
        .map_err(|_| StatusCode::NOT_FOUND)
}

pub async fn reload_graph(
    State(state): State<Arc<AppState>>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    match state.server.reload_graph() {
        Ok(()) => {
            let result = state
                .server
                .engine
                .with_graph(|graph| {
                    serde_json::json!({
                        "status": "ok",
                        "node_count": graph.node_count(),
                        "edge_count": graph.edge_count(),
                    })
                })
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
            Ok(Json(result))
        }
        Err(e) => {
            tracing::error!("Failed to reload graph: {e}");
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

pub async fn get_impact(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<SubgraphResponse>, StatusCode> {
    state
        .server
        .engine
        .with_graph(|graph| collect_subgraph(graph, &id, 3))
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .map(Json)
        .map_err(|_| StatusCode::NOT_FOUND)
}

pub async fn get_graph_browse(
    State(state): State<Arc<AppState>>,
    Query(query): Query<BrowseQuery>,
) -> Json<BrowseResponse> {
    let limit = query.limit.unwrap_or(50).min(200);
    let offset = query.offset.unwrap_or(0);

    let storage = state.server.storage();

    let mut all_nodes = storage.all_graph_nodes().unwrap_or_default();
    let all_edges = storage.all_graph_edges().unwrap_or_default();

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

    // Kind counts (only filtered by namespace + search, not kind)
    let mut kind_filtered_nodes = storage.all_graph_nodes().unwrap_or_default();
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
        let ns_node_ids: std::collections::HashSet<&str> =
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

    Json(BrowseResponse {
        nodes,
        total,
        kinds,
        edge_count,
    })
}

// ── Temporal Endpoints ──────────────────────────────────────────────────

pub async fn get_temporal_changes(
    State(state): State<Arc<AppState>>,
    Query(query): Query<crate::api::types::TemporalChangesQuery>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let from = chrono::DateTime::parse_from_rfc3339(&query.from)
        .map(|dt| dt.to_utc())
        .map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": format!("Invalid 'from' date: {e}")})),
            )
        })?;
    let to = chrono::DateTime::parse_from_rfc3339(&query.to)
        .map(|dt| dt.to_utc())
        .map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": format!("Invalid 'to' date: {e}")})),
            )
        })?;

    let entries = state
        .server
        .engine
        .what_changed(from, to, query.namespace.as_deref())
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": e.to_string()})),
            )
        })?;

    Ok(Json(serde_json::json!({
        "commits": entries.len(),
        "entries": entries,
    })))
}

pub async fn get_temporal_snapshot(
    State(state): State<Arc<AppState>>,
    Query(query): Query<crate::api::types::TemporalSnapshotQuery>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let at = chrono::DateTime::parse_from_rfc3339(&query.at)
        .map(|dt| dt.to_utc())
        .map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": format!("Invalid 'at' date: {e}")})),
            )
        })?;

    let snapshot = state.server.engine.graph_at_time(at).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        )
    })?;

    Ok(Json(serde_json::to_value(snapshot).unwrap_or_default()))
}

// ── Stale Files & Drift ────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct StaleFilesQuery {
    pub namespace: Option<String>,
    pub stale_days: Option<u64>,
}

pub async fn get_stale_files(
    State(state): State<Arc<AppState>>,
    Query(query): Query<StaleFilesQuery>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let stale_days = query.stale_days.unwrap_or(30);
    let files = state
        .server
        .engine
        .find_stale_files(query.namespace.as_deref(), stale_days)
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": e.to_string()})),
            )
        })?;

    Ok(Json(serde_json::json!({
        "stale_days": stale_days,
        "stale_files": files.len(),
        "files": files,
    })))
}

pub async fn get_drift(
    State(state): State<Arc<AppState>>,
    Query(query): Query<crate::api::types::TemporalChangesQuery>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let from = chrono::DateTime::parse_from_rfc3339(&query.from)
        .map(|dt| dt.to_utc())
        .map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": format!("Invalid 'from' date: {e}")})),
            )
        })?;
    let to = chrono::DateTime::parse_from_rfc3339(&query.to)
        .map(|dt| dt.to_utc())
        .map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": format!("Invalid 'to' date: {e}")})),
            )
        })?;

    let report = state
        .server
        .engine
        .detect_drift(from, to, query.namespace.as_deref())
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": e.to_string()})),
            )
        })?;

    Ok(Json(serde_json::to_value(report).unwrap_or_default()))
}

// ── File Content ────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct FileContentQuery {
    pub path: String,
    pub line_start: Option<usize>,
    pub line_end: Option<usize>,
    /// Optional project root to resolve relative paths against.
    pub root: Option<String>,
    /// Namespace to look up the stored root path.
    pub namespace: Option<String>,
}

/// Try to resolve a relative file path to an absolute one.
/// Priority: explicit root → namespace root from DB → CWD → as-is.
fn resolve_file_path(
    path: &str,
    root: Option<&str>,
    storage: &dyn codemem_core::StorageBackend,
    namespace: Option<&str>,
) -> std::path::PathBuf {
    let p = std::path::Path::new(path);
    if p.is_absolute() && p.exists() {
        return p.to_path_buf();
    }

    // If explicit root provided, try that first
    if let Some(root) = root {
        let candidate = std::path::Path::new(root).join(path);
        if candidate.exists() {
            return candidate;
        }
    }

    // Try namespace root from DB (stored by `codemem analyze`)
    if let Some(ns) = namespace {
        if let Ok(Some(ns_root)) = storage.get_namespace_root(ns) {
            let candidate = std::path::Path::new(&ns_root).join(path);
            if candidate.exists() {
                return candidate;
            }
        }
    }

    // Try CWD
    if let Ok(cwd) = std::env::current_dir() {
        let candidate = cwd.join(path);
        if candidate.exists() {
            return candidate;
        }
    }

    // Fallback: return as-is
    p.to_path_buf()
}

#[derive(Debug, Serialize)]
pub struct FileContentResponse {
    pub path: String,
    pub content: String,
    pub total_lines: usize,
    pub line_start: usize,
    pub line_end: usize,
    pub language: String,
}

/// Serve file content for the code viewer. Resolves relative paths
/// against CWD (the directory `codemem ui` was launched from).
/// Only serves files that exist as graph nodes to prevent arbitrary reads.
pub async fn get_file_content(
    State(state): State<Arc<AppState>>,
    Query(query): Query<FileContentQuery>,
) -> Result<Json<FileContentResponse>, (StatusCode, Json<serde_json::Value>)> {
    let file_path = &query.path;

    // Security: verify the file exists as a graph node
    let node_id = format!("file:{file_path}");
    let exists = state
        .server
        .engine
        .storage()
        .get_graph_node(&node_id)
        .ok()
        .flatten()
        .is_some();
    if !exists {
        return Err((
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "File not found in graph"})),
        ));
    }

    // Resolve relative paths using stored namespace root, explicit root, or CWD.
    let resolved = resolve_file_path(
        file_path,
        query.root.as_deref(),
        state.server.engine.storage(),
        query.namespace.as_deref(),
    );

    // Read the file
    let content = std::fs::read_to_string(&resolved).map_err(|e| {
        (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": format!("Cannot read file: {e}")})),
        )
    })?;

    let lines: Vec<&str> = content.lines().collect();
    let total_lines = lines.len();

    let line_start = query.line_start.unwrap_or(1).max(1);
    let line_end = query.line_end.unwrap_or(total_lines).min(total_lines);

    let sliced = lines
        .get(line_start.saturating_sub(1)..line_end)
        .map(|s| s.join("\n"))
        .unwrap_or_default();

    let language = file_path.rsplit('.').next().unwrap_or("").to_string();

    Ok(Json(FileContentResponse {
        path: file_path.to_string(),
        content: sliced,
        total_lines,
        line_start,
        line_end,
        language,
    }))
}
