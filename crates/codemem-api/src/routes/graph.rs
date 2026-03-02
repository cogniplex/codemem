//! Graph exploration routes.

use crate::types::{
    CommunitiesQuery, CommunitiesResponse, GraphEdgeResponse, GraphNodeResponse, NeighborsQuery,
    PagerankEntry, PagerankQuery, PagerankResponse, ShortestPathQuery, SubgraphQuery,
    SubgraphResponse,
};
use crate::AppState;
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    Json,
};
use codemem_core::{GraphBackend, NodeKind};
use std::sync::Arc;

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

    let graph = state
        .server
        .graph()
        .lock()
        .unwrap_or_else(|e| e.into_inner());

    let (nodes, edges) =
        graph.subgraph_top_n(max_nodes, query.namespace.as_deref(), kinds.as_deref());

    let node_responses: Vec<GraphNodeResponse> = nodes
        .into_iter()
        .filter(|n| {
            if let Some(min_c) = query.min_centrality {
                n.centrality >= min_c
            } else {
                true
            }
        })
        .map(|n| GraphNodeResponse {
            id: n.id,
            kind: n.kind.to_string(),
            label: n.label,
            centrality: n.centrality,
            memory_id: n.memory_id,
            namespace: n.namespace,
        })
        .collect();

    let edge_responses: Vec<GraphEdgeResponse> = edges
        .into_iter()
        .map(|e| GraphEdgeResponse {
            id: e.id,
            src: e.src,
            dst: e.dst,
            relationship: e.relationship.to_string(),
            weight: e.weight,
        })
        .collect();

    Json(SubgraphResponse {
        nodes: node_responses,
        edges: edge_responses,
    })
}

pub async fn get_neighbors(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Query(query): Query<NeighborsQuery>,
) -> Result<Json<SubgraphResponse>, StatusCode> {
    let depth = query.depth.unwrap_or(1);
    let graph = state
        .server
        .graph()
        .lock()
        .unwrap_or_else(|e| e.into_inner());

    let neighbor_nodes = graph.bfs(&id, depth).map_err(|_| StatusCode::NOT_FOUND)?;

    let node_ids: std::collections::HashSet<&str> =
        neighbor_nodes.iter().map(|n| n.id.as_str()).collect();

    // Get all edges between these nodes
    let mut edges = Vec::new();
    let mut seen_edges = std::collections::HashSet::new();
    for node in &neighbor_nodes {
        if let Ok(node_edges) = graph.get_edges(&node.id) {
            for edge in node_edges {
                if node_ids.contains(edge.src.as_str())
                    && node_ids.contains(edge.dst.as_str())
                    && seen_edges.insert(edge.id.clone())
                {
                    edges.push(GraphEdgeResponse {
                        id: edge.id,
                        src: edge.src,
                        dst: edge.dst,
                        relationship: edge.relationship.to_string(),
                        weight: edge.weight,
                    });
                }
            }
        }
    }

    let nodes: Vec<GraphNodeResponse> = neighbor_nodes
        .into_iter()
        .map(|n| GraphNodeResponse {
            id: n.id,
            kind: n.kind.to_string(),
            label: n.label,
            centrality: n.centrality,
            memory_id: n.memory_id,
            namespace: n.namespace,
        })
        .collect();

    Ok(Json(SubgraphResponse { nodes, edges }))
}

pub async fn get_communities(
    State(state): State<Arc<AppState>>,
    Query(query): Query<CommunitiesQuery>,
) -> Json<CommunitiesResponse> {
    let resolution = query.resolution.unwrap_or(1.0);
    let graph = state
        .server
        .graph()
        .lock()
        .unwrap_or_else(|e| e.into_inner());

    let assignment = graph.louvain_with_assignment(resolution);
    let num_communities = assignment
        .values()
        .collect::<std::collections::HashSet<_>>()
        .len();

    Json(CommunitiesResponse {
        communities: assignment,
        num_communities,
    })
}

pub async fn get_pagerank(
    State(state): State<Arc<AppState>>,
    Query(query): Query<PagerankQuery>,
) -> Json<PagerankResponse> {
    let top = query.top.unwrap_or(20);
    let graph = state
        .server
        .graph()
        .lock()
        .unwrap_or_else(|e| e.into_inner());

    let scores = graph.pagerank(0.85, 100, 1e-6);

    let mut entries: Vec<PagerankEntry> = scores
        .into_iter()
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

    Json(PagerankResponse { scores: entries })
}

pub async fn get_shortest_path(
    State(state): State<Arc<AppState>>,
    Query(query): Query<ShortestPathQuery>,
) -> Result<Json<Vec<String>>, StatusCode> {
    let graph = state
        .server
        .graph()
        .lock()
        .unwrap_or_else(|e| e.into_inner());

    match graph.shortest_path(&query.from, &query.to) {
        Ok(path) => Ok(Json(path)),
        Err(_) => Err(StatusCode::NOT_FOUND),
    }
}

pub async fn get_impact(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<SubgraphResponse>, StatusCode> {
    let graph = state
        .server
        .graph()
        .lock()
        .unwrap_or_else(|e| e.into_inner());

    let reachable = graph.bfs(&id, 3).map_err(|_| StatusCode::NOT_FOUND)?;

    let nodes: Vec<GraphNodeResponse> = reachable
        .into_iter()
        .map(|n| GraphNodeResponse {
            id: n.id,
            kind: n.kind.to_string(),
            label: n.label,
            centrality: n.centrality,
            memory_id: n.memory_id,
            namespace: n.namespace,
        })
        .collect();

    Ok(Json(SubgraphResponse {
        nodes,
        edges: Vec::new(),
    }))
}
