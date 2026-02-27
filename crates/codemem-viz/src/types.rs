use codemem_core::{Edge, GraphNode};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Deserialize)]
pub struct MemoryQuery {
    pub namespace: Option<String>,
    #[serde(rename = "type")]
    pub memory_type: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct MemoryListItem {
    pub id: String,
    pub content: String,
    pub memory_type: String,
    pub importance: f64,
    pub tags: Vec<String>,
    pub namespace: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct VectorQuery {
    pub namespace: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct VectorPoint {
    pub id: String,
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub memory_type: String,
    pub importance: f64,
    pub namespace: Option<String>,
    pub label: String,
}

#[derive(Debug, Deserialize)]
pub struct GraphNodeQuery {
    pub namespace: Option<String>,
    pub kind: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct EdgeResponse {
    pub id: String,
    pub src: String,
    pub dst: String,
    pub relationship: String,
    pub weight: f64,
}

#[derive(Debug, Deserialize)]
pub struct EdgeQuery {
    pub namespace: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct NeighborQuery {
    pub depth: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct NeighborResponse {
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<Edge>,
}

#[derive(Debug, Deserialize)]
pub struct SearchQuery {
    pub q: Option<String>,
    pub namespace: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct BrowseQuery {
    pub namespace: Option<String>,
    pub kind: Option<String>,
    pub q: Option<String>,
    pub offset: Option<usize>,
    pub limit: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct BrowseNodeItem {
    pub id: String,
    pub kind: String,
    pub label: String,
    pub centrality: f64,
    pub namespace: Option<String>,
    pub degree: usize,
}

#[derive(Debug, Serialize)]
pub struct BrowseResponse {
    pub nodes: Vec<BrowseNodeItem>,
    pub total: usize,
    pub kinds: HashMap<String, usize>,
    pub edge_count: usize,
}
