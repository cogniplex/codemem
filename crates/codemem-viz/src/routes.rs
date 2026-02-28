use axum::{
    extract::{Path, Query, State},
    response::Json,
};
use codemem_core::{GraphNode, MemoryNode, NodeKind, StorageBackend};
use codemem_storage::Storage;
use ndarray::{Array1, Array2};
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
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
    Ok(Json(serde_json::to_value(stats).map_err(lock_err)?))
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
        return Ok(Json(serde_json::to_value(m).map_err(lock_err)?));
    }

    // Fall back to graph node -- prevents 404 when clicking graph-only nodes
    let node = storage.get_graph_node(&id).map_err(lock_err)?;
    match node {
        Some(n) => Ok(Json(serde_json::to_value(n).map_err(lock_err)?)),
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

// -- D3 Graph -----------------------------------------------------------------

pub async fn api_graph_d3(
    State(storage): State<AppState>,
    Query(query): Query<GraphD3Query>,
) -> ApiResult<D3Graph> {
    let storage = storage.lock().map_err(lock_err)?;
    let mut nodes = storage.all_graph_nodes().map_err(lock_err)?;
    let edges = if let Some(ref ns) = query.namespace {
        nodes.retain(|n| n.namespace.as_deref() == Some(ns.as_str()));
        storage.graph_edges_for_namespace(ns)
    } else {
        storage.all_graph_edges()
    }
    .map_err(lock_err)?;

    let node_ids: HashSet<String> = nodes.iter().map(|n| n.id.clone()).collect();

    let d3_nodes: Vec<D3Node> = nodes
        .into_iter()
        .map(|n| D3Node {
            id: n.id,
            label: n.label,
            kind: n.kind.to_string(),
            centrality: n.centrality,
            namespace: n.namespace,
        })
        .collect();

    let d3_links: Vec<D3Link> = edges
        .into_iter()
        .filter(|e| node_ids.contains(&e.src) && node_ids.contains(&e.dst))
        .map(|e| D3Link {
            source: e.src,
            target: e.dst,
            relationship: e.relationship.to_string(),
            weight: e.weight,
        })
        .collect();

    Ok(Json(D3Graph {
        nodes: d3_nodes,
        links: d3_links,
    }))
}

// -- Timeline -----------------------------------------------------------------

pub async fn api_timeline(
    State(storage): State<AppState>,
    Query(query): Query<TimelineQuery>,
) -> ApiResult<TimelineResponse> {
    let storage = storage.lock().map_err(lock_err)?;
    let memories = storage
        .list_memories_filtered(query.namespace.as_deref(), None)
        .map_err(lock_err)?;

    let mut all_types: HashSet<String> = HashSet::new();
    let mut day_map: BTreeMap<String, HashMap<String, usize>> = BTreeMap::new();

    for m in &memories {
        let date = m.created_at.format("%Y-%m-%d").to_string();
        let mtype = m.memory_type.to_string();
        all_types.insert(mtype.clone());
        *day_map.entry(date).or_default().entry(mtype).or_insert(0) += 1;
    }

    let mut types: Vec<String> = all_types.into_iter().collect();
    types.sort();

    let buckets: Vec<TimelineBucket> = day_map
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

    Ok(Json(TimelineResponse { buckets, types }))
}

// -- Distribution -------------------------------------------------------------

pub async fn api_distribution(
    State(storage): State<AppState>,
    Query(query): Query<DistributionQuery>,
) -> ApiResult<DistributionResponse> {
    let storage = storage.lock().map_err(lock_err)?;
    let memories = storage
        .list_memories_filtered(query.namespace.as_deref(), None)
        .map_err(lock_err)?;

    let total = memories.len();
    let mut type_counts: HashMap<String, usize> = HashMap::new();
    let mut importance_histogram: BTreeMap<String, usize> = BTreeMap::new();

    for m in &memories {
        *type_counts.entry(m.memory_type.to_string()).or_insert(0) += 1;

        // Bucket importance into 0.0-0.1, 0.1-0.2, ... 0.9-1.0
        let bucket = (m.importance * 10.0).floor().min(9.0) as u8;
        let label = format!(
            "{:.1}-{:.1}",
            bucket as f64 / 10.0,
            (bucket + 1) as f64 / 10.0
        );
        *importance_histogram.entry(label).or_insert(0) += 1;
    }

    Ok(Json(DistributionResponse {
        type_counts,
        importance_histogram,
        total,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;
    use codemem_core::{Edge as CoreEdge, MemoryType, RelationshipType};

    fn make_test_memory(
        content: &str,
        ns: Option<&str>,
        mtype: MemoryType,
        importance: f64,
        created_at: chrono::DateTime<chrono::Utc>,
    ) -> MemoryNode {
        let id = format!("mem-{:x}", {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            content.hash(&mut h);
            h.finish()
        });
        MemoryNode {
            id,
            content: content.to_string(),
            memory_type: mtype,
            importance,
            confidence: 1.0,
            access_count: 0,
            content_hash: Storage::content_hash(content),
            tags: vec![],
            metadata: HashMap::new(),
            namespace: ns.map(|s| s.to_string()),
            created_at,
            updated_at: created_at,
            last_accessed_at: created_at,
        }
    }

    fn make_test_graph_node(id: &str, kind: NodeKind, label: &str, ns: Option<&str>) -> GraphNode {
        GraphNode {
            id: id.to_string(),
            kind,
            label: label.to_string(),
            payload: HashMap::new(),
            centrality: 0.5,
            memory_id: None,
            namespace: ns.map(|s| s.to_string()),
        }
    }

    fn make_test_edge(id: &str, src: &str, dst: &str) -> CoreEdge {
        CoreEdge {
            id: id.to_string(),
            src: src.to_string(),
            dst: dst.to_string(),
            relationship: RelationshipType::RelatesTo,
            weight: 1.0,
            properties: HashMap::new(),
            created_at: chrono::Utc::now(),
            valid_from: None,
            valid_to: None,
        }
    }

    #[test]
    fn truncate_str_short_string() {
        assert_eq!(truncate_str("hello", 10), "hello");
    }

    #[test]
    fn truncate_str_exact_length() {
        assert_eq!(truncate_str("hello", 5), "hello");
    }

    #[test]
    fn truncate_str_long_string() {
        let result = truncate_str("hello world", 5);
        assert_eq!(result, "hello...");
    }

    #[test]
    fn truncate_str_unicode_boundary() {
        // Multi-byte character: each char is 2 bytes
        let s = "\u{00e9}\u{00e9}\u{00e9}\u{00e9}"; // 4 x e-acute (2 bytes each)
        let result = truncate_str(s, 3);
        // Should back up to char boundary at 2, not split a character
        assert!(result.ends_with("..."));
        // The result should be valid UTF-8
        assert!(result.is_char_boundary(0));
    }

    #[test]
    fn truncate_str_empty() {
        assert_eq!(truncate_str("", 10), "");
    }

    #[tokio::test]
    async fn api_stats_empty_db() {
        let storage = Storage::open_in_memory().unwrap();
        let state: AppState = Arc::new(Mutex::new(storage));
        let result = api_stats(State(state)).await;
        assert!(result.is_ok());
        let json = result.unwrap().0;
        assert_eq!(json["memory_count"], 0);
    }

    #[tokio::test]
    async fn api_namespaces_empty_db() {
        let storage = Storage::open_in_memory().unwrap();
        let state: AppState = Arc::new(Mutex::new(storage));
        let result = api_namespaces(State(state)).await;
        assert!(result.is_ok());
        assert!(result.unwrap().0.is_empty());
    }

    #[tokio::test]
    async fn api_memories_empty_db() {
        let storage = Storage::open_in_memory().unwrap();
        let state: AppState = Arc::new(Mutex::new(storage));
        let query = MemoryQuery {
            namespace: None,
            memory_type: None,
        };
        let result = api_memories(State(state), Query(query)).await;
        assert!(result.is_ok());
        assert!(result.unwrap().0.is_empty());
    }

    #[tokio::test]
    async fn api_vectors_empty_db() {
        let storage = Storage::open_in_memory().unwrap();
        let state: AppState = Arc::new(Mutex::new(storage));
        let query = VectorQuery { namespace: None };
        let result = api_vectors(State(state), Query(query)).await;
        assert!(result.is_ok());
        assert!(result.unwrap().0.is_empty());
    }

    #[tokio::test]
    async fn api_graph_nodes_empty_db() {
        let storage = Storage::open_in_memory().unwrap();
        let state: AppState = Arc::new(Mutex::new(storage));
        let query = GraphNodeQuery {
            namespace: None,
            kind: None,
        };
        let result = api_graph_nodes(State(state), Query(query)).await;
        assert!(result.is_ok());
        assert!(result.unwrap().0.is_empty());
    }

    #[tokio::test]
    async fn api_graph_edges_empty_db() {
        let storage = Storage::open_in_memory().unwrap();
        let state: AppState = Arc::new(Mutex::new(storage));
        let query = EdgeQuery { namespace: None };
        let result = api_graph_edges(State(state), Query(query)).await;
        assert!(result.is_ok());
        assert!(result.unwrap().0.is_empty());
    }

    #[tokio::test]
    async fn api_search_empty_query() {
        let storage = Storage::open_in_memory().unwrap();
        let state: AppState = Arc::new(Mutex::new(storage));
        let query = SearchQuery {
            q: None,
            namespace: None,
        };
        let result = api_search(State(state), Query(query)).await;
        assert!(result.is_ok());
        assert!(result.unwrap().0.is_empty());
    }

    #[tokio::test]
    async fn api_graph_browse_empty_db() {
        let storage = Storage::open_in_memory().unwrap();
        let state: AppState = Arc::new(Mutex::new(storage));
        let query = BrowseQuery {
            namespace: None,
            kind: None,
            q: None,
            offset: None,
            limit: None,
        };
        let result = api_graph_browse(State(state), Query(query)).await;
        assert!(result.is_ok());
        let browse = result.unwrap().0;
        assert!(browse.nodes.is_empty());
        assert_eq!(browse.total, 0);
        assert_eq!(browse.edge_count, 0);
    }

    // -- Integration tests with data ------------------------------------------

    #[tokio::test]
    async fn api_graph_d3_empty_db() {
        let storage = Storage::open_in_memory().unwrap();
        let state: AppState = Arc::new(Mutex::new(storage));
        let query = GraphD3Query { namespace: None };
        let result = api_graph_d3(State(state), Query(query)).await;
        assert!(result.is_ok());
        let d3 = result.unwrap().0;
        assert!(d3.nodes.is_empty());
        assert!(d3.links.is_empty());
    }

    #[tokio::test]
    async fn api_graph_d3_with_data() {
        let storage = Storage::open_in_memory().unwrap();
        let node_a = make_test_graph_node("node-a", NodeKind::Function, "fn_alpha", None);
        let node_b = make_test_graph_node("node-b", NodeKind::Function, "fn_beta", None);
        storage.insert_graph_node(&node_a).unwrap();
        storage.insert_graph_node(&node_b).unwrap();

        let edge = make_test_edge("edge-ab", "node-a", "node-b");
        storage.insert_graph_edge(&edge).unwrap();

        let state: AppState = Arc::new(Mutex::new(storage));
        let query = GraphD3Query { namespace: None };
        let result = api_graph_d3(State(state), Query(query)).await;
        assert!(result.is_ok());
        let d3 = result.unwrap().0;
        assert_eq!(d3.nodes.len(), 2);
        assert_eq!(d3.links.len(), 1);
        assert_eq!(d3.links[0].source, "node-a");
        assert_eq!(d3.links[0].target, "node-b");
    }

    #[tokio::test]
    async fn api_timeline_empty_db() {
        let storage = Storage::open_in_memory().unwrap();
        let state: AppState = Arc::new(Mutex::new(storage));
        let query = TimelineQuery { namespace: None };
        let result = api_timeline(State(state), Query(query)).await;
        assert!(result.is_ok());
        let timeline = result.unwrap().0;
        assert!(timeline.buckets.is_empty());
        assert!(timeline.types.is_empty());
    }

    #[tokio::test]
    async fn api_timeline_with_data() {
        let storage = Storage::open_in_memory().unwrap();

        let date1 = chrono::Utc.with_ymd_and_hms(2024, 1, 15, 12, 0, 0).unwrap();
        let date2 = chrono::Utc.with_ymd_and_hms(2024, 2, 20, 12, 0, 0).unwrap();

        let m1 = make_test_memory(
            "decision about error handling",
            None,
            MemoryType::Decision,
            0.7,
            date1,
        );
        let m2 = make_test_memory(
            "pattern for retry logic",
            None,
            MemoryType::Pattern,
            0.6,
            date2,
        );
        storage.insert_memory(&m1).unwrap();
        storage.insert_memory(&m2).unwrap();

        let state: AppState = Arc::new(Mutex::new(storage));
        let query = TimelineQuery { namespace: None };
        let result = api_timeline(State(state), Query(query)).await;
        assert!(result.is_ok());
        let timeline = result.unwrap().0;
        assert_eq!(timeline.buckets.len(), 2);
        assert_eq!(timeline.buckets[0].date, "2024-01-15");
        assert_eq!(timeline.buckets[1].date, "2024-02-20");
        assert_eq!(timeline.buckets[0].total, 1);
        assert_eq!(timeline.buckets[1].total, 1);
    }

    #[tokio::test]
    async fn api_distribution_empty_db() {
        let storage = Storage::open_in_memory().unwrap();
        let state: AppState = Arc::new(Mutex::new(storage));
        let query = DistributionQuery { namespace: None };
        let result = api_distribution(State(state), Query(query)).await;
        assert!(result.is_ok());
        let dist = result.unwrap().0;
        assert!(dist.type_counts.is_empty());
        assert!(dist.importance_histogram.is_empty());
        assert_eq!(dist.total, 0);
    }

    #[tokio::test]
    async fn api_distribution_with_data() {
        let storage = Storage::open_in_memory().unwrap();

        let now = chrono::Utc::now();
        let m1 = make_test_memory(
            "use Result instead of panic",
            None,
            MemoryType::Decision,
            0.3,
            now,
        );
        let m2 = make_test_memory(
            "builder pattern for configs",
            None,
            MemoryType::Pattern,
            0.8,
            now,
        );
        storage.insert_memory(&m1).unwrap();
        storage.insert_memory(&m2).unwrap();

        let state: AppState = Arc::new(Mutex::new(storage));
        let query = DistributionQuery { namespace: None };
        let result = api_distribution(State(state), Query(query)).await;
        assert!(result.is_ok());
        let dist = result.unwrap().0;
        assert_eq!(dist.total, 2);
        assert_eq!(dist.type_counts.get("decision"), Some(&1));
        assert_eq!(dist.type_counts.get("pattern"), Some(&1));
        // importance 0.3 -> bucket "0.3-0.4", importance 0.8 -> bucket "0.8-0.9"
        assert_eq!(dist.importance_histogram.get("0.3-0.4"), Some(&1));
        assert_eq!(dist.importance_histogram.get("0.8-0.9"), Some(&1));
    }

    #[tokio::test]
    async fn api_memory_detail_found() {
        let storage = Storage::open_in_memory().unwrap();
        let now = chrono::Utc::now();
        let m = make_test_memory(
            "important architectural decision",
            None,
            MemoryType::Decision,
            0.9,
            now,
        );
        let mem_id = m.id.clone();
        storage.insert_memory(&m).unwrap();

        let state: AppState = Arc::new(Mutex::new(storage));
        let result = api_memory_detail(State(state), Path(mem_id)).await;
        assert!(result.is_ok());
        let json = result.unwrap().0;
        assert_eq!(json["content"], "important architectural decision");
    }

    #[tokio::test]
    async fn api_memory_detail_not_found() {
        let storage = Storage::open_in_memory().unwrap();
        let state: AppState = Arc::new(Mutex::new(storage));
        let result = api_memory_detail(State(state), Path("nonexistent-id".to_string())).await;
        assert!(result.is_err());
        let (status, _msg) = result.unwrap_err();
        assert_eq!(status, axum::http::StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn api_graph_neighbors_with_data() {
        let storage = Storage::open_in_memory().unwrap();

        // Create 3 nodes: A -> B -> C
        let node_a = make_test_graph_node("node-a", NodeKind::Module, "module_a", None);
        let node_b = make_test_graph_node("node-b", NodeKind::Function, "fn_b", None);
        let node_c = make_test_graph_node("node-c", NodeKind::Function, "fn_c", None);
        storage.insert_graph_node(&node_a).unwrap();
        storage.insert_graph_node(&node_b).unwrap();
        storage.insert_graph_node(&node_c).unwrap();

        let edge_ab = make_test_edge("edge-ab", "node-a", "node-b");
        let edge_bc = make_test_edge("edge-bc", "node-b", "node-c");
        storage.insert_graph_edge(&edge_ab).unwrap();
        storage.insert_graph_edge(&edge_bc).unwrap();

        let state: AppState = Arc::new(Mutex::new(storage));
        let query = NeighborQuery { depth: Some(2) };
        let result = api_graph_neighbors(
            State(state.clone()),
            Path("node-a".to_string()),
            Query(query),
        )
        .await;
        assert!(result.is_ok());
        let neighbors = result.unwrap().0;
        // BFS from A with depth 2 should reach A, B, and C
        assert_eq!(neighbors.nodes.len(), 3);
        assert_eq!(neighbors.edges.len(), 2);
        let node_ids: HashSet<String> = neighbors.nodes.iter().map(|n| n.id.clone()).collect();
        assert!(node_ids.contains("node-a"));
        assert!(node_ids.contains("node-b"));
        assert!(node_ids.contains("node-c"));
    }

    #[tokio::test]
    async fn api_search_with_results() {
        let storage = Storage::open_in_memory().unwrap();
        let now = chrono::Utc::now();
        let m = make_test_memory(
            "rust pattern matching is powerful",
            None,
            MemoryType::Insight,
            0.7,
            now,
        );
        storage.insert_memory(&m).unwrap();

        let state: AppState = Arc::new(Mutex::new(storage));
        let query = SearchQuery {
            q: Some("pattern".to_string()),
            namespace: None,
        };
        let result = api_search(State(state), Query(query)).await;
        assert!(result.is_ok());
        let results = result.unwrap().0;
        assert_eq!(results.len(), 1);
        assert!(results[0].content.contains("pattern matching"));
    }

    #[tokio::test]
    async fn api_memories_namespace_filter() {
        let storage = Storage::open_in_memory().unwrap();
        let now = chrono::Utc::now();

        let m1 = make_test_memory(
            "memory in project alpha",
            Some("proj-a"),
            MemoryType::Context,
            0.5,
            now,
        );
        let m2 = make_test_memory(
            "memory in project beta",
            Some("proj-b"),
            MemoryType::Context,
            0.5,
            now,
        );
        storage.insert_memory(&m1).unwrap();
        storage.insert_memory(&m2).unwrap();

        let state: AppState = Arc::new(Mutex::new(storage));
        let query = MemoryQuery {
            namespace: Some("proj-a".to_string()),
            memory_type: None,
        };
        let result = api_memories(State(state), Query(query)).await;
        assert!(result.is_ok());
        let items = result.unwrap().0;
        assert_eq!(items.len(), 1);
        assert!(items[0].content.contains("project alpha"));
        assert_eq!(items[0].namespace, Some("proj-a".to_string()));
    }
}
