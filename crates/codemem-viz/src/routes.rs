use axum::{
    extract::{Path, Query, State},
    response::Json,
};
use codemem_core::NodeKind;
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

// ── Stats & Namespaces ───────────────────────────────────────────────────────

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

// ── Memories ─────────────────────────────────────────────────────────────────

pub async fn api_memories(
    State(storage): State<AppState>,
    Query(query): Query<MemoryQuery>,
) -> ApiResult<Vec<MemoryListItem>> {
    let storage = storage.lock().map_err(lock_err)?;
    let conn = storage.connection();

    let mut sql =
        "SELECT id, content, memory_type, importance, tags, namespace FROM memories WHERE 1=1"
            .to_string();
    let mut param_values: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();

    if let Some(ref ns) = query.namespace {
        param_values.push(Box::new(ns.clone()));
        sql.push_str(&format!(" AND namespace = ?{}", param_values.len()));
    }
    if let Some(ref mt) = query.memory_type {
        param_values.push(Box::new(mt.clone()));
        sql.push_str(&format!(" AND memory_type = ?{}", param_values.len()));
    }
    sql.push_str(" ORDER BY created_at DESC");

    let params_refs: Vec<&dyn rusqlite::types::ToSql> =
        param_values.iter().map(|p| p.as_ref()).collect();
    let mut stmt = conn.prepare(&sql).map_err(lock_err)?;

    let rows = stmt
        .query_map(params_refs.as_slice(), |row| {
            let content: String = row.get(1)?;
            let tags_str: String = row.get(4)?;
            Ok(MemoryListItem {
                id: row.get(0)?,
                content: truncate_str(&content, 120),
                memory_type: row.get(2)?,
                importance: row.get(3)?,
                tags: serde_json::from_str(&tags_str).unwrap_or_default(),
                namespace: row.get(5)?,
            })
        })
        .map_err(lock_err)?
        .filter_map(|r| r.ok())
        .collect::<Vec<_>>();

    Ok(Json(rows))
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

    // Fall back to graph node — prevents 404 when clicking graph-only nodes
    let node = storage.get_graph_node(&id).map_err(lock_err)?;
    match node {
        Some(n) => Ok(Json(serde_json::to_value(n).unwrap())),
        None => Err((
            axum::http::StatusCode::NOT_FOUND,
            format!("Memory or node {} not found", id),
        )),
    }
}

// ── Vectors (PCA) ────────────────────────────────────────────────────────────

pub async fn api_vectors(
    State(storage): State<AppState>,
    Query(query): Query<VectorQuery>,
) -> ApiResult<Vec<VectorPoint>> {
    let storage = storage.lock().map_err(lock_err)?;
    let conn = storage.connection();

    // LEFT JOIN both memories and graph_nodes so embeddings linked to either show up
    let sql = "SELECT me.memory_id, me.embedding,
                      m.memory_type, m.importance, m.namespace, m.content,
                      gn.kind, gn.label, gn.namespace AS gn_namespace
               FROM memory_embeddings me
               LEFT JOIN memories m ON me.memory_id = m.id
               LEFT JOIN graph_nodes gn ON me.memory_id = gn.id";

    let mut stmt = conn.prepare(sql).map_err(lock_err)?;

    struct EmbeddingRow {
        memory_id: String,
        embedding: Vec<f32>,
        memory_type: String,
        importance: f64,
        namespace: Option<String>,
        content: String,
    }

    let rows: Vec<EmbeddingRow> = stmt
        .query_map([], |row| {
            let blob: Vec<u8> = row.get(1)?;
            let embedding: Vec<f32> = blob
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();
            let memory_type: Option<String> = row.get(2)?;
            let importance: Option<f64> = row.get(3)?;
            let mem_namespace: Option<String> = row.get(4)?;
            let mem_content: Option<String> = row.get(5)?;
            let gn_kind: Option<String> = row.get(6)?;
            let gn_label: Option<String> = row.get(7)?;
            let gn_namespace: Option<String> = row.get(8)?;

            Ok(EmbeddingRow {
                memory_id: row.get(0)?,
                embedding,
                memory_type: memory_type
                    .unwrap_or_else(|| gn_kind.unwrap_or_else(|| "context".to_string())),
                importance: importance.unwrap_or(0.5),
                namespace: mem_namespace.or(gn_namespace),
                content: mem_content.unwrap_or_else(|| gn_label.unwrap_or_default()),
            })
        })
        .map_err(lock_err)?
        .filter_map(|r| r.ok())
        .collect();

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

// ── Graph ────────────────────────────────────────────────────────────────────

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

// ── Search ───────────────────────────────────────────────────────────────────

pub async fn api_search(
    State(storage): State<AppState>,
    Query(query): Query<SearchQuery>,
) -> ApiResult<Vec<MemoryListItem>> {
    let q = match query.q {
        Some(ref q) if !q.is_empty() => q.clone(),
        _ => return Ok(Json(vec![])),
    };

    let storage = storage.lock().map_err(lock_err)?;
    let conn = storage.connection();
    let pattern = format!("%{}%", q);

    // Search memories
    let (mem_sql, mem_params): (String, Vec<Box<dyn rusqlite::types::ToSql>>) = if let Some(
        ref ns,
    ) =
        query.namespace
    {
        (
                "SELECT id, content, memory_type, importance, tags, namespace FROM memories WHERE content LIKE ?1 AND namespace = ?2 ORDER BY importance DESC LIMIT 100".to_string(),
                vec![Box::new(pattern.clone()) as Box<dyn rusqlite::types::ToSql>, Box::new(ns.clone())],
            )
    } else {
        (
                "SELECT id, content, memory_type, importance, tags, namespace FROM memories WHERE content LIKE ?1 ORDER BY importance DESC LIMIT 100".to_string(),
                vec![Box::new(pattern.clone()) as Box<dyn rusqlite::types::ToSql>],
            )
    };

    let mem_refs: Vec<&dyn rusqlite::types::ToSql> =
        mem_params.iter().map(|p| p.as_ref()).collect();
    let mut mem_stmt = conn.prepare(&mem_sql).map_err(lock_err)?;

    let mut results: Vec<MemoryListItem> = mem_stmt
        .query_map(mem_refs.as_slice(), |row| {
            let content: String = row.get(1)?;
            let tags_str: String = row.get(4)?;
            Ok(MemoryListItem {
                id: row.get(0)?,
                content: truncate_str(&content, 120),
                memory_type: row.get(2)?,
                importance: row.get(3)?,
                tags: serde_json::from_str(&tags_str).unwrap_or_default(),
                namespace: row.get(5)?,
            })
        })
        .map_err(lock_err)?
        .filter_map(|r| r.ok())
        .collect();

    // Also search graph node labels
    let (gn_sql, gn_params): (String, Vec<Box<dyn rusqlite::types::ToSql>>) = if let Some(ref ns) =
        query.namespace
    {
        (
                "SELECT id, label, kind, namespace FROM graph_nodes WHERE label LIKE ?1 AND namespace = ?2 ORDER BY centrality DESC LIMIT 100".to_string(),
                vec![Box::new(pattern.clone()) as Box<dyn rusqlite::types::ToSql>, Box::new(ns.clone())],
            )
    } else {
        (
                "SELECT id, label, kind, namespace FROM graph_nodes WHERE label LIKE ?1 ORDER BY centrality DESC LIMIT 100".to_string(),
                vec![Box::new(pattern) as Box<dyn rusqlite::types::ToSql>],
            )
    };

    let gn_refs: Vec<&dyn rusqlite::types::ToSql> = gn_params.iter().map(|p| p.as_ref()).collect();
    let mut gn_stmt = conn.prepare(&gn_sql).map_err(lock_err)?;

    let seen_ids: HashSet<String> = results.iter().map(|r| r.id.clone()).collect();

    let gn_results: Vec<MemoryListItem> = gn_stmt
        .query_map(gn_refs.as_slice(), |row| {
            let id: String = row.get(0)?;
            let label: String = row.get(1)?;
            let kind: String = row.get(2)?;
            let namespace: Option<String> = row.get(3)?;
            Ok(MemoryListItem {
                id,
                content: truncate_str(&label, 120),
                memory_type: kind,
                importance: 0.5,
                tags: vec![],
                namespace,
            })
        })
        .map_err(lock_err)?
        .filter_map(|r| r.ok())
        .filter(|r| !seen_ids.contains(&r.id))
        .collect();

    results.extend(gn_results);
    Ok(Json(results))
}

// ── Browse (paginated graph nodes) ──────────────────────────────────────────

pub async fn api_graph_browse(
    State(storage): State<AppState>,
    Query(query): Query<BrowseQuery>,
) -> ApiResult<BrowseResponse> {
    let storage = storage.lock().map_err(lock_err)?;
    let conn = storage.connection();

    let limit = query.limit.unwrap_or(50).min(200);
    let offset = query.offset.unwrap_or(0);

    // Build WHERE clause
    let mut conditions = Vec::new();
    let mut param_values: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();

    if let Some(ref ns) = query.namespace {
        param_values.push(Box::new(ns.clone()));
        conditions.push(format!("gn.namespace = ?{}", param_values.len()));
    }
    if let Some(ref kind) = query.kind {
        param_values.push(Box::new(kind.clone()));
        conditions.push(format!("gn.kind = ?{}", param_values.len()));
    }
    if let Some(ref q) = query.q {
        if !q.is_empty() {
            param_values.push(Box::new(format!("%{}%", q)));
            conditions.push(format!("gn.label LIKE ?{}", param_values.len()));
        }
    }

    let where_clause = if conditions.is_empty() {
        String::new()
    } else {
        format!(" WHERE {}", conditions.join(" AND "))
    };

    // Main query: nodes with degree, sorted by centrality
    let main_sql = format!(
        "SELECT gn.id, gn.kind, gn.label, gn.centrality, gn.namespace, \
         (SELECT COUNT(*) FROM graph_edges WHERE src=gn.id OR dst=gn.id) AS degree \
         FROM graph_nodes gn{} ORDER BY gn.centrality DESC LIMIT ?{} OFFSET ?{}",
        where_clause,
        param_values.len() + 1,
        param_values.len() + 2,
    );

    let mut main_params = param_values
        .iter()
        .map(|p| p.as_ref() as &dyn rusqlite::types::ToSql)
        .collect::<Vec<_>>();
    let limit_i64 = limit as i64;
    let offset_i64 = offset as i64;
    main_params.push(&limit_i64);
    main_params.push(&offset_i64);

    let mut stmt = conn.prepare(&main_sql).map_err(lock_err)?;
    let nodes: Vec<BrowseNodeItem> = stmt
        .query_map(main_params.as_slice(), |row| {
            Ok(BrowseNodeItem {
                id: row.get(0)?,
                kind: row.get(1)?,
                label: row.get(2)?,
                centrality: row.get::<_, f64>(3).unwrap_or(0.0),
                namespace: row.get(4)?,
                degree: row.get::<_, i64>(5).unwrap_or(0) as usize,
            })
        })
        .map_err(lock_err)?
        .filter_map(|r| r.ok())
        .collect();

    // Total count
    let count_sql = format!("SELECT COUNT(*) FROM graph_nodes gn{}", where_clause);
    let count_params: Vec<&dyn rusqlite::types::ToSql> =
        param_values.iter().map(|p| p.as_ref()).collect();
    let total: usize = conn
        .query_row(&count_sql, count_params.as_slice(), |row| {
            row.get::<_, i64>(0)
        })
        .map_err(lock_err)? as usize;

    // Kind counts (only filter by namespace + search, not by kind itself)
    let mut kind_conditions = Vec::new();
    let mut kind_param_values: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();

    if let Some(ref ns) = query.namespace {
        kind_param_values.push(Box::new(ns.clone()));
        kind_conditions.push(format!("namespace = ?{}", kind_param_values.len()));
    }
    if let Some(ref q) = query.q {
        if !q.is_empty() {
            kind_param_values.push(Box::new(format!("%{}%", q)));
            kind_conditions.push(format!("label LIKE ?{}", kind_param_values.len()));
        }
    }

    let kind_where = if kind_conditions.is_empty() {
        String::new()
    } else {
        format!(" WHERE {}", kind_conditions.join(" AND "))
    };

    let kind_sql = format!(
        "SELECT kind, COUNT(*) FROM graph_nodes{} GROUP BY kind",
        kind_where
    );
    let kind_params: Vec<&dyn rusqlite::types::ToSql> =
        kind_param_values.iter().map(|p| p.as_ref()).collect();
    let mut kind_stmt = conn.prepare(&kind_sql).map_err(lock_err)?;
    let kinds: HashMap<String, usize> = kind_stmt
        .query_map(kind_params.as_slice(), |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)? as usize))
        })
        .map_err(lock_err)?
        .filter_map(|r| r.ok())
        .collect();

    // Edge count (filtered by namespace if provided)
    let edge_count: usize = if let Some(ref ns) = query.namespace {
        conn.query_row(
            "SELECT COUNT(*) FROM graph_edges ge \
             JOIN graph_nodes s ON ge.src = s.id \
             JOIN graph_nodes d ON ge.dst = d.id \
             WHERE s.namespace = ?1 AND d.namespace = ?1",
            [ns],
            |row| row.get::<_, i64>(0),
        )
        .map_err(lock_err)? as usize
    } else {
        conn.query_row("SELECT COUNT(*) FROM graph_edges", [], |row| {
            row.get::<_, i64>(0)
        })
        .map_err(lock_err)? as usize
    };

    Ok(Json(BrowseResponse {
        nodes,
        total,
        kinds,
        edge_count,
    }))
}
