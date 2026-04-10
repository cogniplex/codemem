//! Memory CRUD and search routes.

use crate::api::types::{
    IdResponse, MemoryItem, MemoryListQuery, MemoryListResponse, MessageResponse,
    ScoreBreakdownResponse, SearchQuery, SearchResponse, SearchResultItem, StoreMemoryRequest,
    UpdateMemoryRequest,
};
use crate::api::AppState;
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    Json,
};
use codemem_core::{MemoryNode, MemoryType, SearchResult};
use std::sync::Arc;

/// Run a closure on a plain OS thread (no tokio runtime in TLS).
///
/// `tokio::task::spawn_blocking` threads still carry the runtime Handle,
/// which causes `reqwest::blocking::Client::send()` to detect a nested
/// runtime and panic.  A plain `std::thread` has no tokio context so
/// `reqwest::blocking` works correctly.
pub(crate) fn run_blocking<F, T>(f: F) -> T
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(f());
    });
    rx.recv().expect("blocking thread panicked")
}

pub async fn list_memories(
    State(state): State<Arc<AppState>>,
    Query(query): Query<MemoryListQuery>,
) -> Json<MemoryListResponse> {
    let storage = state.server.storage();
    let offset = query.offset.unwrap_or(0);
    let limit = query.limit.unwrap_or(50).min(200);

    let memories = storage
        .list_memories_filtered(query.namespace.as_deref(), query.memory_type.as_deref())
        .unwrap_or_default();

    let total = memories.len();
    let page: Vec<MemoryItem> = memories
        .into_iter()
        .skip(offset)
        .take(limit)
        .map(memory_to_item)
        .collect();

    Json(MemoryListResponse {
        memories: page,
        total,
        offset,
        limit,
    })
}

pub async fn get_memory(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<MemoryItem>, StatusCode> {
    let storage = state.server.storage();
    match storage.get_memory(&id) {
        Ok(Some(m)) => Ok(Json(memory_to_item(m))),
        Ok(None) => Err(StatusCode::NOT_FOUND),
        Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
    }
}

pub async fn store_memory(
    State(state): State<Arc<AppState>>,
    Json(req): Json<StoreMemoryRequest>,
) -> Result<(StatusCode, Json<IdResponse>), (StatusCode, Json<MessageResponse>)> {
    let memory_type = req
        .memory_type
        .as_deref()
        .and_then(|t| t.parse::<MemoryType>().ok())
        .unwrap_or(MemoryType::Context);

    let mut memory = MemoryNode::new(req.content.clone(), memory_type);
    let id = memory.id.clone();
    memory.importance = req.importance.unwrap_or(0.5);
    memory.tags = req.tags.unwrap_or_default();
    memory.namespace = req.namespace;

    let server = Arc::clone(&state.server);
    let result = run_blocking(move || server.engine.persist_memory(&memory));

    if let Err(e) = result {
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(MessageResponse {
                message: e.to_string(),
            }),
        ));
    }

    Ok((StatusCode::CREATED, Json(IdResponse { id })))
}

pub async fn update_memory(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Json(req): Json<UpdateMemoryRequest>,
) -> Result<Json<MessageResponse>, (StatusCode, Json<MessageResponse>)> {
    let storage = state.server.storage();

    // Check if memory exists (no-touch to avoid bumping access_count on read-only check)
    match storage.get_memory_no_touch(&id) {
        Ok(Some(_)) => {}
        Ok(None) => {
            return Err((
                StatusCode::NOT_FOUND,
                Json(MessageResponse {
                    message: "Memory not found".to_string(),
                }),
            ))
        }
        Err(e) => {
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(MessageResponse {
                    message: e.to_string(),
                }),
            ))
        }
    }

    let server = Arc::clone(&state.server);
    let content = req.content.clone();
    let importance = req.importance;
    let result = run_blocking(move || {
        if let Some(ref content) = content {
            server.engine.update_memory(&id, content, importance)
        } else if let Some(importance) = importance {
            server.engine.update_importance(&id, importance)
        } else {
            Ok(())
        }
    });

    if let Err(e) = result {
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(MessageResponse {
                message: e.to_string(),
            }),
        ));
    }

    Ok(Json(MessageResponse {
        message: "Updated".to_string(),
    }))
}

pub async fn delete_memory(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<MessageResponse>, StatusCode> {
    // Use engine's full delete pipeline: storage (cascade) → vector → graph → BM25
    match state.server.engine.delete_memory(&id) {
        Ok(true) => Ok(Json(MessageResponse {
            message: "Deleted".to_string(),
        })),
        Ok(false) => Err(StatusCode::NOT_FOUND),
        Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
    }
}

pub async fn search_memories(
    State(state): State<Arc<AppState>>,
    Query(query): Query<SearchQuery>,
) -> Json<SearchResponse> {
    let k = query.k.unwrap_or(10);
    let q_str = query.q.clone();
    let ns = query.namespace.clone();
    let mt = query
        .memory_type
        .as_deref()
        .and_then(|t| t.parse::<MemoryType>().ok());

    let server = Arc::clone(&state.server);
    let results = run_blocking(move || {
        let rq = codemem_engine::RecallQuery {
            query: &q_str,
            k,
            memory_type_filter: mt,
            namespace_filter: ns.as_deref(),
            ..codemem_engine::RecallQuery::new(&q_str, k)
        };
        server.recall(&rq)
    });

    let results = match results {
        Ok(results) => results.into_iter().map(search_result_to_item).collect(),
        Err(e) => {
            tracing::warn!("Search failed: {e}");
            Vec::new()
        }
    };

    Json(SearchResponse {
        results,
        query: query.q,
        k,
    })
}

fn search_result_to_item(r: SearchResult) -> SearchResultItem {
    let bd = &r.score_breakdown;
    SearchResultItem {
        id: r.memory.id,
        content: r.memory.content,
        memory_type: r.memory.memory_type.to_string(),
        score: r.score,
        score_breakdown: ScoreBreakdownResponse {
            vector_similarity: bd.vector_similarity,
            graph_strength: bd.graph_strength,
            token_overlap: bd.token_overlap,
            temporal: bd.temporal,
            tag_matching: bd.tag_matching,
            importance: bd.importance,
            confidence: bd.confidence,
            recency: bd.recency,
        },
        tags: r.memory.tags,
        namespace: r.memory.namespace,
    }
}

fn memory_to_item(m: MemoryNode) -> MemoryItem {
    MemoryItem {
        id: m.id,
        content: m.content,
        memory_type: m.memory_type.to_string(),
        importance: m.importance,
        confidence: m.confidence,
        access_count: m.access_count,
        tags: m.tags,
        namespace: m.namespace,
        created_at: m.created_at.to_rfc3339(),
        updated_at: m.updated_at.to_rfc3339(),
    }
}
