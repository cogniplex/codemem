//! Memory CRUD and search routes.

use crate::types::{
    IdResponse, MemoryItem, MemoryListQuery, MemoryListResponse, MessageResponse,
    ScoreBreakdownResponse, SearchQuery, SearchResponse, SearchResultItem, StoreMemoryRequest,
    UpdateMemoryRequest,
};
use crate::AppState;
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    Json,
};
use codemem_core::{MemoryNode, MemoryType, VectorBackend};
use std::sync::Arc;

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
    let now = chrono::Utc::now();
    let id = uuid::Uuid::new_v4().to_string();
    let memory_type = req
        .memory_type
        .as_deref()
        .and_then(|t| t.parse::<MemoryType>().ok())
        .unwrap_or(MemoryType::Context);
    let hash = codemem_storage::Storage::content_hash(&req.content);

    let memory = MemoryNode {
        id: id.clone(),
        content: req.content.clone(),
        memory_type,
        importance: req.importance.unwrap_or(0.5),
        confidence: 1.0,
        access_count: 0,
        content_hash: hash,
        tags: req.tags.unwrap_or_default(),
        metadata: Default::default(),
        namespace: req.namespace,
        created_at: now,
        updated_at: now,
        last_accessed_at: now,
    };

    let storage = state.server.storage();
    if let Err(e) = storage.insert_memory(&memory) {
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(MessageResponse {
                message: e.to_string(),
            }),
        ));
    }

    // Auto-embed if provider available
    if let Some(emb_mutex) = state.server.embeddings() {
        if let Ok(emb) = emb_mutex.lock() {
            if let Ok(embedding) = emb.embed(&req.content) {
                let _ = storage.store_embedding(&id, &embedding);
                if let Ok(mut vec) = state.server.vector().lock() {
                    let _ = vec.insert(&id, &embedding);
                }
            }
        }
    }

    Ok((StatusCode::CREATED, Json(IdResponse { id })))
}

pub async fn update_memory(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Json(req): Json<UpdateMemoryRequest>,
) -> Result<Json<MessageResponse>, (StatusCode, Json<MessageResponse>)> {
    let storage = state.server.storage();

    // Check if memory exists
    match storage.get_memory(&id) {
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

    if let Some(ref content) = req.content {
        if let Err(e) = storage.update_memory(&id, content, req.importance) {
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(MessageResponse {
                    message: e.to_string(),
                }),
            ));
        }

        // Re-embed if content changed
        if let Some(emb_mutex) = state.server.embeddings() {
            if let Ok(emb) = emb_mutex.lock() {
                if let Ok(embedding) = emb.embed(content) {
                    let _ = storage.store_embedding(&id, &embedding);
                    if let Ok(mut vec) = state.server.vector().lock() {
                        let _ = vec.remove(&id);
                        let _ = vec.insert(&id, &embedding);
                    }
                }
            }
        }
    }

    Ok(Json(MessageResponse {
        message: "Updated".to_string(),
    }))
}

pub async fn delete_memory(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<MessageResponse>, StatusCode> {
    let storage = state.server.storage();
    match storage.delete_memory(&id) {
        Ok(true) => {
            let _ = storage.delete_embedding(&id);
            if let Ok(mut vec) = state.server.vector().lock() {
                let _ = vec.remove(&id);
            }
            Ok(Json(MessageResponse {
                message: "Deleted".to_string(),
            }))
        }
        Ok(false) => Err(StatusCode::NOT_FOUND),
        Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
    }
}

pub async fn search_memories(
    State(state): State<Arc<AppState>>,
    Query(query): Query<SearchQuery>,
) -> Json<SearchResponse> {
    let k = query.k.unwrap_or(10);

    // Use the MCP server's handle_request to leverage the full hybrid scoring pipeline
    let mut args = serde_json::json!({
        "query": query.q,
        "k": k,
    });
    if let Some(ref ns) = query.namespace {
        args["namespace"] = serde_json::Value::String(ns.clone());
    }
    if let Some(ref mt) = query.memory_type {
        args["memory_type"] = serde_json::Value::String(mt.clone());
    }

    let params = serde_json::json!({
        "name": "recall_memory",
        "arguments": args,
    });

    let response = state.server.handle_request(
        "tools/call",
        Some(&params),
        serde_json::Value::Number(0.into()),
    );

    // Parse the tool result to extract search results
    let results = if let Some(result) = response.result {
        parse_recall_result(&result)
    } else {
        Vec::new()
    };

    Json(SearchResponse {
        results,
        query: query.q,
        k,
    })
}

fn parse_recall_result(result: &serde_json::Value) -> Vec<SearchResultItem> {
    // The recall_memory tool returns a ToolResult with content[0].text containing
    // formatted results. We parse what we can from the JSON structure.
    let text = result
        .get("content")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("text"))
        .and_then(|t| t.as_str())
        .unwrap_or("");

    // Try to parse the text as JSON (the tool formats results as JSON)
    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(text) {
        if let Some(arr) = parsed.as_array() {
            return arr
                .iter()
                .filter_map(|item| {
                    Some(SearchResultItem {
                        id: item.get("id")?.as_str()?.to_string(),
                        content: item
                            .get("content")
                            .and_then(|c| c.as_str())
                            .unwrap_or("")
                            .to_string(),
                        memory_type: item
                            .get("memory_type")
                            .and_then(|t| t.as_str())
                            .unwrap_or("context")
                            .to_string(),
                        score: item
                            .get("score")
                            .and_then(|s| s.as_f64().or_else(|| s.as_str().and_then(|t| t.parse().ok())))
                            .unwrap_or(0.0),
                        score_breakdown: ScoreBreakdownResponse {
                            vector_similarity: get_f64(item, "vector_similarity"),
                            graph_strength: get_f64(item, "graph_strength"),
                            token_overlap: get_f64(item, "token_overlap"),
                            temporal: get_f64(item, "temporal"),
                            tag_matching: get_f64(item, "tag_matching"),
                            importance: get_f64(item, "importance"),
                            confidence: get_f64(item, "confidence"),
                            recency: get_f64(item, "recency"),
                        },
                        tags: item
                            .get("tags")
                            .and_then(|t| t.as_array())
                            .map(|arr| {
                                arr.iter()
                                    .filter_map(|v| v.as_str().map(String::from))
                                    .collect()
                            })
                            .unwrap_or_default(),
                        namespace: item
                            .get("namespace")
                            .and_then(|n| n.as_str())
                            .map(String::from),
                    })
                })
                .collect();
        }
    }

    Vec::new()
}

fn get_f64(value: &serde_json::Value, key: &str) -> f64 {
    value
        .get("breakdown")
        .and_then(|b| b.get(key))
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0)
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
