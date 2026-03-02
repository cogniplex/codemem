//! SSE event streams for real-time updates.

use axum::{
    extract::State,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse,
    },
};
use std::convert::Infallible;
use std::sync::Arc;
use tokio_stream::wrappers::BroadcastStream;
use tokio_stream::StreamExt;

use crate::AppState;

/// SSE stream for indexing progress events.
pub async fn indexing_events(
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    let rx = state.indexing_events.subscribe();
    let stream = BroadcastStream::new(rx).filter_map(|result| {
        match result {
            Ok(progress) => Some(Ok::<_, Infallible>(
                Event::default()
                    .event("indexing")
                    .json_data(serde_json::json!({
                        "files_scanned": progress.files_scanned,
                        "files_parsed": progress.files_parsed,
                        "total_symbols": progress.total_symbols,
                        "current_file": progress.current_file,
                    }))
                    .unwrap_or_else(|_| Event::default().data("error")),
            )),
            Err(_) => None,
        }
    });

    Sse::new(stream).keep_alive(KeepAlive::default())
}

/// SSE stream for file watcher events.
pub async fn watch_events(
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    let rx = state.watch_events.subscribe();
    let stream = BroadcastStream::new(rx).filter_map(|result| {
        match result {
            Ok(event) => {
                let (path, event_type) = match &event {
                    codemem_watch::WatchEvent::FileChanged(p) => {
                        (p.to_string_lossy().to_string(), "changed")
                    }
                    codemem_watch::WatchEvent::FileCreated(p) => {
                        (p.to_string_lossy().to_string(), "created")
                    }
                    codemem_watch::WatchEvent::FileDeleted(p) => {
                        (p.to_string_lossy().to_string(), "deleted")
                    }
                };
                Some(Ok::<_, Infallible>(
                    Event::default()
                        .event("watch")
                        .json_data(serde_json::json!({
                            "path": path,
                            "event_type": event_type,
                            "timestamp": chrono::Utc::now().to_rfc3339(),
                        }))
                        .unwrap_or_else(|_| Event::default().data("error")),
                ))
            }
            Err(_) => None,
        }
    });

    Sse::new(stream).keep_alive(KeepAlive::default())
}
