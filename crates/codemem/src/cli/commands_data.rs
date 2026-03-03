//! Serve, ingest, and watch commands.

use codemem_core::{StorageBackend, VectorBackend};
use std::sync::Arc;

/// Build the shared server components (storage, vector, graph, embeddings).
/// Returns (McpServer, storage_for_api) — the second is a separate
/// Storage handle suitable for async task boundaries in the API.
fn build_server() -> anyhow::Result<(crate::mcp::McpServer, codemem_storage::Storage)> {
    let db_path = super::codemem_db_path();

    let storage = codemem_storage::Storage::open(&db_path)?;
    let mut vector = codemem_storage::HnswIndex::with_defaults()?;

    let index_path = db_path.with_extension("idx");
    if index_path.exists() {
        if let Err(e) = vector.load(&index_path) {
            tracing::warn!("Stale or corrupt vector index, rebuilding: {e}");
            vector = super::rebuild_vector_index(&storage)?;
            let _ = vector.save(&index_path);
        }
    } else {
        // No .idx file — rebuild from stored embeddings if any exist
        let emb_count = storage.list_all_embeddings().map(|e| e.len()).unwrap_or(0);
        if emb_count > 0 {
            tracing::info!("No vector index file, rebuilding from {emb_count} stored embeddings");
            vector = super::rebuild_vector_index(&storage)?;
            let _ = vector.save(&index_path);
        }
    }

    let graph = codemem_storage::GraphEngine::from_storage(&storage)?;

    let embeddings = match codemem_embeddings::from_env() {
        Ok(provider) => Some(provider),
        Err(e) => {
            tracing::warn!("Embedding provider unavailable, continuing without embeddings: {e}");
            None
        }
    };

    let server = crate::mcp::McpServer::new(Box::new(storage), vector, graph, embeddings);

    // Open a second storage connection for the API layer (async-safe)
    let api_storage = codemem_storage::Storage::open(&db_path)?;

    Ok((server, api_storage))
}

fn start_background_watcher() {
    let db_path = super::codemem_db_path();
    if std::env::var("CODEMEM_NO_WATCH").as_deref() != Ok("1") {
        if let Ok(cwd) = std::env::current_dir() {
            tracing::info!("Background file watcher started for {}", cwd.display());
            std::thread::spawn(move || {
                if let Err(e) = run_watcher_loop(&db_path, &cwd, true) {
                    tracing::warn!("Background file watcher stopped: {e}");
                }
            });
        }
    }
}

pub(crate) fn cmd_serve(api: bool, http: bool, port: u16) -> anyhow::Result<()> {
    let (server, api_storage) = build_server()?;
    start_background_watcher();

    match (api, http) {
        // Pure stdio mode (backwards compatible)
        (false, false) => {
            tracing::info!(
                "Codemem MCP server ready (stdio mode, db: {})",
                super::codemem_db_path().display()
            );
            server.run()?;
        }
        // REST API + embedded frontend + stdio MCP
        (true, false) => {
            let server = Arc::new(server);
            let api_server = crate::api::ApiServer::new(Arc::clone(&server), api_storage);

            // Run stdio in a background thread, HTTP in the main tokio runtime
            let server_for_stdio = Arc::clone(&server);
            std::thread::spawn(move || {
                tracing::info!("stdio MCP transport running in background");
                if let Err(e) = server_for_stdio.run() {
                    tracing::warn!("stdio transport stopped: {e}");
                }
            });

            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(async {
                api_server
                    .serve(port, false)
                    .await
                    .map_err(|e| anyhow::anyhow!("API server error: {e}"))
            })?;
        }
        // HTTP MCP + REST API + embedded frontend (no stdio)
        (true, true) => {
            let server = Arc::new(server);
            let api_server = crate::api::ApiServer::new(Arc::clone(&server), api_storage);

            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(async {
                api_server
                    .serve(port, true)
                    .await
                    .map_err(|e| anyhow::anyhow!("API server error: {e}"))
            })?;
        }
        // HTTP MCP only (for remote MCP clients) — use the API server with MCP mounted
        (false, true) => {
            let server = Arc::new(server);
            let api_server = crate::api::ApiServer::new(Arc::clone(&server), api_storage);

            tracing::info!("Codemem MCP HTTP transport on http://localhost:{port}/mcp");
            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(async {
                api_server
                    .serve(port, true)
                    .await
                    .map_err(|e| anyhow::anyhow!("API server error: {e}"))
            })?;
        }
    }

    Ok(())
}

/// Convenience alias for `serve --api` with auto-open browser.
pub(crate) fn cmd_ui(port: u16, no_open: bool) -> anyhow::Result<()> {
    if !no_open {
        let url = format!("http://localhost:{port}");
        // Open browser in background before starting server
        std::thread::spawn(move || {
            // Brief delay to let the server start
            std::thread::sleep(std::time::Duration::from_millis(500));
            let _ = open_browser(&url);
        });
    }
    cmd_serve(true, false, port)
}

fn open_browser(url: &str) -> std::io::Result<()> {
    #[cfg(target_os = "macos")]
    std::process::Command::new("open").arg(url).spawn()?;
    #[cfg(target_os = "linux")]
    std::process::Command::new("xdg-open").arg(url).spawn()?;
    #[cfg(target_os = "windows")]
    std::process::Command::new("cmd")
        .args(["/c", "start", url])
        .spawn()?;
    Ok(())
}

pub(crate) fn cmd_ingest() -> anyhow::Result<()> {
    use std::io::BufRead;

    let mut input = String::new();
    let stdin = std::io::stdin();
    let _ = stdin.lock().read_line(&mut input);

    if input.trim().is_empty() {
        return Ok(());
    }

    let payload = codemem_engine::hooks::parse_payload(&input)?;
    let extracted = codemem_engine::hooks::extract(&payload)?;

    if let Some(mut extracted) = extracted {
        let db_path = super::codemem_db_path();
        let storage = codemem_storage::Storage::open(&db_path)?;

        // Build the set of existing graph node IDs so we can detect
        // Read-then-Edit/Write patterns and create edges.
        let existing_node_ids: std::collections::HashSet<String> = storage
            .all_graph_nodes()
            .unwrap_or_default()
            .into_iter()
            .map(|n| n.id)
            .collect();

        // Resolve edges based on previously-seen file nodes
        codemem_engine::hooks::resolve_edges(&mut extracted, &existing_node_ids);

        // Dedup on raw content hash (before compression) for consistency
        let hash = codemem_engine::hooks::content_hash(&extracted.content);
        let now = chrono::Utc::now();
        let id = uuid::Uuid::new_v4().to_string();

        // Compress observation via LLM if configured
        let compressor = super::compress::CompressProvider::from_env();
        let tool_name = extracted
            .metadata
            .get("tool")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();
        let file_path = extracted.metadata.get("file_path").and_then(|v| v.as_str());
        let search_pattern = extracted
            .metadata
            .get("pattern")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let file_path_owned = file_path.map(|s| s.to_string());
        let (content, compressed) =
            if let Some(summary) = compressor.compress(&extracted.content, &tool_name, file_path) {
                extracted
                    .metadata
                    .insert("compressed".to_string(), serde_json::Value::Bool(true));
                extracted.metadata.insert(
                    "original_len".to_string(),
                    serde_json::json!(extracted.content.len()),
                );
                (summary, true)
            } else {
                (extracted.content.clone(), false)
            };

        // Use current working directory as namespace
        let namespace = std::env::current_dir()
            .ok()
            .map(|p| p.to_string_lossy().to_string());

        let memory = codemem_core::MemoryNode {
            id: id.clone(),
            content: content.clone(),
            memory_type: extracted.memory_type,
            importance: 0.5,
            confidence: 1.0,
            access_count: 0,
            content_hash: hash,
            tags: extracted.tags,
            metadata: extracted.metadata,
            namespace: namespace.clone(),
            created_at: now,
            updated_at: now,
            last_accessed_at: now,
        };

        match storage.insert_memory(&memory) {
            Ok(()) => {
                tracing::info!(
                    "Stored memory {} ({}){}",
                    id,
                    memory.memory_type,
                    if compressed { " [compressed]" } else { "" }
                );

                // Record session activity for trigger-based auto-insights
                if let Some(sid) = payload.session_id.as_deref() {
                    if !sid.is_empty() {
                        let directory = file_path_owned.as_deref().and_then(|fp| {
                            std::path::Path::new(fp)
                                .parent()
                                .map(|p| p.to_string_lossy().to_string())
                        });
                        let _ = storage.record_session_activity(
                            sid,
                            &tool_name,
                            file_path_owned.as_deref(),
                            directory.as_deref(),
                            search_pattern.as_deref(),
                        );

                        // Check triggers and store auto-insights
                        let auto_insights = codemem_engine::hooks::check_triggers(
                            &storage,
                            sid,
                            &tool_name,
                            file_path_owned.as_deref(),
                            search_pattern.as_deref(),
                        );
                        for insight in &auto_insights {
                            let insight_hash =
                                codemem_engine::hooks::content_hash(&insight.content);
                            let insight_now = chrono::Utc::now();
                            let mut insight_metadata = std::collections::HashMap::new();
                            insight_metadata.insert(
                                "session_id".to_string(),
                                serde_json::Value::String(sid.to_string()),
                            );
                            insight_metadata.insert(
                                "auto_insight_tag".to_string(),
                                serde_json::Value::String(insight.dedup_tag.clone()),
                            );
                            insight_metadata.insert(
                                "source".to_string(),
                                serde_json::Value::String("auto_insight".to_string()),
                            );
                            let insight_memory = codemem_core::MemoryNode {
                                id: uuid::Uuid::new_v4().to_string(),
                                content: insight.content.clone(),
                                memory_type: codemem_core::MemoryType::Insight,
                                importance: insight.importance,
                                confidence: 0.8,
                                access_count: 0,
                                content_hash: insight_hash,
                                tags: insight.tags.clone(),
                                metadata: insight_metadata,
                                namespace: namespace.clone(),
                                created_at: insight_now,
                                updated_at: insight_now,
                                last_accessed_at: insight_now,
                            };
                            match storage.insert_memory(&insight_memory) {
                                Ok(()) => {
                                    tracing::info!("Auto-insight stored: {}", insight.dedup_tag);
                                }
                                Err(codemem_core::CodememError::Duplicate(_)) => {}
                                Err(e) => {
                                    tracing::debug!("Failed to store auto-insight: {e}");
                                }
                            }
                        }
                    }
                }

                // Auto-embed and index with error recovery (Task 5)
                match codemem_embeddings::from_env() {
                    Ok(emb_service) => {
                        if let Ok(embedding) = emb_service.embed(&content) {
                            let _ = storage.store_embedding(&id, &embedding);

                            // Load and update vector index
                            let index_path = db_path.with_extension("idx");
                            let mut vector = codemem_storage::HnswIndex::with_defaults()?;
                            if index_path.exists() {
                                if let Err(e) = vector.load(&index_path) {
                                    // Stale vector index: rebuild from stored embeddings
                                    tracing::warn!(
                                        "Stale vector index during ingest, rebuilding: {e}"
                                    );
                                    vector = super::rebuild_vector_index(&storage)?;
                                }
                            }
                            if vector.insert(&id, &embedding).is_ok() {
                                let _ = vector.save(&index_path);
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!("Embedding model unavailable, skipping embedding: {e}");
                    }
                }

                // Store graph node if present
                if let Some(ref node) = extracted.graph_node {
                    let _ = storage.insert_graph_node(node);
                }

                // Store any pending graph edges
                let edges = codemem_engine::hooks::materialize_edges(&extracted.graph_edges, &id);
                for edge in &edges {
                    if let Err(e) = storage.insert_graph_edge(edge) {
                        tracing::debug!("Failed to store graph edge {}: {e}", edge.id);
                    } else {
                        tracing::info!(
                            "Stored graph edge {} ({} -> {})",
                            edge.id,
                            edge.src,
                            edge.dst
                        );
                    }
                }
            }
            Err(codemem_core::CodememError::Duplicate(_)) => {
                tracing::debug!("Skipped duplicate content");
            }
            Err(e) => return Err(e.into()),
        }
    }

    Ok(())
}

// ── Watch Command ─────────────────────────────────────────────────────────

pub(crate) fn cmd_watch(watch_dir: &std::path::Path) -> anyhow::Result<()> {
    if !watch_dir.is_dir() {
        anyhow::bail!("Not a directory: {}", watch_dir.display());
    }

    println!(
        "Watching {} for file changes (Ctrl+C to stop)",
        watch_dir.display()
    );
    run_watcher_loop(&super::codemem_db_path(), watch_dir, false)
}

/// A single file change event with metadata collected from the filesystem.
struct FileChange {
    relative_path: String,
    language: String,
    event_type: &'static str, // "created", "modified", "deleted"
    line_count: Option<usize>,
    byte_size: Option<u64>,
    /// Parent directory (module/package grouping)
    directory: String,
}

/// Batch window duration: events within this window are consolidated.
const BATCH_WINDOW: std::time::Duration = std::time::Duration::from_secs(5);

/// Core watch loop used by both `cmd_watch` (foreground) and `cmd_serve` (background).
///
/// Consolidates file change events within a 5-second window into a single
/// rich context memory instead of creating one memory per file.
///
/// Opens its own `Storage`, `HnswIndex`, and embedding provider so it can run
/// independently from the MCP server without lock contention.
/// When `quiet` is true, uses `tracing::info!` instead of `println!`.
pub(crate) fn run_watcher_loop(
    db_path: &std::path::Path,
    watch_dir: &std::path::Path,
    quiet: bool,
) -> anyhow::Result<()> {
    let storage = codemem_storage::Storage::open(db_path)?;

    let emb_service = codemem_embeddings::from_env().ok();

    let index_path = db_path.with_extension("idx");
    let mut vector = codemem_storage::HnswIndex::with_defaults()?;
    if index_path.exists() {
        let _ = vector.load(&index_path);
    }

    let watcher = codemem_engine::watch::FileWatcher::new(watch_dir)?;

    let receiver = watcher.receiver();
    let mut changes_since_save = 0usize;
    let mut batch: Vec<FileChange> = Vec::new();

    loop {
        // If we have a batch in progress, use a timeout; otherwise block
        let event = if batch.is_empty() {
            match receiver.recv() {
                Ok(e) => Some(e),
                Err(_) => break,
            }
        } else {
            match receiver.recv_timeout(BATCH_WINDOW) {
                Ok(e) => Some(e),
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => None,
                Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
            }
        };

        if let Some(event) = event {
            let (path, event_type) = match &event {
                codemem_engine::watch::WatchEvent::FileChanged(p) => (p.clone(), "modified"),
                codemem_engine::watch::WatchEvent::FileCreated(p) => (p.clone(), "created"),
                codemem_engine::watch::WatchEvent::FileDeleted(p) => (p.clone(), "deleted"),
            };

            let language = codemem_engine::watch::detect_language(&path)
                .unwrap_or("unknown")
                .to_string();

            let relative = path
                .strip_prefix(watch_dir)
                .unwrap_or(&path)
                .to_string_lossy()
                .to_string();

            let directory = std::path::Path::new(&relative)
                .parent()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_else(|| ".".to_string());

            // Collect file metadata
            let (line_count, byte_size) = if event_type != "deleted" {
                let byte_size = std::fs::metadata(&path).ok().map(|m| m.len());
                let line_count = std::fs::read_to_string(&path)
                    .ok()
                    .map(|c| c.lines().count());
                (line_count, byte_size)
            } else {
                (None, None)
            };

            if !quiet {
                println!("  [{event_type}] {relative} ({language})");
            }

            batch.push(FileChange {
                relative_path: relative,
                language,
                event_type,
                line_count,
                byte_size,
                directory,
            });

            // If batch is getting large, flush early
            if batch.len() >= 50 {
                changes_since_save += flush_batch(
                    &batch,
                    watch_dir,
                    &storage,
                    &emb_service,
                    &mut vector,
                    quiet,
                );
                batch.clear();
            }

            continue;
        }

        // Timeout reached — flush the batch
        if !batch.is_empty() {
            changes_since_save += flush_batch(
                &batch,
                watch_dir,
                &storage,
                &emb_service,
                &mut vector,
                quiet,
            );
            batch.clear();
        }

        // Periodically save vector index
        if changes_since_save >= 10 {
            let _ = vector.save(&index_path);
            changes_since_save = 0;
        }
    }

    // Flush remaining batch
    if !batch.is_empty() {
        flush_batch(
            &batch,
            watch_dir,
            &storage,
            &emb_service,
            &mut vector,
            quiet,
        );
    }

    // Final save
    if changes_since_save > 0 {
        let _ = vector.save(&index_path);
    }

    Ok(())
}

/// Flush a batch of file changes into a single consolidated context memory.
fn flush_batch(
    batch: &[FileChange],
    watch_dir: &std::path::Path,
    storage: &codemem_storage::Storage,
    emb_service: &Option<Box<dyn codemem_embeddings::EmbeddingProvider>>,
    vector: &mut codemem_storage::HnswIndex,
    quiet: bool,
) -> usize {
    if batch.is_empty() {
        return 0;
    }

    // Skip trivial changes (1-2 files, all just modifications) — these create noise.
    // Only store memories for significant batches (3+ files, or any created/deleted).
    let created = batch.iter().filter(|f| f.event_type == "created").count();
    let deleted = batch.iter().filter(|f| f.event_type == "deleted").count();
    if batch.len() <= 2 && created == 0 && deleted == 0 {
        if !quiet {
            tracing::debug!(
                "[batch] Skipping trivial change ({} modified files)",
                batch.len()
            );
        }
        return 0;
    }

    let now = chrono::Utc::now();
    let id = uuid::Uuid::new_v4().to_string();

    // Count by event type (created/deleted already computed above for threshold check)
    let modified = batch.iter().filter(|f| f.event_type == "modified").count();

    // Group by language
    let mut lang_counts: std::collections::BTreeMap<&str, usize> =
        std::collections::BTreeMap::new();
    for f in batch {
        *lang_counts.entry(&f.language).or_insert(0) += 1;
    }

    // Group by directory
    let mut dir_counts: std::collections::BTreeMap<&str, usize> = std::collections::BTreeMap::new();
    for f in batch {
        *dir_counts.entry(&f.directory).or_insert(0) += 1;
    }

    // Total lines/bytes across non-deleted files
    let total_lines: usize = batch.iter().filter_map(|f| f.line_count).sum();
    let total_bytes: u64 = batch.iter().filter_map(|f| f.byte_size).sum();

    // Build a rich summary
    let mut summary_parts = Vec::new();

    // Header
    let mut changes_desc = Vec::new();
    if created > 0 {
        changes_desc.push(format!("{created} created"));
    }
    if modified > 0 {
        changes_desc.push(format!("{modified} modified"));
    }
    if deleted > 0 {
        changes_desc.push(format!("{deleted} deleted"));
    }
    summary_parts.push(format!(
        "File changes: {} files ({})",
        batch.len(),
        changes_desc.join(", ")
    ));

    // Languages
    let lang_list: Vec<String> = lang_counts
        .iter()
        .map(|(l, c)| format!("{l}: {c}"))
        .collect();
    summary_parts.push(format!("Languages: {}", lang_list.join(", ")));

    // Directories (top 5)
    let mut dir_sorted: Vec<_> = dir_counts.iter().collect();
    dir_sorted.sort_by(|a, b| b.1.cmp(a.1));
    let dir_list: Vec<String> = dir_sorted
        .iter()
        .take(5)
        .map(|(d, c)| format!("{d}/ ({c})"))
        .collect();
    summary_parts.push(format!("Directories: {}", dir_list.join(", ")));

    // File list
    let file_list: Vec<String> = batch
        .iter()
        .take(20)
        .map(|f| {
            let info = match f.event_type {
                "deleted" => "deleted".to_string(),
                _ => {
                    let lines = f
                        .line_count
                        .map(|l| format!("{l} lines"))
                        .unwrap_or_default();
                    let size = f.byte_size.map(|b| format!("{b}B")).unwrap_or_default();
                    [lines, size]
                        .iter()
                        .filter(|s| !s.is_empty())
                        .cloned()
                        .collect::<Vec<_>>()
                        .join(", ")
                }
            };
            format!("  {} [{}] {}", f.event_type, f.language, f.relative_path)
                + if info.is_empty() {
                    String::new()
                } else {
                    format!(" ({info})")
                }
                .as_str()
        })
        .collect();
    summary_parts.push(format!("Files:\n{}", file_list.join("\n")));
    if batch.len() > 20 {
        summary_parts.push(format!("  ... and {} more", batch.len() - 20));
    }

    let raw_summary = summary_parts.join("\n");

    // Try LLM summarization for a more meaningful description
    let compressor = super::compress::CompressProvider::from_env();
    let file_list_brief: Vec<&str> = batch
        .iter()
        .take(10)
        .map(|f| f.relative_path.as_str())
        .collect();
    let content = if let Some(summary) = compressor.summarize_batch(&raw_summary) {
        format!("{summary}\n\n---\nFiles: {}", file_list_brief.join(", "))
    } else {
        raw_summary
    };

    // Build rich metadata
    let mut metadata = std::collections::HashMap::new();
    metadata.insert("source".to_string(), serde_json::json!("file_watcher"));
    metadata.insert("file_count".to_string(), serde_json::json!(batch.len()));
    metadata.insert("created_count".to_string(), serde_json::json!(created));
    metadata.insert("modified_count".to_string(), serde_json::json!(modified));
    metadata.insert("deleted_count".to_string(), serde_json::json!(deleted));
    metadata.insert("total_lines".to_string(), serde_json::json!(total_lines));
    metadata.insert("total_bytes".to_string(), serde_json::json!(total_bytes));
    metadata.insert(
        "languages".to_string(),
        serde_json::json!(lang_counts
            .iter()
            .map(|(k, v)| (k.to_string(), *v))
            .collect::<std::collections::HashMap<String, usize>>()),
    );
    metadata.insert(
        "directories".to_string(),
        serde_json::json!(dir_counts
            .iter()
            .map(|(k, v)| (k.to_string(), *v))
            .collect::<std::collections::HashMap<String, usize>>()),
    );
    metadata.insert(
        "files".to_string(),
        serde_json::json!(batch
            .iter()
            .map(|f| f.relative_path.clone())
            .collect::<Vec<_>>()),
    );

    // Collect tags from all languages + directories
    let mut tags: Vec<String> = vec!["file_watch".to_string()];
    for lang in lang_counts.keys() {
        if *lang != "unknown" {
            tags.push(lang.to_string());
        }
    }
    // Add top directory as tag
    if let Some((top_dir, _)) = dir_sorted.first() {
        let dir_tag = top_dir.replace('/', "::");
        if !dir_tag.is_empty() && dir_tag != "." {
            tags.push(dir_tag);
        }
    }

    // Importance scales with batch size: more files = more significant change
    let importance = (0.3 + (batch.len() as f64 * 0.05).min(0.5)).min(0.8);

    let hash = codemem_storage::Storage::content_hash(&content);

    let memory = codemem_core::MemoryNode {
        id: id.clone(),
        content,
        memory_type: codemem_core::MemoryType::Context,
        importance,
        confidence: 1.0,
        access_count: 0,
        content_hash: hash,
        tags,
        metadata,
        namespace: Some(watch_dir.to_string_lossy().to_string()),
        created_at: now,
        updated_at: now,
        last_accessed_at: now,
    };

    let mut embedded = 0usize;
    match storage.insert_memory(&memory) {
        Ok(()) => {
            if let Some(ref emb) = emb_service {
                if let Ok(embedding) = emb.embed(&memory.content) {
                    let _ = storage.store_embedding(&id, &embedding);
                    let _ = vector.insert(&id, &embedding);
                    embedded = 1;
                }
            }
            if quiet {
                tracing::info!(
                    "[batch] {} files consolidated into memory {}",
                    batch.len(),
                    id
                );
            } else {
                println!(
                    "  [batch] {} files consolidated into memory {}",
                    batch.len(),
                    id
                );
            }
        }
        Err(codemem_core::CodememError::Duplicate(_)) => {}
        Err(e) => {
            tracing::warn!("Failed to store watch batch memory: {e}");
        }
    }

    embedded
}
