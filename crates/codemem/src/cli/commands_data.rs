//! Serve, ingest, and watch commands.

use codemem_core::GraphBackend;
use std::sync::Arc;

/// Build the shared server components via the engine's unified constructor.
fn build_server() -> anyhow::Result<crate::mcp::McpServer> {
    let db_path = super::codemem_db_path();
    let server = crate::mcp::McpServer::from_db_path(&db_path)?;
    Ok(server)
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
    let server = build_server()?;
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
            let api_server = crate::api::ApiServer::new(Arc::clone(&server));

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
            let api_server = crate::api::ApiServer::new(Arc::clone(&server));

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
            let api_server = crate::api::ApiServer::new(Arc::clone(&server));

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
        let engine = codemem_engine::CodememEngine::from_db_path(&db_path)?;

        // Build the set of existing graph node IDs so we can detect
        // Read-then-Edit/Write patterns and create edges.
        let existing_node_ids: std::collections::HashSet<String> = engine
            .storage()
            .all_graph_nodes()
            .unwrap_or_default()
            .into_iter()
            .map(|n| n.id)
            .collect();

        // Resolve edges based on previously-seen file nodes
        codemem_engine::hooks::resolve_edges(&mut extracted, &existing_node_ids);

        // Dedup on raw content hash (before compression) for consistency
        let hash = codemem_engine::hooks::content_hash(&extracted.content);

        // Compress observation via LLM if configured
        let compressor = codemem_engine::compress::CompressProvider::from_env();
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

        // Use directory basename as namespace (not full path)
        let namespace = std::env::current_dir()
            .ok()
            .map(|p| super::namespace_from_path(&p.to_string_lossy()).to_string());

        let mut memory = codemem_core::MemoryNode::new(content.clone(), extracted.memory_type);
        let id = memory.id.clone();
        memory.content_hash = hash;
        memory.tags = extracted.tags;
        memory.metadata = extracted.metadata;
        memory.namespace = namespace.clone();
        memory.session_id = extracted.session_id.clone();

        // Use engine.persist_memory() for the full pipeline (storage + BM25 + graph + embedding + vector)
        match engine.persist_memory(&memory) {
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
                        let _ = engine.storage().record_session_activity(
                            sid,
                            &tool_name,
                            file_path_owned.as_deref(),
                            directory.as_deref(),
                            search_pattern.as_deref(),
                        );

                        // Check triggers and store auto-insights
                        let auto_insights = codemem_engine::hooks::check_triggers(
                            engine.storage(),
                            sid,
                            &tool_name,
                            file_path_owned.as_deref(),
                            search_pattern.as_deref(),
                        );
                        for insight in &auto_insights {
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
                            let mut insight_memory = codemem_core::MemoryNode::new(
                                insight.content.clone(),
                                codemem_core::MemoryType::Insight,
                            );
                            insight_memory.importance = insight.importance;
                            insight_memory.confidence = 0.8;
                            insight_memory.tags = insight.tags.clone();
                            insight_memory.metadata = insight_metadata;
                            insight_memory.namespace = namespace.clone();
                            insight_memory.session_id = payload.session_id.clone();
                            match engine.persist_memory(&insight_memory) {
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

                // Store graph node if present (e.g., file nodes from hooks)
                if let Some(ref node) = extracted.graph_node {
                    let _ = engine.storage().insert_graph_node(node);
                    if let Ok(mut graph) = engine.lock_graph() {
                        let _ = graph.add_node(node.clone());
                    }
                }

                // Store any pending graph edges
                let edges = codemem_engine::hooks::materialize_edges(&extracted.graph_edges, &id);
                for edge in &edges {
                    match engine.add_edge(edge.clone()) {
                        Err(e) => {
                            tracing::debug!("Failed to store graph edge {}: {e}", edge.id);
                        }
                        Ok(()) => {
                            tracing::info!(
                                "Stored graph edge {} ({} -> {})",
                                edge.id,
                                edge.src,
                                edge.dst
                            );
                        }
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
pub(crate) const BATCH_WINDOW: std::time::Duration = std::time::Duration::from_secs(5);

/// Compute importance score for a batch of file changes.
/// Scales from 0.3 (1 file) to 0.8 (10+ files).
pub(crate) fn batch_importance(file_count: usize) -> f64 {
    (0.3 + (file_count as f64 * 0.05).min(0.5)).min(0.8)
}

/// Returns true when a batch of file changes is trivial
/// (1-2 files, all modifications, no creates/deletes).
pub(crate) fn is_trivial_batch(file_count: usize, created: usize, deleted: usize) -> bool {
    file_count <= 2 && created == 0 && deleted == 0
}

/// Core watch loop used by both `cmd_watch` (foreground) and `cmd_serve` (background).
///
/// Consolidates file change events within a 5-second window into a single
/// rich context memory instead of creating one memory per file.
///
/// Opens its own `CodememEngine` so it can run independently from the MCP
/// server without lock contention.
/// When `quiet` is true, uses `tracing::info!` instead of `println!`.
pub(crate) fn run_watcher_loop(
    db_path: &std::path::Path,
    watch_dir: &std::path::Path,
    quiet: bool,
) -> anyhow::Result<()> {
    let engine = codemem_engine::CodememEngine::from_db_path(db_path)?;

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
                changes_since_save += flush_batch(&batch, watch_dir, &engine, quiet);
                batch.clear();
            }

            continue;
        }

        // Timeout reached — flush the batch
        if !batch.is_empty() {
            changes_since_save += flush_batch(&batch, watch_dir, &engine, quiet);
            batch.clear();
        }

        // Periodically save vector index
        if changes_since_save >= 10 {
            engine.save_index();
            changes_since_save = 0;
        }
    }

    // Flush remaining batch
    if !batch.is_empty() {
        flush_batch(&batch, watch_dir, &engine, quiet);
    }

    // Final save
    if changes_since_save > 0 {
        engine.save_index();
    }

    Ok(())
}

/// Flush a batch of file changes into a single consolidated context memory.
fn flush_batch(
    batch: &[FileChange],
    watch_dir: &std::path::Path,
    engine: &codemem_engine::CodememEngine,
    quiet: bool,
) -> usize {
    if batch.is_empty() {
        return 0;
    }

    // Skip trivial changes (1-2 files, all just modifications) — these create noise.
    // Only store memories for significant batches (3+ files, or any created/deleted).
    let created = batch.iter().filter(|f| f.event_type == "created").count();
    let deleted = batch.iter().filter(|f| f.event_type == "deleted").count();
    if is_trivial_batch(batch.len(), created, deleted) {
        if !quiet {
            tracing::debug!(
                "[batch] Skipping trivial change ({} modified files)",
                batch.len()
            );
        }
        return 0;
    }

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
    let compressor = codemem_engine::compress::CompressProvider::from_env();
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
    let importance = batch_importance(batch.len());

    let mut memory = codemem_core::MemoryNode::new(content, codemem_core::MemoryType::Context);
    memory.id = id.clone();
    memory.importance = importance;
    memory.tags = tags;
    memory.metadata = metadata;
    memory.namespace = Some(super::namespace_from_path(&watch_dir.to_string_lossy()).to_string());

    match engine.persist_memory(&memory) {
        Ok(()) => {
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
            1
        }
        Err(codemem_core::CodememError::Duplicate(_)) => 0,
        Err(e) => {
            tracing::warn!("Failed to store watch batch memory: {e}");
            0
        }
    }
}

#[cfg(test)]
#[path = "tests/commands_data_tests.rs"]
mod tests;
