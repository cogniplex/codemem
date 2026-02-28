//! Serve, ingest, and watch commands.

use codemem_core::VectorBackend;

pub(crate) fn cmd_serve() -> anyhow::Result<()> {
    let db_path = crate::codemem_db_path();

    // Task 5: Graceful error recovery for corrupt/missing ONNX model and stale vector index.
    // McpServer::from_db_path handles most of this, but we wrap it to catch vector index issues.
    let storage = codemem_storage::Storage::open(&db_path)?;
    let mut vector = codemem_vector::HnswIndex::with_defaults()?;

    let index_path = db_path.with_extension("idx");
    if index_path.exists() {
        if let Err(e) = vector.load(&index_path) {
            tracing::warn!("Stale or corrupt vector index, rebuilding: {e}");
            vector = crate::rebuild_vector_index(&storage)?;
            let _ = vector.save(&index_path);
        }
    }

    let graph = codemem_graph::GraphEngine::from_storage(&storage)?;

    // Try loading embeddings; log warning and continue without if unavailable
    let embeddings = match codemem_embeddings::from_env() {
        Ok(provider) => Some(provider),
        Err(e) => {
            tracing::warn!("Embedding provider unavailable, continuing without embeddings: {e}");
            None
        }
    };

    let server = codemem_mcp::McpServer::new(Box::new(storage), vector, graph, embeddings);
    tracing::info!(
        "Codemem MCP server ready (stdio mode, db: {})",
        db_path.display()
    );
    server.run()?;
    Ok(())
}

pub(crate) fn cmd_ingest() -> anyhow::Result<()> {
    use std::io::Read;

    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input)?;

    if input.trim().is_empty() {
        return Ok(());
    }

    let payload = codemem_hooks::parse_payload(&input)?;
    let extracted = codemem_hooks::extract(&payload)?;

    if let Some(mut extracted) = extracted {
        let db_path = crate::codemem_db_path();
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
        codemem_hooks::resolve_edges(&mut extracted, &existing_node_ids);

        // Dedup on raw content hash (before compression) for consistency
        let hash = codemem_hooks::content_hash(&extracted.content);
        let now = chrono::Utc::now();
        let id = uuid::Uuid::new_v4().to_string();

        // Compress observation via LLM if configured
        let compressor = crate::compress::CompressProvider::from_env();
        let tool_name = extracted
            .metadata
            .get("tool")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        let file_path = extracted.metadata.get("file_path").and_then(|v| v.as_str());
        let (content, compressed) =
            if let Some(summary) = compressor.compress(&extracted.content, tool_name, file_path) {
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
            namespace,
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

                // Auto-embed and index with error recovery (Task 5)
                match codemem_embeddings::from_env() {
                    Ok(emb_service) => {
                        if let Ok(embedding) = emb_service.embed(&content) {
                            let _ = storage.store_embedding(&id, &embedding);

                            // Load and update vector index
                            let index_path = db_path.with_extension("idx");
                            let mut vector = codemem_vector::HnswIndex::with_defaults()?;
                            if index_path.exists() {
                                if let Err(e) = vector.load(&index_path) {
                                    // Stale vector index: rebuild from stored embeddings
                                    tracing::warn!(
                                        "Stale vector index during ingest, rebuilding: {e}"
                                    );
                                    vector = crate::rebuild_vector_index(&storage)?;
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
                let edges = codemem_hooks::materialize_edges(&extracted.graph_edges, &id);
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

    let db_path = crate::codemem_db_path();
    let storage = codemem_storage::Storage::open(&db_path)?;

    let emb_service = codemem_embeddings::from_env().ok();

    let index_path = db_path.with_extension("idx");
    let mut vector = codemem_vector::HnswIndex::with_defaults()?;
    if index_path.exists() {
        let _ = vector.load(&index_path);
    }

    let watcher = codemem_watch::FileWatcher::new(watch_dir)?;
    println!(
        "Watching {} for file changes (Ctrl+C to stop)",
        watch_dir.display()
    );

    let receiver = watcher.receiver();
    let mut changes_since_save = 0usize;

    while let Ok(event) = receiver.recv() {
        let path = match &event {
            codemem_watch::WatchEvent::FileChanged(p)
            | codemem_watch::WatchEvent::FileCreated(p)
            | codemem_watch::WatchEvent::FileDeleted(p) => p.clone(),
        };

        let language = codemem_watch::detect_language(&path).unwrap_or("unknown");

        let relative = path
            .strip_prefix(watch_dir)
            .unwrap_or(&path)
            .to_string_lossy();

        match &event {
            codemem_watch::WatchEvent::FileDeleted(_) => {
                println!("  [deleted] {relative}");
            }
            _ => {
                // Index the changed file
                if let Ok(content) = std::fs::read_to_string(&path) {
                    let now = chrono::Utc::now();
                    let id = uuid::Uuid::new_v4().to_string();
                    let hash = codemem_storage::Storage::content_hash(&content);

                    let mut metadata = std::collections::HashMap::new();
                    metadata.insert(
                        "file_path".to_string(),
                        serde_json::Value::String(relative.to_string()),
                    );
                    metadata.insert(
                        "language".to_string(),
                        serde_json::Value::String(language.to_string()),
                    );
                    metadata.insert(
                        "source".to_string(),
                        serde_json::Value::String("file_watcher".to_string()),
                    );

                    let summary = format!(
                        "[{language}] File {} ({})",
                        relative,
                        match &event {
                            codemem_watch::WatchEvent::FileCreated(_) => "created",
                            _ => "modified",
                        }
                    );

                    let memory = codemem_core::MemoryNode {
                        id: id.clone(),
                        content: summary,
                        memory_type: codemem_core::MemoryType::Context,
                        importance: 0.3,
                        confidence: 1.0,
                        access_count: 0,
                        content_hash: hash,
                        tags: vec![language.to_string(), "file_watch".to_string()],
                        metadata,
                        namespace: Some(watch_dir.to_string_lossy().to_string()),
                        created_at: now,
                        updated_at: now,
                        last_accessed_at: now,
                    };

                    match storage.insert_memory(&memory) {
                        Ok(()) => {
                            // Auto-embed if available
                            if let Some(ref emb) = emb_service {
                                if let Ok(embedding) = emb.embed(&memory.content) {
                                    let _ = storage.store_embedding(&id, &embedding);
                                    let _ = vector.insert(&id, &embedding);
                                    changes_since_save += 1;
                                }
                            }
                            println!("  [indexed] {relative} ({language})");
                        }
                        Err(codemem_core::CodememError::Duplicate(_)) => {
                            // Skip duplicate content
                        }
                        Err(e) => {
                            tracing::warn!("Failed to store watch memory: {e}");
                        }
                    }
                }
            }
        }

        // Periodically save vector index
        if changes_since_save >= 10 {
            let _ = vector.save(&index_path);
            changes_since_save = 0;
        }
    }

    // Final save
    if changes_since_save > 0 {
        let _ = vector.save(&index_path);
    }

    Ok(())
}
