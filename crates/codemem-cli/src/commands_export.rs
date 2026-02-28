//! Export, import, and index commands.

use codemem_core::{StorageBackend, VectorBackend};
use std::io::Write;

pub(crate) fn cmd_export(
    namespace: Option<&str>,
    memory_type: Option<&str>,
    output: Option<&std::path::Path>,
) -> anyhow::Result<()> {
    let db_path = crate::codemem_db_path();
    let storage = codemem_storage::Storage::open(&db_path)?;

    let memory_type_filter: Option<codemem_core::MemoryType> =
        memory_type.and_then(|s| s.parse().ok());

    let ids = match namespace {
        Some(ns) => storage.list_memory_ids_for_namespace(ns)?,
        None => storage.list_memory_ids()?,
    };

    let mut writer: Box<dyn std::io::Write> = match output {
        Some(path) => Box::new(std::fs::File::create(path)?),
        None => Box::new(std::io::stdout()),
    };

    let mut count = 0usize;

    for id in &ids {
        if let Some(memory) = storage.get_memory(id)? {
            // Apply memory_type filter
            if let Some(ref filter_type) = memory_type_filter {
                if memory.memory_type != *filter_type {
                    continue;
                }
            }

            // Get edges for this memory
            let edges: Vec<serde_json::Value> = storage
                .get_edges_for_node(id)
                .unwrap_or_default()
                .iter()
                .map(|e| {
                    serde_json::json!({
                        "id": e.id,
                        "src": e.src,
                        "dst": e.dst,
                        "relationship": e.relationship.to_string(),
                        "weight": e.weight,
                    })
                })
                .collect();

            let obj = serde_json::json!({
                "id": memory.id,
                "content": memory.content,
                "memory_type": memory.memory_type.to_string(),
                "importance": memory.importance,
                "confidence": memory.confidence,
                "tags": memory.tags,
                "namespace": memory.namespace,
                "metadata": memory.metadata,
                "created_at": memory.created_at.to_rfc3339(),
                "updated_at": memory.updated_at.to_rfc3339(),
                "edges": edges,
            });

            // JSONL: one JSON object per line (compact)
            let line = serde_json::to_string(&obj)?;
            writeln!(writer, "{line}")?;
            count += 1;
        }
    }

    // Print count to stderr (so it doesn't mix with JSONL output on stdout)
    eprintln!("Exported {count} memories.");
    Ok(())
}

pub(crate) fn cmd_import(
    input: Option<&std::path::Path>,
    skip_duplicates: bool,
) -> anyhow::Result<()> {
    use std::io::BufRead;

    let db_path = crate::codemem_db_path();
    let storage = codemem_storage::Storage::open(&db_path)?;

    // Try loading embeddings for auto-embedding
    let emb_service = codemem_embeddings::from_env().ok();

    let mut vector = codemem_vector::HnswIndex::with_defaults()?;
    let index_path = db_path.with_extension("idx");
    if index_path.exists() {
        let _ = vector.load(&index_path);
    }

    let reader: Box<dyn BufRead> = match input {
        Some(path) => Box::new(std::io::BufReader::new(std::fs::File::open(path)?)),
        None => Box::new(std::io::BufReader::new(std::io::stdin())),
    };

    let mut imported = 0usize;
    let mut skipped = 0usize;

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let mem_val: serde_json::Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("Skipping invalid JSON line: {e}");
                skipped += 1;
                continue;
            }
        };

        let content = match mem_val.get("content").and_then(|v| v.as_str()) {
            Some(c) if !c.is_empty() => c.to_string(),
            _ => {
                eprintln!("Skipping line without content");
                skipped += 1;
                continue;
            }
        };

        let memory_type: codemem_core::MemoryType = mem_val
            .get("memory_type")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse().ok())
            .unwrap_or(codemem_core::MemoryType::Context);

        let importance = mem_val
            .get("importance")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.5);

        let confidence = mem_val
            .get("confidence")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0);

        let tags: Vec<String> = mem_val
            .get("tags")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        let namespace = mem_val
            .get("namespace")
            .and_then(|v| v.as_str())
            .map(String::from);

        let metadata: std::collections::HashMap<String, serde_json::Value> = mem_val
            .get("metadata")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or_default();

        let now = chrono::Utc::now();
        let id = uuid::Uuid::new_v4().to_string();
        let hash = codemem_storage::Storage::content_hash(&content);

        let memory = codemem_core::MemoryNode {
            id: id.clone(),
            content: content.clone(),
            memory_type,
            importance,
            confidence,
            access_count: 0,
            content_hash: hash,
            tags,
            metadata,
            namespace,
            created_at: now,
            updated_at: now,
            last_accessed_at: now,
        };

        match storage.insert_memory(&memory) {
            Ok(()) => {
                // Auto-embed if available
                if let Some(ref emb) = emb_service {
                    if let Ok(embedding) = emb.embed(&content) {
                        let _ = storage.store_embedding(&id, &embedding);
                        let _ = vector.insert(&id, &embedding);
                    }
                }
                imported += 1;
            }
            Err(codemem_core::CodememError::Duplicate(_)) => {
                if skip_duplicates {
                    skipped += 1;
                } else {
                    eprintln!("Duplicate content detected (use --skip-duplicates to ignore)");
                    skipped += 1;
                }
            }
            Err(e) => {
                eprintln!("Failed to import memory: {e}");
                skipped += 1;
            }
        }
    }

    // Save vector index if we embedded anything
    if imported > 0 && emb_service.is_some() {
        let _ = vector.save(&index_path);
    }

    eprintln!("Imported: {imported}, Skipped: {skipped}");
    Ok(())
}

pub(crate) fn cmd_index(root: &std::path::Path, verbose: bool) -> anyhow::Result<()> {
    let db_path = crate::codemem_db_path();
    let storage = codemem_storage::Storage::open(&db_path)?;

    // Load incremental state
    let mut change_detector = codemem_index::incremental::ChangeDetector::new();
    change_detector.load_from_storage(&storage);

    let mut indexer = codemem_index::Indexer::with_change_detector(change_detector);

    println!("Indexing {}...", root.display());
    let result = indexer.index_directory(root)?;

    println!("  Files scanned:  {}", result.files_scanned);
    println!("  Files parsed:   {}", result.files_parsed);
    println!("  Files skipped:  {}", result.files_skipped);
    println!("  Symbols found:  {}", result.total_symbols);
    println!("  References:     {}", result.total_references);

    // Collect all symbols and references
    let mut all_symbols = Vec::new();
    let mut all_references = Vec::new();
    for pr in &result.parse_results {
        all_symbols.extend(pr.symbols.clone());
        all_references.extend(pr.references.clone());
    }

    // Resolve references into edges
    let mut resolver = codemem_index::ReferenceResolver::new();
    resolver.add_symbols(&all_symbols);
    let edges = resolver.resolve_all(&all_references);
    println!("  Edges resolved: {}", edges.len());

    // Persist symbols as graph nodes (using batch insert for performance)
    let namespace = root.to_string_lossy().to_string();
    let now = chrono::Utc::now();

    print!("  Storing graph nodes...");
    std::io::Write::flush(&mut std::io::stdout()).ok();

    // Collect all nodes into a Vec for batch insert
    let graph_nodes: Vec<codemem_core::GraphNode> = all_symbols
        .iter()
        .map(|sym| {
            let kind = match sym.kind {
                codemem_index::SymbolKind::Function => codemem_core::NodeKind::Function,
                codemem_index::SymbolKind::Method => codemem_core::NodeKind::Method,
                codemem_index::SymbolKind::Class => codemem_core::NodeKind::Class,
                codemem_index::SymbolKind::Struct => codemem_core::NodeKind::Class,
                codemem_index::SymbolKind::Enum => codemem_core::NodeKind::Class,
                codemem_index::SymbolKind::Interface => codemem_core::NodeKind::Interface,
                codemem_index::SymbolKind::Type => codemem_core::NodeKind::Type,
                codemem_index::SymbolKind::Constant => codemem_core::NodeKind::Constant,
                codemem_index::SymbolKind::Module => codemem_core::NodeKind::Module,
                codemem_index::SymbolKind::Test => codemem_core::NodeKind::Test,
            };

            let mut payload = std::collections::HashMap::new();
            payload.insert(
                "signature".to_string(),
                serde_json::Value::String(sym.signature.clone()),
            );
            payload.insert(
                "file_path".to_string(),
                serde_json::Value::String(sym.file_path.clone()),
            );
            payload.insert("line_start".to_string(), serde_json::json!(sym.line_start));
            payload.insert("line_end".to_string(), serde_json::json!(sym.line_end));

            codemem_core::GraphNode {
                id: format!("sym:{}", sym.qualified_name),
                kind,
                label: sym.name.clone(),
                payload,
                centrality: 0.0,
                memory_id: None,
                namespace: Some(namespace.clone()),
            }
        })
        .collect();

    let nodes_stored = graph_nodes.len();
    storage.insert_graph_nodes_batch(&graph_nodes)?;

    // Collect all edges into a Vec for batch insert
    let graph_edges: Vec<codemem_core::Edge> = edges
        .iter()
        .map(|edge| codemem_core::Edge {
            id: format!(
                "ref:{}->{}:{}",
                edge.source_qualified_name, edge.target_qualified_name, edge.relationship
            ),
            src: format!("sym:{}", edge.source_qualified_name),
            dst: format!("sym:{}", edge.target_qualified_name),
            relationship: edge.relationship,
            weight: 1.0,
            properties: std::collections::HashMap::new(),
            created_at: now,
            valid_from: None,
            valid_to: None,
        })
        .collect();

    let edges_stored = graph_edges.len();
    storage.insert_graph_edges_batch(&graph_edges)?;

    println!(" done");
    println!("  Graph nodes:    {}", nodes_stored);
    println!("  Graph edges:    {}", edges_stored);

    // Embed symbol signatures for semantic code search
    let mut symbols_embedded = 0usize;
    if let Ok(emb_service) = codemem_embeddings::from_env() {
        let index_path = db_path.with_extension("idx");
        let mut vector = codemem_vector::HnswIndex::with_defaults()?;
        if index_path.exists() {
            let _ = vector.load(&index_path);
        }

        let total = all_symbols.len();
        println!("  Embedding {} symbols...", total);

        // Prepare all texts and IDs up front for batch embedding
        let embed_data: Vec<(String, String)> = all_symbols
            .iter()
            .map(|sym| {
                let mut text = format!("{}: {}", sym.qualified_name, sym.signature);
                if let Some(ref doc) = sym.doc_comment {
                    text.push('\n');
                    text.push_str(doc);
                }
                (format!("sym:{}", sym.qualified_name), text)
            })
            .collect();

        let mut all_sym_pairs: Vec<(String, Vec<f32>)> = Vec::new();
        for (batch_idx, chunk) in embed_data.chunks(32).enumerate() {
            let texts: Vec<&str> = chunk.iter().map(|(_, t)| t.as_str()).collect();
            if let Ok(embeddings) = emb_service.embed_batch(&texts) {
                for ((sym_id, _), embedding) in chunk.iter().zip(embeddings) {
                    let _ = vector.insert(sym_id, &embedding);
                    all_sym_pairs.push((sym_id.clone(), embedding));
                    symbols_embedded += 1;
                }
            }
            let done = (batch_idx + 1) * 32;
            print!("\r  Embedding symbols: {}/{}", done.min(total), total);
            std::io::Write::flush(&mut std::io::stdout()).ok();
        }

        // Batch store all symbol embeddings
        let sym_batch_refs: Vec<(&str, &[f32])> = all_sym_pairs
            .iter()
            .map(|(id, emb)| (id.as_str(), emb.as_slice()))
            .collect();
        let _ = storage.store_embeddings_batch(&sym_batch_refs);

        println!(); // newline after progress
        if symbols_embedded > 0 {
            let _ = vector.save(&index_path);
        }
        println!("  Symbols embedded: {}", symbols_embedded);
    }

    // Save incremental state
    indexer.change_detector().save_to_storage(&storage)?;

    if verbose {
        println!("\nSymbols:");
        for sym in &all_symbols {
            println!(
                "  {} {} [{}] {}:{}-{}",
                sym.visibility,
                sym.kind,
                sym.qualified_name,
                sym.file_path,
                sym.line_start,
                sym.line_end
            );
        }
    }

    println!("\nDone. Run `codemem stats` to see updated totals.");
    Ok(())
}
