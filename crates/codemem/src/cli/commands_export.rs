//! Export, import, and index commands.

use codemem_core::VectorBackend;

pub(crate) fn cmd_export(
    namespace: Option<&str>,
    memory_type: Option<&str>,
    output: Option<&std::path::Path>,
    format: &str,
) -> anyhow::Result<()> {
    let db_path = super::codemem_db_path();
    let storage = codemem_storage::Storage::open(&db_path)?;

    let memory_type_filter: Option<codemem_core::MemoryType> =
        memory_type.and_then(|s| s.parse().ok());

    let ids = match namespace {
        Some(ns) => storage.list_memory_ids_for_namespace(ns)?,
        None => storage.list_memory_ids()?,
    };

    // Collect all matching memories into JSON objects
    let mut records: Vec<serde_json::Value> = Vec::new();

    for id in &ids {
        if let Some(memory) = storage.get_memory(id)? {
            if let Some(ref filter_type) = memory_type_filter {
                if memory.memory_type != *filter_type {
                    continue;
                }
            }

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

            records.push(serde_json::json!({
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
            }));
        }
    }

    let mut writer: Box<dyn std::io::Write> = match output {
        Some(path) => Box::new(std::fs::File::create(path)?),
        None => Box::new(std::io::stdout()),
    };

    let count = records.len();

    match format {
        "jsonl" => write_jsonl(&mut writer, &records)?,
        "json" => write_json(&mut writer, &records)?,
        "csv" => write_csv(&mut writer, &records)?,
        "markdown" | "md" => write_markdown(&mut writer, &records, namespace, memory_type)?,
        other => {
            anyhow::bail!("Unknown format: {other}. Supported formats: jsonl, json, csv, markdown");
        }
    }

    eprintln!("Exported {count} memories.");
    Ok(())
}

/// RFC 4180 CSV field escaping: double-quote fields containing commas, CR/LF, or quotes.
fn csv_escape(field: &str) -> String {
    if field.contains(',') || field.contains('\n') || field.contains('\r') || field.contains('"') {
        format!("\"{}\"", field.replace('"', "\"\""))
    } else {
        field.to_string()
    }
}

fn write_jsonl(
    writer: &mut dyn std::io::Write,
    records: &[serde_json::Value],
) -> anyhow::Result<()> {
    for obj in records {
        let line = serde_json::to_string(obj)?;
        writeln!(writer, "{line}")?;
    }
    Ok(())
}

fn write_json(
    writer: &mut dyn std::io::Write,
    records: &[serde_json::Value],
) -> anyhow::Result<()> {
    let pretty = serde_json::to_string_pretty(records)?;
    writeln!(writer, "{pretty}")?;
    Ok(())
}

fn write_csv(writer: &mut dyn std::io::Write, records: &[serde_json::Value]) -> anyhow::Result<()> {
    writeln!(
        writer,
        "id,content,memory_type,importance,confidence,tags,namespace,created_at,updated_at"
    )?;
    for obj in records {
        writeln!(
            writer,
            "{},{},{},{},{},{},{},{},{}",
            csv_escape(obj["id"].as_str().unwrap_or("")),
            csv_escape(obj["content"].as_str().unwrap_or("")),
            csv_escape(obj["memory_type"].as_str().unwrap_or("")),
            obj["importance"].as_f64().unwrap_or(0.0),
            obj["confidence"].as_f64().unwrap_or(0.0),
            csv_escape(
                &obj["tags"]
                    .as_array()
                    .map(|a| a
                        .iter()
                        .filter_map(|v| v.as_str())
                        .collect::<Vec<_>>()
                        .join(";"))
                    .unwrap_or_default()
            ),
            csv_escape(obj["namespace"].as_str().unwrap_or("")),
            csv_escape(obj["created_at"].as_str().unwrap_or("")),
            csv_escape(obj["updated_at"].as_str().unwrap_or("")),
        )?;
    }
    Ok(())
}

fn write_markdown(
    writer: &mut dyn std::io::Write,
    records: &[serde_json::Value],
    namespace: Option<&str>,
    memory_type: Option<&str>,
) -> anyhow::Result<()> {
    let count = records.len();
    writeln!(writer, "# Codemem Export")?;
    writeln!(writer)?;
    writeln!(writer, "**Total memories:** {count}")?;
    if let Some(ns) = namespace {
        writeln!(writer, "**Namespace:** {ns}")?;
    }
    if let Some(mt) = memory_type {
        writeln!(writer, "**Type filter:** {mt}")?;
    }
    writeln!(writer)?;

    // Group by memory type
    let mut by_type: std::collections::BTreeMap<String, Vec<&serde_json::Value>> =
        std::collections::BTreeMap::new();
    for obj in records {
        let mt = obj["memory_type"].as_str().unwrap_or("unknown").to_string();
        by_type.entry(mt).or_default().push(obj);
    }

    for (mt, memories) in &by_type {
        writeln!(writer, "## {} ({} memories)", mt, memories.len())?;
        writeln!(writer)?;
        for obj in memories {
            let id = obj["id"].as_str().unwrap_or("?");
            let content = obj["content"].as_str().unwrap_or("");
            let importance = obj["importance"].as_f64().unwrap_or(0.0);
            let tags = obj["tags"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                })
                .unwrap_or_default();

            writeln!(writer, "### `{id}`")?;
            writeln!(writer)?;
            writeln!(writer, "- **Importance:** {importance:.2}")?;
            if !tags.is_empty() {
                writeln!(writer, "- **Tags:** {tags}")?;
            }
            writeln!(writer)?;
            writeln!(writer, "{content}")?;
            writeln!(writer)?;
            writeln!(writer, "---")?;
            writeln!(writer)?;
        }
    }
    Ok(())
}

pub(crate) fn cmd_import(
    input: Option<&std::path::Path>,
    skip_duplicates: bool,
) -> anyhow::Result<()> {
    use std::io::BufRead;

    let db_path = super::codemem_db_path();
    let engine = codemem_engine::CodememEngine::from_db_path(&db_path)?;

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
            id,
            content,
            memory_type,
            importance,
            confidence,
            access_count: 0,
            content_hash: hash,
            tags,
            metadata,
            namespace,
            session_id: None,
            created_at: now,
            updated_at: now,
            last_accessed_at: now,
        };

        match engine.persist_memory(&memory) {
            Ok(()) => {
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

    eprintln!("Imported: {imported}, Skipped: {skipped}");
    Ok(())
}

pub(crate) fn cmd_index(root: &std::path::Path, verbose: bool) -> anyhow::Result<()> {
    let db_path = super::codemem_db_path();
    let engine = codemem_engine::CodememEngine::from_db_path(&db_path)?;

    // Load incremental state
    let mut change_detector = codemem_engine::index::incremental::ChangeDetector::new();
    change_detector.load_from_storage(engine.storage());

    let mut indexer = codemem_engine::Indexer::with_change_detector(change_detector);

    println!("Indexing {}...", root.display());
    let resolved = indexer.index_and_resolve(root)?;

    println!("  Files scanned:  {}", resolved.index.files_scanned);
    println!("  Files parsed:   {}", resolved.index.files_parsed);
    println!("  Files skipped:  {}", resolved.index.files_skipped);
    println!("  Symbols found:  {}", resolved.index.total_symbols);
    println!("  References:     {}", resolved.index.total_references);
    println!("  Edges resolved: {}", resolved.edges.len());

    let all_symbols = resolved.symbols;
    let edges = resolved.edges;

    // Persist symbols as graph nodes (using batch insert for performance)
    let namespace = root
        .file_name()
        .and_then(|f| f.to_str())
        .unwrap_or_else(|| root.to_str().unwrap_or("unknown"))
        .to_string();
    let now = chrono::Utc::now();

    print!("  Storing graph nodes...");
    std::io::Write::flush(&mut std::io::stdout()).ok();

    // Collect all nodes into a Vec for batch insert
    let graph_nodes: Vec<codemem_core::GraphNode> = all_symbols
        .iter()
        .map(|sym| {
            let kind = codemem_core::NodeKind::from(sym.kind);

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
    engine.storage().insert_graph_nodes_batch(&graph_nodes)?;

    // Collect all edges, filtering out edges that reference nodes not in the graph
    // (e.g., external stdlib symbols that weren't indexed)
    let node_ids: std::collections::HashSet<String> =
        graph_nodes.iter().map(|n| n.id.clone()).collect();
    let graph_edges: Vec<codemem_core::Edge> = edges
        .iter()
        .filter(|edge| {
            let src = format!("sym:{}", edge.source_qualified_name);
            let dst = format!("sym:{}", edge.target_qualified_name);
            node_ids.contains(&src) && node_ids.contains(&dst)
        })
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
    engine.storage().insert_graph_edges_batch(&graph_edges)?;

    println!(" done");
    println!("  Graph nodes:    {}", nodes_stored);
    println!("  Graph edges:    {}", edges_stored);

    // Embed symbol signatures for semantic code search
    let mut symbols_embedded = 0usize;
    if let Ok(Some(emb_guard)) = engine.lock_embeddings() {
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
            if let Ok(embeddings) = emb_guard.embed_batch(&texts) {
                for ((sym_id, _), embedding) in chunk.iter().zip(embeddings) {
                    if let Ok(mut vec) = engine.lock_vector() {
                        let _ = vec.insert(sym_id, &embedding);
                    }
                    all_sym_pairs.push((sym_id.clone(), embedding));
                    symbols_embedded += 1;
                }
            }
            let done = (batch_idx + 1) * 32;
            print!("\r  Embedding symbols: {}/{}", done.min(total), total);
            std::io::Write::flush(&mut std::io::stdout()).ok();
        }
        drop(emb_guard);

        // Batch store all symbol embeddings
        let sym_batch_refs: Vec<(&str, &[f32])> = all_sym_pairs
            .iter()
            .map(|(id, emb)| (id.as_str(), emb.as_slice()))
            .collect();
        let _ = engine.storage().store_embeddings_batch(&sym_batch_refs);

        println!(); // newline after progress
        if symbols_embedded > 0 {
            engine.save_index();
        }
        println!("  Symbols embedded: {}", symbols_embedded);
    }

    // Save incremental state
    indexer
        .change_detector()
        .save_to_storage(engine.storage())?;

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

#[cfg(test)]
#[path = "tests/commands_export_tests.rs"]
mod tests;
