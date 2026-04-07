//! Export, import, and index commands.

pub(crate) fn cmd_export(
    namespace: Option<&str>,
    memory_type: Option<&str>,
    output: Option<&std::path::Path>,
    format: &str,
) -> anyhow::Result<()> {
    let db_path = super::codemem_db_path();
    let storage = codemem_engine::Storage::open(&db_path)?;

    let memory_type_filter: Option<codemem_core::MemoryType> =
        memory_type.and_then(|s| s.parse().ok());

    let ids = match namespace {
        Some(ns) => storage.list_memory_ids_for_namespace(ns)?,
        None => storage.list_memory_ids()?,
    };

    // Collect all matching memories into JSON objects
    let mut records: Vec<serde_json::Value> = Vec::new();

    for id in &ids {
        if let Some(memory) = storage.get_memory_no_touch(id)? {
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

        let mut memory = codemem_core::MemoryNode::new(content, memory_type);
        memory.importance = importance;
        memory.confidence = confidence;
        memory.tags = tags;
        memory.metadata = metadata;
        memory.namespace = namespace;

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


#[cfg(test)]
#[path = "tests/commands_export_tests.rs"]
mod tests;
