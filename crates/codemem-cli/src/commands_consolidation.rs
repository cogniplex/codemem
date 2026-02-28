//! Consolidation cycle commands.

use codemem_core::{StorageBackend, VectorBackend};

pub(crate) fn cmd_consolidate(cycle: &str) -> anyhow::Result<()> {
    let db_path = crate::codemem_db_path();
    let storage = codemem_storage::Storage::open(&db_path)?;

    println!("Running {} consolidation cycle...", cycle);

    let affected = match cycle {
        "decay" => consolidate_decay(&storage)?,
        "creative" => consolidate_creative(&storage)?,
        "cluster" => consolidate_cluster(&storage, &db_path)?,
        "forget" => consolidate_forget(&storage, &db_path)?,
        _ => {
            anyhow::bail!(
                "Unknown cycle type: '{}'. Valid types: decay, creative, cluster, forget",
                cycle
            );
        }
    };

    // Log the consolidation run
    if let Err(e) = storage.insert_consolidation_log(cycle, affected) {
        tracing::warn!("Failed to log consolidation run: {e}");
    }

    Ok(())
}

pub(crate) fn cmd_consolidate_status() -> anyhow::Result<()> {
    let db_path = crate::codemem_db_path();
    let storage = codemem_storage::Storage::open(&db_path)?;
    let runs = storage.last_consolidation_runs()?;

    if runs.is_empty() {
        println!("No consolidation runs recorded yet.");
        return Ok(());
    }

    println!("Last consolidation runs:");
    for entry in &runs {
        let dt = chrono::DateTime::from_timestamp(entry.run_at, 0)
            .map(|t| t.format("%Y-%m-%d %H:%M:%S UTC").to_string())
            .unwrap_or_else(|| "unknown".to_string());
        println!(
            "  {:<10} last run: {}  ({} affected)",
            entry.cycle_type, dt, entry.affected_count
        );
    }

    // Show cycles that have never been run
    let all_cycles = ["decay", "creative", "cluster", "forget"];
    for cycle in &all_cycles {
        if !runs.iter().any(|r| r.cycle_type == *cycle) {
            println!("  {:<10} never run", cycle);
        }
    }

    Ok(())
}

/// Decay cycle: find memories not accessed in 30+ days, reduce importance by 10%.
fn consolidate_decay(storage: &codemem_storage::Storage) -> anyhow::Result<usize> {
    let threshold_ts = (chrono::Utc::now() - chrono::Duration::days(30)).timestamp();
    let count = storage.decay_stale_memories(threshold_ts, 0.9)?;
    tracing::info!("Decay cycle complete: {} memories affected", count);
    println!("Decayed {} memories (importance reduced by 10%).", count);
    Ok(count)
}

/// Creative cycle: find pairs of memories with similar tags but different types,
/// create RELATES_TO edges between them if not already connected.
fn consolidate_creative(storage: &codemem_storage::Storage) -> anyhow::Result<usize> {
    // Load all memories with their id, type, and tags
    let parsed = storage.list_memories_for_creative()?;

    // Load existing RELATES_TO edges to avoid duplicates
    let all_edges = storage.all_graph_edges()?;
    let existing_edges: std::collections::HashSet<(String, String)> = all_edges
        .into_iter()
        .filter(|e| e.relationship == codemem_core::RelationshipType::RelatesTo)
        .map(|e| (e.src, e.dst))
        .collect();

    let mut new_connections = 0usize;
    let now = chrono::Utc::now();

    // Collect nodes and edges to insert in batch
    let mut nodes_to_insert: Vec<codemem_core::GraphNode> = Vec::new();
    let mut edges_to_insert: Vec<codemem_core::Edge> = Vec::new();
    let mut inserted_node_ids: std::collections::HashSet<String> = std::collections::HashSet::new();

    for i in 0..parsed.len() {
        for j in (i + 1)..parsed.len() {
            let (ref id_a, ref type_a, ref tags_a) = parsed[i];
            let (ref id_b, ref type_b, ref tags_b) = parsed[j];

            // Different types required
            if type_a == type_b {
                continue;
            }

            // Must have at least one overlapping tag
            let has_common_tag = tags_a.iter().any(|t| tags_b.contains(t));
            if !has_common_tag {
                continue;
            }

            // Check not already connected in either direction
            if existing_edges.contains(&(id_a.clone(), id_b.clone()))
                || existing_edges.contains(&(id_b.clone(), id_a.clone()))
            {
                continue;
            }

            // Ensure both nodes exist in graph_nodes (upsert memory-type nodes)
            for id in [id_a, id_b] {
                if !inserted_node_ids.contains(id) {
                    nodes_to_insert.push(codemem_core::GraphNode {
                        id: id.clone(),
                        kind: codemem_core::NodeKind::Memory,
                        label: id.clone(),
                        payload: std::collections::HashMap::new(),
                        centrality: 0.0,
                        memory_id: Some(id.clone()),
                        namespace: None,
                    });
                    inserted_node_ids.insert(id.clone());
                }
            }

            let edge_id = format!("{id_a}-RELATES_TO-{id_b}");
            edges_to_insert.push(codemem_core::Edge {
                id: edge_id,
                src: id_a.clone(),
                dst: id_b.clone(),
                relationship: codemem_core::RelationshipType::RelatesTo,
                weight: 1.0,
                properties: std::collections::HashMap::new(),
                created_at: now,
                valid_from: None,
                valid_to: None,
            });

            new_connections += 1;
        }
    }

    // Batch insert nodes and edges
    if !nodes_to_insert.is_empty() {
        let _ = storage.insert_graph_nodes_batch(&nodes_to_insert);
    }
    if !edges_to_insert.is_empty() {
        let _ = storage.insert_graph_edges_batch(&edges_to_insert);
    }

    tracing::info!(
        "Creative cycle complete: {} new connections",
        new_connections
    );
    println!(
        "Creative cycle: created {} new RELATES_TO connections.",
        new_connections
    );
    Ok(new_connections)
}

/// Cluster cycle: find memories with the same content_hash prefix (first 8 chars)
/// and merge duplicates by keeping the one with highest importance, deleting others.
fn consolidate_cluster(
    storage: &codemem_storage::Storage,
    db_path: &std::path::Path,
) -> anyhow::Result<usize> {
    // Load all memories to extract id, content_hash, importance
    let all_memories = storage.list_memories_filtered(None, None)?;

    // Group by first 8 chars of content_hash
    let mut groups: std::collections::HashMap<String, Vec<(String, f64)>> =
        std::collections::HashMap::new();
    for mem in &all_memories {
        let prefix = if mem.content_hash.len() >= 8 {
            mem.content_hash[..8].to_string()
        } else {
            mem.content_hash.clone()
        };
        groups
            .entry(prefix)
            .or_default()
            .push((mem.id.clone(), mem.importance));
    }

    let mut merged_count = 0usize;
    let mut ids_to_delete: Vec<String> = Vec::new();

    for (_prefix, mut members) in groups {
        if members.len() <= 1 {
            continue;
        }

        // Sort by importance descending; keep the first (highest), delete the rest
        members.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        for (id, _importance) in members.iter().skip(1) {
            ids_to_delete.push(id.clone());
            merged_count += 1;
        }
    }

    // Delete the duplicates
    for id in &ids_to_delete {
        let _ = storage.delete_memory(id);
        // Also clean up embeddings
        let _ = storage.delete_embedding(id);
    }

    // Rebuild vector index if we deleted anything
    if merged_count > 0 {
        if let Ok(vector) = crate::rebuild_vector_index(storage) {
            let index_path = db_path.with_extension("idx");
            let _ = vector.save(&index_path);
        }
    }

    tracing::info!("Cluster cycle complete: {} duplicates merged", merged_count);
    println!(
        "Cluster cycle: merged {} duplicate memories (by content_hash prefix).",
        merged_count
    );
    Ok(merged_count)
}

/// Forget cycle: delete memories with importance < 0.1 and access_count == 0.
fn consolidate_forget(
    storage: &codemem_storage::Storage,
    db_path: &std::path::Path,
) -> anyhow::Result<usize> {
    // Find memories to forget
    let ids = storage.find_forgettable(0.1)?;

    let count = ids.len();

    for id in &ids {
        let _ = storage.delete_memory(id);
        let _ = storage.delete_embedding(id);
    }

    // Rebuild vector index if we deleted anything
    if count > 0 {
        if let Ok(vector) = crate::rebuild_vector_index(storage) {
            let index_path = db_path.with_extension("idx");
            let _ = vector.save(&index_path);
        }
    }

    tracing::info!("Forget cycle complete: {} memories deleted", count);
    println!(
        "Forget cycle: deleted {} forgotten memories (importance < 0.1, never accessed).",
        count
    );
    Ok(count)
}
