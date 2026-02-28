//! Search & stats commands.

use codemem_core::{StorageBackend, VectorBackend};

pub(crate) fn cmd_search(query: &str, k: usize, namespace: Option<&str>) -> anyhow::Result<()> {
    let db_path = crate::codemem_db_path();
    let storage = codemem_storage::Storage::open(&db_path)?;

    // Try loading embeddings for vector search
    let emb_service = codemem_embeddings::from_env().ok();

    let mut vector = codemem_vector::HnswIndex::with_defaults()?;
    let index_path = db_path.with_extension("idx");
    if index_path.exists() {
        vector.load(&index_path)?;
    }

    // Try vector search first
    let vector_results: Vec<(String, f32)> = if let Some(ref emb) = emb_service {
        match emb.embed(query) {
            Ok(query_embedding) => vector.search(&query_embedding, k * 2).unwrap_or_default(),
            Err(_) => vec![],
        }
    } else {
        vec![]
    };

    if !vector_results.is_empty() {
        println!("Top {} results for: \"{}\" (vector search)\n", k, query);
        let mut shown = 0;
        for (id, distance) in &vector_results {
            if shown >= k {
                break;
            }
            let similarity = 1.0 - *distance as f64;
            if let Some(memory) = storage.get_memory(id)? {
                // Apply namespace filter
                if let Some(ns) = namespace {
                    if memory.namespace.as_deref() != Some(ns) {
                        continue;
                    }
                }
                println!(
                    "  [{:.3}] [{}] {}",
                    similarity, memory.memory_type, memory.id
                );
                println!("         {}", crate::truncate_str(&memory.content, 120));
                if !memory.tags.is_empty() {
                    println!("         tags: {}", memory.tags.join(", "));
                }
                println!();
                shown += 1;
            } else if let Some(node) = storage.get_graph_node(id)? {
                // Fallback: symbol/graph node (e.g. sym:* IDs from code indexing)
                if let Some(ns) = namespace {
                    if node.namespace.as_deref() != Some(ns) {
                        continue;
                    }
                }
                println!("  [{:.3}] [{}] {}", similarity, node.kind, node.id);
                println!("         {}", crate::truncate_str(&node.label, 120));
                println!();
                shown += 1;
            }
        }
        return Ok(());
    }

    // Fallback: text search
    let ids = if let Some(ns) = namespace {
        storage.list_memory_ids_for_namespace(ns)?
    } else {
        storage.list_memory_ids()?
    };

    if ids.is_empty() {
        println!("No memories stored yet.");
        return Ok(());
    }

    let query_lower = query.to_lowercase();
    println!(
        "Searching {} memories for: \"{}\" (text search)\n",
        ids.len(),
        query
    );

    let mut found = 0;
    for id in &ids {
        if found >= k {
            break;
        }
        if let Some(memory) = storage.get_memory(id)? {
            if memory.content.to_lowercase().contains(&query_lower) {
                println!(
                    "  [{}] {} (importance: {:.1})",
                    memory.memory_type, memory.id, memory.importance
                );
                println!("    {}", crate::truncate_str(&memory.content, 120));
                println!();
                found += 1;
            }
        }
    }

    // Also search graph nodes by label
    let gn_results = storage.search_graph_nodes(&query_lower, namespace, k)?;
    for node in &gn_results {
        if found >= k {
            break;
        }
        if let Some(filter_ns) = namespace {
            if node.namespace.as_deref() != Some(filter_ns) {
                continue;
            }
        }
        println!("  [{}] {} (graph node)", node.kind, node.id);
        println!("    {}", crate::truncate_str(&node.label, 120));
        println!();
        found += 1;
    }

    if found == 0 {
        println!("No matching memories or graph nodes found.");
    }

    Ok(())
}

pub(crate) fn cmd_stats() -> anyhow::Result<()> {
    let db_path = crate::codemem_db_path();
    let storage = codemem_storage::Storage::open(&db_path)?;
    let stats = storage.stats()?;

    println!("Codemem Statistics");
    println!("  Memories:    {}", stats.memory_count);
    println!("  Embeddings:  {}", stats.embedding_count);
    println!("  Graph nodes: {}", stats.node_count);
    println!("  Graph edges: {}", stats.edge_count);

    // Vector index
    let index_path = db_path.with_extension("idx");
    if index_path.exists() {
        if let Ok(mut vector) = codemem_vector::HnswIndex::with_defaults() {
            if vector.load(&index_path).is_ok() {
                let vstats = vector.stats();
                println!("  Vector indexed: {}", vstats.count);
            }
        }
    }

    // Embedding provider
    match codemem_embeddings::from_env() {
        Ok(provider) => println!(
            "  Embedding provider: {} ({}d)",
            provider.name(),
            provider.dimensions()
        ),
        Err(_) => println!("  Embedding provider: not configured"),
    }

    Ok(())
}
