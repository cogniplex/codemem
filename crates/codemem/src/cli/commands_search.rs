//! Search & stats commands.

use codemem_core::VectorBackend;

pub(crate) fn cmd_search(query: &str, k: usize, namespace: Option<&str>) -> anyhow::Result<()> {
    let db_path = super::codemem_db_path();
    let engine = codemem_engine::CodememEngine::from_db_path(&db_path)?;

    // Try vector search first
    let vector_results: Vec<(String, f32)> = if let Ok(Some(emb)) = engine.lock_embeddings() {
        match emb.embed(query) {
            Ok(query_embedding) => {
                drop(emb);
                engine
                    .lock_vector()?
                    .search(&query_embedding, k * 2)
                    .unwrap_or_default()
            }
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
            if let Some(memory) = engine.storage.get_memory(id)? {
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
                println!("         {}", super::truncate_str(&memory.content, 120));
                if !memory.tags.is_empty() {
                    println!("         tags: {}", memory.tags.join(", "));
                }
                println!();
                shown += 1;
            } else if let Some(node) = engine.storage.get_graph_node(id)? {
                // Fallback: symbol/graph node (e.g. sym:* IDs from code indexing)
                if let Some(ns) = namespace {
                    if node.namespace.as_deref() != Some(ns) {
                        continue;
                    }
                }
                println!("  [{:.3}] [{}] {}", similarity, node.kind, node.id);
                println!("         {}", super::truncate_str(&node.label, 120));
                println!();
                shown += 1;
            }
        }
        return Ok(());
    }

    // Fallback: text search
    let ids = if let Some(ns) = namespace {
        engine.storage.list_memory_ids_for_namespace(ns)?
    } else {
        engine.storage.list_memory_ids()?
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
        if let Some(memory) = engine.storage.get_memory(id)? {
            if memory.content.to_lowercase().contains(&query_lower) {
                println!(
                    "  [{}] {} (importance: {:.1})",
                    memory.memory_type, memory.id, memory.importance
                );
                println!("    {}", super::truncate_str(&memory.content, 120));
                println!();
                found += 1;
            }
        }
    }

    // Also search graph nodes by label
    let gn_results = engine
        .storage
        .search_graph_nodes(&query_lower, namespace, k)?;
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
        println!("    {}", super::truncate_str(&node.label, 120));
        println!();
        found += 1;
    }

    if found == 0 {
        println!("No matching memories or graph nodes found.");
    }

    Ok(())
}

pub(crate) fn cmd_stats() -> anyhow::Result<()> {
    let db_path = super::codemem_db_path();
    let engine = codemem_engine::CodememEngine::from_db_path(&db_path)?;
    let stats = engine.storage.stats()?;

    println!("Codemem Statistics");
    println!("  Memories:    {}", stats.memory_count);
    println!("  Embeddings:  {}", stats.embedding_count);
    println!("  Graph nodes: {}", stats.node_count);
    println!("  Graph edges: {}", stats.edge_count);

    // Vector index
    {
        let vector = engine.lock_vector()?;
        let vstats = vector.stats();
        println!("  Vector indexed: {}", vstats.count);
    }

    // Embedding provider
    match engine.lock_embeddings()? {
        Some(emb) => println!(
            "  Embedding provider: {} ({}d)",
            emb.name(),
            emb.dimensions()
        ),
        None => println!("  Embedding provider: not configured"),
    }

    Ok(())
}
