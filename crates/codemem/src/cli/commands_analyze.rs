use codemem_core::GraphBackend;
use std::path::Path;

/// Full analysis pipeline: index → enrich → PageRank → clusters.
/// Mirrors the MCP `analyze_codebase` tool but with CLI progress output.
///
/// Uses `persist_index_results_with_progress` for the full persistence pipeline
/// (nodes, edges, embeddings in chunks of 64 with progress callbacks, compaction).
pub(crate) fn cmd_analyze(root: &Path, namespace: Option<&str>, days: u64) -> anyhow::Result<()> {
    let db_path = super::codemem_db_path();
    let engine = codemem_engine::CodememEngine::from_db_path(&db_path)?;

    let path_str = root
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("Path is not valid UTF-8"))?;

    // Namespace: use directory basename (not full path) to match store_memory convention
    let ns = root
        .file_name()
        .and_then(|f| f.to_str())
        .unwrap_or(path_str);

    // ── Step 1: Index ───────────────────────────────────────────────────────
    println!("Step 1/4: Indexing {}...", root.display());

    let mut change_detector = codemem_engine::index::incremental::ChangeDetector::new();
    change_detector.load_from_storage(&*engine.storage);

    let mut indexer = codemem_engine::Indexer::with_change_detector(change_detector);
    let resolved = indexer.index_and_resolve(root)?;

    println!("  Files parsed:     {}", resolved.index.files_parsed);
    println!("  Files skipped:    {}", resolved.index.files_skipped);
    println!("  Symbols:          {}", resolved.index.total_symbols);

    let persist_result =
        engine.persist_index_results_with_progress(&resolved, Some(ns), |done, total| {
            print!("\r  Embedding: {done}/{total}");
            std::io::Write::flush(&mut std::io::stdout()).ok();
        })?;
    // Clear the progress line
    if persist_result.symbols_embedded + persist_result.chunks_embedded > 0 {
        println!();
    }

    println!("  Edges resolved:   {}", persist_result.edges_resolved);
    println!("  Chunks stored:    {}", persist_result.chunks_stored);
    println!(
        "  Embedded:         {} symbols, {} chunks",
        persist_result.symbols_embedded, persist_result.chunks_embedded
    );
    if persist_result.chunks_pruned > 0 || persist_result.symbols_pruned > 0 {
        println!(
            "  Compacted:        {} chunks, {} symbols pruned",
            persist_result.chunks_pruned, persist_result.symbols_pruned
        );
    }

    indexer
        .change_detector()
        .save_to_storage(&*engine.storage)?;

    // ── Step 2: Enrich ──────────────────────────────────────────────────────
    println!("\nStep 2/4: Enriching...");

    // Use explicit --namespace if provided, otherwise default to the derived basename
    let enrich_ns = namespace.unwrap_or(ns);

    match engine.enrich_git_history(path_str, days, Some(enrich_ns)) {
        Ok(r) => println!("  Git history:      {} insights stored", r.insights_stored),
        Err(e) => println!("  Git history:      skipped ({e})"),
    }

    match engine.enrich_security(Some(enrich_ns)) {
        Ok(r) => println!("  Security:         {} insights stored", r.insights_stored),
        Err(e) => println!("  Security:         skipped ({e})"),
    }

    match engine.enrich_performance(10, Some(enrich_ns)) {
        Ok(r) => println!("  Performance:      {} insights stored", r.insights_stored),
        Err(e) => println!("  Performance:      skipped ({e})"),
    }

    // ── Step 3: PageRank ────────────────────────────────────────────────────
    println!("\nStep 3/4: Computing PageRank...");

    let graph = engine.lock_graph()?;
    let scores = graph.pagerank(0.85, 100, 1e-6);
    let mut sorted: Vec<(String, f64)> = scores.into_iter().collect();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    sorted.truncate(10);

    println!("  Top 10 nodes by PageRank:");
    for (id, score) in &sorted {
        let node = graph.get_node(id).ok().flatten();
        let kind = node
            .as_ref()
            .map(|n| n.kind.to_string())
            .unwrap_or_default();
        let label = node.as_ref().map(|n| n.label.as_str()).unwrap_or_default();
        println!("    {score:.6}  [{kind}] {label}  ({id})");
    }

    // ── Step 4: Clusters ────────────────────────────────────────────────────
    println!("\nStep 4/4: Detecting clusters...");

    let communities = graph.louvain_communities(1.0);
    println!("  Communities found: {}", communities.len());

    drop(graph);

    // Save indexes
    engine.save_index();

    println!("\nDone. Run `codemem stats` to see updated totals.");
    Ok(())
}
