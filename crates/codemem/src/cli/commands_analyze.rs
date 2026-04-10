use std::path::Path;

/// Full analysis pipeline: index → enrich → PageRank → clusters.
/// Uses the unified `engine.analyze()` method.
pub(crate) fn cmd_analyze(
    root: &Path,
    namespace: Option<&str>,
    days: u64,
    skip_scip: bool,
    skip_embed: bool,
    skip_enrich: bool,
    force: bool,
) -> anyhow::Result<()> {
    let db_path = super::codemem_db_path();
    let engine = codemem_engine::CodememEngine::from_db_path(&db_path)?;

    let path_str = root
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("Path is not valid UTF-8"))?;

    // Namespace: use directory basename (not full path) to match store_memory convention
    let ns = namespace.unwrap_or_else(|| {
        root.file_name()
            .and_then(|f| f.to_str())
            .unwrap_or(path_str)
    });

    // Store the namespace → root path mapping for the UI file content viewer
    if let Ok(canonical) = root.canonicalize() {
        let _ = engine
            .storage()
            .set_namespace_root(ns, &canonical.to_string_lossy());
    }

    // Load incremental state (skip when forcing full re-index)
    let change_detector = if force {
        None
    } else {
        let mut cd = codemem_engine::index::incremental::ChangeDetector::new();
        cd.load_from_storage(engine.storage(), ns);
        Some(cd)
    };

    println!("Analyzing {}...", root.display());

    let options = codemem_engine::AnalyzeOptions {
        path: root,
        namespace: ns,
        git_days: days,
        change_detector,
        progress: Some(Box::new(|progress| {
            let codemem_engine::AnalyzeProgress::Embedding { done, total } = progress;
            print!("\r  Embedding: {done}/{total}");
            std::io::Write::flush(&mut std::io::stdout()).ok();
        })),
        skip_scip,
        skip_embed,
        skip_enrich,
        force,
    };

    let result = engine.analyze(options)?;

    // Clear the progress line if we embedded anything
    if result.symbols_embedded + result.chunks_embedded > 0 {
        println!();
    }

    // Step 1: Index summary
    println!("\nStep 1/4: Index");
    if result.scip_files_covered > 0 {
        println!(
            "  SCIP:             {} files covered, {} nodes, {} edges",
            result.scip_files_covered, result.scip_nodes_created, result.scip_edges_created
        );
    }
    println!("  Files parsed:     {}", result.files_parsed);
    println!("  Files skipped:    {}", result.files_skipped);
    println!("  Symbols:          {}", result.symbols_found);
    println!("  Edges resolved:   {}", result.edges_resolved);
    println!("  Chunks stored:    {}", result.chunks_stored);
    println!(
        "  Embedded:         {} symbols, {} chunks",
        result.symbols_embedded, result.chunks_embedded
    );
    if result.chunks_pruned > 0 || result.symbols_pruned > 0 {
        println!(
            "  Compacted:        {} chunks, {} symbols pruned",
            result.chunks_pruned, result.symbols_pruned
        );
    }

    // Step 2: Enrichment summary
    println!("\nStep 2/4: Enrichment");
    if let Some(obj) = result.enrichment_results.as_object() {
        for (name, detail) in obj {
            let status = if detail.get("error").is_some() {
                format!("skipped ({})", detail["error"].as_str().unwrap_or("error"))
            } else {
                "done".to_string()
            };
            println!("  {name:20} {status}");
        }
    }
    println!(
        "  {} analyses: {} insights stored",
        result.enrichment_results.as_object().map_or(0, |o| o.len()),
        result.total_insights
    );

    // Step 3: PageRank
    println!("\nStep 3/4: Top nodes by PageRank");
    for node in &result.top_nodes {
        let kind = node.kind.as_deref().unwrap_or("");
        let label = node.label.as_deref().unwrap_or("");
        println!("    {:.6}  [{}] {}  ({})", node.score, kind, label, node.id);
    }

    // Step 4: Clusters
    println!("\nStep 4/4: Clusters");
    println!("  Communities found: {}", result.community_count);

    println!("\nDone. Run `codemem stats` to see updated totals.");
    Ok(())
}

#[cfg(test)]
#[path = "tests/commands_analyze_tests.rs"]
mod tests;
