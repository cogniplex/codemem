//! Consolidation cycle commands.

use codemem_engine::CodememEngine;

pub(crate) fn cmd_consolidate(cycle: &str) -> anyhow::Result<()> {
    let db_path = super::codemem_db_path();
    let engine = CodememEngine::from_db_path(&db_path)?;

    println!("Running {} consolidation cycle...", cycle);

    let result = match cycle {
        "decay" => engine.consolidate_decay(None)?,
        "creative" => engine.consolidate_creative()?,
        "cluster" => engine.consolidate_cluster(None)?,
        "forget" => engine.consolidate_forget(None, None, None)?,
        _ => {
            anyhow::bail!(
                "Unknown cycle type: '{}'. Valid types: decay, creative, cluster, forget",
                cycle
            );
        }
    };

    println!(
        "{} cycle complete: {} affected.",
        result.cycle, result.affected
    );

    Ok(())
}

pub(crate) fn cmd_consolidate_status() -> anyhow::Result<()> {
    let db_path = super::codemem_db_path();
    let engine = CodememEngine::from_db_path(&db_path)?;
    let entries = engine.consolidation_status()?;

    if entries.is_empty() {
        println!("No consolidation runs recorded yet.");
        return Ok(());
    }

    println!("Last consolidation runs:");
    for entry in &entries {
        println!(
            "  {:<10} last run: {}  ({} affected)",
            entry.cycle_type, entry.last_run, entry.affected_count
        );
    }

    // Show cycles that have never been run
    let all_cycles = ["decay", "creative", "cluster", "forget"];
    for cycle in &all_cycles {
        if !entries.iter().any(|r| r.cycle_type == *cycle) {
            println!("  {:<10} never run", cycle);
        }
    }

    Ok(())
}
