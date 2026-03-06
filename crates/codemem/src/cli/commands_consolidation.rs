//! Consolidation cycle commands.

use codemem_engine::CodememEngine;

/// All valid consolidation cycle types.
pub(crate) const VALID_CYCLES: &[&str] = &["decay", "creative", "cluster", "forget"];

/// Returns `true` if the given cycle name is a known consolidation cycle.
pub(crate) fn is_valid_cycle(cycle: &str) -> bool {
    VALID_CYCLES.contains(&cycle)
}

pub(crate) fn cmd_consolidate(cycle: &str) -> anyhow::Result<()> {
    if !is_valid_cycle(cycle) {
        anyhow::bail!(
            "Unknown cycle type: '{}'. Valid types: {}",
            cycle,
            VALID_CYCLES.join(", ")
        );
    }

    let db_path = super::codemem_db_path();
    let engine = CodememEngine::from_db_path(&db_path)?;

    println!("Running {} consolidation cycle...", cycle);

    let result = match cycle {
        "decay" => engine.consolidate_decay(None)?,
        "creative" => engine.consolidate_creative()?,
        "cluster" => engine.consolidate_cluster(None)?,
        "forget" => engine.consolidate_forget(None, None, None)?,
        _ => unreachable!("is_valid_cycle check above"),
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
    for cycle in VALID_CYCLES {
        if !entries.iter().any(|r| r.cycle_type == *cycle) {
            println!("  {:<10} never run", cycle);
        }
    }

    Ok(())
}

#[cfg(test)]
#[path = "tests/commands_consolidation_tests.rs"]
mod tests;
