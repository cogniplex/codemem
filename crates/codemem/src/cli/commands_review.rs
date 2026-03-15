//! `codemem review` command: read a unified diff from stdin and compute blast radius.

use std::io::Read;

pub(crate) fn cmd_review(depth: usize, format: &str) -> anyhow::Result<()> {
    // Read diff from stdin
    let mut diff = String::new();
    std::io::stdin().read_to_string(&mut diff)?;

    if diff.trim().is_empty() {
        eprintln!("No diff provided on stdin. Usage: git diff main..HEAD | codemem review");
        return Ok(());
    }

    let db_path = super::codemem_db_path();
    let engine = codemem_engine::CodememEngine::from_db_path(&db_path)?;

    let report = engine.blast_radius(&diff, depth)?;

    match format {
        "text" => {
            println!("=== Blast Radius Report ===\n");
            println!("Changed symbols: {}", report.changed_symbols.len());
            for sym in &report.changed_symbols {
                println!(
                    "  {} ({}) {}",
                    sym.label,
                    sym.kind,
                    sym.file_path.as_deref().unwrap_or("")
                );
            }

            println!("\nDirect dependents: {}", report.direct_dependents.len());
            for dep in &report.direct_dependents {
                println!(
                    "  {} ({}) {}",
                    dep.label,
                    dep.kind,
                    dep.file_path.as_deref().unwrap_or("")
                );
            }

            println!(
                "\nTransitive dependents: {}",
                report.transitive_dependents.len()
            );
            println!("Affected files: {}", report.affected_files.len());
            for f in &report.affected_files {
                println!("  {f}");
            }

            println!("\nRisk score: {:.2}", report.risk_score);

            if !report.missing_changes.is_empty() {
                println!("\nPotentially missing changes:");
                for mc in &report.missing_changes {
                    println!("  {} — {}", mc.symbol, mc.reason);
                }
            }

            if !report.relevant_memories.is_empty() {
                println!("\nRelevant memories:");
                for mem in &report.relevant_memories {
                    println!(
                        "  [{}] {}",
                        mem.memory_type,
                        if mem.content.len() > 80 {
                            format!("{}...", &mem.content[..80])
                        } else {
                            mem.content.clone()
                        }
                    );
                }
            }
        }
        _ => {
            // Default: JSON
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
    }

    Ok(())
}
