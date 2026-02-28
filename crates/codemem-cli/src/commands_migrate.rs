//! `codemem migrate` — show and apply pending schema migrations.

use colored::Colorize;

pub(crate) fn cmd_migrate() -> anyhow::Result<()> {
    let db_path = crate::codemem_db_path();

    if !db_path.exists() {
        println!(
            "{}: Database not found at {}. Run `codemem init` first.",
            "Error".red().bold(),
            db_path.display()
        );
        return Ok(());
    }

    let storage = codemem_storage::Storage::open(&db_path)?;
    let current = storage.schema_version()?;

    println!("Database: {}", db_path.display());
    println!("Current schema version: {}", current.to_string().cyan());

    // The Storage::open() call already runs all pending migrations.
    // We report the result.
    let after = storage.schema_version()?;
    if after > current {
        println!(
            "{}",
            format!("Applied migrations: v{current} → v{after}")
                .green()
                .bold()
        );
    } else {
        println!("{}", "All migrations are up to date.".green());
    }

    Ok(())
}
