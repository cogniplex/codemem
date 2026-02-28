//! `codemem doctor` — health checks for the Codemem installation.

use colored::Colorize;
use codemem_core::VectorBackend;

pub(crate) fn cmd_doctor() -> anyhow::Result<()> {
    println!("{}", "Codemem Doctor".bold());
    println!();

    let mut all_ok = true;

    // 1. Database existence and integrity
    let db_path = crate::codemem_db_path();
    if db_path.exists() {
        print_check("Database file exists", true, &db_path.display().to_string());
        match codemem_storage::Storage::open(&db_path) {
            Ok(storage) => {
                print_check("Database opens successfully", true, "");

                // PRAGMA integrity_check
                match storage.integrity_check() {
                    Ok(true) => print_check("Database integrity check", true, "ok"),
                    Ok(false) => {
                        print_check("Database integrity check", false, "CORRUPTION DETECTED");
                        all_ok = false;
                    }
                    Err(e) => {
                        print_check("Database integrity check", false, &e.to_string());
                        all_ok = false;
                    }
                }

                // Schema version
                match storage.schema_version() {
                    Ok(v) => print_check("Schema version", true, &format!("v{v}")),
                    Err(e) => {
                        print_check("Schema version", false, &e.to_string());
                        all_ok = false;
                    }
                }

                // Memory count
                match storage.stats() {
                    Ok(stats) => {
                        let count = stats.memory_count;
                        print_check("Memory store", true, &format!("{count} memories"));
                    }
                    Err(e) => {
                        print_check("Memory store", false, &e.to_string());
                        all_ok = false;
                    }
                }
            }
            Err(e) => {
                print_check("Database opens", false, &e.to_string());
                all_ok = false;
            }
        }
    } else {
        print_check("Database file exists", false, "Run `codemem init` first");
        all_ok = false;
    }

    // 2. Vector index
    let index_path = db_path.with_extension("idx");
    if index_path.exists() {
        match codemem_vector::HnswIndex::with_defaults() {
            Ok(mut vector) => match vector.load(&index_path) {
                Ok(()) => {
                    let count = vector.len();
                    print_check("Vector index", true, &format!("{count} vectors"));
                }
                Err(e) => {
                    print_check("Vector index", false, &format!("load error: {e}"));
                    all_ok = false;
                }
            },
            Err(e) => {
                print_check("Vector index", false, &e.to_string());
                all_ok = false;
            }
        }
    } else {
        print_check("Vector index", true, "not yet created (will be created on first use)");
    }

    // 3. Embedding model probe
    match codemem_embeddings::from_env() {
        Ok(emb) => {
            match emb.embed("hello world") {
                Ok(v) => print_check(
                    "Embedding provider",
                    true,
                    &format!("{}-dim vectors", v.len()),
                ),
                Err(e) => {
                    print_check("Embedding provider", false, &format!("probe failed: {e}"));
                    all_ok = false;
                }
            }
        }
        Err(e) => {
            // Not an error if using default Candle provider — it may need a model download
            print_check(
                "Embedding provider",
                false,
                &format!("not available: {e}"),
            );
            all_ok = false;
        }
    }

    // 4. Config validation
    let config_path = codemem_core::CodememConfig::default_path();
    if config_path.exists() {
        match codemem_core::CodememConfig::load(&config_path) {
            Ok(_) => print_check("Configuration", true, &config_path.display().to_string()),
            Err(e) => {
                print_check("Configuration", false, &format!("parse error: {e}"));
                all_ok = false;
            }
        }
    } else {
        print_check("Configuration", true, "using defaults (no config.toml)");
    }

    println!();
    if all_ok {
        println!("{}", "All checks passed!".green().bold());
    } else {
        println!(
            "{}",
            "Some checks failed. See above for details.".red().bold()
        );
    }

    Ok(())
}

fn print_check(name: &str, ok: bool, detail: &str) {
    let status = if ok {
        "OK".green().bold().to_string()
    } else {
        "FAIL".red().bold().to_string()
    };
    if detail.is_empty() {
        println!("  [{status}] {name}");
    } else {
        println!("  [{status}] {name}: {detail}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn print_check_ok_does_not_panic() {
        print_check("test", true, "");
    }

    #[test]
    fn print_check_fail_with_detail_does_not_panic() {
        print_check("test", false, "some detail");
    }
}
