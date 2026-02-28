//! codemem-cli: CLI entry point for the Codemem memory engine.

mod commands_consolidation;
mod commands_data;
mod commands_export;
mod commands_init;
mod commands_lifecycle;
mod commands_search;
mod compress;

use clap::{Parser, Subcommand};
use codemem_core::{StorageBackend, VectorBackend};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "codemem",
    about = "Persistent memory engine for AI coding assistants"
)]
#[command(version, propagate_version = true)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Initialize Codemem in the current project
    Init {
        /// Project directory (defaults to current directory)
        #[arg(short, long)]
        path: Option<PathBuf>,

        /// Skip embedding model download (useful for CI/testing)
        #[arg(long)]
        skip_model: bool,
    },

    /// Search memories
    Search {
        /// Search query
        query: String,

        /// Number of results
        #[arg(short, long, default_value = "10")]
        k: usize,

        /// Filter results to a specific namespace (e.g. project path)
        #[arg(long)]
        namespace: Option<String>,
    },

    /// Show database statistics
    Stats,

    /// Start MCP server (stdio)
    Serve,

    /// Process hook payload from stdin
    Ingest,

    /// Run memory consolidation cycles
    Consolidate {
        /// Cycle type: decay, creative, cluster, forget
        #[arg(short, long, default_value = "decay")]
        cycle: String,

        /// Show last run status for each consolidation cycle
        #[arg(long)]
        status: bool,
    },

    /// Start interactive visualization dashboard
    Viz {
        /// Port to listen on
        #[arg(short, long, default_value = "4242")]
        port: u16,

        /// Don't open browser automatically
        #[arg(long)]
        no_open: bool,
    },

    /// Index a codebase for structural analysis
    Index {
        /// Path to index (defaults to current directory)
        #[arg(short, long)]
        path: Option<PathBuf>,

        /// Show detailed symbol output
        #[arg(short, long)]
        verbose: bool,
    },

    /// Export memories to JSONL format
    Export {
        /// Filter by namespace
        #[arg(long)]
        namespace: Option<String>,
        /// Filter by memory type
        #[arg(long)]
        memory_type: Option<String>,
        /// Output file (defaults to stdout)
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Import memories from JSONL format
    Import {
        /// Input file (defaults to stdin)
        #[arg(short, long)]
        input: Option<PathBuf>,
        /// Skip duplicates (by content hash)
        #[arg(long)]
        skip_duplicates: bool,
    },

    /// Watch a directory for file changes and re-index
    Watch {
        /// Directory to watch (defaults to current directory)
        #[arg(short, long)]
        path: Option<PathBuf>,
    },

    /// Manage memory sessions
    Sessions {
        #[command(subcommand)]
        action: SessionAction,
    },

    /// SessionStart hook: inject prior context into a new session (reads JSON from stdin)
    Context,

    /// UserPromptSubmit hook: record a user prompt (reads JSON from stdin)
    Prompt,

    /// Stop hook: generate session summary (reads JSON from stdin)
    Summarize,
}

#[derive(Subcommand)]
enum SessionAction {
    /// List all sessions
    List {
        /// Filter by namespace
        #[arg(long)]
        namespace: Option<String>,
    },
    /// Start a new session
    Start {
        /// Optional namespace
        #[arg(long)]
        namespace: Option<String>,
    },
    /// End an active session
    End {
        /// Session ID to end
        id: String,
        /// Optional summary of what was accomplished
        #[arg(short, long)]
        summary: Option<String>,
    },
}

fn main() -> anyhow::Result<()> {
    // Initialize tracing to stderr (stdout reserved for JSON-RPC in serve mode)
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("codemem=info".parse().expect("valid tracing directive")),
        )
        .with_target(false)
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Init { path, skip_model } => {
            let project_dir = match path {
                Some(p) => p,
                None => std::env::current_dir()?,
            };
            commands_init::cmd_init(&project_dir, skip_model)?;
        }
        Commands::Search {
            query,
            k,
            namespace,
        } => {
            commands_search::cmd_search(&query, k, namespace.as_deref())?;
        }
        Commands::Stats => {
            commands_search::cmd_stats()?;
        }
        Commands::Serve => {
            commands_data::cmd_serve()?;
        }
        Commands::Ingest => {
            commands_data::cmd_ingest()?;
        }
        Commands::Consolidate { cycle, status } => {
            if status {
                commands_consolidation::cmd_consolidate_status()?;
            } else {
                commands_consolidation::cmd_consolidate(&cycle)?;
            }
        }
        Commands::Viz { port, no_open } => {
            let db_path = codemem_db_path();
            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(codemem_viz::serve(db_path, port, !no_open))?;
        }
        Commands::Index { path, verbose } => {
            let project_dir = match path {
                Some(p) => p,
                None => std::env::current_dir()?,
            };
            commands_export::cmd_index(&project_dir, verbose)?;
        }
        Commands::Export {
            namespace,
            memory_type,
            output,
        } => {
            commands_export::cmd_export(
                namespace.as_deref(),
                memory_type.as_deref(),
                output.as_deref(),
            )?;
        }
        Commands::Import {
            input,
            skip_duplicates,
        } => {
            commands_export::cmd_import(input.as_deref(), skip_duplicates)?;
        }
        Commands::Watch { path } => {
            let watch_dir = match path {
                Some(p) => p,
                None => std::env::current_dir()?,
            };
            commands_data::cmd_watch(&watch_dir)?;
        }
        Commands::Sessions { action } => match action {
            SessionAction::List { namespace } => {
                commands_lifecycle::cmd_sessions_list(namespace.as_deref())?;
            }
            SessionAction::Start { namespace } => {
                commands_lifecycle::cmd_sessions_start(namespace.as_deref())?;
            }
            SessionAction::End { id, summary } => {
                commands_lifecycle::cmd_sessions_end(&id, summary.as_deref())?;
            }
        },
        Commands::Context => {
            commands_lifecycle::cmd_context()?;
        }
        Commands::Prompt => {
            commands_lifecycle::cmd_prompt()?;
        }
        Commands::Summarize => {
            commands_lifecycle::cmd_summarize()?;
        }
    }

    Ok(())
}

// ── Helpers (shared across modules) ────────────────────────────────────────

/// Rebuild the HNSW vector index from all stored embeddings in the database.
pub(crate) fn rebuild_vector_index(
    storage: &codemem_storage::Storage,
) -> anyhow::Result<codemem_vector::HnswIndex> {
    let mut vector = codemem_vector::HnswIndex::with_defaults()?;
    let embeddings = storage.list_all_embeddings()?;
    for (id, floats) in &embeddings {
        let _ = vector.insert(id, floats);
    }
    tracing::info!("Rebuilt vector index with {} entries", embeddings.len());
    Ok(vector)
}

/// Return the system-wide Codemem database path: ~/.codemem/codemem.db
pub(crate) fn codemem_db_path() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".codemem")
        .join("codemem.db")
}

pub(crate) fn truncate_str(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}...", &s[..max])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn truncate_str_short() {
        assert_eq!(truncate_str("hi", 10), "hi");
    }

    #[test]
    fn truncate_str_exact() {
        assert_eq!(truncate_str("hello", 5), "hello");
    }

    #[test]
    fn truncate_str_long() {
        assert_eq!(truncate_str("hello world", 5), "hello...");
    }

    #[test]
    fn truncate_str_empty() {
        assert_eq!(truncate_str("", 5), "");
    }

    #[test]
    fn parse_search_command() {
        let cli = Cli::try_parse_from(["codemem", "search", "query text"]).unwrap();
        match cli.command {
            Commands::Search {
                query,
                k,
                namespace,
            } => {
                assert_eq!(query, "query text");
                assert_eq!(k, 10); // default
                assert!(namespace.is_none());
            }
            _ => panic!("Expected Search command"),
        }
    }

    #[test]
    fn parse_search_with_options() {
        let cli = Cli::try_parse_from([
            "codemem",
            "search",
            "my query",
            "-k",
            "5",
            "--namespace",
            "my-project",
        ])
        .unwrap();
        match cli.command {
            Commands::Search {
                query,
                k,
                namespace,
            } => {
                assert_eq!(query, "my query");
                assert_eq!(k, 5);
                assert_eq!(namespace, Some("my-project".to_string()));
            }
            _ => panic!("Expected Search command"),
        }
    }

    #[test]
    fn parse_stats_command() {
        let cli = Cli::try_parse_from(["codemem", "stats"]).unwrap();
        assert!(matches!(cli.command, Commands::Stats));
    }

    #[test]
    fn parse_init_command() {
        let cli = Cli::try_parse_from(["codemem", "init"]).unwrap();
        match cli.command {
            Commands::Init { path, skip_model } => {
                assert!(path.is_none());
                assert!(!skip_model);
            }
            _ => panic!("Expected Init command"),
        }
    }

    #[test]
    fn parse_init_with_skip_model() {
        let cli = Cli::try_parse_from(["codemem", "init", "--skip-model"]).unwrap();
        match cli.command {
            Commands::Init { skip_model, .. } => {
                assert!(skip_model);
            }
            _ => panic!("Expected Init command"),
        }
    }

    #[test]
    fn parse_export_command() {
        let cli = Cli::try_parse_from([
            "codemem",
            "export",
            "--namespace",
            "test-ns",
            "--memory-type",
            "decision",
        ])
        .unwrap();
        match cli.command {
            Commands::Export {
                namespace,
                memory_type,
                output,
            } => {
                assert_eq!(namespace, Some("test-ns".to_string()));
                assert_eq!(memory_type, Some("decision".to_string()));
                assert!(output.is_none());
            }
            _ => panic!("Expected Export command"),
        }
    }

    #[test]
    fn parse_import_command() {
        let cli = Cli::try_parse_from(["codemem", "import", "--skip-duplicates"]).unwrap();
        match cli.command {
            Commands::Import {
                input,
                skip_duplicates,
            } => {
                assert!(input.is_none());
                assert!(skip_duplicates);
            }
            _ => panic!("Expected Import command"),
        }
    }

    #[test]
    fn parse_consolidate_command() {
        let cli = Cli::try_parse_from(["codemem", "consolidate", "-c", "creative"]).unwrap();
        match cli.command {
            Commands::Consolidate { cycle, status } => {
                assert_eq!(cycle, "creative");
                assert!(!status);
            }
            _ => panic!("Expected Consolidate command"),
        }
    }

    #[test]
    fn parse_sessions_list() {
        let cli = Cli::try_parse_from(["codemem", "sessions", "list"]).unwrap();
        match cli.command {
            Commands::Sessions { action } => match action {
                SessionAction::List { namespace } => {
                    assert!(namespace.is_none());
                }
                _ => panic!("Expected List action"),
            },
            _ => panic!("Expected Sessions command"),
        }
    }

    #[test]
    fn parse_sessions_end() {
        let cli =
            Cli::try_parse_from(["codemem", "sessions", "end", "sess-123", "-s", "done"]).unwrap();
        match cli.command {
            Commands::Sessions { action } => match action {
                SessionAction::End { id, summary } => {
                    assert_eq!(id, "sess-123");
                    assert_eq!(summary, Some("done".to_string()));
                }
                _ => panic!("Expected End action"),
            },
            _ => panic!("Expected Sessions command"),
        }
    }

    #[test]
    fn parse_viz_defaults() {
        let cli = Cli::try_parse_from(["codemem", "viz"]).unwrap();
        match cli.command {
            Commands::Viz { port, no_open } => {
                assert_eq!(port, 4242);
                assert!(!no_open);
            }
            _ => panic!("Expected Viz command"),
        }
    }

    #[test]
    fn parse_unknown_command_fails() {
        assert!(Cli::try_parse_from(["codemem", "unknown"]).is_err());
    }

    #[test]
    fn codemem_db_path_returns_valid_path() {
        let path = codemem_db_path();
        assert!(path.ends_with("codemem.db"));
        assert!(path.to_string_lossy().contains(".codemem"));
    }
}
