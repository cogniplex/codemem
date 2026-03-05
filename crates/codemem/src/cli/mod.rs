//! codemem-cli: CLI entry point for the Codemem memory engine.

mod commands_analyze;
mod commands_config;
mod commands_consolidation;
mod commands_data;
mod commands_doctor;
mod commands_export;
mod commands_init;
mod commands_lifecycle;
mod commands_migrate;
mod commands_search;

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

    /// Start MCP server (stdio by default, composable with --api and --http)
    Serve {
        /// Enable REST API + embedded frontend on HTTP
        #[arg(long)]
        api: bool,
        /// Use HTTP transport for MCP (instead of stdio)
        #[arg(long)]
        http: bool,
        /// HTTP server port (used when --api or --http is set)
        #[arg(long, default_value = "4242")]
        port: u16,
    },

    /// Open the control plane UI (alias for `serve --api`, auto-opens browser)
    Ui {
        /// HTTP server port
        #[arg(long, default_value = "4242")]
        port: u16,
        /// Don't open browser automatically
        #[arg(long)]
        no_open: bool,
    },

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

    /// Index a codebase for structural analysis
    Index {
        /// Path to index (defaults to current directory)
        #[arg(short, long)]
        path: Option<PathBuf>,

        /// Show detailed symbol output
        #[arg(short, long)]
        verbose: bool,
    },

    /// Full analysis pipeline: index → enrich → PageRank → clusters
    Analyze {
        /// Path to analyze (defaults to current directory)
        #[arg(short, long)]
        path: Option<PathBuf>,

        /// Namespace scope for enrichment
        #[arg(long)]
        namespace: Option<String>,

        /// Days of git history to analyze
        #[arg(long, default_value = "90")]
        days: u64,
    },

    /// Export memories to JSONL, JSON, CSV, or Markdown format
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
        /// Output format: jsonl (default), json, csv, markdown
        #[arg(short, long, default_value = "jsonl")]
        format: String,
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

    /// Run health checks on the Codemem installation
    Doctor,

    /// Manage Codemem configuration
    Config {
        #[command(subcommand)]
        action: ConfigAction,
    },

    /// Run pending database schema migrations
    Migrate,

    /// SessionStart hook: inject prior context into a new session (reads JSON from stdin)
    Context,

    /// UserPromptSubmit hook: record a user prompt (reads JSON from stdin)
    Prompt,

    /// Stop hook: generate session summary (reads JSON from stdin)
    Summarize,
}

#[derive(Subcommand)]
enum ConfigAction {
    /// Get a configuration value by dot-separated key path
    Get {
        /// Key path (e.g. "scoring.vector_similarity")
        key: String,
    },
    /// Set a configuration value
    Set {
        /// Key path (e.g. "scoring.vector_similarity")
        key: String,
        /// Value to set (JSON-compatible)
        value: String,
    },
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

pub fn run() -> anyhow::Result<()> {
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
        Commands::Serve { api, http, port } => {
            commands_data::cmd_serve(api, http, port)?;
        }
        Commands::Ui { port, no_open } => {
            commands_data::cmd_ui(port, no_open)?;
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
        Commands::Index { path, verbose } => {
            let project_dir = match path {
                Some(p) => p,
                None => std::env::current_dir()?,
            };
            commands_export::cmd_index(&project_dir, verbose)?;
        }
        Commands::Analyze {
            path,
            namespace,
            days,
        } => {
            let project_dir = match path {
                Some(p) => p,
                None => std::env::current_dir()?,
            };
            commands_analyze::cmd_analyze(&project_dir, namespace.as_deref(), days)?;
        }
        Commands::Export {
            namespace,
            memory_type,
            output,
            format,
        } => {
            commands_export::cmd_export(
                namespace.as_deref(),
                memory_type.as_deref(),
                output.as_deref(),
                &format,
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
        Commands::Doctor => {
            commands_doctor::cmd_doctor()?;
        }
        Commands::Config { action } => match action {
            ConfigAction::Get { key } => {
                commands_config::cmd_config_get(&key)?;
            }
            ConfigAction::Set { key, value } => {
                commands_config::cmd_config_set(&key, &value)?;
            }
        },
        Commands::Migrate => {
            commands_migrate::cmd_migrate()?;
        }
        Commands::Sessions { action } => {
            // H3: Open lightweight storage instead of the full engine
            // (avoids loading vector index, building graph, populating BM25).
            let db_path = codemem_db_path();
            let storage = codemem_storage::Storage::open(&db_path)?;
            match action {
                SessionAction::List { namespace } => {
                    commands_lifecycle::cmd_sessions_list(&storage, namespace.as_deref())?;
                }
                SessionAction::Start { namespace } => {
                    commands_lifecycle::cmd_sessions_start(&storage, namespace.as_deref())?;
                }
                SessionAction::End { id, summary } => {
                    commands_lifecycle::cmd_sessions_end(&storage, &id, summary.as_deref())?;
                }
            }
        }
        Commands::Context => {
            // H3: Open lightweight storage instead of the full engine.
            let db_path = codemem_db_path();
            match codemem_storage::Storage::open(&db_path) {
                Ok(storage) => {
                    commands_lifecycle::cmd_context(&storage)?;
                }
                Err(e) => {
                    // H2: Only suppress missing-file errors (DB doesn't exist yet).
                    // Log other errors (corruption, permission denied, lock poisoning)
                    // so they're visible in traces rather than silently swallowed.
                    if db_path.exists() {
                        tracing::warn!("Failed to open storage for context: {e}");
                    }
                    println!("{}", serde_json::to_string(&serde_json::json!({}))?);
                }
            }
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
) -> anyhow::Result<codemem_storage::HnswIndex> {
    let mut vector = codemem_storage::HnswIndex::with_defaults()?;
    let embeddings = storage.list_all_embeddings()?;
    for (id, floats) in &embeddings {
        if let Err(e) = vector.insert(id, floats) {
            tracing::warn!("Failed to insert embedding for {id}: {e}");
        }
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
    crate::truncate_str(s, max)
}

#[cfg(test)]
#[path = "tests/main_tests.rs"]
mod tests;
