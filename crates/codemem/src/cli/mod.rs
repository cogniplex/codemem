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
mod commands_review;
mod commands_search;

use clap::{Parser, Subcommand};
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

    /// Open the control plane UI (alias for `serve --api`, auto-opens browser)
    Ui {
        /// HTTP server port
        #[arg(long, default_value = "4242")]
        port: u16,
        /// Don't open browser automatically
        #[arg(long)]
        no_open: bool,
    },

    /// Run memory consolidation cycles
    Consolidate {
        /// Cycle type: decay, creative, cluster, forget
        #[arg(short, long, default_value = "decay")]
        cycle: String,

        /// Show last run status for each consolidation cycle
        #[arg(long)]
        status: bool,
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

        /// Skip SCIP indexing (fast, ast-grep only)
        #[arg(long)]
        skip_scip: bool,

        /// Skip embedding phase (store graph without vectorizing)
        #[arg(long)]
        skip_embed: bool,

        /// Skip enrichment phase (no git/complexity/security analysis)
        #[arg(long)]
        skip_enrich: bool,

        /// Force re-index even when file SHAs haven't changed
        #[arg(long)]
        force: bool,
    },

    /// Review a diff: map changed lines to symbols, compute blast radius
    Review {
        /// Base ref for scope context (e.g., "main"). Sets the overlay base
        /// so memories/graph data from the base branch is included in analysis.
        #[arg(long, default_value = "main")]
        base: String,
        /// Traversal depth for transitive impact analysis
        #[arg(long, default_value = "2")]
        depth: usize,
        /// Output format: json (default) or text
        #[arg(short, long, default_value = "json")]
        format: String,
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

    /// MCP server, hooks, and integration commands
    Mcp {
        #[command(subcommand)]
        action: McpAction,
    },

    // ── Hidden backward-compat aliases ──────────────────────────────

    /// (hidden) Alias for `mcp serve`
    #[command(hide = true)]
    Serve {
        #[arg(long)]
        api: bool,
        #[arg(long)]
        http: bool,
        #[arg(long, default_value = "4242")]
        port: u16,
    },

    /// (hidden) Alias for `mcp ingest`
    #[command(hide = true)]
    Ingest,

    /// (hidden) Alias for `mcp context`
    #[command(hide = true)]
    Context,

    /// (hidden) Alias for `mcp prompt`
    #[command(hide = true)]
    Prompt,

    /// (hidden) Alias for `mcp summarize`
    #[command(hide = true)]
    Summarize,

    /// (hidden) Alias for `mcp agent-result`
    #[command(hide = true)]
    AgentResult,

    /// (hidden) Alias for `mcp agent-start`
    #[command(hide = true)]
    AgentStart,

    /// (hidden) Alias for `mcp tool-error`
    #[command(hide = true)]
    ToolError,

    /// (hidden) Alias for `mcp session-close`
    #[command(hide = true)]
    SessionClose,

    /// (hidden) Alias for `mcp checkpoint`
    #[command(hide = true)]
    Checkpoint,
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
enum McpAction {
    /// Start MCP server (stdio by default, or --http for HTTP transport)
    Serve {
        /// Use HTTP transport for MCP (instead of stdio)
        #[arg(long)]
        http: bool,
        /// HTTP server port (used when --http is set)
        #[arg(long, default_value = "4242")]
        port: u16,
    },

    /// Process hook payload from stdin
    Ingest,

    /// SessionStart hook: inject prior context into a new session (reads JSON from stdin)
    Context,

    /// UserPromptSubmit hook: record a user prompt (reads JSON from stdin)
    Prompt,

    /// Stop hook: generate session summary (reads JSON from stdin)
    Summarize,

    /// SubagentStop hook: capture subagent findings (reads JSON from stdin)
    AgentResult,

    /// SubagentStart hook: note agent spawn (reads JSON from stdin)
    AgentStart,

    /// PostToolUseFailure hook: capture tool errors (reads JSON from stdin)
    ToolError,

    /// SessionEnd hook: clean session close (reads JSON from stdin)
    SessionClose,

    /// PreCompact hook: checkpoint before context compaction (reads JSON from stdin)
    Checkpoint,
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
        Commands::Ui { port, no_open } => {
            commands_data::cmd_ui(port, no_open)?;
        }
        Commands::Consolidate { cycle, status } => {
            if status {
                commands_consolidation::cmd_consolidate_status()?;
            } else {
                commands_consolidation::cmd_consolidate(&cycle)?;
            }
        }
        Commands::Analyze {
            path,
            namespace,
            days,
            skip_scip,
            skip_embed,
            skip_enrich,
            force,
        } => {
            let project_dir = match path {
                Some(p) => p,
                None => std::env::current_dir()?,
            };
            commands_analyze::cmd_analyze(
                &project_dir,
                namespace.as_deref(),
                days,
                skip_scip,
                skip_embed,
                skip_enrich,
                force,
            )?;
        }
        Commands::Review {
            base,
            depth,
            format,
        } => {
            commands_review::cmd_review(&base, depth, &format)?;
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
        Commands::Mcp { action } => {
            run_mcp_action(action)?;
        }
        Commands::Sessions { action } => {
            // H3: Open lightweight storage instead of the full engine
            // (avoids loading vector index, building graph, populating BM25).
            let db_path = codemem_db_path();
            let storage = codemem_engine::Storage::open(&db_path)?;
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

        // ── Hidden backward-compat aliases → delegate to mcp subcommand ──
        Commands::Serve { api, http, port } => {
            if api {
                // Old `codemem serve --api` → now `codemem ui`
                commands_data::cmd_ui(port, false)?;
            } else {
                run_mcp_action(McpAction::Serve { http, port })?;
            }
        }
        Commands::Ingest => {
            run_mcp_action(McpAction::Ingest)?;
        }
        Commands::Context => {
            run_mcp_action(McpAction::Context)?;
        }
        Commands::Prompt => {
            run_mcp_action(McpAction::Prompt)?;
        }
        Commands::Summarize => {
            run_mcp_action(McpAction::Summarize)?;
        }
        Commands::AgentResult => {
            run_mcp_action(McpAction::AgentResult)?;
        }
        Commands::AgentStart => {
            run_mcp_action(McpAction::AgentStart)?;
        }
        Commands::ToolError => {
            run_mcp_action(McpAction::ToolError)?;
        }
        Commands::SessionClose => {
            run_mcp_action(McpAction::SessionClose)?;
        }
        Commands::Checkpoint => {
            run_mcp_action(McpAction::Checkpoint)?;
        }
    }

    Ok(())
}

fn run_mcp_action(action: McpAction) -> anyhow::Result<()> {
    match action {
        McpAction::Serve { http, port } => {
            commands_data::cmd_serve(false, http, port)?;
        }
        McpAction::Ingest => {
            commands_data::cmd_ingest()?;
        }
        McpAction::Context => {
            let db_path = codemem_db_path();
            match codemem_engine::Storage::open_without_migrations(&db_path) {
                Ok(storage) => {
                    commands_lifecycle::cmd_context(&storage)?;
                }
                Err(e) => {
                    if db_path.exists() {
                        tracing::warn!("Failed to open storage for context: {e}");
                    }
                    println!("{}", serde_json::to_string(&serde_json::json!({}))?);
                }
            }
        }
        McpAction::Prompt => {
            commands_lifecycle::cmd_prompt()?;
        }
        McpAction::Summarize => {
            commands_lifecycle::cmd_summarize()?;
        }
        McpAction::AgentResult => {
            commands_lifecycle::cmd_agent_result()?;
        }
        McpAction::AgentStart => {
            commands_lifecycle::cmd_agent_start()?;
        }
        McpAction::ToolError => {
            commands_lifecycle::cmd_tool_error()?;
        }
        McpAction::SessionClose => {
            commands_lifecycle::cmd_session_close()?;
        }
        McpAction::Checkpoint => {
            commands_lifecycle::cmd_checkpoint()?;
        }
    }
    Ok(())
}

/// Derive a short namespace from a working-directory path.
/// Returns the directory basename (e.g. `/Users/me/project` → `"project"`).
pub(crate) fn namespace_from_path(path: &str) -> &str {
    std::path::Path::new(path)
        .file_name()
        .and_then(|f| f.to_str())
        .unwrap_or(path)
}

/// Return the system-wide Codemem database path: ~/.codemem/codemem.db
pub(crate) fn codemem_db_path() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".codemem")
        .join("codemem.db")
}

#[cfg(test)]
#[path = "tests/main_tests.rs"]
mod tests;
