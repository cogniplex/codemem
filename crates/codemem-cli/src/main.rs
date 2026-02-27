//! codemem-cli: CLI entry point for the Codemem memory engine.

mod compress;

use clap::{Parser, Subcommand};
use codemem_core::VectorBackend;
use rusqlite::params;
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
                .add_directive("codemem=info".parse().unwrap()),
        )
        .with_target(false)
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Init { path, skip_model } => {
            let project_dir = path.unwrap_or_else(|| std::env::current_dir().unwrap());
            cmd_init(&project_dir, skip_model)?;
        }
        Commands::Search {
            query,
            k,
            namespace,
        } => {
            cmd_search(&query, k, namespace.as_deref())?;
        }
        Commands::Stats => {
            cmd_stats()?;
        }
        Commands::Serve => {
            cmd_serve()?;
        }
        Commands::Ingest => {
            cmd_ingest()?;
        }
        Commands::Consolidate { cycle, status } => {
            if status {
                cmd_consolidate_status()?;
            } else {
                cmd_consolidate(&cycle)?;
            }
        }
        Commands::Viz { port, no_open } => {
            let db_path = codemem_db_path();
            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(codemem_viz::serve(db_path, port, !no_open))?;
        }
        Commands::Index { path, verbose } => {
            let project_dir = path.unwrap_or_else(|| std::env::current_dir().unwrap());
            cmd_index(&project_dir, verbose)?;
        }
        Commands::Export {
            namespace,
            memory_type,
            output,
        } => {
            cmd_export(
                namespace.as_deref(),
                memory_type.as_deref(),
                output.as_deref(),
            )?;
        }
        Commands::Import {
            input,
            skip_duplicates,
        } => {
            cmd_import(input.as_deref(), skip_duplicates)?;
        }
        Commands::Watch { path } => {
            let watch_dir = path.unwrap_or_else(|| std::env::current_dir().unwrap());
            cmd_watch(&watch_dir)?;
        }
        Commands::Sessions { action } => match action {
            SessionAction::List { namespace } => {
                cmd_sessions_list(namespace.as_deref())?;
            }
            SessionAction::Start { namespace } => {
                cmd_sessions_start(namespace.as_deref())?;
            }
            SessionAction::End { id, summary } => {
                cmd_sessions_end(&id, summary.as_deref())?;
            }
        },
        Commands::Context => {
            cmd_context()?;
        }
        Commands::Prompt => {
            cmd_prompt()?;
        }
        Commands::Summarize => {
            cmd_summarize()?;
        }
    }

    Ok(())
}

// ── AI Assistant Detection ─────────────────────────────────────────────────

/// Represents a detected AI coding assistant installation.
struct DetectedAssistant {
    name: &'static str,
    config_dir: PathBuf,
    in_path: bool,
}

/// Check whether `name` is found on PATH (e.g. "claude", "cursor", "windsurf").
fn is_in_path(name: &str) -> bool {
    // Use the `which` approach: iterate PATH directories
    if let Ok(path_var) = std::env::var("PATH") {
        for dir in std::env::split_paths(&path_var) {
            let candidate = dir.join(name);
            if candidate.is_file() {
                return true;
            }
        }
    }
    false
}

/// Detect which AI coding assistants are installed on the system.
fn detect_assistants() -> Vec<DetectedAssistant> {
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    let mut found = Vec::new();

    // Claude Code: ~/.claude/ directory or `claude` in PATH
    let claude_dir = home.join(".claude");
    let claude_in_path = is_in_path("claude");
    if claude_dir.is_dir() || claude_in_path {
        found.push(DetectedAssistant {
            name: "Claude Code",
            config_dir: claude_dir,
            in_path: claude_in_path,
        });
    }

    // Cursor: ~/.cursor/ directory
    let cursor_dir = home.join(".cursor");
    let cursor_in_path = is_in_path("cursor");
    if cursor_dir.is_dir() || cursor_in_path {
        found.push(DetectedAssistant {
            name: "Cursor",
            config_dir: cursor_dir,
            in_path: cursor_in_path,
        });
    }

    // Windsurf: ~/.windsurf/ directory
    let windsurf_dir = home.join(".windsurf");
    let windsurf_in_path = is_in_path("windsurf");
    if windsurf_dir.is_dir() || windsurf_in_path {
        found.push(DetectedAssistant {
            name: "Windsurf",
            config_dir: windsurf_dir,
            in_path: windsurf_in_path,
        });
    }

    found
}

// ── Init Command ───────────────────────────────────────────────────────────

fn cmd_init(project_dir: &std::path::Path, skip_model: bool) -> anyhow::Result<()> {
    println!(
        "Codemem init: setting up memory engine for {}\n",
        project_dir.display()
    );

    // Track status for final summary
    let mut status_lines: Vec<String> = Vec::new();

    // ── Step 1: Detect AI assistants ───────────────────────────────────────
    let assistants = detect_assistants();
    if assistants.is_empty() {
        println!("[detect] No AI coding assistants detected (Claude Code, Cursor, Windsurf)");
        println!(
            "         Codemem will still set up hooks and MCP config in the project directory.\n"
        );
    } else {
        println!("[detect] Found AI coding assistants:");
        for asst in &assistants {
            let path_info = if asst.in_path { " (in PATH)" } else { "" };
            println!(
                "         - {}{} ({})",
                asst.name,
                path_info,
                asst.config_dir.display()
            );
        }
        println!();
    }

    let has_claude = assistants.iter().any(|a| a.name == "Claude Code");
    let has_cursor = assistants.iter().any(|a| a.name == "Cursor");
    let has_windsurf = assistants.iter().any(|a| a.name == "Windsurf");

    // ── Step 2: Create global ~/.codemem directory and database ─────────────
    let global_dir = dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".codemem");
    std::fs::create_dir_all(global_dir.join("models"))?;

    let db_path = codemem_db_path();
    let _storage = codemem_storage::Storage::open(&db_path)?;
    println!("[database] Initialized at {}", db_path.display());
    status_lines.push(format!("Database: {}", db_path.display()));

    // ── Step 3: Write lifecycle hooks into Claude Code settings ────────────
    // Register 4 hooks: SessionStart, UserPromptSubmit, PostToolUse, Stop
    // Claude Code reads hooks from .claude/settings.json in the PROJECT directory
    {
        let hooks_dir = project_dir.join(".claude");
        std::fs::create_dir_all(&hooks_dir)?;
        let settings_path = hooks_dir.join("settings.json");

        let mut settings: serde_json::Value = if settings_path.exists() {
            let content = std::fs::read_to_string(&settings_path)?;
            serde_json::from_str(&content).unwrap_or(serde_json::json!({}))
        } else {
            serde_json::json!({})
        };

        let hooks = settings
            .as_object_mut()
            .unwrap()
            .entry("hooks")
            .or_insert_with(|| serde_json::json!({}));

        if !hooks.is_object() {
            *hooks = serde_json::json!({});
        }

        // Define all 4 codemem hooks
        // matcher is a regex string filtering tool names (PostToolUse) or session type (SessionStart)
        // UserPromptSubmit and Stop don't support matchers — omit the field
        let hook_defs: Vec<(&str, &str, serde_json::Value)> = vec![
            (
                "SessionStart",
                "codemem context",
                serde_json::json!([{
                    "hooks": [{
                        "type": "command",
                        "command": "codemem context",
                        "timeout": 10000
                    }]
                }]),
            ),
            (
                "UserPromptSubmit",
                "codemem prompt",
                serde_json::json!([{
                    "hooks": [{
                        "type": "command",
                        "command": "codemem prompt",
                        "timeout": 5000
                    }]
                }]),
            ),
            (
                "PostToolUse",
                "codemem ingest",
                serde_json::json!([{
                    "matcher": "Read|Glob|Grep|Edit|Write|MultiEdit",
                    "hooks": [{
                        "type": "command",
                        "command": "codemem ingest",
                        "timeout": 5000
                    }]
                }]),
            ),
            (
                "Stop",
                "codemem summarize",
                serde_json::json!([{
                    "hooks": [{
                        "type": "command",
                        "command": "codemem summarize",
                        "timeout": 10000
                    }]
                }]),
            ),
        ];

        let mut hooks_added = 0;
        let mut hooks_skipped = 0;

        for (event_name, cmd_name, hook_value) in &hook_defs {
            let event_hooks = hooks
                .as_object_mut()
                .unwrap()
                .entry(*event_name)
                .or_insert_with(|| serde_json::json!([]));

            if !event_hooks.is_array() {
                *event_hooks = serde_json::json!([]);
            }

            // Check if an codemem hook already exists for this event
            let already_exists = event_hooks.as_array().unwrap().iter().any(|h| {
                h.get("hooks")
                    .and_then(|arr| arr.as_array())
                    .map(|arr| {
                        arr.iter().any(|entry| {
                            entry
                                .get("command")
                                .and_then(|c| c.as_str())
                                .map(|c| c.starts_with("codemem "))
                                .unwrap_or(false)
                        })
                    })
                    .unwrap_or(false)
            });

            if already_exists {
                hooks_skipped += 1;
            } else {
                // Append the hook entries from the value array
                if let Some(entries) = hook_value.as_array() {
                    for entry in entries {
                        event_hooks.as_array_mut().unwrap().push(entry.clone());
                    }
                }
                println!("[hooks] Added {} -> {} hook", event_name, cmd_name);
                hooks_added += 1;
            }
        }

        if hooks_added > 0 {
            status_lines.push(format!(
                "Hooks: {} lifecycle hooks configured ({})",
                hooks_added,
                settings_path.display()
            ));
        }
        if hooks_skipped > 0 {
            println!("[hooks] {} hook(s) already present, skipped", hooks_skipped);
            if hooks_added == 0 {
                status_lines.push("Hooks: all already configured (no changes)".to_string());
            }
        }

        std::fs::write(&settings_path, serde_json::to_string_pretty(&settings)?)?;
    }

    // ── Step 4: Register codemem as MCP server (.mcp.json) ─────────────────
    // Write to .mcp.json in the project directory, merging non-destructively
    {
        let mcp_json_path = project_dir.join(".mcp.json");

        let mut mcp_config: serde_json::Value = if mcp_json_path.exists() {
            let content = std::fs::read_to_string(&mcp_json_path)?;
            serde_json::from_str(&content).unwrap_or(serde_json::json!({}))
        } else {
            serde_json::json!({})
        };

        // Ensure mcpServers object exists
        let servers = mcp_config
            .as_object_mut()
            .unwrap()
            .entry("mcpServers")
            .or_insert_with(|| serde_json::json!({}));

        if !servers.is_object() {
            *servers = serde_json::json!({});
        }

        let servers_map = servers.as_object_mut().unwrap();

        if servers_map.contains_key("codemem") {
            println!(
                "[mcp] Codemem MCP server already registered in {}",
                mcp_json_path.display()
            );
            status_lines.push("MCP: already registered (no changes)".to_string());
        } else {
            servers_map.insert(
                "codemem".to_string(),
                serde_json::json!({
                    "command": "codemem",
                    "args": ["serve"]
                }),
            );
            println!(
                "[mcp] Registered codemem MCP server in {}",
                mcp_json_path.display()
            );
            status_lines.push(format!("MCP: codemem serve ({})", mcp_json_path.display()));
        }

        std::fs::write(&mcp_json_path, serde_json::to_string_pretty(&mcp_config)?)?;
    }

    // ── Step 5: Download embedding model ──────────────────────────────────
    if skip_model {
        println!("[model] Skipped (--skip-model)");
        status_lines.push("Model: skipped (--skip-model)".to_string());
    } else {
        let model_dir = codemem_embeddings::EmbeddingService::default_model_dir();
        if model_dir.join("model.safetensors").exists() {
            println!(
                "[model] Embedding model already downloaded at {}",
                model_dir.display()
            );
            status_lines.push("Model: already present".to_string());
        } else {
            println!("[model] Downloading embedding model (BAAI/bge-base-en-v1.5)...");
            match codemem_embeddings::EmbeddingService::download_model(&model_dir) {
                Ok(_) => {
                    println!("[model] Downloaded to {}", model_dir.display());
                    status_lines.push(format!("Model: downloaded to {}", model_dir.display()));
                }
                Err(e) => {
                    println!("[model] Download failed: {e}");
                    println!("        Codemem will work without embeddings (text search only)");
                    println!("        Re-run `codemem init` later to retry download");
                    status_lines.push("Model: download failed (text search only)".to_string());
                }
            }
        }
    }

    // ── Step 6: Batch embed existing memories ─────────────────────────────
    if !skip_model {
        batch_embed_existing(&db_path);
    }

    // ── Final Summary ─────────────────────────────────────────────────────
    println!("\n{}", "=".repeat(60));
    println!("Codemem initialization complete\n");

    // Detected assistants
    if assistants.is_empty() {
        println!("  Assistants: none detected");
    } else {
        let names: Vec<&str> = assistants.iter().map(|a| a.name).collect();
        println!("  Assistants: {}", names.join(", "));
    }

    for line in &status_lines {
        println!("  {}", line);
    }

    // Helpful notes for detected assistants
    println!();
    if has_claude {
        println!("  Claude Code: hooks and MCP server configured. Ready to use.");
    }
    if has_cursor {
        println!("  Cursor: .mcp.json configured. Add PostToolUse hooks manually if supported.");
    }
    if has_windsurf {
        println!("  Windsurf: .mcp.json configured. Add PostToolUse hooks manually if supported.");
    }
    if !has_claude && !has_cursor && !has_windsurf {
        println!("  No assistants detected. Install Claude Code, Cursor, or Windsurf,");
        println!("  then re-run `codemem init` to auto-configure integration.");
    }

    println!();
    println!("Next steps:");
    println!("  1. Start a coding session -- codemem will passively capture context");
    println!("  2. Search your memories: codemem search \"<query>\"");
    println!("  3. View stats: codemem stats");

    Ok(())
}

/// Batch-embed existing memories that don't yet have embeddings,
/// and update the vector index. Called at the end of cmd_init().
fn batch_embed_existing(db_path: &std::path::Path) {
    let storage = match codemem_storage::Storage::open(db_path) {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!("Could not open storage for batch embedding: {e}");
            return;
        }
    };

    let emb_service = match codemem_embeddings::from_env() {
        Ok(s) => s,
        Err(_) => {
            // Embedding provider not available, skip batch embedding
            return;
        }
    };

    // Find memories that lack embeddings
    let conn = storage.connection();
    let mut stmt = match conn.prepare(
        "SELECT m.id, m.content FROM memories m
         LEFT JOIN memory_embeddings me ON m.id = me.memory_id
         WHERE me.memory_id IS NULL",
    ) {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!("Failed to query unembedded memories: {e}");
            return;
        }
    };

    let rows: Vec<(String, String)> = match stmt.query_map([], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
    }) {
        Ok(r) => r.filter_map(|r| r.ok()).collect(),
        Err(e) => {
            tracing::warn!("Failed to collect unembedded memories: {e}");
            return;
        }
    };

    if rows.is_empty() {
        tracing::info!("All memories already have embeddings");
        return;
    }

    tracing::info!("Batch embedding {} memories without embeddings", rows.len());
    println!("Embedding {} existing memories...", rows.len());

    let index_path = db_path.with_extension("idx");
    let mut vector = match codemem_vector::HnswIndex::with_defaults() {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!("Failed to create vector index: {e}");
            return;
        }
    };
    if index_path.exists() {
        let _ = vector.load(&index_path);
    }

    let mut embedded = 0usize;
    let total = rows.len();

    // Wrap in a transaction so each store_embedding doesn't trigger an fsync
    let _ = storage.connection().execute_batch("BEGIN");

    for (batch_idx, chunk) in rows.chunks(32).enumerate() {
        let texts: Vec<&str> = chunk.iter().map(|(_, content)| content.as_str()).collect();
        if let Ok(embeddings) = emb_service.embed_batch(&texts) {
            for ((id, _), embedding) in chunk.iter().zip(embeddings) {
                let _ = storage.store_embedding(id, &embedding);
                let _ = vector.insert(id, &embedding);
                embedded += 1;
            }
        }
        let done = (batch_idx + 1) * 32;
        print!("\r  Embedding: {}/{}", done.min(total), total);
        std::io::Write::flush(&mut std::io::stdout()).ok();
    }

    let _ = storage.connection().execute_batch("COMMIT");

    if total > 0 {
        println!(); // newline after progress
    }
    if embedded > 0 {
        let _ = vector.save(&index_path);
        tracing::info!("Batch embedded {} of {} memories", embedded, total);
        println!("  Embedded and indexed {} memories.", embedded);
    }
}

fn cmd_search(query: &str, k: usize, namespace: Option<&str>) -> anyhow::Result<()> {
    let db_path = codemem_db_path();
    let storage = codemem_storage::Storage::open(&db_path)?;

    // Try loading embeddings for vector search
    let emb_service = codemem_embeddings::from_env().ok();

    let mut vector = codemem_vector::HnswIndex::with_defaults()?;
    let index_path = db_path.with_extension("idx");
    if index_path.exists() {
        vector.load(&index_path)?;
    }

    // Try vector search first
    let vector_results: Vec<(String, f32)> = if let Some(ref emb) = emb_service {
        match emb.embed(query) {
            Ok(query_embedding) => vector.search(&query_embedding, k * 2).unwrap_or_default(),
            Err(_) => vec![],
        }
    } else {
        vec![]
    };

    if !vector_results.is_empty() {
        println!("Top {} results for: \"{}\" (vector search)\n", k, query);
        let mut shown = 0;
        for (id, distance) in &vector_results {
            if shown >= k {
                break;
            }
            let similarity = 1.0 - *distance as f64;
            if let Some(memory) = storage.get_memory(id)? {
                // Apply namespace filter
                if let Some(ns) = namespace {
                    if memory.namespace.as_deref() != Some(ns) {
                        continue;
                    }
                }
                println!(
                    "  [{:.3}] [{}] {}",
                    similarity, memory.memory_type, memory.id
                );
                println!("         {}", truncate_str(&memory.content, 120));
                if !memory.tags.is_empty() {
                    println!("         tags: {}", memory.tags.join(", "));
                }
                println!();
                shown += 1;
            } else if let Some(node) = storage.get_graph_node(id)? {
                // Fallback: symbol/graph node (e.g. sym:* IDs from code indexing)
                if let Some(ns) = namespace {
                    if node.namespace.as_deref() != Some(ns) {
                        continue;
                    }
                }
                println!("  [{:.3}] [{}] {}", similarity, node.kind, node.id);
                println!("         {}", truncate_str(&node.label, 120));
                println!();
                shown += 1;
            }
        }
        return Ok(());
    }

    // Fallback: text search
    let ids = if let Some(ns) = namespace {
        storage.list_memory_ids_for_namespace(ns)?
    } else {
        storage.list_memory_ids()?
    };

    if ids.is_empty() {
        println!("No memories stored yet.");
        return Ok(());
    }

    let query_lower = query.to_lowercase();
    println!(
        "Searching {} memories for: \"{}\" (text search)\n",
        ids.len(),
        query
    );

    let mut found = 0;
    for id in &ids {
        if found >= k {
            break;
        }
        if let Some(memory) = storage.get_memory(id)? {
            if memory.content.to_lowercase().contains(&query_lower) {
                println!(
                    "  [{}] {} (importance: {:.1})",
                    memory.memory_type, memory.id, memory.importance
                );
                println!("    {}", truncate_str(&memory.content, 120));
                println!();
                found += 1;
            }
        }
    }

    // Also search graph nodes by label
    let conn = storage.connection();
    let pattern = format!("%{}%", query_lower);
    let mut gn_stmt = conn.prepare(
        "SELECT id, kind, label, namespace FROM graph_nodes WHERE LOWER(label) LIKE ?1 ORDER BY centrality DESC LIMIT ?2",
    )?;
    let gn_rows: Vec<(String, String, String, Option<String>)> = gn_stmt
        .query_map(params![pattern, k as i64], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, Option<String>>(3)?,
            ))
        })?
        .filter_map(|r| r.ok())
        .collect();

    for (id, kind, label, ns) in &gn_rows {
        if found >= k {
            break;
        }
        if let Some(filter_ns) = namespace {
            if ns.as_deref() != Some(filter_ns) {
                continue;
            }
        }
        println!("  [{}] {} (graph node)", kind, id);
        println!("    {}", truncate_str(label, 120));
        println!();
        found += 1;
    }

    if found == 0 {
        println!("No matching memories or graph nodes found.");
    }

    Ok(())
}

fn cmd_stats() -> anyhow::Result<()> {
    let db_path = codemem_db_path();
    let storage = codemem_storage::Storage::open(&db_path)?;
    let stats = storage.stats()?;

    println!("Codemem Statistics");
    println!("  Memories:    {}", stats.memory_count);
    println!("  Embeddings:  {}", stats.embedding_count);
    println!("  Graph nodes: {}", stats.node_count);
    println!("  Graph edges: {}", stats.edge_count);

    // Vector index
    let index_path = db_path.with_extension("idx");
    if index_path.exists() {
        if let Ok(mut vector) = codemem_vector::HnswIndex::with_defaults() {
            if vector.load(&index_path).is_ok() {
                let vstats = vector.stats();
                println!("  Vector indexed: {}", vstats.count);
            }
        }
    }

    // Embedding provider
    match codemem_embeddings::from_env() {
        Ok(provider) => println!(
            "  Embedding provider: {} ({}d)",
            provider.name(),
            provider.dimensions()
        ),
        Err(_) => println!("  Embedding provider: not configured"),
    }

    Ok(())
}

fn cmd_serve() -> anyhow::Result<()> {
    let db_path = codemem_db_path();

    // Task 5: Graceful error recovery for corrupt/missing ONNX model and stale vector index.
    // McpServer::from_db_path handles most of this, but we wrap it to catch vector index issues.
    let storage = codemem_storage::Storage::open(&db_path)?;
    let mut vector = codemem_vector::HnswIndex::with_defaults()?;

    let index_path = db_path.with_extension("idx");
    if index_path.exists() {
        if let Err(e) = vector.load(&index_path) {
            tracing::warn!("Stale or corrupt vector index, rebuilding: {e}");
            vector = rebuild_vector_index(&storage)?;
            let _ = vector.save(&index_path);
        }
    }

    let graph = codemem_graph::GraphEngine::from_storage(&storage)?;

    // Try loading embeddings; log warning and continue without if unavailable
    let embeddings = match codemem_embeddings::from_env() {
        Ok(provider) => Some(provider),
        Err(e) => {
            tracing::warn!("Embedding provider unavailable, continuing without embeddings: {e}");
            None
        }
    };

    let server = codemem_mcp::McpServer::new(storage, vector, graph, embeddings);
    tracing::info!(
        "Codemem MCP server ready (stdio mode, db: {})",
        db_path.display()
    );
    server.run()?;
    Ok(())
}

fn cmd_ingest() -> anyhow::Result<()> {
    use std::io::Read;

    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input)?;

    if input.trim().is_empty() {
        return Ok(());
    }

    let payload = codemem_hooks::parse_payload(&input)?;
    let extracted = codemem_hooks::extract(&payload)?;

    if let Some(mut extracted) = extracted {
        let db_path = codemem_db_path();
        let storage = codemem_storage::Storage::open(&db_path)?;

        // Build the set of existing graph node IDs so we can detect
        // Read-then-Edit/Write patterns and create edges.
        let existing_node_ids: std::collections::HashSet<String> = storage
            .all_graph_nodes()
            .unwrap_or_default()
            .into_iter()
            .map(|n| n.id)
            .collect();

        // Resolve edges based on previously-seen file nodes
        codemem_hooks::resolve_edges(&mut extracted, &existing_node_ids);

        // Dedup on raw content hash (before compression) for consistency
        let hash = codemem_hooks::content_hash(&extracted.content);
        let now = chrono::Utc::now();
        let id = uuid::Uuid::new_v4().to_string();

        // Compress observation via LLM if configured
        let compressor = compress::CompressProvider::from_env();
        let tool_name = extracted
            .metadata
            .get("tool")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        let file_path = extracted.metadata.get("file_path").and_then(|v| v.as_str());
        let (content, compressed) =
            if let Some(summary) = compressor.compress(&extracted.content, tool_name, file_path) {
                extracted
                    .metadata
                    .insert("compressed".to_string(), serde_json::Value::Bool(true));
                extracted.metadata.insert(
                    "original_len".to_string(),
                    serde_json::json!(extracted.content.len()),
                );
                (summary, true)
            } else {
                (extracted.content.clone(), false)
            };

        // Use current working directory as namespace
        let namespace = std::env::current_dir()
            .ok()
            .map(|p| p.to_string_lossy().to_string());

        let memory = codemem_core::MemoryNode {
            id: id.clone(),
            content: content.clone(),
            memory_type: extracted.memory_type,
            importance: 0.5,
            confidence: 1.0,
            access_count: 0,
            content_hash: hash,
            tags: extracted.tags,
            metadata: extracted.metadata,
            namespace,
            created_at: now,
            updated_at: now,
            last_accessed_at: now,
        };

        match storage.insert_memory(&memory) {
            Ok(()) => {
                tracing::info!(
                    "Stored memory {} ({}){}",
                    id,
                    memory.memory_type,
                    if compressed { " [compressed]" } else { "" }
                );

                // Auto-embed and index with error recovery (Task 5)
                match codemem_embeddings::from_env() {
                    Ok(emb_service) => {
                        if let Ok(embedding) = emb_service.embed(&content) {
                            let _ = storage.store_embedding(&id, &embedding);

                            // Load and update vector index
                            let index_path = db_path.with_extension("idx");
                            let mut vector = codemem_vector::HnswIndex::with_defaults()?;
                            if index_path.exists() {
                                if let Err(e) = vector.load(&index_path) {
                                    // Stale vector index: rebuild from stored embeddings
                                    tracing::warn!(
                                        "Stale vector index during ingest, rebuilding: {e}"
                                    );
                                    vector = rebuild_vector_index(&storage)?;
                                }
                            }
                            if vector.insert(&id, &embedding).is_ok() {
                                let _ = vector.save(&index_path);
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!("Embedding model unavailable, skipping embedding: {e}");
                    }
                }

                // Store graph node if present
                if let Some(ref node) = extracted.graph_node {
                    let _ = storage.insert_graph_node(node);
                }

                // Store any pending graph edges
                let edges = codemem_hooks::materialize_edges(&extracted.graph_edges, &id);
                for edge in &edges {
                    if let Err(e) = storage.insert_graph_edge(edge) {
                        tracing::debug!("Failed to store graph edge {}: {e}", edge.id);
                    } else {
                        tracing::info!(
                            "Stored graph edge {} ({} -> {})",
                            edge.id,
                            edge.src,
                            edge.dst
                        );
                    }
                }
            }
            Err(codemem_core::CodememError::Duplicate(_)) => {
                tracing::debug!("Skipped duplicate content");
            }
            Err(e) => return Err(e.into()),
        }
    }

    Ok(())
}

// ── Watch Command ─────────────────────────────────────────────────────────

fn cmd_watch(watch_dir: &std::path::Path) -> anyhow::Result<()> {
    if !watch_dir.is_dir() {
        anyhow::bail!("Not a directory: {}", watch_dir.display());
    }

    let db_path = codemem_db_path();
    let storage = codemem_storage::Storage::open(&db_path)?;

    let emb_service = codemem_embeddings::from_env().ok();

    let index_path = db_path.with_extension("idx");
    let mut vector = codemem_vector::HnswIndex::with_defaults()?;
    if index_path.exists() {
        let _ = vector.load(&index_path);
    }

    let watcher = codemem_watch::FileWatcher::new(watch_dir)?;
    println!(
        "Watching {} for file changes (Ctrl+C to stop)",
        watch_dir.display()
    );

    let receiver = watcher.receiver();
    let mut changes_since_save = 0usize;

    while let Ok(event) = receiver.recv() {
        let path = match &event {
            codemem_watch::WatchEvent::FileChanged(p)
            | codemem_watch::WatchEvent::FileCreated(p)
            | codemem_watch::WatchEvent::FileDeleted(p) => p.clone(),
        };

        let language = codemem_watch::detect_language(&path).unwrap_or("unknown");

        let relative = path
            .strip_prefix(watch_dir)
            .unwrap_or(&path)
            .to_string_lossy();

        match &event {
            codemem_watch::WatchEvent::FileDeleted(_) => {
                println!("  [deleted] {relative}");
            }
            _ => {
                // Index the changed file
                if let Ok(content) = std::fs::read_to_string(&path) {
                    let now = chrono::Utc::now();
                    let id = uuid::Uuid::new_v4().to_string();
                    let hash = codemem_storage::Storage::content_hash(&content);

                    let mut metadata = std::collections::HashMap::new();
                    metadata.insert(
                        "file_path".to_string(),
                        serde_json::Value::String(relative.to_string()),
                    );
                    metadata.insert(
                        "language".to_string(),
                        serde_json::Value::String(language.to_string()),
                    );
                    metadata.insert(
                        "source".to_string(),
                        serde_json::Value::String("file_watcher".to_string()),
                    );

                    let summary = format!(
                        "[{language}] File {} ({})",
                        relative,
                        match &event {
                            codemem_watch::WatchEvent::FileCreated(_) => "created",
                            _ => "modified",
                        }
                    );

                    let memory = codemem_core::MemoryNode {
                        id: id.clone(),
                        content: summary,
                        memory_type: codemem_core::MemoryType::Context,
                        importance: 0.3,
                        confidence: 1.0,
                        access_count: 0,
                        content_hash: hash,
                        tags: vec![language.to_string(), "file_watch".to_string()],
                        metadata,
                        namespace: Some(watch_dir.to_string_lossy().to_string()),
                        created_at: now,
                        updated_at: now,
                        last_accessed_at: now,
                    };

                    match storage.insert_memory(&memory) {
                        Ok(()) => {
                            // Auto-embed if available
                            if let Some(ref emb) = emb_service {
                                if let Ok(embedding) = emb.embed(&memory.content) {
                                    let _ = storage.store_embedding(&id, &embedding);
                                    let _ = vector.insert(&id, &embedding);
                                    changes_since_save += 1;
                                }
                            }
                            println!("  [indexed] {relative} ({language})");
                        }
                        Err(codemem_core::CodememError::Duplicate(_)) => {
                            // Skip duplicate content
                        }
                        Err(e) => {
                            tracing::warn!("Failed to store watch memory: {e}");
                        }
                    }
                }
            }
        }

        // Periodically save vector index
        if changes_since_save >= 10 {
            let _ = vector.save(&index_path);
            changes_since_save = 0;
        }
    }

    // Final save
    if changes_since_save > 0 {
        let _ = vector.save(&index_path);
    }

    Ok(())
}

// ── Sessions Commands ─────────────────────────────────────────────────────

fn cmd_sessions_list(namespace: Option<&str>) -> anyhow::Result<()> {
    let db_path = codemem_db_path();
    let storage = codemem_storage::Storage::open(&db_path)?;
    let sessions = storage.list_sessions(namespace)?;

    if sessions.is_empty() {
        println!("No sessions recorded yet.");
        return Ok(());
    }

    println!("Sessions ({}):\n", sessions.len());
    for session in &sessions {
        let started = session.started_at.format("%Y-%m-%d %H:%M:%S UTC");
        let ended = session
            .ended_at
            .map(|t| t.format("%Y-%m-%d %H:%M:%S UTC").to_string())
            .unwrap_or_else(|| "active".to_string());
        let ns = session.namespace.as_deref().unwrap_or("(global)");

        println!("  {} [{}]", session.id, ns);
        println!("    Started: {}  Ended: {}", started, ended);
        println!("    Memories: {}", session.memory_count);
        if let Some(ref summary) = session.summary {
            println!("    Summary: {}", summary);
        }
        println!();
    }

    Ok(())
}

fn cmd_sessions_start(namespace: Option<&str>) -> anyhow::Result<()> {
    let db_path = codemem_db_path();
    let storage = codemem_storage::Storage::open(&db_path)?;

    let session_id = uuid::Uuid::new_v4().to_string();
    storage.start_session(&session_id, namespace)?;

    println!("Session started: {}", session_id);
    if let Some(ns) = namespace {
        println!("  Namespace: {}", ns);
    }
    println!(
        "\nUse `codemem sessions end {}` to close this session.",
        session_id
    );

    Ok(())
}

fn cmd_sessions_end(id: &str, summary: Option<&str>) -> anyhow::Result<()> {
    let db_path = codemem_db_path();
    let storage = codemem_storage::Storage::open(&db_path)?;
    storage.end_session(id, summary)?;

    println!("Session ended: {}", id);
    if let Some(s) = summary {
        println!("  Summary: {}", s);
    }

    Ok(())
}

// ── Lifecycle Hooks (SessionStart, UserPromptSubmit, Stop) ─────────────────

/// SessionStart hook: query recent memories and inject context into the new session.
///
/// Reads JSON `{session_id, cwd}` from stdin.
/// Outputs JSON `{hookSpecificOutput: {additionalContext: "..."}}` to stdout.
fn cmd_context() -> anyhow::Result<()> {
    use std::io::Read;

    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input)?;

    let payload: serde_json::Value = if input.trim().is_empty() {
        serde_json::json!({})
    } else {
        serde_json::from_str(&input).unwrap_or(serde_json::json!({}))
    };

    let cwd = payload.get("cwd").and_then(|v| v.as_str()).unwrap_or("");
    let session_id = payload
        .get("session_id")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    let db_path = codemem_db_path();
    let storage = match codemem_storage::Storage::open(&db_path) {
        Ok(s) => s,
        Err(_) => {
            // No database yet — nothing to inject
            let output = serde_json::json!({});
            println!("{}", serde_json::to_string(&output)?);
            return Ok(());
        }
    };

    // Auto-start a session for this project
    if !session_id.is_empty() {
        let namespace = if cwd.is_empty() { None } else { Some(cwd) };
        let _ = storage.start_session(session_id, namespace);
    }

    let namespace = if cwd.is_empty() { None } else { Some(cwd) };

    // Gather context from multiple sources
    let mut sections: Vec<String> = Vec::new();

    // 1. Recent sessions with summaries
    if let Ok(sessions) = storage.list_sessions(namespace) {
        let with_summaries: Vec<_> = sessions
            .iter()
            .filter(|s| s.summary.is_some() && s.ended_at.is_some())
            .take(5)
            .collect();
        if !with_summaries.is_empty() {
            let mut sec = String::from("### Recent Sessions\n\n");
            sec.push_str("| Date | Memories | Summary |\n|------|----------|---------|\n");
            for s in &with_summaries {
                let date = s.started_at.format("%Y-%m-%d %H:%M");
                let summary = s.summary.as_deref().unwrap_or("-");
                sec.push_str(&format!(
                    "| {} | {} | {} |\n",
                    date,
                    s.memory_count,
                    truncate_str(summary, 80)
                ));
            }
            sections.push(sec);
        }
    }

    // 2. Recent Decision and Insight memories (highest signal)
    let memory_ids = if let Some(ns) = namespace {
        storage
            .list_memory_ids_for_namespace(ns)
            .unwrap_or_default()
    } else {
        storage.list_memory_ids().unwrap_or_default()
    };

    let mut recent_memories: Vec<codemem_core::MemoryNode> = Vec::new();
    for id in memory_ids.iter().rev().take(200) {
        if let Ok(Some(m)) = storage.get_memory(id) {
            if matches!(
                m.memory_type,
                codemem_core::MemoryType::Decision
                    | codemem_core::MemoryType::Insight
                    | codemem_core::MemoryType::Pattern
            ) {
                recent_memories.push(m);
            }
            if recent_memories.len() >= 15 {
                break;
            }
        }
    }

    if !recent_memories.is_empty() {
        let mut sec = String::from("### Key Memories\n\n");
        sec.push_str("| Type | Tags | Content |\n|------|------|---------|\n");
        for m in &recent_memories {
            let tags = if m.tags.is_empty() {
                "-".to_string()
            } else {
                m.tags
                    .iter()
                    .take(3)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            sec.push_str(&format!(
                "| {} | {} | {} |\n",
                m.memory_type,
                tags,
                truncate_str(&m.content.replace('\n', " "), 80)
            ));
        }
        sec.push_str("\n*Use `recall_memory` MCP tool for full details.*");
        sections.push(sec);
    }

    // 3. File hotspots (most frequently touched files)
    if let Ok(hotspots) = storage.get_file_hotspots(5, namespace) {
        if !hotspots.is_empty() {
            let mut sec = String::from("### File Hotspots\n\n");
            for (path, count, _ids) in &hotspots {
                sec.push_str(&format!(
                    "- `{}` ({} interactions)\n",
                    short_path(path),
                    count
                ));
            }
            sections.push(sec);
        }
    }

    // 4. Detected patterns
    let mut pattern_items: Vec<String> = Vec::new();
    if let Ok(searches) = storage.get_repeated_searches(3, namespace) {
        for (pattern, count, _ids) in searches.iter().take(3) {
            pattern_items.push(format!(
                "- Repeated search: \"{}\" ({} times)",
                pattern, count
            ));
        }
    }
    if !pattern_items.is_empty() {
        let mut sec = String::from("### Detected Patterns\n\n");
        for item in &pattern_items {
            sec.push_str(item);
            sec.push('\n');
        }
        sections.push(sec);
    }

    // 5. Stats overview
    if let Ok(stats) = storage.stats() {
        if stats.memory_count > 0 {
            let sec = format!(
                "### Codemem Stats\n\n{} memories, {} graph nodes, {} edges, {} embeddings",
                stats.memory_count, stats.node_count, stats.edge_count, stats.embedding_count
            );
            sections.push(sec);
        }
    }

    // Build the final context
    if sections.is_empty() {
        let output = serde_json::json!({});
        println!("{}", serde_json::to_string(&output)?);
    } else {
        let mut context = String::from("<codemem-context>\n# Codemem Memory Context\n\n");
        context.push_str("Prior knowledge from this project's memory graph. ");
        context.push_str("Use `recall_memory` and `search_code` MCP tools for details.\n\n");
        for sec in &sections {
            context.push_str(sec);
            context.push_str("\n\n");
        }
        context.push_str("</codemem-context>");

        let output = serde_json::json!({
            "hookSpecificOutput": {
                "additionalContext": context
            }
        });
        println!("{}", serde_json::to_string(&output)?);
    }

    Ok(())
}

/// UserPromptSubmit hook: record the user's prompt as a Context memory.
///
/// Reads JSON `{session_id, cwd, prompt}` from stdin.
/// Outputs JSON `{continue: true}` to stdout.
fn cmd_prompt() -> anyhow::Result<()> {
    use std::io::Read;

    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input)?;

    if input.trim().is_empty() {
        println!("{{}}");
        return Ok(());
    }

    let payload: serde_json::Value = serde_json::from_str(&input).unwrap_or(serde_json::json!({}));

    let prompt = payload.get("prompt").and_then(|v| v.as_str()).unwrap_or("");
    let session_id = payload.get("session_id").and_then(|v| v.as_str());
    let cwd = payload.get("cwd").and_then(|v| v.as_str());

    // Skip empty or very short prompts
    if prompt.len() < 5 {
        let output = serde_json::json!({"continue": true});
        println!("{}", serde_json::to_string(&output)?);
        return Ok(());
    }

    let db_path = codemem_db_path();
    let storage = match codemem_storage::Storage::open(&db_path) {
        Ok(s) => s,
        Err(_) => {
            let output = serde_json::json!({"continue": true});
            println!("{}", serde_json::to_string(&output)?);
            return Ok(());
        }
    };

    // Auto-start session if needed
    if let Some(sid) = session_id {
        if !sid.is_empty() {
            let _ = storage.start_session(sid, cwd);
        }
    }

    // Store prompt as a Context memory
    let content = format!("User prompt: {}", truncate_str(prompt, 2000));
    let content_hash = codemem_hooks::content_hash(&content);

    let now = chrono::Utc::now();
    let memory = codemem_core::MemoryNode {
        id: uuid::Uuid::new_v4().to_string(),
        content,
        memory_type: codemem_core::MemoryType::Context,
        importance: 0.3,
        confidence: 1.0,
        tags: vec!["prompt".to_string()],
        metadata: {
            let mut m = std::collections::HashMap::new();
            m.insert(
                "source".to_string(),
                serde_json::Value::String("UserPromptSubmit".to_string()),
            );
            if let Some(sid) = session_id {
                m.insert(
                    "session_id".to_string(),
                    serde_json::Value::String(sid.to_string()),
                );
            }
            m
        },
        namespace: cwd.map(|s| s.to_string()),
        content_hash,
        created_at: now,
        updated_at: now,
        last_accessed_at: now,
        access_count: 0,
    };

    let _ = storage.insert_memory(&memory);

    let output = serde_json::json!({"continue": true});
    println!("{}", serde_json::to_string(&output)?);

    Ok(())
}

/// Stop hook: build a session summary from captured memories and store it.
///
/// Reads JSON `{session_id, cwd}` from stdin.
/// Outputs JSON `{continue: true}` to stdout.
fn cmd_summarize() -> anyhow::Result<()> {
    use std::io::Read;

    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input)?;

    if input.trim().is_empty() {
        println!("{{}}");
        return Ok(());
    }

    let payload: serde_json::Value = serde_json::from_str(&input).unwrap_or(serde_json::json!({}));

    let session_id = payload
        .get("session_id")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let cwd = payload.get("cwd").and_then(|v| v.as_str());

    if session_id.is_empty() {
        let output = serde_json::json!({"continue": true});
        println!("{}", serde_json::to_string(&output)?);
        return Ok(());
    }

    let db_path = codemem_db_path();
    let storage = match codemem_storage::Storage::open(&db_path) {
        Ok(s) => s,
        Err(_) => {
            let output = serde_json::json!({"continue": true});
            println!("{}", serde_json::to_string(&output)?);
            return Ok(());
        }
    };

    let namespace = cwd;

    // Look up session start time to filter memories by creation time
    let session_start = storage
        .list_sessions(namespace)
        .unwrap_or_default()
        .into_iter()
        .find(|s| s.id == session_id)
        .map(|s| s.started_at)
        .unwrap_or_else(chrono::Utc::now);

    // Collect all memories created during this session
    let all_ids = if let Some(ns) = namespace {
        storage
            .list_memory_ids_for_namespace(ns)
            .unwrap_or_default()
    } else {
        storage.list_memory_ids().unwrap_or_default()
    };

    let mut session_memories: Vec<codemem_core::MemoryNode> = Vec::new();
    let mut files_read: Vec<String> = Vec::new();
    let mut files_edited: Vec<String> = Vec::new();
    let mut searches: Vec<String> = Vec::new();
    let mut decisions: Vec<String> = Vec::new();
    let mut prompts: Vec<String> = Vec::new();

    for id in &all_ids {
        if let Ok(Some(m)) = storage.get_memory(id) {
            // Filter to memories created during this session
            if m.created_at < session_start {
                continue;
            }
            // Categorize
            let tool = m
                .metadata
                .get("tool")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let source = m
                .metadata
                .get("source")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let file_path = m
                .metadata
                .get("file_path")
                .and_then(|v| v.as_str())
                .unwrap_or("");

            match tool {
                "Read" => {
                    if !file_path.is_empty() && !files_read.contains(&file_path.to_string()) {
                        files_read.push(file_path.to_string());
                    }
                }
                "Edit" | "Write" => {
                    if !file_path.is_empty() && !files_edited.contains(&file_path.to_string()) {
                        files_edited.push(file_path.to_string());
                    }
                }
                "Grep" | "Glob" => {
                    if let Some(pat) = m.metadata.get("pattern").and_then(|v| v.as_str()) {
                        if !searches.contains(&pat.to_string()) {
                            searches.push(pat.to_string());
                        }
                    }
                }
                _ => {}
            }

            if source == "UserPromptSubmit" {
                // Extract the prompt text (strip "User prompt: " prefix)
                let text = m
                    .content
                    .strip_prefix("User prompt: ")
                    .unwrap_or(&m.content);
                prompts.push(truncate_str(text, 120).to_string());
            }

            if m.memory_type == codemem_core::MemoryType::Decision {
                decisions.push(truncate_str(&m.content, 120).to_string());
            }

            session_memories.push(m);
        }
    }

    // Build summary
    let mut summary_parts: Vec<String> = Vec::new();

    if !prompts.is_empty() {
        summary_parts.push(format!(
            "Requests: {}",
            prompts
                .iter()
                .take(3)
                .cloned()
                .collect::<Vec<_>>()
                .join("; ")
        ));
    }

    if !files_read.is_empty() {
        summary_parts.push(format!(
            "Investigated {} file(s): {}",
            files_read.len(),
            files_read
                .iter()
                .take(5)
                .map(|p| short_path(p))
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }

    if !files_edited.is_empty() {
        summary_parts.push(format!(
            "Modified {} file(s): {}",
            files_edited.len(),
            files_edited
                .iter()
                .take(5)
                .map(|p| short_path(p))
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }

    if !decisions.is_empty() {
        summary_parts.push(format!(
            "Decisions: {}",
            decisions
                .iter()
                .take(3)
                .cloned()
                .collect::<Vec<_>>()
                .join("; ")
        ));
    }

    if !searches.is_empty() {
        summary_parts.push(format!(
            "Searched: {}",
            searches
                .iter()
                .take(5)
                .cloned()
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }

    let summary_text = if summary_parts.is_empty() {
        format!("{} memories captured.", session_memories.len())
    } else {
        summary_parts.join(". ")
    };

    // Store the summary as an Insight memory
    if !session_memories.is_empty() {
        let content_hash = codemem_hooks::content_hash(&summary_text);
        let now = chrono::Utc::now();
        let summary_memory = codemem_core::MemoryNode {
            id: uuid::Uuid::new_v4().to_string(),
            content: format!("Session summary: {}", summary_text),
            memory_type: codemem_core::MemoryType::Insight,
            importance: 0.7,
            confidence: 1.0,
            tags: vec!["session-summary".to_string()],
            metadata: {
                let mut m = std::collections::HashMap::new();
                m.insert(
                    "session_id".to_string(),
                    serde_json::Value::String(session_id.to_string()),
                );
                m.insert(
                    "files_read".to_string(),
                    serde_json::json!(files_read.len()),
                );
                m.insert(
                    "files_edited".to_string(),
                    serde_json::json!(files_edited.len()),
                );
                m.insert(
                    "total_memories".to_string(),
                    serde_json::json!(session_memories.len()),
                );
                m
            },
            namespace: namespace.map(|s| s.to_string()),
            content_hash,
            created_at: now,
            updated_at: now,
            last_accessed_at: now,
            access_count: 0,
        };
        let _ = storage.insert_memory(&summary_memory);
    }

    // End the session with summary
    let _ = storage.end_session(session_id, Some(&summary_text));

    let output = serde_json::json!({"continue": true});
    println!("{}", serde_json::to_string(&output)?);

    Ok(())
}

/// Shorten an absolute path to just filename or last 2 components.
fn short_path(path: &str) -> String {
    let parts: Vec<&str> = path.rsplitn(3, '/').collect();
    if parts.len() >= 2 {
        format!("{}/{}", parts[1], parts[0])
    } else {
        path.to_string()
    }
}

// ── Consolidation Cycles ───────────────────────────────────────────────────

fn cmd_consolidate(cycle: &str) -> anyhow::Result<()> {
    let db_path = codemem_db_path();
    let storage = codemem_storage::Storage::open(&db_path)?;

    println!("Running {} consolidation cycle...", cycle);

    let affected = match cycle {
        "decay" => consolidate_decay(&storage)?,
        "creative" => consolidate_creative(&storage)?,
        "cluster" => consolidate_cluster(&storage, &db_path)?,
        "forget" => consolidate_forget(&storage, &db_path)?,
        _ => {
            anyhow::bail!(
                "Unknown cycle type: '{}'. Valid types: decay, creative, cluster, forget",
                cycle
            );
        }
    };

    // Log the consolidation run
    if let Err(e) = storage.insert_consolidation_log(cycle, affected) {
        tracing::warn!("Failed to log consolidation run: {e}");
    }

    Ok(())
}

fn cmd_consolidate_status() -> anyhow::Result<()> {
    let db_path = codemem_db_path();
    let storage = codemem_storage::Storage::open(&db_path)?;
    let runs = storage.last_consolidation_runs()?;

    if runs.is_empty() {
        println!("No consolidation runs recorded yet.");
        return Ok(());
    }

    println!("Last consolidation runs:");
    for entry in &runs {
        let dt = chrono::DateTime::from_timestamp(entry.run_at, 0)
            .map(|t| t.format("%Y-%m-%d %H:%M:%S UTC").to_string())
            .unwrap_or_else(|| "unknown".to_string());
        println!(
            "  {:<10} last run: {}  ({} affected)",
            entry.cycle_type, dt, entry.affected_count
        );
    }

    // Show cycles that have never been run
    let all_cycles = ["decay", "creative", "cluster", "forget"];
    for cycle in &all_cycles {
        if !runs.iter().any(|r| r.cycle_type == *cycle) {
            println!("  {:<10} never run", cycle);
        }
    }

    Ok(())
}

/// Decay cycle: find memories not accessed in 30+ days, reduce importance by 10%.
fn consolidate_decay(storage: &codemem_storage::Storage) -> anyhow::Result<usize> {
    let conn = storage.connection();
    let threshold_ts = (chrono::Utc::now() - chrono::Duration::days(30)).timestamp();

    let count = conn.execute(
        "UPDATE memories SET importance = importance * 0.9, updated_at = ?1
         WHERE last_accessed_at < ?2",
        params![chrono::Utc::now().timestamp(), threshold_ts],
    )?;

    tracing::info!("Decay cycle complete: {} memories affected", count);
    println!("Decayed {} memories (importance reduced by 10%).", count);
    Ok(count)
}

/// Creative cycle: find pairs of memories with similar tags but different types,
/// create RELATES_TO edges between them if not already connected.
fn consolidate_creative(storage: &codemem_storage::Storage) -> anyhow::Result<usize> {
    let conn = storage.connection();

    // Load all memories with their id, type, and tags
    let mut stmt = conn.prepare("SELECT id, memory_type, tags FROM memories")?;

    let rows: Vec<(String, String, String)> = stmt
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
            ))
        })?
        .filter_map(|r| r.ok())
        .collect();

    // Parse tags for each memory
    let parsed: Vec<(String, String, Vec<String>)> = rows
        .into_iter()
        .map(|(id, mtype, tags_json)| {
            let tags: Vec<String> = serde_json::from_str(&tags_json).unwrap_or_default();
            (id, mtype, tags)
        })
        .collect();

    // Load existing edges to avoid duplicates
    let mut edge_stmt =
        conn.prepare("SELECT src, dst FROM graph_edges WHERE relationship = 'RELATES_TO'")?;
    let existing_edges: std::collections::HashSet<(String, String)> = edge_stmt
        .query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?
        .filter_map(|r| r.ok())
        .collect();

    let mut new_connections = 0usize;
    let now = chrono::Utc::now();

    for i in 0..parsed.len() {
        for j in (i + 1)..parsed.len() {
            let (ref id_a, ref type_a, ref tags_a) = parsed[i];
            let (ref id_b, ref type_b, ref tags_b) = parsed[j];

            // Different types required
            if type_a == type_b {
                continue;
            }

            // Must have at least one overlapping tag
            let has_common_tag = tags_a.iter().any(|t| tags_b.contains(t));
            if !has_common_tag {
                continue;
            }

            // Check not already connected in either direction
            if existing_edges.contains(&(id_a.clone(), id_b.clone()))
                || existing_edges.contains(&(id_b.clone(), id_a.clone()))
            {
                continue;
            }

            // Ensure both nodes exist in graph_nodes (upsert memory-type nodes)
            conn.execute(
                "INSERT OR IGNORE INTO graph_nodes (id, kind, label, payload, centrality, memory_id)
                 VALUES (?1, 'memory', ?1, '{}', 0.0, ?1)",
                params![id_a],
            )?;
            conn.execute(
                "INSERT OR IGNORE INTO graph_nodes (id, kind, label, payload, centrality, memory_id)
                 VALUES (?1, 'memory', ?1, '{}', 0.0, ?1)",
                params![id_b],
            )?;

            let edge_id = format!("{id_a}-RELATES_TO-{id_b}");
            let edge = codemem_core::Edge {
                id: edge_id.clone(),
                src: id_a.clone(),
                dst: id_b.clone(),
                relationship: codemem_core::RelationshipType::RelatesTo,
                weight: 1.0,
                properties: std::collections::HashMap::new(),
                created_at: now,
            };

            if storage.insert_graph_edge(&edge).is_ok() {
                new_connections += 1;
            }
        }
    }

    tracing::info!(
        "Creative cycle complete: {} new connections",
        new_connections
    );
    println!(
        "Creative cycle: created {} new RELATES_TO connections.",
        new_connections
    );
    Ok(new_connections)
}

/// Cluster cycle: find memories with the same content_hash prefix (first 8 chars)
/// and merge duplicates by keeping the one with highest importance, deleting others.
fn consolidate_cluster(
    storage: &codemem_storage::Storage,
    db_path: &std::path::Path,
) -> anyhow::Result<usize> {
    let conn = storage.connection();

    // Find groups of memories sharing the same 8-char content_hash prefix
    let mut stmt =
        conn.prepare("SELECT id, content_hash, importance FROM memories ORDER BY content_hash")?;

    let rows: Vec<(String, String, f64)> = stmt
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, f64>(2)?,
            ))
        })?
        .filter_map(|r| r.ok())
        .collect();

    // Group by first 8 chars of content_hash
    let mut groups: std::collections::HashMap<String, Vec<(String, f64)>> =
        std::collections::HashMap::new();
    for (id, hash, importance) in &rows {
        let prefix = if hash.len() >= 8 {
            hash[..8].to_string()
        } else {
            hash.clone()
        };
        groups
            .entry(prefix)
            .or_default()
            .push((id.clone(), *importance));
    }

    let mut merged_count = 0usize;
    let mut ids_to_delete: Vec<String> = Vec::new();

    for (_prefix, mut members) in groups {
        if members.len() <= 1 {
            continue;
        }

        // Sort by importance descending; keep the first (highest), delete the rest
        members.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        for (id, _importance) in members.iter().skip(1) {
            ids_to_delete.push(id.clone());
            merged_count += 1;
        }
    }

    // Delete the duplicates
    for id in &ids_to_delete {
        let _ = storage.delete_memory(id);
        // Also clean up embeddings and graph nodes
        let _ = conn.execute(
            "DELETE FROM memory_embeddings WHERE memory_id = ?1",
            params![id],
        );
        let _ = conn.execute("DELETE FROM graph_nodes WHERE memory_id = ?1", params![id]);
    }

    // Rebuild vector index if we deleted anything
    if merged_count > 0 {
        if let Ok(vector) = rebuild_vector_index(storage) {
            let index_path = db_path.with_extension("idx");
            let _ = vector.save(&index_path);
        }
    }

    tracing::info!("Cluster cycle complete: {} duplicates merged", merged_count);
    println!(
        "Cluster cycle: merged {} duplicate memories (by content_hash prefix).",
        merged_count
    );
    Ok(merged_count)
}

/// Forget cycle: delete memories with importance < 0.1 and access_count == 0.
fn consolidate_forget(
    storage: &codemem_storage::Storage,
    db_path: &std::path::Path,
) -> anyhow::Result<usize> {
    let conn = storage.connection();

    // Find memories to forget
    let mut stmt =
        conn.prepare("SELECT id FROM memories WHERE importance < 0.1 AND access_count = 0")?;

    let ids: Vec<String> = stmt
        .query_map([], |row| row.get(0))?
        .filter_map(|r| r.ok())
        .collect();

    let count = ids.len();

    for id in &ids {
        let _ = storage.delete_memory(id);
        let _ = conn.execute(
            "DELETE FROM memory_embeddings WHERE memory_id = ?1",
            params![id],
        );
        let _ = conn.execute("DELETE FROM graph_nodes WHERE memory_id = ?1", params![id]);
    }

    // Rebuild vector index if we deleted anything
    if count > 0 {
        if let Ok(vector) = rebuild_vector_index(storage) {
            let index_path = db_path.with_extension("idx");
            let _ = vector.save(&index_path);
        }
    }

    tracing::info!("Forget cycle complete: {} memories deleted", count);
    println!(
        "Forget cycle: deleted {} forgotten memories (importance < 0.1, never accessed).",
        count
    );
    Ok(count)
}

// ── Helpers ────────────────────────────────────────────────────────────────

/// Rebuild the HNSW vector index from all stored embeddings in the database.
fn rebuild_vector_index(
    storage: &codemem_storage::Storage,
) -> anyhow::Result<codemem_vector::HnswIndex> {
    let conn = storage.connection();
    let mut vector = codemem_vector::HnswIndex::with_defaults()?;

    let mut stmt = conn.prepare("SELECT memory_id, embedding FROM memory_embeddings")?;

    let rows: Vec<(String, Vec<u8>)> = stmt
        .query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, Vec<u8>>(1)?))
        })?
        .filter_map(|r| r.ok())
        .collect();

    for (id, blob) in &rows {
        let floats: Vec<f32> = blob
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        let _ = vector.insert(id, &floats);
    }

    tracing::info!("Rebuilt vector index with {} entries", rows.len());
    Ok(vector)
}

/// Return the system-wide Codemem database path: ~/.codemem/codemem.db
fn codemem_db_path() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".codemem")
        .join("codemem.db")
}

fn truncate_str(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}...", &s[..max])
    }
}

// ── Export/Import Commands ──────────────────────────────────────────────────

fn cmd_export(
    namespace: Option<&str>,
    memory_type: Option<&str>,
    output: Option<&std::path::Path>,
) -> anyhow::Result<()> {
    let db_path = codemem_db_path();
    let storage = codemem_storage::Storage::open(&db_path)?;

    let memory_type_filter: Option<codemem_core::MemoryType> =
        memory_type.and_then(|s| s.parse().ok());

    let ids = match namespace {
        Some(ns) => storage.list_memory_ids_for_namespace(ns)?,
        None => storage.list_memory_ids()?,
    };

    let mut writer: Box<dyn std::io::Write> = match output {
        Some(path) => Box::new(std::fs::File::create(path)?),
        None => Box::new(std::io::stdout()),
    };

    let mut count = 0usize;

    for id in &ids {
        if let Some(memory) = storage.get_memory(id)? {
            // Apply memory_type filter
            if let Some(ref filter_type) = memory_type_filter {
                if memory.memory_type != *filter_type {
                    continue;
                }
            }

            // Get edges for this memory
            let edges: Vec<serde_json::Value> = storage
                .get_edges_for_node(id)
                .unwrap_or_default()
                .iter()
                .map(|e| {
                    serde_json::json!({
                        "id": e.id,
                        "src": e.src,
                        "dst": e.dst,
                        "relationship": e.relationship.to_string(),
                        "weight": e.weight,
                    })
                })
                .collect();

            let obj = serde_json::json!({
                "id": memory.id,
                "content": memory.content,
                "memory_type": memory.memory_type.to_string(),
                "importance": memory.importance,
                "confidence": memory.confidence,
                "tags": memory.tags,
                "namespace": memory.namespace,
                "metadata": memory.metadata,
                "created_at": memory.created_at.to_rfc3339(),
                "updated_at": memory.updated_at.to_rfc3339(),
                "edges": edges,
            });

            // JSONL: one JSON object per line (compact)
            let line = serde_json::to_string(&obj)?;
            writeln!(writer, "{line}")?;
            count += 1;
        }
    }

    // Print count to stderr (so it doesn't mix with JSONL output on stdout)
    eprintln!("Exported {count} memories.");
    Ok(())
}

fn cmd_import(input: Option<&std::path::Path>, skip_duplicates: bool) -> anyhow::Result<()> {
    use std::io::BufRead;

    let db_path = codemem_db_path();
    let storage = codemem_storage::Storage::open(&db_path)?;

    // Try loading embeddings for auto-embedding
    let emb_service = codemem_embeddings::from_env().ok();

    let mut vector = codemem_vector::HnswIndex::with_defaults()?;
    let index_path = db_path.with_extension("idx");
    if index_path.exists() {
        let _ = vector.load(&index_path);
    }

    let reader: Box<dyn BufRead> = match input {
        Some(path) => Box::new(std::io::BufReader::new(std::fs::File::open(path)?)),
        None => Box::new(std::io::BufReader::new(std::io::stdin())),
    };

    let mut imported = 0usize;
    let mut skipped = 0usize;

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let mem_val: serde_json::Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("Skipping invalid JSON line: {e}");
                skipped += 1;
                continue;
            }
        };

        let content = match mem_val.get("content").and_then(|v| v.as_str()) {
            Some(c) if !c.is_empty() => c.to_string(),
            _ => {
                eprintln!("Skipping line without content");
                skipped += 1;
                continue;
            }
        };

        let memory_type: codemem_core::MemoryType = mem_val
            .get("memory_type")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse().ok())
            .unwrap_or(codemem_core::MemoryType::Context);

        let importance = mem_val
            .get("importance")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.5);

        let confidence = mem_val
            .get("confidence")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0);

        let tags: Vec<String> = mem_val
            .get("tags")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        let namespace = mem_val
            .get("namespace")
            .and_then(|v| v.as_str())
            .map(String::from);

        let metadata: std::collections::HashMap<String, serde_json::Value> = mem_val
            .get("metadata")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or_default();

        let now = chrono::Utc::now();
        let id = uuid::Uuid::new_v4().to_string();
        let hash = codemem_storage::Storage::content_hash(&content);

        let memory = codemem_core::MemoryNode {
            id: id.clone(),
            content: content.clone(),
            memory_type,
            importance,
            confidence,
            access_count: 0,
            content_hash: hash,
            tags,
            metadata,
            namespace,
            created_at: now,
            updated_at: now,
            last_accessed_at: now,
        };

        match storage.insert_memory(&memory) {
            Ok(()) => {
                // Auto-embed if available
                if let Some(ref emb) = emb_service {
                    if let Ok(embedding) = emb.embed(&content) {
                        let _ = storage.store_embedding(&id, &embedding);
                        let _ = vector.insert(&id, &embedding);
                    }
                }
                imported += 1;
            }
            Err(codemem_core::CodememError::Duplicate(_)) => {
                if skip_duplicates {
                    skipped += 1;
                } else {
                    eprintln!("Duplicate content detected (use --skip-duplicates to ignore)");
                    skipped += 1;
                }
            }
            Err(e) => {
                eprintln!("Failed to import memory: {e}");
                skipped += 1;
            }
        }
    }

    // Save vector index if we embedded anything
    if imported > 0 && emb_service.is_some() {
        let _ = vector.save(&index_path);
    }

    eprintln!("Imported: {imported}, Skipped: {skipped}");
    Ok(())
}

fn cmd_index(root: &std::path::Path, verbose: bool) -> anyhow::Result<()> {
    let db_path = codemem_db_path();
    let storage = codemem_storage::Storage::open(&db_path)?;

    // Load incremental state
    let mut change_detector = codemem_index::incremental::ChangeDetector::new();
    change_detector.load_from_storage(&storage);

    let mut indexer = codemem_index::Indexer::with_change_detector(change_detector);

    println!("Indexing {}...", root.display());
    let result = indexer.index_directory(root)?;

    println!("  Files scanned:  {}", result.files_scanned);
    println!("  Files parsed:   {}", result.files_parsed);
    println!("  Files skipped:  {}", result.files_skipped);
    println!("  Symbols found:  {}", result.total_symbols);
    println!("  References:     {}", result.total_references);

    // Collect all symbols and references
    let mut all_symbols = Vec::new();
    let mut all_references = Vec::new();
    for pr in &result.parse_results {
        all_symbols.extend(pr.symbols.clone());
        all_references.extend(pr.references.clone());
    }

    // Resolve references into edges
    let mut resolver = codemem_index::ReferenceResolver::new();
    resolver.add_symbols(&all_symbols);
    let edges = resolver.resolve_all(&all_references);
    println!("  Edges resolved: {}", edges.len());

    // Persist symbols as graph nodes (wrapped in a transaction for performance)
    let namespace = root.to_string_lossy().to_string();
    let now = chrono::Utc::now();
    let mut nodes_stored = 0usize;
    let mut edges_stored = 0usize;

    print!("  Storing graph nodes...");
    std::io::Write::flush(&mut std::io::stdout()).ok();

    // Without a transaction, each INSERT is an implicit transaction with fsync —
    // catastrophically slow for thousands of inserts. Wrapping in BEGIN/COMMIT
    // batches all writes into a single disk sync.
    storage.connection().execute_batch("BEGIN")?;

    for sym in &all_symbols {
        let kind = match sym.kind {
            codemem_index::SymbolKind::Function => codemem_core::NodeKind::Function,
            codemem_index::SymbolKind::Method => codemem_core::NodeKind::Method,
            codemem_index::SymbolKind::Class => codemem_core::NodeKind::Class,
            codemem_index::SymbolKind::Struct => codemem_core::NodeKind::Class,
            codemem_index::SymbolKind::Enum => codemem_core::NodeKind::Class,
            codemem_index::SymbolKind::Interface => codemem_core::NodeKind::Interface,
            codemem_index::SymbolKind::Type => codemem_core::NodeKind::Type,
            codemem_index::SymbolKind::Constant => codemem_core::NodeKind::Constant,
            codemem_index::SymbolKind::Module => codemem_core::NodeKind::Module,
            codemem_index::SymbolKind::Test => codemem_core::NodeKind::Test,
        };

        let mut payload = std::collections::HashMap::new();
        payload.insert(
            "signature".to_string(),
            serde_json::Value::String(sym.signature.clone()),
        );
        payload.insert(
            "file_path".to_string(),
            serde_json::Value::String(sym.file_path.clone()),
        );
        payload.insert("line_start".to_string(), serde_json::json!(sym.line_start));
        payload.insert("line_end".to_string(), serde_json::json!(sym.line_end));

        let node = codemem_core::GraphNode {
            id: format!("sym:{}", sym.qualified_name),
            kind,
            label: sym.name.clone(),
            payload,
            centrality: 0.0,
            memory_id: None,
            namespace: Some(namespace.clone()),
        };

        if storage.insert_graph_node(&node).is_ok() {
            nodes_stored += 1;
        }
    }

    for edge in &edges {
        let e = codemem_core::Edge {
            id: format!(
                "ref:{}->{}:{}",
                edge.source_qualified_name, edge.target_qualified_name, edge.relationship
            ),
            src: format!("sym:{}", edge.source_qualified_name),
            dst: format!("sym:{}", edge.target_qualified_name),
            relationship: edge.relationship,
            weight: 1.0,
            properties: std::collections::HashMap::new(),
            created_at: now,
        };
        if storage.insert_graph_edge(&e).is_ok() {
            edges_stored += 1;
        }
    }

    storage.connection().execute_batch("COMMIT")?;

    println!(" done");
    println!("  Graph nodes:    {}", nodes_stored);
    println!("  Graph edges:    {}", edges_stored);

    // Embed symbol signatures for semantic code search
    let mut symbols_embedded = 0usize;
    if let Ok(emb_service) = codemem_embeddings::from_env() {
        let index_path = db_path.with_extension("idx");
        let mut vector = codemem_vector::HnswIndex::with_defaults()?;
        if index_path.exists() {
            let _ = vector.load(&index_path);
        }

        let total = all_symbols.len();
        println!("  Embedding {} symbols...", total);

        // Prepare all texts and IDs up front for batch embedding
        let embed_data: Vec<(String, String)> = all_symbols
            .iter()
            .map(|sym| {
                let mut text = format!("{}: {}", sym.qualified_name, sym.signature);
                if let Some(ref doc) = sym.doc_comment {
                    text.push('\n');
                    text.push_str(doc);
                }
                (format!("sym:{}", sym.qualified_name), text)
            })
            .collect();

        // Wrap embedding storage in a transaction for performance
        storage.connection().execute_batch("BEGIN")?;

        for (batch_idx, chunk) in embed_data.chunks(32).enumerate() {
            let texts: Vec<&str> = chunk.iter().map(|(_, t)| t.as_str()).collect();
            if let Ok(embeddings) = emb_service.embed_batch(&texts) {
                for ((sym_id, _), embedding) in chunk.iter().zip(embeddings) {
                    let _ = storage.store_embedding(sym_id, &embedding);
                    let _ = vector.insert(sym_id, &embedding);
                    symbols_embedded += 1;
                }
            }
            let done = (batch_idx + 1) * 32;
            print!("\r  Embedding symbols: {}/{}", done.min(total), total);
            std::io::Write::flush(&mut std::io::stdout()).ok();
        }

        storage.connection().execute_batch("COMMIT")?;

        println!(); // newline after progress
        if symbols_embedded > 0 {
            let _ = vector.save(&index_path);
        }
        println!("  Symbols embedded: {}", symbols_embedded);
    }

    // Save incremental state
    indexer.change_detector().save_to_storage(&storage)?;

    if verbose {
        println!("\nSymbols:");
        for sym in &all_symbols {
            println!(
                "  {} {} [{}] {}:{}-{}",
                sym.visibility,
                sym.kind,
                sym.qualified_name,
                sym.file_path,
                sym.line_start,
                sym.line_end
            );
        }
    }

    println!("\nDone. Run `codemem stats` to see updated totals.");
    Ok(())
}
