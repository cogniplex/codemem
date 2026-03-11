//! Init & setup commands.

use codemem_core::{StorageBackend, VectorBackend};
use std::path::PathBuf;

/// Represents a detected AI coding assistant installation.
pub(crate) struct DetectedAssistant {
    pub(crate) name: &'static str,
    pub(crate) config_dir: PathBuf,
    pub(crate) in_path: bool,
}

/// Check whether `name` is found on PATH (e.g. "claude", "cursor", "windsurf").
pub(crate) fn is_in_path(name: &str) -> bool {
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
pub(crate) fn detect_assistants() -> Vec<DetectedAssistant> {
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

pub(crate) fn cmd_init(project_dir: &std::path::Path, skip_model: bool) -> anyhow::Result<()> {
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

    let db_path = super::codemem_db_path();
    let _storage = codemem_engine::Storage::open(&db_path)?;
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
            .expect("settings initialized as JSON object")
            .entry("hooks")
            .or_insert_with(|| serde_json::json!({}));

        if !hooks.is_object() {
            *hooks = serde_json::json!({});
        }

        // Define all codemem lifecycle hooks covering the Claude Code hooks spec.
        // matcher is a regex filtering tool names or session source. Omitted = fires for all.
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
                    "matcher": "Edit|Write|MultiEdit",
                    "hooks": [{
                        "type": "command",
                        "command": "codemem ingest",
                        "timeout": 5000
                    }]
                }]),
            ),
            (
                "PostToolUseFailure",
                "codemem tool-error",
                serde_json::json!([{
                    "hooks": [{
                        "type": "command",
                        "command": "codemem tool-error",
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
            (
                "SubagentStop",
                "codemem agent-result",
                serde_json::json!([{
                    "hooks": [{
                        "type": "command",
                        "command": "codemem agent-result",
                        "timeout": 5000
                    }]
                }]),
            ),
            (
                "SubagentStart",
                "codemem agent-start",
                serde_json::json!([{
                    "hooks": [{
                        "type": "command",
                        "command": "codemem agent-start",
                        "timeout": 5000
                    }]
                }]),
            ),
            (
                "SessionEnd",
                "codemem session-close",
                serde_json::json!([{
                    "hooks": [{
                        "type": "command",
                        "command": "codemem session-close",
                        "timeout": 5000
                    }]
                }]),
            ),
            (
                "PreCompact",
                "codemem checkpoint",
                serde_json::json!([{
                    "hooks": [{
                        "type": "command",
                        "command": "codemem checkpoint",
                        "timeout": 5000
                    }]
                }]),
            ),
        ];

        let mut hooks_added = 0;
        let mut hooks_skipped = 0;

        for (event_name, cmd_name, hook_value) in &hook_defs {
            let event_hooks = hooks
                .as_object_mut()
                .expect("hooks verified as JSON object above")
                .entry(*event_name)
                .or_insert_with(|| serde_json::json!([]));

            if !event_hooks.is_array() {
                *event_hooks = serde_json::json!([]);
            }

            // Check if an codemem hook already exists for this event
            let already_exists = event_hooks
                .as_array()
                .expect("event_hooks ensured as array above")
                .iter()
                .any(|h| {
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
                        event_hooks
                            .as_array_mut()
                            .expect("event_hooks ensured as array above")
                            .push(entry.clone());
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

    // ── Step 3b: Install code-mapper agent team (8 agent definitions) ─────
    {
        let agents_dir = project_dir.join(".claude").join("agents");
        std::fs::create_dir_all(&agents_dir)?;

        let agent_files: &[(&str, &str)] = &[
            (
                "code-mapper.md",
                include_str!("../../assets/agents/code-mapper.md"),
            ),
            (
                "baseline-scanner.md",
                include_str!("../../assets/agents/baseline-scanner.md"),
            ),
            (
                "symbol-analyst.md",
                include_str!("../../assets/agents/symbol-analyst.md"),
            ),
            (
                "api-mapper.md",
                include_str!("../../assets/agents/api-mapper.md"),
            ),
            (
                "pattern-hunter.md",
                include_str!("../../assets/agents/pattern-hunter.md"),
            ),
            (
                "architecture-reviewer.md",
                include_str!("../../assets/agents/architecture-reviewer.md"),
            ),
            (
                "security-reviewer.md",
                include_str!("../../assets/agents/security-reviewer.md"),
            ),
            (
                "test-mapper.md",
                include_str!("../../assets/agents/test-mapper.md"),
            ),
        ];

        let mut installed = 0;
        let mut skipped = 0;
        for (name, content) in agent_files {
            let agent_path = agents_dir.join(name);
            if agent_path.exists() {
                skipped += 1;
            } else {
                std::fs::write(&agent_path, content)?;
                installed += 1;
            }
        }

        if installed > 0 {
            println!(
                "[agents] Installed {} agent definitions → .claude/agents/",
                installed
            );
            status_lines.push(format!(
                "Agents: {} code-mapper team agents installed",
                installed
            ));
        }
        if skipped > 0 {
            println!("[agents] {} agent(s) already present, skipped", skipped);
            if installed == 0 {
                status_lines.push("Agents: all already present (no changes)".to_string());
            }
        }

        // Legacy cleanup: remove old skills directory if it exists
        let legacy_skills_dir = project_dir
            .join(".claude")
            .join("skills")
            .join("code-mapper");
        if legacy_skills_dir.is_dir() {
            if let Err(e) = std::fs::remove_dir_all(&legacy_skills_dir) {
                println!(
                    "[agents] Warning: could not remove legacy skills dir: {}",
                    e
                );
            } else {
                println!("[agents] Removed legacy .claude/skills/code-mapper/ directory");
            }
        }
    }

    // ── Step 3c: Install codemem skill (tool quick-reference) ─────────────
    {
        let skill_dir = project_dir.join(".claude").join("skills").join("codemem");
        std::fs::create_dir_all(&skill_dir)?;
        let skill_path = skill_dir.join("SKILL.md");

        if skill_path.exists() {
            println!("[skills] codemem skill already installed, skipped");
        } else {
            let skill_content = include_str!("../../assets/skills/SKILL.md");
            std::fs::write(&skill_path, skill_content)?;
            println!("[skills] Installed codemem tool guide → .claude/skills/codemem/SKILL.md");
            status_lines.push("Skill: /codemem tool reference installed".to_string());
        }
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
            .expect("mcp_config initialized as JSON object")
            .entry("mcpServers")
            .or_insert_with(|| serde_json::json!({}));

        if !servers.is_object() {
            *servers = serde_json::json!({});
        }

        let servers_map = servers
            .as_object_mut()
            .expect("servers ensured as JSON object above");

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
        let model_dir = codemem_engine::EmbeddingService::default_model_dir();
        if model_dir.join("model.safetensors").exists() {
            println!(
                "[model] Embedding model already downloaded at {}",
                model_dir.display()
            );
            status_lines.push("Model: already present".to_string());
        } else {
            println!("[model] Downloading embedding model (BAAI/bge-base-en-v1.5)...");
            match codemem_engine::EmbeddingService::download_default_model() {
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
pub(crate) fn batch_embed_existing(db_path: &std::path::Path) {
    let storage = match codemem_engine::Storage::open(db_path) {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!("Could not open storage for batch embedding: {e}");
            return;
        }
    };

    let config = codemem_core::CodememConfig::load_or_default();
    let emb_service = match codemem_engine::embeddings_from_env(Some(&config.embedding)) {
        Ok(s) => s,
        Err(_) => {
            // Embedding provider not available, skip batch embedding
            return;
        }
    };

    // Find memories that lack embeddings
    let rows = match storage.find_unembedded_memories() {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!("Failed to query unembedded memories: {e}");
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
    let mut vector = match codemem_engine::HnswIndex::with_defaults() {
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
    let batch_size = config.embedding.batch_size;

    let mut all_pairs: Vec<(String, Vec<f32>)> = Vec::new();
    for (batch_idx, chunk) in rows.chunks(batch_size).enumerate() {
        let texts: Vec<&str> = chunk.iter().map(|(_, content)| content.as_str()).collect();
        if let Ok(embeddings) = emb_service.embed_batch(&texts) {
            for ((id, _), embedding) in chunk.iter().zip(embeddings) {
                let _ = vector.insert(id, &embedding);
                all_pairs.push((id.clone(), embedding));
                embedded += 1;
            }
        }
        let done = (batch_idx + 1) * batch_size;
        print!("\r  Embedding: {}/{}", done.min(total), total);
        std::io::Write::flush(&mut std::io::stdout()).ok();
    }

    // Batch store all embeddings
    let batch_refs: Vec<(&str, &[f32])> = all_pairs
        .iter()
        .map(|(id, emb)| (id.as_str(), emb.as_slice()))
        .collect();
    let _ = storage.store_embeddings_batch(&batch_refs);

    if total > 0 {
        println!(); // newline after progress
    }
    if embedded > 0 {
        let _ = vector.save(&index_path);
        tracing::info!("Batch embedded {} of {} memories", embedded, total);
        println!("  Embedded and indexed {} memories.", embedded);
    }
}

#[cfg(test)]
#[path = "tests/commands_init_tests.rs"]
mod tests;
