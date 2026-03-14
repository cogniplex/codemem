//! Sessions & lifecycle hook commands.

use codemem_core::StorageBackend;

use super::namespace_from_path;

/// Alias for backwards compatibility within this module.
fn namespace_from_cwd(cwd: &str) -> &str {
    namespace_from_path(cwd)
}

// ── Sessions Commands ─────────────────────────────────────────────────────

/// H3: Session commands use `&dyn StorageBackend` instead of `&CodememEngine`
/// to avoid loading the full vector/graph/BM25 engine for lightweight storage-only ops.
pub(crate) fn cmd_sessions_list(
    storage: &dyn StorageBackend,
    namespace: Option<&str>,
) -> anyhow::Result<()> {
    let sessions = storage.list_sessions(namespace, usize::MAX)?;

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

pub(crate) fn cmd_sessions_start(
    storage: &dyn StorageBackend,
    namespace: Option<&str>,
) -> anyhow::Result<()> {
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

pub(crate) fn cmd_sessions_end(
    storage: &dyn StorageBackend,
    id: &str,
    summary: Option<&str>,
) -> anyhow::Result<()> {
    storage.end_session(id, summary)?;

    println!("Session ended: {}", id);
    if let Some(s) = summary {
        println!("  Summary: {}", s);
    }

    Ok(())
}

// ── Shared helpers ────────────────────────────────────────────────────────

/// Read a single-line JSON payload from stdin. Returns an empty object on
/// failure or empty input.
fn read_hook_payload() -> serde_json::Value {
    use std::io::BufRead;

    let mut input = String::new();
    let stdin = std::io::stdin();
    if let Err(e) = stdin.lock().read_line(&mut input) {
        tracing::warn!("Failed to read hook payload from stdin: {e}");
    }

    if input.trim().is_empty() {
        serde_json::json!({})
    } else {
        serde_json::from_str(&input).unwrap_or(serde_json::json!({}))
    }
}

/// Common fields extracted from every hook payload.
struct HookContext<'a> {
    session_id: &'a str,
    namespace: Option<&'a str>,
    /// Scope context derived from cwd (git branch detection).
    scope: Option<codemem_core::ScopeContext>,
}

/// Extract the common session_id / cwd / namespace fields from a payload.
/// The returned references borrow from `payload`.
fn extract_hook_context(payload: &serde_json::Value) -> HookContext<'_> {
    let cwd_raw = payload.get("cwd").and_then(|v| v.as_str()).unwrap_or("");
    let (namespace, scope) = if cwd_raw.is_empty() {
        (None, None)
    } else {
        let scope = codemem_core::ScopeContext::from_local(std::path::Path::new(cwd_raw));
        let ns = namespace_from_cwd(cwd_raw);
        (Some(ns), Some(scope))
    };
    let session_id = payload
        .get("session_id")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    HookContext {
        session_id,
        namespace,
        scope,
    }
}

// ── Lifecycle Hooks ──────────────────────────────────────────────────────

/// SessionStart hook: query recent memories and inject context into the new session.
///
/// Handles `source` field: "startup" (full context), "resume" (brief note),
/// "compact" (checkpoint + full context), "clear" (treated like startup).
pub(crate) fn cmd_context(storage: &dyn StorageBackend) -> anyhow::Result<()> {
    let payload = read_hook_payload();
    let ctx = extract_hook_context(&payload);
    let source = payload
        .get("source")
        .and_then(|v| v.as_str())
        .unwrap_or("startup");

    // Auto-start a session for this project
    if !ctx.session_id.is_empty() {
        if let Err(e) = storage.start_session(ctx.session_id, ctx.namespace) {
            tracing::warn!("Failed to start session {}: {e}", ctx.session_id);
        }
    }

    // On "resume", skip heavy context — the session is continuing with context intact
    if source == "resume" {
        let context = "<codemem-context>\nSession resumed. Prior context still available. \
                        Use `recall_memory` and `search_code` MCP tools as needed.\n</codemem-context>";
        let output = serde_json::json!({
            "hookSpecificOutput": {
                "additionalContext": context
            }
        });
        println!("{}", serde_json::to_string(&output)?);
        return Ok(());
    }

    // On "compact", save a checkpoint before context is lost
    if source == "compact" {
        save_compact_checkpoint(storage, ctx.session_id, ctx.namespace);
    }

    let namespace = ctx.namespace;

    // Gather context from multiple sources
    let mut sections: Vec<String> = Vec::new();

    // 1. Recent sessions with summaries
    if let Ok(sessions) = storage.list_sessions(namespace, usize::MAX) {
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
                    crate::truncate_str(summary, 80)
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
        if let Ok(Some(m)) = storage.get_memory_no_touch(id) {
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
                crate::truncate_str(&m.content.replace('\n', " "), 80)
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
        sec.push_str("\n*Consider storing a permanent memory for any recurring pattern.*");
        sections.push(sec);
    }

    // 4b. Pending analysis from file changes
    let pending_ids = if let Some(ns) = namespace {
        storage
            .list_memory_ids_for_namespace(ns)
            .unwrap_or_default()
    } else {
        storage.list_memory_ids().unwrap_or_default()
    };
    let mut pending_files: Vec<String> = Vec::new();
    for id in pending_ids.iter().rev().take(100) {
        if let Ok(Some(m)) = storage.get_memory_no_touch(id) {
            if m.tags.contains(&"pending-analysis".to_string()) {
                if let Some(files) = m.metadata.get("files").and_then(|v| v.as_array()) {
                    for f in files {
                        if let Some(fp) = f.as_str() {
                            if !pending_files.contains(&fp.to_string()) {
                                pending_files.push(fp.to_string());
                            }
                        }
                    }
                }
            }
        }
    }
    if !pending_files.is_empty() {
        let mut sec = format!(
            "### Pending Analysis\n\n{} file(s) modified since last analysis:\n",
            pending_files.len()
        );
        for f in pending_files.iter().take(10) {
            sec.push_str(&format!("- `{}`\n", short_path(f)));
        }
        if pending_files.len() > 10 {
            sec.push_str(&format!("- ...and {} more\n", pending_files.len() - 10));
        }
        sec.push_str("\n*Consider running the code-mapper agent to analyze these changes.*");
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
        // Usage tips for the assistant
        context.push_str(
            "> **Tips:** Use `store_memory` to save decisions and insights. \
             Use `recall_memory` before exploring code you may have seen before. \
             Run `detect_patterns` to spot repeated workflows. \
             Use `session_checkpoint` mid-session to capture progress. \
             Tag memories with project areas (e.g. `auth`, `api`) for better recall. \
             Important findings deserve `importance >= 0.7`.\n\n",
        );
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
pub(crate) fn cmd_prompt() -> anyhow::Result<()> {
    let payload = read_hook_payload();
    let ctx = extract_hook_context(&payload);

    let prompt = payload.get("prompt").and_then(|v| v.as_str()).unwrap_or("");
    let session_id = if ctx.session_id.is_empty() {
        None
    } else {
        Some(ctx.session_id)
    };
    let cwd = ctx.namespace;
    let scope = ctx.scope;

    // Skip empty or very short prompts
    if prompt.len() < 5 {
        let output = serde_json::json!({"continue": true});
        println!("{}", serde_json::to_string(&output)?);
        return Ok(());
    }

    let db_path = super::codemem_db_path();
    let storage = match codemem_engine::Storage::open_without_migrations(&db_path) {
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
            if let Err(e) = storage.start_session(sid, cwd) {
                tracing::warn!("Failed to start session {sid}: {e}");
            }
        }
    }

    // Skip trivial prompts — "commit this", "clear", "continue", etc. are noise
    let trimmed_prompt = prompt.trim();
    if trimmed_prompt.len() < 30 || trimmed_prompt.split_whitespace().count() < 5 {
        let output = serde_json::json!({"continue": true});
        println!("{}", serde_json::to_string(&output)?);
        return Ok(());
    }

    // Store prompt as a Context memory via the full persist pipeline
    // (BM25 + graph node + embedding + auto-linking)
    let content = format!("User prompt: {}", crate::truncate_str(prompt, 2000));

    let mut memory = codemem_core::MemoryNode::new(content, codemem_core::MemoryType::Context);
    memory.importance = 0.3;
    memory.tags = vec!["prompt".to_string()];
    memory.metadata = {
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
    };
    memory.namespace = cwd.map(|s| s.to_string());

    // Use the engine's persist_memory pipeline for consistent indexing
    match codemem_engine::CodememEngine::from_db_path(&db_path) {
        Ok(engine) => {
            engine.set_active_session(session_id.map(|s| s.to_string()));
            engine.set_scope(scope);
            if let Err(e) = engine.persist_memory(&memory) {
                tracing::warn!("Failed to persist prompt memory: {e}");
            }
        }
        Err(_) => {
            // Fallback to raw storage insert if engine init fails
            if let Err(e) = storage.insert_memory(&memory) {
                tracing::warn!("Failed to insert prompt memory: {e}");
            }
        }
    }

    let output = serde_json::json!({"continue": true});
    println!("{}", serde_json::to_string(&output)?);

    Ok(())
}

/// Stop hook: build a session summary from captured memories and store it.
///
/// Handles `stop_hook_active` (skip if true to avoid loops) and
/// `last_assistant_message` (included in summary for richer context).
pub(crate) fn cmd_summarize() -> anyhow::Result<()> {
    let payload = read_hook_payload();

    // If stop_hook_active is true, we're in a stop-hook loop — bail to avoid recursion
    if payload
        .get("stop_hook_active")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
    {
        println!(
            "{}",
            serde_json::to_string(&serde_json::json!({"continue": true}))?
        );
        return Ok(());
    }

    let ctx = extract_hook_context(&payload);
    let last_message = payload
        .get("last_assistant_message")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    if ctx.session_id.is_empty() {
        let output = serde_json::json!({"continue": true});
        println!("{}", serde_json::to_string(&output)?);
        return Ok(());
    }

    let session_id = ctx.session_id;
    let namespace = ctx.namespace;

    let db_path = super::codemem_db_path();
    let storage = match codemem_engine::Storage::open_without_migrations(&db_path) {
        Ok(s) => s,
        Err(_) => {
            let output = serde_json::json!({"continue": true});
            println!("{}", serde_json::to_string(&output)?);
            return Ok(());
        }
    };

    // Look up session start time to filter memories by creation time.
    // Use UFCS to call the trait method (2-arg) instead of the concrete 1-arg
    // convenience method, so this code would survive a refactor to &dyn StorageBackend.
    let session_start = StorageBackend::list_sessions(&storage, namespace, usize::MAX)
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

    for id in &all_ids {
        if let Ok(Some(m)) = storage.get_memory_no_touch(id) {
            if m.created_at >= session_start {
                session_memories.push(m);
            }
        }
    }

    let cat = categorize_memories(&session_memories);
    let summary_text = {
        let mut s = build_session_summary(&cat);
        if s.is_empty() {
            s = format!("{} memories captured.", session_memories.len());
        }
        // Append a truncated excerpt of Claude's final response for richer context
        if !last_message.is_empty() {
            let excerpt = crate::truncate_str(last_message, 200);
            // Avoid double period if summary already ends with one
            let sep = if s.ends_with('.') { "" } else { "." };
            s.push_str(&format!("{sep} Final response: {excerpt}"));
        }
        s
    };
    let has_substance = has_substance(&cat);

    // Aliases for downstream metadata references
    let files_read = &cat.files_read;
    let files_edited = &cat.files_edited;

    // Build the engine once for both memory persists (if needed)
    let engine = if (has_substance && !session_memories.is_empty()) || !files_edited.is_empty() {
        codemem_engine::CodememEngine::from_db_path(&db_path)
            .ok()
            .inspect(|eng| {
                eng.set_active_session(Some(session_id.to_string()));
            })
    } else {
        None
    };

    // Collect persist errors but never abort — end_session must always run.
    let mut persist_errors: Vec<String> = Vec::new();

    if has_substance && !session_memories.is_empty() {
        let mut summary_memory = codemem_core::MemoryNode::new(
            format!("Session summary: {}", summary_text),
            codemem_core::MemoryType::Insight,
        );
        summary_memory.importance = 0.7;
        summary_memory.tags = vec!["session-summary".to_string()];
        summary_memory.metadata = {
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
        };
        summary_memory.namespace = namespace.map(|s| s.to_string());
        // Use the engine's persist_memory pipeline for consistent indexing
        let result = match &engine {
            Some(eng) => eng.persist_memory(&summary_memory),
            None => storage.insert_memory(&summary_memory),
        };
        if let Err(e) = result {
            persist_errors.push(format!("session summary: {e}"));
        }
    }

    // Store a change-tracking memory for the code-mapper to pick up
    if !files_edited.is_empty() {
        let change_content = format!(
            "Files modified in session {}: {}",
            session_id,
            files_edited.join(", ")
        );
        let mut change_memory =
            codemem_core::MemoryNode::new(change_content, codemem_core::MemoryType::Context);
        change_memory.importance = 0.4;
        change_memory.tags = vec!["pending-analysis".to_string(), "file-changes".to_string()];
        change_memory.metadata = {
            let mut m = std::collections::HashMap::new();
            m.insert("session_id".into(), serde_json::json!(session_id));
            m.insert("files".into(), serde_json::json!(files_edited));
            m.insert(
                "timestamp".into(),
                serde_json::json!(chrono::Utc::now().to_rfc3339()),
            );
            m
        };
        change_memory.namespace = namespace.map(|s| s.to_string());
        // Use the engine's persist_memory pipeline for consistent indexing
        let result = match &engine {
            Some(eng) => eng.persist_memory(&change_memory),
            None => storage.insert_memory(&change_memory),
        };
        if let Err(e) = result {
            persist_errors.push(format!("change-tracking memory: {e}"));
        }
    }

    // End the session unconditionally — memory persist failures must not prevent this.
    if let Err(e) = storage.end_session(session_id, Some(&summary_text)) {
        tracing::warn!("Failed to end session {session_id}: {e}");
    }

    // Log any memory persist failures as warnings after the session is safely ended.
    for err in &persist_errors {
        tracing::warn!("Failed to persist {err}");
    }

    let output = serde_json::json!({"continue": true});
    println!("{}", serde_json::to_string(&output)?);

    Ok(())
}

// ── New Lifecycle Hooks ───────────────────────────────────────────────────

/// SubagentStop hook: capture subagent findings from `last_assistant_message`.
pub(crate) fn cmd_agent_result() -> anyhow::Result<()> {
    let payload = read_hook_payload();
    let ctx = extract_hook_context(&payload);

    // Skip if stop_hook_active (avoid loops)
    if payload
        .get("stop_hook_active")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
    {
        println!("{}", serde_json::to_string(&serde_json::json!({}))?);
        return Ok(());
    }

    let agent_type = payload
        .get("agent_type")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    let agent_id = payload
        .get("agent_id")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let last_message = payload
        .get("last_assistant_message")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    // Skip if no meaningful content
    if last_message.len() < 20 {
        println!("{}", serde_json::to_string(&serde_json::json!({}))?);
        return Ok(());
    }

    let db_path = super::codemem_db_path();
    let storage = match codemem_engine::Storage::open_without_migrations(&db_path) {
        Ok(s) => s,
        Err(_) => {
            println!("{}", serde_json::to_string(&serde_json::json!({}))?);
            return Ok(());
        }
    };

    let content = format!(
        "Agent {} result: {}",
        agent_type,
        crate::truncate_str(last_message, 2000)
    );

    let mut memory = codemem_core::MemoryNode::new(content, codemem_core::MemoryType::Insight);
    memory.importance = 0.5;
    memory.tags = vec!["agent-result".to_string(), format!("agent:{agent_type}")];
    memory.metadata = {
        let mut m = std::collections::HashMap::new();
        m.insert("source".into(), serde_json::json!("SubagentStop"));
        m.insert("agent_type".into(), serde_json::json!(agent_type));
        if !agent_id.is_empty() {
            m.insert("agent_id".into(), serde_json::json!(agent_id));
        }
        if !ctx.session_id.is_empty() {
            m.insert("session_id".into(), serde_json::json!(ctx.session_id));
        }
        m
    };
    memory.namespace = ctx.namespace.map(|s| s.to_string());

    if let Err(e) = storage.insert_memory(&memory) {
        tracing::warn!("Failed to persist agent result: {e}");
    }

    println!("{}", serde_json::to_string(&serde_json::json!({}))?);
    Ok(())
}

/// SubagentStart hook: log agent spawn without storing a memory.
///
/// Agent spawns are transient operational events — storing each one as a
/// MemoryNode at importance 0.2 would clutter recall. We just log it and
/// let SubagentStop capture the valuable output.
pub(crate) fn cmd_agent_start() -> anyhow::Result<()> {
    let payload = read_hook_payload();

    let agent_type = payload
        .get("agent_type")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    let agent_id = payload
        .get("agent_id")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    tracing::info!("Subagent started: {agent_type} ({agent_id})");

    println!("{{}}");
    Ok(())
}

/// PostToolUseFailure hook: capture tool error patterns.
pub(crate) fn cmd_tool_error() -> anyhow::Result<()> {
    let payload = read_hook_payload();
    let ctx = extract_hook_context(&payload);

    // Skip user interrupts — not a real error
    if payload
        .get("is_interrupt")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
    {
        println!("{}", serde_json::to_string(&serde_json::json!({}))?);
        return Ok(());
    }

    let tool_name = payload
        .get("tool_name")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    let error = payload.get("error").and_then(|v| v.as_str()).unwrap_or("");

    if error.is_empty() {
        println!("{}", serde_json::to_string(&serde_json::json!({}))?);
        return Ok(());
    }

    let db_path = super::codemem_db_path();
    let storage = match codemem_engine::Storage::open_without_migrations(&db_path) {
        Ok(s) => s,
        Err(_) => {
            println!("{}", serde_json::to_string(&serde_json::json!({}))?);
            return Ok(());
        }
    };

    // Extract relevant tool_input info for context
    let tool_input = &payload["tool_input"];
    let input_context = tool_input
        .get("file_path")
        .and_then(|v| v.as_str())
        .or_else(|| tool_input.get("command").and_then(|v| v.as_str()))
        .or_else(|| tool_input.get("pattern").and_then(|v| v.as_str()))
        .unwrap_or("");

    let content = if input_context.is_empty() {
        format!(
            "Tool error ({tool_name}): {}",
            crate::truncate_str(error, 1000)
        )
    } else {
        format!(
            "Tool error ({tool_name} on {input_context}): {}",
            crate::truncate_str(error, 1000)
        )
    };

    let mut memory = codemem_core::MemoryNode::new(content, codemem_core::MemoryType::Context);
    memory.importance = 0.4;
    memory.tags = vec![
        "error".to_string(),
        "tool-failure".to_string(),
        format!("tool:{tool_name}"),
    ];
    memory.metadata = {
        let mut m = std::collections::HashMap::new();
        m.insert("source".into(), serde_json::json!("PostToolUseFailure"));
        m.insert("tool_name".into(), serde_json::json!(tool_name));
        m.insert(
            "error".into(),
            serde_json::json!(crate::truncate_str(error, 500)),
        );
        if !ctx.session_id.is_empty() {
            m.insert("session_id".into(), serde_json::json!(ctx.session_id));
        }
        m
    };
    memory.namespace = ctx.namespace.map(|s| s.to_string());

    if let Err(e) = storage.insert_memory(&memory) {
        tracing::warn!("Failed to persist tool error: {e}");
    }

    println!("{}", serde_json::to_string(&serde_json::json!({}))?);
    Ok(())
}

/// SessionEnd hook: cleanly close the session with the termination reason.
pub(crate) fn cmd_session_close() -> anyhow::Result<()> {
    let payload = read_hook_payload();
    let ctx = extract_hook_context(&payload);

    let reason = payload
        .get("reason")
        .and_then(|v| v.as_str())
        .unwrap_or("other");

    if ctx.session_id.is_empty() {
        println!("{{}}");
        return Ok(());
    }

    let db_path = super::codemem_db_path();
    let storage = match codemem_engine::Storage::open_without_migrations(&db_path) {
        Ok(s) => s,
        Err(_) => {
            println!("{{}}");
            return Ok(());
        }
    };

    // Check if the Stop hook already ended this session with a rich summary.
    // If so, skip to avoid overwriting it with a terse "Session ended: {reason}".
    let already_ended = StorageBackend::list_sessions(&storage, ctx.namespace, usize::MAX)
        .unwrap_or_default()
        .iter()
        .any(|s| s.id == ctx.session_id && s.ended_at.is_some());

    if !already_ended {
        let summary = format!("Session ended: {reason}");
        if let Err(e) = storage.end_session(ctx.session_id, Some(&summary)) {
            tracing::warn!("Failed to end session {}: {e}", ctx.session_id);
        }
    }

    println!("{{}}");
    Ok(())
}

/// PreCompact hook: save a checkpoint memory before context compaction.
pub(crate) fn cmd_checkpoint() -> anyhow::Result<()> {
    let payload = read_hook_payload();
    let ctx = extract_hook_context(&payload);

    let db_path = super::codemem_db_path();
    let storage = match codemem_engine::Storage::open_without_migrations(&db_path) {
        Ok(s) => s,
        Err(_) => {
            println!("{{}}");
            return Ok(());
        }
    };

    save_compact_checkpoint(&storage, ctx.session_id, ctx.namespace);

    println!("{{}}");
    Ok(())
}

/// Save a compact checkpoint memory summarizing recent session state.
/// Used by both PreCompact hook and SessionStart(compact).
fn save_compact_checkpoint(
    storage: &dyn StorageBackend,
    session_id: &str,
    namespace: Option<&str>,
) {
    // Gather recent memories to build a brief state summary
    let memory_ids = if let Some(ns) = namespace {
        storage
            .list_memory_ids_for_namespace(ns)
            .unwrap_or_default()
    } else {
        storage.list_memory_ids().unwrap_or_default()
    };

    // Batch-fetch recent memories in a single DB roundtrip.
    // We only need 5 Decision/Insight/Pattern items; ~1/3 of memories are
    // high-signal in a typical session, so 15 is sufficient headroom.
    let batch_ids: Vec<&str> = memory_ids
        .iter()
        .rev()
        .take(15)
        .map(|s| s.as_str())
        .collect();
    let batch = storage.get_memories_batch(&batch_ids).unwrap_or_default();

    let mut recent_items: Vec<String> = Vec::new();
    for m in &batch {
        if matches!(
            m.memory_type,
            codemem_core::MemoryType::Decision
                | codemem_core::MemoryType::Insight
                | codemem_core::MemoryType::Pattern
        ) {
            recent_items.push(crate::truncate_str(&m.content.replace('\n', " "), 100).to_string());
            if recent_items.len() >= 5 {
                break;
            }
        }
    }

    let summary = if recent_items.is_empty() {
        "Pre-compact checkpoint: no key memories in recent context.".to_string()
    } else {
        format!(
            "Pre-compact checkpoint. Key context: {}",
            recent_items.join("; ")
        )
    };

    let mut memory = codemem_core::MemoryNode::new(summary, codemem_core::MemoryType::Context);
    memory.importance = 0.5;
    memory.tags = vec!["pre-compact".to_string(), "checkpoint".to_string()];
    memory.metadata = {
        let mut m = std::collections::HashMap::new();
        m.insert("source".into(), serde_json::json!("PreCompact"));
        if !session_id.is_empty() {
            m.insert("session_id".into(), serde_json::json!(session_id));
        }
        m.insert(
            "timestamp".into(),
            serde_json::json!(chrono::Utc::now().to_rfc3339()),
        );
        m
    };
    memory.namespace = namespace.map(|s| s.to_string());

    if let Err(e) = storage.insert_memory(&memory) {
        tracing::warn!("Failed to persist compact checkpoint: {e}");
    }
}

/// Categorized session activity extracted from memories.
struct SessionCategories {
    files_read: Vec<String>,
    files_edited: Vec<String>,
    searches: Vec<String>,
    decisions: Vec<String>,
    prompts: Vec<String>,
}

/// Categorize memories by tool type, source, and memory type.
///
/// Mirrors the logic in `cmd_summarize` but operates on an in-memory slice
/// so it can be tested without storage.
fn categorize_memories(memories: &[codemem_core::MemoryNode]) -> SessionCategories {
    let mut files_read: Vec<String> = Vec::new();
    let mut files_edited: Vec<String> = Vec::new();
    let mut searches: Vec<String> = Vec::new();
    let mut decisions: Vec<String> = Vec::new();
    let mut prompts: Vec<String> = Vec::new();

    for m in memories {
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
            let text = m
                .content
                .strip_prefix("User prompt: ")
                .unwrap_or(&m.content);
            prompts.push(crate::truncate_str(text, 120).to_string());
        }

        if m.memory_type == codemem_core::MemoryType::Decision {
            decisions.push(crate::truncate_str(&m.content, 120).to_string());
        }
    }

    SessionCategories {
        files_read,
        files_edited,
        searches,
        decisions,
        prompts,
    }
}

/// Build a human-readable session summary from categorized activity.
fn build_session_summary(cat: &SessionCategories) -> String {
    let mut parts: Vec<String> = Vec::new();

    if !cat.prompts.is_empty() {
        parts.push(format!(
            "Requests: {}",
            cat.prompts
                .iter()
                .take(3)
                .cloned()
                .collect::<Vec<_>>()
                .join("; ")
        ));
    }

    if !cat.files_read.is_empty() {
        parts.push(format!(
            "Investigated {} file(s): {}",
            cat.files_read.len(),
            cat.files_read
                .iter()
                .take(5)
                .map(|p| short_path(p))
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }

    if !cat.files_edited.is_empty() {
        parts.push(format!(
            "Modified {} file(s): {}",
            cat.files_edited.len(),
            cat.files_edited
                .iter()
                .take(5)
                .map(|p| short_path(p))
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }

    if !cat.decisions.is_empty() {
        parts.push(format!(
            "Decisions: {}",
            cat.decisions
                .iter()
                .take(3)
                .cloned()
                .collect::<Vec<_>>()
                .join("; ")
        ));
    }

    if !cat.searches.is_empty() {
        parts.push(format!(
            "Searched: {}",
            cat.searches
                .iter()
                .take(5)
                .cloned()
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }

    if parts.is_empty() {
        String::new()
    } else {
        parts.join(". ")
    }
}

/// Whether a session has enough substance to warrant storing a summary.
fn has_substance(cat: &SessionCategories) -> bool {
    !cat.files_edited.is_empty() || !cat.decisions.is_empty() || cat.files_read.len() >= 5
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

#[cfg(test)]
#[path = "tests/commands_lifecycle_tests.rs"]
mod tests;
