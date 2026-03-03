//! Sessions & lifecycle hook commands.

// ── Sessions Commands ─────────────────────────────────────────────────────

pub(crate) fn cmd_sessions_list(namespace: Option<&str>) -> anyhow::Result<()> {
    let db_path = super::codemem_db_path();
    let storage = codemem_storage::Storage::open_without_migrations(&db_path)?;
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

pub(crate) fn cmd_sessions_start(namespace: Option<&str>) -> anyhow::Result<()> {
    let db_path = super::codemem_db_path();
    let storage = codemem_storage::Storage::open_without_migrations(&db_path)?;

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

pub(crate) fn cmd_sessions_end(id: &str, summary: Option<&str>) -> anyhow::Result<()> {
    let db_path = super::codemem_db_path();
    let storage = codemem_storage::Storage::open_without_migrations(&db_path)?;
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
pub(crate) fn cmd_context() -> anyhow::Result<()> {
    use std::io::BufRead;

    // Read a single line — hook payloads are one JSON object per line.
    // Using read_line instead of read_to_string avoids hanging when the
    // hook runner doesn't close stdin after writing the payload.
    let mut input = String::new();
    let stdin = std::io::stdin();
    let _ = stdin.lock().read_line(&mut input);

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

    let db_path = super::codemem_db_path();
    let storage = match codemem_storage::Storage::open_without_migrations(&db_path) {
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
                    super::truncate_str(summary, 80)
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
                super::truncate_str(&m.content.replace('\n', " "), 80)
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
///
/// Reads JSON `{session_id, cwd, prompt}` from stdin.
/// Outputs JSON `{continue: true}` to stdout.
pub(crate) fn cmd_prompt() -> anyhow::Result<()> {
    use std::io::BufRead;

    let mut input = String::new();
    let stdin = std::io::stdin();
    let _ = stdin.lock().read_line(&mut input);

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

    let db_path = super::codemem_db_path();
    let storage = match codemem_storage::Storage::open_without_migrations(&db_path) {
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

    // Skip trivial prompts — "commit this", "clear", "continue", etc. are noise
    let trimmed_prompt = prompt.trim();
    if trimmed_prompt.len() < 30 || trimmed_prompt.split_whitespace().count() < 5 {
        let output = serde_json::json!({"continue": true});
        println!("{}", serde_json::to_string(&output)?);
        return Ok(());
    }

    // Store prompt as a Context memory
    let content = format!("User prompt: {}", super::truncate_str(prompt, 2000));
    let content_hash = codemem_engine::hooks::content_hash(&content);

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
pub(crate) fn cmd_summarize() -> anyhow::Result<()> {
    use std::io::BufRead;

    let mut input = String::new();
    let stdin = std::io::stdin();
    let _ = stdin.lock().read_line(&mut input);

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

    let db_path = super::codemem_db_path();
    let storage = match codemem_storage::Storage::open_without_migrations(&db_path) {
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
        if let Ok(Some(m)) = storage.get_memory_no_touch(id) {
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
                prompts.push(super::truncate_str(text, 120).to_string());
            }

            if m.memory_type == codemem_core::MemoryType::Decision {
                decisions.push(super::truncate_str(&m.content, 120).to_string());
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

    // Only store session summary when there's real substance:
    // - at least one file was edited, or
    // - at least one decision was made, or
    // - non-trivial investigation (5+ files read)
    // Skip summaries that are just echoed prompts — those aren't insights.
    let has_substance = !files_edited.is_empty() || !decisions.is_empty() || files_read.len() >= 5;
    if has_substance && !session_memories.is_empty() {
        let content_hash = codemem_engine::hooks::content_hash(&summary_text);
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

    // Store a change-tracking memory for the code-mapper to pick up
    if !files_edited.is_empty() {
        let change_content = format!(
            "Files modified in session {}: {}",
            session_id,
            files_edited.join(", ")
        );
        let change_hash = codemem_engine::hooks::content_hash(&change_content);
        let now = chrono::Utc::now();
        let change_memory = codemem_core::MemoryNode {
            id: uuid::Uuid::new_v4().to_string(),
            content: change_content,
            memory_type: codemem_core::MemoryType::Context,
            importance: 0.4,
            confidence: 1.0,
            tags: vec!["pending-analysis".to_string(), "file-changes".to_string()],
            metadata: {
                let mut m = std::collections::HashMap::new();
                m.insert("session_id".into(), serde_json::json!(session_id));
                m.insert("files".into(), serde_json::json!(files_edited));
                m.insert(
                    "timestamp".into(),
                    serde_json::json!(chrono::Utc::now().to_rfc3339()),
                );
                m
            },
            namespace: namespace.map(|s| s.to_string()),
            content_hash: change_hash,
            created_at: now,
            updated_at: now,
            last_accessed_at: now,
            access_count: 0,
        };
        let _ = storage.insert_memory(&change_memory);
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

#[cfg(test)]
#[path = "tests/commands_lifecycle_tests.rs"]
mod tests;
