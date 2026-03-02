//! Enrichment tools: git history, security, and performance analysis.
//!
//! Each tool annotates existing graph nodes with additional metadata and stores
//! Insight-type memories tagged with `track:*` tags for the Insights UI.

use crate::types::ToolResult;
use crate::McpServer;
use codemem_core::{Edge, GraphBackend, MemoryNode, MemoryType, NodeKind, RelationshipType};
use codemem_storage::Storage;
use serde_json::{json, Value};
use std::collections::HashMap;

// ── Shared Helper ───────────────────────────────────────────────────────────

/// Store an Insight memory with content-hash dedup. Silently skips duplicates.
fn store_insight(
    storage: &dyn codemem_core::StorageBackend,
    content: &str,
    track: &str,
    tags: &[&str],
    importance: f64,
    namespace: Option<&str>,
) {
    let hash = Storage::content_hash(content);
    let now = chrono::Utc::now();
    let mut all_tags: Vec<String> = vec![format!("track:{track}")];
    all_tags.extend(tags.iter().map(|t| t.to_string()));

    let memory = MemoryNode {
        id: uuid::Uuid::new_v4().to_string(),
        content: content.to_string(),
        memory_type: MemoryType::Insight,
        importance: importance.clamp(0.0, 1.0),
        confidence: 0.8,
        access_count: 0,
        content_hash: hash,
        tags: all_tags,
        metadata: HashMap::from([
            ("track".into(), json!(track)),
            ("generated_by".into(), json!("enrichment_pipeline")),
        ]),
        namespace: namespace.map(String::from),
        created_at: now,
        updated_at: now,
        last_accessed_at: now,
    };
    // Err(Duplicate) silently skips — running enrichment twice won't create duplicates
    let _ = storage.insert_memory(&memory);
}

// ── Tool 1: enrich_git_history ──────────────────────────────────────────────

impl McpServer {
    pub(crate) fn tool_enrich_git_history(&self, args: &Value) -> ToolResult {
        let path = match args.get("path").and_then(|v| v.as_str()) {
            Some(p) if !p.is_empty() => p,
            _ => return ToolResult::tool_error("Missing required 'path' parameter (repo root)"),
        };

        let days = args.get("days").and_then(|v| v.as_u64()).unwrap_or(90);
        let namespace = args.get("namespace").and_then(|v| v.as_str());

        // Run git log
        let output = match std::process::Command::new("git")
            .args([
                "-C",
                path,
                "log",
                &format!("--format=COMMIT:%H|%an|%aI"),
                "--name-only",
                &format!("--since={days} days ago"),
            ])
            .output()
        {
            Ok(o) => o,
            Err(e) => return ToolResult::tool_error(format!("Failed to run git: {e}")),
        };

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return ToolResult::tool_error(format!("git log failed: {stderr}"));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);

        // Parse commits
        struct Commit {
            author: String,
            _date: String,
            files: Vec<String>,
        }

        let mut commits: Vec<Commit> = Vec::new();

        for block in stdout.split("COMMIT:").skip(1) {
            let mut lines = block.lines();
            if let Some(header) = lines.next() {
                let parts: Vec<&str> = header.splitn(3, '|').collect();
                if parts.len() >= 3 {
                    let author = parts[1].to_string();
                    let date = parts[2].to_string();
                    let files: Vec<String> = lines
                        .filter(|l| !l.trim().is_empty())
                        .map(|l| l.trim().to_string())
                        .collect();
                    if !files.is_empty() {
                        commits.push(Commit {
                            author,
                            _date: date,
                            files,
                        });
                    }
                }
            }
        }

        let total_commits = commits.len();

        // Aggregate per-file stats
        struct FileStats {
            commit_count: usize,
            authors: Vec<String>,
        }

        let mut file_stats: HashMap<String, FileStats> = HashMap::new();
        let mut author_file_count: HashMap<String, usize> = HashMap::new();
        let mut author_commit_count: HashMap<String, usize> = HashMap::new();

        // Co-change tracking: count how many commits each file pair appears in together
        let mut co_change_count: HashMap<(String, String), usize> = HashMap::new();

        for commit in &commits {
            *author_commit_count.entry(commit.author.clone()).or_default() += 1;

            for file in &commit.files {
                let stats = file_stats.entry(file.clone()).or_insert(FileStats {
                    commit_count: 0,
                    authors: Vec::new(),
                });
                stats.commit_count += 1;
                if !stats.authors.contains(&commit.author) {
                    stats.authors.push(commit.author.clone());
                }
                *author_file_count.entry(commit.author.clone()).or_default() += 1;
            }

            // Track co-changes (pairs of files in same commit)
            let mut sorted_files: Vec<&String> = commit.files.iter().collect();
            sorted_files.sort();
            for i in 0..sorted_files.len() {
                for j in (i + 1)..sorted_files.len() {
                    let key = (sorted_files[i].clone(), sorted_files[j].clone());
                    *co_change_count.entry(key).or_default() += 1;
                }
            }
        }

        // Annotate graph nodes with git stats
        let mut files_annotated = 0;
        {
            let mut graph = match self.lock_graph() {
                Ok(g) => g,
                Err(e) => return ToolResult::tool_error(format!("Lock error: {e}")),
            };

            for (file_path, stats) in &file_stats {
                // Try to find a matching File node in the graph
                let node_id = format!("file:{file_path}");
                if let Ok(Some(mut node)) = graph.get_node(&node_id) {
                    node.payload
                        .insert("git_commit_count".into(), json!(stats.commit_count));
                    node.payload
                        .insert("git_authors".into(), json!(stats.authors));
                    let churn_rate = if days > 0 {
                        stats.commit_count as f64 / (days as f64 / 30.0)
                    } else {
                        0.0
                    };
                    node.payload
                        .insert("git_churn_rate".into(), json!(churn_rate));
                    let _ = graph.add_node(node);
                    files_annotated += 1;
                }
            }
        }

        // Create co-change edges (threshold: 2+ co-occurrences)
        let co_change_threshold = 2;
        let mut co_change_edges_created = 0;
        {
            let mut graph = match self.lock_graph() {
                Ok(g) => g,
                Err(e) => return ToolResult::tool_error(format!("Lock error: {e}")),
            };

            for ((file_a, file_b), count) in &co_change_count {
                if *count < co_change_threshold {
                    continue;
                }
                let src_id = format!("file:{file_a}");
                let dst_id = format!("file:{file_b}");

                // Only create edge if both nodes exist in the graph
                if graph.get_node(&src_id).ok().flatten().is_none()
                    || graph.get_node(&dst_id).ok().flatten().is_none()
                {
                    continue;
                }

                let weight = if total_commits > 0 {
                    *count as f64 / total_commits as f64
                } else {
                    0.0
                };

                let edge = Edge {
                    id: format!("cochange:{}:{}", file_a, file_b),
                    src: src_id,
                    dst: dst_id,
                    relationship: RelationshipType::CoChanged,
                    weight,
                    properties: HashMap::from([("commit_count".into(), json!(count))]),
                    created_at: chrono::Utc::now(),
                    valid_from: None,
                    valid_to: None,
                };
                if graph.add_edge(edge).is_ok() {
                    co_change_edges_created += 1;
                }
            }
        }

        // Store insights
        let mut insights_stored = 0;

        // High-activity files
        for (file_path, stats) in &file_stats {
            if stats.commit_count > 10 {
                let authors_str = stats.authors.join(", ");
                let content = format!(
                    "High activity: {} — {} commits in the last {} days by {}",
                    file_path, stats.commit_count, days, authors_str
                );
                let importance = (stats.commit_count as f64 / 50.0).min(1.0).max(0.4);
                store_insight(
                    &*self.storage,
                    &content,
                    "activity",
                    &["git-history"],
                    importance,
                    namespace,
                );
                insights_stored += 1;
            }
        }

        // Co-change patterns
        for ((file_a, file_b), count) in &co_change_count {
            if *count >= 3 {
                let content = format!(
                    "Co-change pattern: {} and {} change together in {} commits — likely coupled",
                    file_a, file_b, count
                );
                store_insight(
                    &*self.storage,
                    &content,
                    "activity",
                    &["git-history", "coupling"],
                    0.6,
                    namespace,
                );
                insights_stored += 1;
            }
        }

        // Most active contributors
        let mut author_vec: Vec<_> = author_commit_count.iter().collect();
        author_vec.sort_by(|a, b| b.1.cmp(a.1));
        for (author, commit_count) in author_vec.iter().take(3) {
            let file_count = author_file_count.get(*author).unwrap_or(&0);
            let content = format!(
                "Most active contributor: {} with {} commits across {} files",
                author, commit_count, file_count
            );
            store_insight(
                &*self.storage,
                &content,
                "activity",
                &["git-history", "contributor"],
                0.5,
                namespace,
            );
            insights_stored += 1;
        }

        ToolResult::text(
            serde_json::to_string_pretty(&json!({
                "total_commits": total_commits,
                "files_annotated": files_annotated,
                "co_change_edges_created": co_change_edges_created,
                "insights_stored": insights_stored,
                "unique_authors": author_commit_count.len(),
            }))
            .unwrap_or_default(),
        )
    }

    // ── Tool 2: enrich_security ──────────────────────────────────────────────

    pub(crate) fn tool_enrich_security(&self, args: &Value) -> ToolResult {
        let namespace = args.get("namespace").and_then(|v| v.as_str());

        let security_pattern = regex::Regex::new(
            r"(?i)(auth|secret|key|password|token|credential|\.env|private|encrypt|decrypt|cert|permission|rbac|oauth|jwt|session|cookie|csrf|xss|injection|sanitize)"
        ).expect("valid regex");

        let security_fn_pattern = regex::Regex::new(
            r"(?i)(hash|verify|sign|encrypt|authenticate|authorize|validate_token|check_permission)"
        ).expect("valid regex");

        let graph = match self.lock_graph() {
            Ok(g) => g,
            Err(e) => return ToolResult::tool_error(format!("Lock error: {e}")),
        };

        let all_nodes = graph.get_all_nodes();
        drop(graph);

        let mut sensitive_files: Vec<String> = Vec::new();
        let mut endpoints: Vec<String> = Vec::new();
        let mut security_functions: Vec<(String, String)> = Vec::new(); // (name, file)
        let mut nodes_to_annotate: Vec<(String, Vec<String>)> = Vec::new();

        for node in &all_nodes {
            match node.kind {
                NodeKind::File => {
                    if security_pattern.is_match(&node.label) {
                        sensitive_files.push(node.label.clone());
                        nodes_to_annotate.push((
                            node.id.clone(),
                            vec!["sensitive".into(), "auth_related".into()],
                        ));
                    }
                }
                NodeKind::Endpoint => {
                    endpoints.push(node.label.clone());
                    nodes_to_annotate.push((
                        node.id.clone(),
                        vec!["exposed_endpoint".into()],
                    ));
                }
                NodeKind::Function | NodeKind::Method => {
                    if security_fn_pattern.is_match(&node.label) {
                        let file = node
                            .payload
                            .get("file_path")
                            .and_then(|v| v.as_str())
                            .unwrap_or("unknown")
                            .to_string();
                        security_functions.push((node.label.clone(), file));
                        nodes_to_annotate.push((
                            node.id.clone(),
                            vec!["security_function".into()],
                        ));
                    }
                }
                _ => {}
            }
        }

        // Annotate nodes with security flags
        {
            let mut graph = match self.lock_graph() {
                Ok(g) => g,
                Err(e) => return ToolResult::tool_error(format!("Lock error: {e}")),
            };
            for (node_id, flags) in &nodes_to_annotate {
                if let Ok(Some(mut node)) = graph.get_node(node_id) {
                    node.payload
                        .insert("security_flags".into(), json!(flags));
                    let _ = graph.add_node(node);
                }
            }
        }

        // Count security-relevant memories by scanning filtered content
        let security_memory_count = {
            let keywords = ["auth", "security", "token", "credential", "password"];
            let memories = self
                .storage
                .list_memories_filtered(namespace, None)
                .unwrap_or_default();
            memories
                .iter()
                .filter(|m| {
                    let lower = m.content.to_lowercase();
                    keywords.iter().any(|kw| lower.contains(kw))
                })
                .count()
        };

        // Store insights
        let mut insights_stored = 0;

        for file_path in &sensitive_files {
            let content = format!(
                "Sensitive file: {} — contains security-critical code (auth/credentials)",
                file_path
            );
            store_insight(
                &*self.storage,
                &content,
                "security",
                &["severity:high"],
                0.8,
                namespace,
            );
            insights_stored += 1;
        }

        if !endpoints.is_empty() {
            let content = format!(
                "{} exposed API endpoints detected — review access controls",
                endpoints.len()
            );
            store_insight(
                &*self.storage,
                &content,
                "security",
                &["severity:medium", "endpoints"],
                0.7,
                namespace,
            );
            insights_stored += 1;
        }

        for (name, file) in &security_functions {
            let content = format!(
                "Security-critical function: {} in {} — ensure proper testing",
                name, file
            );
            store_insight(
                &*self.storage,
                &content,
                "security",
                &["severity:medium"],
                0.6,
                namespace,
            );
            insights_stored += 1;
        }

        if security_memory_count > 0 {
            let content = format!(
                "{} security-related memories found — security context is being tracked",
                security_memory_count
            );
            store_insight(
                &*self.storage,
                &content,
                "security",
                &["severity:info"],
                0.4,
                namespace,
            );
            insights_stored += 1;
        }

        ToolResult::text(
            serde_json::to_string_pretty(&json!({
                "sensitive_file_count": sensitive_files.len(),
                "endpoint_count": endpoints.len(),
                "security_function_count": security_functions.len(),
                "security_memory_count": security_memory_count,
                "insights_stored": insights_stored,
            }))
            .unwrap_or_default(),
        )
    }

    // ── Tool 3: enrich_performance ──────────────────────────────────────────

    pub(crate) fn tool_enrich_performance(&self, args: &Value) -> ToolResult {
        let namespace = args.get("namespace").and_then(|v| v.as_str());
        let top = args.get("top").and_then(|v| v.as_u64()).unwrap_or(10) as usize;

        let mut graph = match self.lock_graph() {
            Ok(g) => g,
            Err(e) => return ToolResult::tool_error(format!("Lock error: {e}")),
        };

        let all_nodes = graph.get_all_nodes();

        // 1. Compute coupling (in-degree + out-degree) for each node via get_edges
        let mut high_coupling_count = 0;
        let mut coupling_data: Vec<(String, String, usize)> = Vec::new(); // (id, label, degree)

        for node in &all_nodes {
            let degree = graph.get_edges(&node.id).map(|e| e.len()).unwrap_or(0);
            coupling_data.push((node.id.clone(), node.label.clone(), degree));

            if degree > 15 {
                high_coupling_count += 1;
            }
        }

        // Annotate nodes with coupling scores
        for (node_id, _label, degree) in &coupling_data {
            if let Ok(Some(mut node)) = graph.get_node(node_id) {
                node.payload
                    .insert("coupling_score".into(), json!(degree));
                let _ = graph.add_node(node);
            }
        }

        // 2. Compute dependency depth via topological layers
        let layers = graph.topological_layers();
        let max_depth = layers.len();

        for (layer_idx, layer) in layers.iter().enumerate() {
            for node_id in layer {
                if let Ok(Some(mut node)) = graph.get_node(node_id) {
                    node.payload
                        .insert("dependency_layer".into(), json!(layer_idx));
                    let _ = graph.add_node(node);
                }
            }
        }

        // 3. PageRank for critical path (File nodes only)
        let mut file_pagerank: Vec<(String, String, f64)> = Vec::new();
        for node in &all_nodes {
            if node.kind == NodeKind::File {
                let pr = graph.get_pagerank(&node.id);
                if pr > 0.0 {
                    file_pagerank.push((node.id.clone(), node.label.clone(), pr));
                }
            }
        }
        file_pagerank.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        for (node_id, _label, rank) in file_pagerank.iter().take(top) {
            if let Ok(Some(mut node)) = graph.get_node(node_id) {
                node.payload
                    .insert("critical_path_rank".into(), json!(rank));
                let _ = graph.add_node(node);
            }
        }

        // 4. File complexity from symbol counts
        let mut file_symbol_counts: HashMap<String, usize> = HashMap::new();
        for node in &all_nodes {
            match node.kind {
                NodeKind::Function | NodeKind::Method | NodeKind::Class | NodeKind::Interface | NodeKind::Type => {
                    if let Some(file_path) = node.payload.get("file_path").and_then(|v| v.as_str()) {
                        *file_symbol_counts.entry(file_path.to_string()).or_default() += 1;
                    }
                }
                _ => {}
            }
        }

        for (file_path, sym_count) in &file_symbol_counts {
            let node_id = format!("file:{file_path}");
            if let Ok(Some(mut node)) = graph.get_node(&node_id) {
                node.payload
                    .insert("symbol_count".into(), json!(sym_count));
                let _ = graph.add_node(node);
            }
        }

        drop(graph);

        // Store insights
        let mut insights_stored = 0;

        // High-coupling nodes
        coupling_data.sort_by(|a, b| b.2.cmp(&a.2));
        for (_, label, degree) in coupling_data.iter().take(top) {
            if *degree > 15 {
                let content = format!(
                    "High coupling: {} has {} dependencies — refactoring risk",
                    label, degree
                );
                store_insight(
                    &*self.storage,
                    &content,
                    "performance",
                    &["coupling"],
                    0.7,
                    namespace,
                );
                insights_stored += 1;
            }
        }

        // Deep dependency chain
        if max_depth > 5 {
            let content = format!(
                "Deep dependency chain: {} layers — impacts build and test times",
                max_depth
            );
            store_insight(
                &*self.storage,
                &content,
                "performance",
                &["dependency-depth"],
                0.6,
                namespace,
            );
            insights_stored += 1;
        }

        // Critical bottleneck (top PageRank file)
        if let Some((_, label, _)) = file_pagerank.first() {
            let content = format!(
                "Critical bottleneck: {} — highest centrality file, changes cascade widely",
                label
            );
            store_insight(
                &*self.storage,
                &content,
                "performance",
                &["critical-path"],
                0.8,
                namespace,
            );
            insights_stored += 1;
        }

        // Complex files (high symbol count)
        let mut complex_files: Vec<_> = file_symbol_counts.iter().collect();
        complex_files.sort_by(|a, b| b.1.cmp(a.1));
        for (file_path, sym_count) in complex_files.iter().take(top) {
            if **sym_count > 20 {
                let content = format!(
                    "Complex file: {} — {} symbols",
                    file_path, sym_count
                );
                store_insight(
                    &*self.storage,
                    &content,
                    "performance",
                    &["complexity"],
                    0.5,
                    namespace,
                );
                insights_stored += 1;
            }
        }

        let critical_files: Vec<_> = file_pagerank
            .iter()
            .take(top)
            .map(|(_, label, score)| json!({"file": label, "pagerank": score}))
            .collect();

        ToolResult::text(
            serde_json::to_string_pretty(&json!({
                "high_coupling_count": high_coupling_count,
                "max_depth": max_depth,
                "critical_files": critical_files,
                "insights_stored": insights_stored,
            }))
            .unwrap_or_default(),
        )
    }

}
