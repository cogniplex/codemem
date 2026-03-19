//! Temporal graph layer: commit and PR nodes with symbol-level ModifiedBy edges.
//!
//! Replays git history (default 90 days) to build a layered graph where each
//! commit is a node connected to the symbols/files it modified. PRs are detected
//! from merge/squash commit patterns and connected to their commits via PartOf edges.

use crate::review::parse_diff;
use crate::CodememEngine;
use codemem_core::{CodememError, Edge, GraphNode, NodeKind, RelationshipType};
use serde_json::json;
use std::collections::{HashMap, HashSet};

/// Git's empty tree SHA — used as the diff parent for root commits.
const EMPTY_TREE_SHA: &str = "4b825dc642cb6eb9a060e54bf899d69f82d7a419";

/// Result of temporal graph ingestion.
#[derive(Debug, Default)]
pub struct TemporalIngestResult {
    pub commits_processed: usize,
    pub commits_skipped: usize,
    pub pr_nodes_created: usize,
    pub modified_by_edges: usize,
    pub part_of_edges: usize,
    pub symbols_expired: usize,
}

/// Parsed commit from git log.
pub(crate) struct ParsedCommit {
    hash: String,
    short_hash: String,
    parents: Vec<String>,
    author: String,
    date: chrono::DateTime<chrono::Utc>,
    subject: String,
    files: Vec<String>,
}

/// Detected PR from commit patterns.
struct DetectedPR {
    /// PR number (from commit subject).
    number: String,
    /// Commit hashes belonging to this PR.
    commits: Vec<String>,
    /// Whether this was a squash merge (single commit).
    squash: bool,
    /// Timestamp from the merge/squash commit.
    merged_at: chrono::DateTime<chrono::Utc>,
    /// Subject of the merge commit (used as PR title).
    title: String,
    /// Author of the merge commit.
    author: String,
}

/// Check if a commit looks like a bot/CI commit that should be compacted.
fn is_bot_commit(author: &str, files: &[String]) -> bool {
    let bot_author = author.contains("[bot]")
        || author.ends_with("-bot")
        || author.ends_with("bot)")
        || author == "renovate"
        || author == "github-actions";

    if bot_author {
        return true;
    }

    // All files are lock/generated files
    if !files.is_empty()
        && files.iter().all(|f| {
            f.ends_with(".lock")
                || f.ends_with("lock.json")
                || f.ends_with("lock.yaml")
                || f == "CHANGELOG.md"
                || f == "Cargo.lock"
                || f == "bun.lock"
                || f == "yarn.lock"
                || f == "package-lock.json"
                || f == "pnpm-lock.yaml"
                || f == "Gemfile.lock"
                || f == "poetry.lock"
                || f == "go.sum"
        })
    {
        return true;
    }

    false
}

/// Extract PR number from a commit subject.
/// Matches: `feat: add foo (#123)`, `Merge pull request #123 from ...`
fn extract_pr_number(subject: &str) -> Option<String> {
    // Squash merge: "subject (#123)"
    if let Some(start) = subject.rfind("(#") {
        if let Some(end) = subject[start..].find(')') {
            let num = &subject[start + 2..start + end];
            if num.chars().all(|c| c.is_ascii_digit()) {
                return Some(num.to_string());
            }
        }
    }
    // GitHub merge commit: "Merge pull request #123 from ..."
    if let Some(rest) = subject.strip_prefix("Merge pull request #") {
        let num: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
        if !num.is_empty() {
            return Some(num);
        }
    }
    None
}

impl CodememEngine {
    /// Ingest git history into the temporal graph layer.
    ///
    /// Creates Commit nodes, PullRequest nodes, and ModifiedBy edges.
    /// Detects squash/merge PRs and compacts bot commits.
    pub fn ingest_git_temporal(
        &self,
        path: &str,
        days: u64,
        namespace: Option<&str>,
    ) -> Result<TemporalIngestResult, CodememError> {
        let mut result = TemporalIngestResult::default();
        let ns = namespace.unwrap_or("");

        // ── Step 1: Parse git log with parent hashes and subject ────────
        let commits = self.parse_git_log(path, days)?;
        if commits.is_empty() {
            return Ok(result);
        }

        // ── Step 2: Check for incremental ingestion ────────────────────
        let last_ingested = self.get_last_ingested_commit(ns);
        let commits: Vec<ParsedCommit> = if let Some(ref last_hash) = last_ingested {
            // Skip commits we've already processed
            let skip_idx = commits.iter().position(|c| c.hash == *last_hash);
            match skip_idx {
                Some(idx) => commits.into_iter().take(idx).collect(),
                None => commits, // Last commit not found, process all
            }
        } else {
            commits
        };

        if commits.is_empty() {
            return Ok(result);
        }

        // ── Step 3: Compact bot/repetitive commits ──────────────────────
        let (real_commits, bot_groups) = compact_bot_commits(commits);
        result.commits_skipped = bot_groups.values().map(|g| g.len().saturating_sub(1)).sum();

        // ── Step 4: Create commit nodes and ModifiedBy edges ────────────
        let now = chrono::Utc::now();
        let mut commit_nodes = Vec::new();
        let mut edges = Vec::new();

        for commit in &real_commits {
            let commit_id = format!("commit:{}", commit.hash);

            let node = GraphNode {
                id: commit_id.clone(),
                kind: NodeKind::Commit,
                label: format!("{} {}", commit.short_hash, commit.subject),
                payload: {
                    let mut p = HashMap::new();
                    p.insert("hash".into(), json!(commit.hash));
                    p.insert("short_hash".into(), json!(commit.short_hash));
                    p.insert("author".into(), json!(commit.author));
                    p.insert("date".into(), json!(commit.date.to_rfc3339()));
                    p.insert("subject".into(), json!(commit.subject));
                    p.insert("parents".into(), json!(commit.parents));
                    p.insert("files_changed".into(), json!(commit.files.len()));
                    p
                },
                centrality: 0.0,
                memory_id: None,
                namespace: Some(ns.to_string()),
                valid_from: Some(commit.date),
                valid_to: None,
            };
            commit_nodes.push(node);

            // File-level ModifiedBy edges
            for file in &commit.files {
                let file_id = format!("file:{file}");
                edges.push(Edge {
                    id: format!("modby:{file_id}:{}", commit.hash),
                    src: file_id,
                    dst: commit_id.clone(),
                    relationship: RelationshipType::ModifiedBy,
                    weight: 0.4,
                    properties: {
                        let mut p = HashMap::new();
                        p.insert("commit_date".into(), json!(commit.date.to_rfc3339()));
                        p
                    },
                    created_at: now,
                    valid_from: Some(commit.date),
                    valid_to: None,
                });
                result.modified_by_edges += 1;
            }

            result.commits_processed += 1;
        }

        // ── Step 5: Symbol-level ModifiedBy edges (via diff) ────────────
        // Only for recent commits (last 30 days) to limit cost
        let symbol_cutoff = now - chrono::Duration::days(30);
        for commit in &real_commits {
            if commit.date < symbol_cutoff {
                continue;
            }
            let symbol_edges = self.commit_symbol_edges(path, commit, ns);
            edges.extend(symbol_edges);
        }

        // ── Step 6: Create compacted bot commit nodes ───────────────────
        for (key, group) in &bot_groups {
            if group.is_empty() {
                continue;
            }
            let representative = &group[0];
            let commit_id = format!("commit:{}", representative.hash);
            let node = GraphNode {
                id: commit_id,
                kind: NodeKind::Commit,
                label: format!("{} [{}x] {}", representative.short_hash, group.len(), key),
                payload: {
                    let mut p = HashMap::new();
                    p.insert("hash".into(), json!(representative.hash));
                    p.insert("author".into(), json!(representative.author));
                    p.insert("date".into(), json!(representative.date.to_rfc3339()));
                    p.insert("compacted_count".into(), json!(group.len()));
                    p.insert("bot".into(), json!(true));
                    p
                },
                centrality: 0.0,
                memory_id: None,
                namespace: Some(ns.to_string()),
                valid_from: Some(representative.date),
                valid_to: None,
            };
            commit_nodes.push(node);
        }

        // ── Step 7: Detect PRs and create PR nodes + PartOf edges ───────
        let prs = detect_prs(&real_commits);
        for pr in &prs {
            let pr_id = format!("pr:{}", pr.number);
            let node = GraphNode {
                id: pr_id.clone(),
                kind: NodeKind::PullRequest,
                label: format!("#{} {}", pr.number, pr.title),
                payload: {
                    let mut p = HashMap::new();
                    p.insert("number".into(), json!(pr.number));
                    p.insert("title".into(), json!(pr.title));
                    p.insert("author".into(), json!(pr.author));
                    p.insert("squash".into(), json!(pr.squash));
                    p.insert("commit_count".into(), json!(pr.commits.len()));
                    p
                },
                centrality: 0.0,
                memory_id: None,
                namespace: Some(ns.to_string()),
                valid_from: Some(pr.merged_at),
                valid_to: None,
            };
            commit_nodes.push(node);
            result.pr_nodes_created += 1;

            for commit_hash in &pr.commits {
                let commit_id = format!("commit:{commit_hash}");
                edges.push(Edge {
                    id: format!("partof:{commit_id}:{pr_id}"),
                    src: commit_id,
                    dst: pr_id.clone(),
                    relationship: RelationshipType::PartOf,
                    weight: 0.4,
                    properties: HashMap::new(),
                    created_at: now,
                    valid_from: Some(pr.merged_at),
                    valid_to: None,
                });
                result.part_of_edges += 1;
            }
        }

        // ── Step 8: Detect deleted symbols ──────────────────────────────
        result.symbols_expired = self.expire_deleted_symbols(path, &real_commits, ns)?;

        // ── Step 9: Persist to storage and in-memory graph ──────────────
        // Collect edge endpoints that don't exist as commit/PR nodes we're
        // about to insert.  These need placeholder rows in graph_nodes
        // BEFORE we insert edges, otherwise the FK constraint fails.
        let commit_node_ids: HashSet<&str> = commit_nodes.iter().map(|n| n.id.as_str()).collect();
        let mut placeholder_ids = HashSet::new();
        let mut placeholders = Vec::new();
        for edge in &edges {
            for endpoint_id in [&edge.src, &edge.dst] {
                if commit_node_ids.contains(endpoint_id.as_str()) {
                    continue;
                }
                if !placeholder_ids.insert(endpoint_id.clone()) {
                    continue; // already queued
                }
                // Only create if missing from storage
                if matches!(self.storage.get_graph_node(endpoint_id), Ok(Some(_))) {
                    continue;
                }
                let kind = if endpoint_id.starts_with("file:") {
                    NodeKind::File
                } else if endpoint_id.starts_with("sym:") {
                    NodeKind::Function
                } else if endpoint_id.starts_with("commit:") {
                    NodeKind::Commit
                } else if endpoint_id.starts_with("pr:") {
                    NodeKind::PullRequest
                } else {
                    NodeKind::External
                };
                let label = endpoint_id
                    .find(':')
                    .map(|i| &endpoint_id[i + 1..])
                    .unwrap_or(endpoint_id)
                    .to_string();
                placeholders.push(GraphNode {
                    id: endpoint_id.clone(),
                    kind,
                    label,
                    payload: HashMap::new(),
                    centrality: 0.0,
                    memory_id: None,
                    namespace: None,
                    valid_from: None,
                    valid_to: None,
                });
            }
        }

        if !placeholders.is_empty() {
            self.storage.insert_graph_nodes_batch(&placeholders)?;
        }
        self.storage.insert_graph_nodes_batch(&commit_nodes)?;
        self.storage.insert_graph_edges_batch(&edges)?;

        // Single lock scope for both nodes and edges to ensure atomic
        // visibility to concurrent readers.
        {
            let mut graph = self.lock_graph()?;
            for node in placeholders {
                let _ = graph.add_node(node);
            }
            for node in commit_nodes {
                let _ = graph.add_node(node);
            }
            self.add_edges_with_placeholders(&mut **graph, &edges)?;
        }

        // Record last ingested commit for incremental runs
        if let Some(latest) = real_commits.first() {
            self.record_last_ingested_commit(ns, &latest.hash);
        }

        Ok(result)
    }

    /// Ensure all edge endpoints exist in the in-memory graph, creating placeholder
    /// nodes as needed, then add the edges. Logs warnings for any remaining failures.
    ///
    /// Placeholder nodes are also persisted to storage so they survive restarts.
    /// Callers must hold the graph lock; this avoids a double-lock window where
    /// concurrent readers could see nodes without their edges.
    pub(crate) fn add_edges_with_placeholders(
        &self,
        graph: &mut dyn codemem_core::GraphBackend,
        edges: &[Edge],
    ) -> Result<(), CodememError> {
        let mut warn_count = 0u32;
        let mut total_failures = 0u32;

        for edge in edges {
            // Ensure src node exists
            for endpoint_id in [&edge.src, &edge.dst] {
                if graph.get_node(endpoint_id)?.is_none() {
                    let kind = if endpoint_id.starts_with("file:") {
                        NodeKind::File
                    } else if endpoint_id.starts_with("sym:") {
                        NodeKind::Function
                    } else if endpoint_id.starts_with("commit:") {
                        NodeKind::Commit
                    } else if endpoint_id.starts_with("pr:") {
                        NodeKind::PullRequest
                    } else {
                        NodeKind::External
                    };

                    let label = endpoint_id
                        .find(':')
                        .map(|i| &endpoint_id[i + 1..])
                        .unwrap_or(endpoint_id)
                        .to_string();

                    let placeholder = GraphNode {
                        id: endpoint_id.clone(),
                        kind,
                        label,
                        payload: HashMap::new(),
                        centrality: 0.0,
                        memory_id: None,
                        namespace: None,
                        valid_from: None,
                        valid_to: None,
                    };
                    // Persist to storage so placeholder survives restarts
                    let _ = self.storage.insert_graph_node(&placeholder);
                    let _ = graph.add_node(placeholder);
                }
            }

            if let Err(e) = graph.add_edge(edge.clone()) {
                total_failures += 1;
                if warn_count < 5 {
                    tracing::warn!(
                        "Failed to add edge {} ({} -> {}): {e}",
                        edge.id,
                        edge.src,
                        edge.dst
                    );
                    warn_count += 1;
                }
            }
        }

        if total_failures > 0 && total_failures > warn_count {
            tracing::warn!(
                "... and {} more edge insertion failures (total: {})",
                total_failures - warn_count,
                total_failures
            );
        }

        Ok(())
    }

    /// Parse git log output into structured commits.
    fn parse_git_log(&self, path: &str, days: u64) -> Result<Vec<ParsedCommit>, CodememError> {
        let output = std::process::Command::new("git")
            .args([
                "-C",
                path,
                "log",
                "--format=COMMIT:%H|%P|%an|%aI|%s",
                "--name-only",
                "--diff-filter=AMDRT",
                &format!("--since={days} days ago"),
            ])
            .output()
            .map_err(|e| CodememError::Internal(format!("Failed to run git: {e}")))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(CodememError::Internal(format!("git log failed: {stderr}")));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut commits = Vec::new();

        for block in stdout.split("COMMIT:").skip(1) {
            let mut lines = block.lines();
            if let Some(header) = lines.next() {
                let parts: Vec<&str> = header.splitn(5, '|').collect();
                if parts.len() >= 5 {
                    let hash = parts[0].to_string();
                    let short_hash = hash[..hash.len().min(7)].to_string();
                    let parents: Vec<String> =
                        parts[1].split_whitespace().map(|s| s.to_string()).collect();
                    let author = parts[2].to_string();
                    let date = match chrono::DateTime::parse_from_rfc3339(parts[3]) {
                        Ok(dt) => dt.with_timezone(&chrono::Utc),
                        Err(e) => {
                            tracing::warn!(
                                "Skipping commit {}: unparseable date {:?}: {e}",
                                &parts[0][..parts[0].len().min(7)],
                                parts[3]
                            );
                            continue;
                        }
                    };
                    let subject = parts[4].to_string();
                    let files: Vec<String> = lines
                        .filter(|l| !l.trim().is_empty())
                        .map(|l| l.trim().to_string())
                        .collect();

                    commits.push(ParsedCommit {
                        hash,
                        short_hash,
                        parents,
                        author,
                        date,
                        subject,
                        files,
                    });
                }
            }
        }

        Ok(commits)
    }

    /// Get symbol-level ModifiedBy edges for a single commit by running git diff.
    fn commit_symbol_edges(&self, path: &str, commit: &ParsedCommit, namespace: &str) -> Vec<Edge> {
        let mut edges = Vec::new();
        let parent = commit
            .parents
            .first()
            .map(|s| s.as_str())
            .unwrap_or(EMPTY_TREE_SHA);

        let diff_output = std::process::Command::new("git")
            .args(["-C", path, "diff", parent, &commit.hash, "--unified=0"])
            .output();

        let diff_text = match diff_output {
            Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).to_string(),
            _ => return edges,
        };

        let hunks = parse_diff(&diff_text);
        if hunks.is_empty() {
            return edges;
        }

        // Build file→symbols map from graph
        let graph = match self.lock_graph() {
            Ok(g) => g,
            Err(e) => {
                tracing::warn!("Failed to lock graph for symbol-level diff: {e}");
                return edges;
            }
        };
        let all_nodes = graph.get_all_nodes();

        let mut file_symbols: HashMap<&str, Vec<(&str, u32, u32)>> = HashMap::new();
        for node in &all_nodes {
            if matches!(
                node.kind,
                NodeKind::Function
                    | NodeKind::Method
                    | NodeKind::Class
                    | NodeKind::Trait
                    | NodeKind::Interface
                    | NodeKind::Enum
            ) {
                if let (Some(fp), Some(ls), Some(le)) = (
                    node.payload.get("file_path").and_then(|v| v.as_str()),
                    node.payload
                        .get("line_start")
                        .and_then(|v| v.as_u64())
                        .map(|v| v as u32),
                    node.payload
                        .get("line_end")
                        .and_then(|v| v.as_u64())
                        .map(|v| v as u32),
                ) {
                    if node.namespace.as_deref() == Some(namespace) || namespace.is_empty() {
                        file_symbols.entry(fp).or_default().push((&node.id, ls, le));
                    }
                }
            }
        }
        drop(graph);

        let commit_id = format!("commit:{}", commit.hash);
        let now = chrono::Utc::now();
        let mut seen = HashSet::new();

        for hunk in &hunks {
            if let Some(symbols) = file_symbols.get(hunk.file_path.as_str()) {
                let changed_lines: HashSet<u32> = hunk
                    .added_lines
                    .iter()
                    .chain(hunk.removed_lines.iter())
                    .copied()
                    .collect();

                for &(sym_id, line_start, line_end) in symbols {
                    if changed_lines
                        .iter()
                        .any(|&l| l >= line_start && l <= line_end)
                        && seen.insert(sym_id)
                    {
                        edges.push(Edge {
                            id: format!("modby:{}:{}", sym_id, commit.hash),
                            src: sym_id.to_string(),
                            dst: commit_id.clone(),
                            relationship: RelationshipType::ModifiedBy,
                            weight: 0.4,
                            properties: {
                                let mut p = HashMap::new();
                                p.insert("commit_date".into(), json!(commit.date.to_rfc3339()));
                                p.insert("symbol_level".into(), json!(true));
                                p
                            },
                            created_at: now,
                            valid_from: Some(commit.date),
                            valid_to: None,
                        });
                    }
                }
            }
        }

        edges
    }

    /// Set valid_to on symbols/files that were deleted in the given commits.
    ///
    /// Uses `git log --diff-filter=D` to find deleted files, then collects
    /// expired nodes before writing — avoids holding the graph lock during
    /// storage writes (deadlock risk).
    pub(crate) fn expire_deleted_symbols(
        &self,
        path: &str,
        commits: &[ParsedCommit],
        namespace: &str,
    ) -> Result<usize, CodememError> {
        // Find deleted files from the already-parsed commits' time range
        let since = commits
            .last()
            .map(|c| c.date.to_rfc3339())
            .unwrap_or_else(|| "90 days ago".to_string());

        let output = std::process::Command::new("git")
            .args([
                "-C",
                path,
                "log",
                "--format=COMMIT:%H|%aI",
                "--diff-filter=D",
                "--name-only",
                &format!("--since={since}"),
            ])
            .output()
            .map_err(|e| CodememError::Internal(format!("Failed to run git: {e}")))?;

        if !output.status.success() {
            return Ok(0);
        }

        let stdout = String::from_utf8_lossy(&output.stdout);

        // Parse deletion events: (date, set of deleted file paths)
        let mut deletions: Vec<(chrono::DateTime<chrono::Utc>, HashSet<String>)> = Vec::new();
        for block in stdout.split("COMMIT:").skip(1) {
            let mut lines = block.lines();
            let date = lines
                .next()
                .and_then(|h| {
                    let parts: Vec<&str> = h.splitn(2, '|').collect();
                    parts.get(1).and_then(|d| {
                        chrono::DateTime::parse_from_rfc3339(d)
                            .ok()
                            .map(|dt| dt.with_timezone(&chrono::Utc))
                    })
                })
                .unwrap_or_else(chrono::Utc::now);

            let files: HashSet<String> = lines
                .filter(|l| !l.trim().is_empty())
                .map(|l| l.trim().to_string())
                .collect();

            if !files.is_empty() {
                deletions.push((date, files));
            }
        }

        if deletions.is_empty() {
            return Ok(0);
        }

        // Filter out files that currently exist in the working tree
        // (they were deleted then re-created, so should not be expired)
        for (_date, deleted_files) in &mut deletions {
            deleted_files.retain(|f| {
                let full_path = std::path::Path::new(path).join(f);
                !full_path.exists()
            });
        }
        deletions.retain(|(_, files)| !files.is_empty());

        if deletions.is_empty() {
            return Ok(0);
        }

        // Phase 1: collect expired nodes under graph lock (read-only)
        let expired_nodes: Vec<GraphNode> = {
            let graph = self.lock_graph()?;
            let all_nodes = graph.get_all_nodes();
            let mut to_expire = Vec::new();

            for (date, deleted_files) in &deletions {
                for node in &all_nodes {
                    if node.valid_to.is_some() {
                        continue;
                    }
                    if !namespace.is_empty() && node.namespace.as_deref() != Some(namespace) {
                        continue;
                    }

                    let should_expire = match node.kind {
                        NodeKind::File => {
                            let fp = node.id.strip_prefix("file:").unwrap_or(&node.id);
                            deleted_files.contains(fp)
                        }
                        _ => node
                            .payload
                            .get("file_path")
                            .and_then(|v| v.as_str())
                            .is_some_and(|fp| deleted_files.contains(fp)),
                    };

                    if should_expire {
                        let mut expired_node = node.clone();
                        expired_node.valid_to = Some(*date);
                        to_expire.push(expired_node);
                    }
                }
            }
            to_expire
        };
        // Graph lock dropped here

        // Phase 2: write to storage and in-memory graph separately
        let count = expired_nodes.len();
        if !expired_nodes.is_empty() {
            self.storage.insert_graph_nodes_batch(&expired_nodes)?;
            let mut graph = self.lock_graph()?;
            for node in expired_nodes {
                let _ = graph.add_node(node);
            }
        }

        Ok(count)
    }

    /// Get the last ingested commit hash for incremental processing.
    fn get_last_ingested_commit(&self, namespace: &str) -> Option<String> {
        let sentinel_id = format!("commit:_HEAD:{namespace}");
        if let Ok(Some(node)) = self.storage.get_graph_node(&sentinel_id) {
            node.payload
                .get("hash")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        } else {
            None
        }
    }

    /// Record the last ingested commit hash for incremental processing.
    fn record_last_ingested_commit(&self, namespace: &str, hash: &str) {
        let sentinel_id = format!("commit:_HEAD:{namespace}");
        let node = GraphNode {
            id: sentinel_id,
            kind: NodeKind::Commit,
            label: format!("_HEAD:{namespace}"),
            payload: {
                let mut p = HashMap::new();
                p.insert("hash".into(), json!(hash));
                p.insert("sentinel".into(), json!(true));
                p
            },
            centrality: 0.0,
            memory_id: None,
            namespace: Some(namespace.to_string()),
            valid_from: None,
            valid_to: None,
        };
        let _ = self.storage.insert_graph_node(&node);
    }
}

/// Separate real commits from bot/repetitive commits.
/// Bot commits are grouped by (author, file pattern) key.
fn compact_bot_commits(
    commits: Vec<ParsedCommit>,
) -> (Vec<ParsedCommit>, HashMap<String, Vec<ParsedCommit>>) {
    let mut real = Vec::new();
    let mut bot_groups: HashMap<String, Vec<ParsedCommit>> = HashMap::new();

    for commit in commits {
        if is_bot_commit(&commit.author, &commit.files) {
            let key = format!(
                "{}:{}",
                commit.author,
                commit
                    .files
                    .first()
                    .map(|f| f.as_str())
                    .unwrap_or("unknown")
            );
            bot_groups.entry(key).or_default().push(commit);
        } else {
            real.push(commit);
        }
    }

    (real, bot_groups)
}

/// Detect PRs from commit patterns.
fn detect_prs(commits: &[ParsedCommit]) -> Vec<DetectedPR> {
    let mut prs = Vec::new();
    let mut seen_prs: HashSet<String> = HashSet::new();

    for commit in commits {
        if let Some(pr_number) = extract_pr_number(&commit.subject) {
            if seen_prs.contains(&pr_number) {
                continue;
            }
            seen_prs.insert(pr_number.clone());

            let is_merge = commit.parents.len() > 1;
            let is_squash = commit.parents.len() == 1;

            // For squash merges: single commit = single PR
            // For merge commits: collect commits between this merge and the previous one
            let commit_hashes = if is_squash {
                vec![commit.hash.clone()]
            } else if is_merge && commit.parents.len() == 2 {
                // The second parent is the branch head; commits between
                // first parent and this merge are the PR's commits.
                // For simplicity, just reference the merge commit itself.
                vec![commit.hash.clone()]
            } else {
                vec![commit.hash.clone()]
            };

            prs.push(DetectedPR {
                number: pr_number,
                commits: commit_hashes,
                squash: is_squash,
                merged_at: commit.date,
                title: commit.subject.clone(),
                author: commit.author.clone(),
            });
        }
    }

    prs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_pr_number_squash() {
        assert_eq!(
            extract_pr_number("feat: add foo (#123)"),
            Some("123".to_string())
        );
        assert_eq!(
            extract_pr_number("fix: something (#42)"),
            Some("42".to_string())
        );
    }

    #[test]
    fn extract_pr_number_merge() {
        assert_eq!(
            extract_pr_number("Merge pull request #456 from org/branch"),
            Some("456".to_string())
        );
    }

    #[test]
    fn extract_pr_number_none() {
        assert_eq!(extract_pr_number("chore: update deps"), None);
        assert_eq!(extract_pr_number("fix bug in #parser"), None);
    }

    #[test]
    fn bot_detection() {
        assert!(is_bot_commit("dependabot[bot]", &[]));
        assert!(is_bot_commit("renovate", &[]));
        assert!(is_bot_commit("some-user", &["Cargo.lock".to_string()]));
        assert!(is_bot_commit(
            "some-user",
            &["package-lock.json".to_string()]
        ));
        assert!(!is_bot_commit("some-user", &["src/main.rs".to_string()]));
    }

    #[test]
    fn compact_separates_bots() {
        let commits = vec![
            ParsedCommit {
                hash: "aaa".into(),
                short_hash: "aaa".into(),
                parents: vec![],
                author: "dev".into(),
                date: chrono::Utc::now(),
                subject: "feat: real work".into(),
                files: vec!["src/main.rs".into()],
            },
            ParsedCommit {
                hash: "bbb".into(),
                short_hash: "bbb".into(),
                parents: vec![],
                author: "dependabot[bot]".into(),
                date: chrono::Utc::now(),
                subject: "chore: bump deps".into(),
                files: vec!["Cargo.lock".into()],
            },
        ];
        let (real, bots) = compact_bot_commits(commits);
        assert_eq!(real.len(), 1);
        assert_eq!(real[0].hash, "aaa");
        assert_eq!(bots.len(), 1);
    }

    #[test]
    fn detect_prs_from_squash() {
        let commits = vec![
            ParsedCommit {
                hash: "abc123".into(),
                short_hash: "abc123".into(),
                parents: vec!["def456".into()],
                author: "dev".into(),
                date: chrono::Utc::now(),
                subject: "feat: add feature (#10)".into(),
                files: vec!["src/lib.rs".into()],
            },
            ParsedCommit {
                hash: "xyz789".into(),
                short_hash: "xyz789".into(),
                parents: vec!["abc123".into()],
                author: "dev".into(),
                date: chrono::Utc::now(),
                subject: "fix: plain commit".into(),
                files: vec!["src/main.rs".into()],
            },
        ];
        let prs = detect_prs(&commits);
        assert_eq!(prs.len(), 1);
        assert_eq!(prs[0].number, "10");
        assert!(prs[0].squash);
        assert_eq!(prs[0].commits, vec!["abc123"]);
    }
}
