//! Git history enrichment: file activity, co-changes, contributors.

use super::EnrichResult;
use crate::CodememEngine;
use codemem_core::{CodememError, Edge, GraphBackend, RelationshipType};
use serde_json::json;
use std::collections::{HashMap, HashSet};

impl CodememEngine {
    /// Enrich the graph with git history analysis: file activity, co-changes, contributors.
    pub fn enrich_git_history(
        &self,
        path: &str,
        days: u64,
        namespace: Option<&str>,
    ) -> Result<EnrichResult, CodememError> {
        // Run git log
        let output = std::process::Command::new("git")
            .args([
                "-C",
                path,
                "log",
                "--format=COMMIT:%H|%an|%aI",
                "--name-only",
                &format!("--since={days} days ago"),
            ])
            .output()
            .map_err(|e| CodememError::Internal(format!("Failed to run git: {e}")))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(CodememError::Internal(format!("git log failed: {stderr}")));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);

        // Parse commits
        struct Commit {
            author: String,
            date: chrono::DateTime<chrono::Utc>,
            files: Vec<String>,
        }

        let mut commits: Vec<Commit> = Vec::new();

        for block in stdout.split("COMMIT:").skip(1) {
            let mut lines = block.lines();
            if let Some(header) = lines.next() {
                let parts: Vec<&str> = header.splitn(3, '|').collect();
                if parts.len() >= 3 {
                    let author = parts[1].to_string();
                    let date = chrono::DateTime::parse_from_rfc3339(parts[2])
                        .map(|dt| dt.with_timezone(&chrono::Utc))
                        .unwrap_or_else(|_| chrono::Utc::now());
                    let files: Vec<String> = lines
                        .filter(|l| !l.trim().is_empty())
                        .map(|l| l.trim().to_string())
                        .collect();
                    if !files.is_empty() {
                        commits.push(Commit {
                            author,
                            date,
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
            authors: HashSet<String>,
        }

        let mut file_stats: HashMap<String, FileStats> = HashMap::new();
        let mut author_file_count: HashMap<String, usize> = HashMap::new();
        let mut author_commit_count: HashMap<String, usize> = HashMap::new();

        // Co-change tracking with temporal info
        struct CoChangeInfo {
            count: usize,
            earliest: chrono::DateTime<chrono::Utc>,
            latest: chrono::DateTime<chrono::Utc>,
        }
        let mut co_change_info: HashMap<(String, String), CoChangeInfo> = HashMap::new();

        for commit in &commits {
            *author_commit_count
                .entry(commit.author.clone())
                .or_default() += 1;

            for file in &commit.files {
                let stats = file_stats.entry(file.clone()).or_insert(FileStats {
                    commit_count: 0,
                    authors: HashSet::new(),
                });
                stats.commit_count += 1;
                stats.authors.insert(commit.author.clone());
                *author_file_count.entry(commit.author.clone()).or_default() += 1;
            }

            // Track co-changes (pairs of files in same commit)
            // Skip bulk refactor commits (>50 files) to avoid O(N^2) explosion
            let mut sorted_files: Vec<&String> = commit.files.iter().collect();
            if sorted_files.len() > 50 {
                continue;
            }
            sorted_files.sort();
            for i in 0..sorted_files.len() {
                for j in (i + 1)..sorted_files.len() {
                    let key = (sorted_files[i].clone(), sorted_files[j].clone());
                    let entry = co_change_info.entry(key).or_insert(CoChangeInfo {
                        count: 0,
                        earliest: commit.date,
                        latest: commit.date,
                    });
                    entry.count += 1;
                    if commit.date < entry.earliest {
                        entry.earliest = commit.date;
                    }
                    if commit.date > entry.latest {
                        entry.latest = commit.date;
                    }
                }
            }
        }

        // Annotate graph nodes with git stats
        let mut files_annotated = 0;
        {
            let mut graph = self.lock_graph()?;

            for (file_path, stats) in &file_stats {
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

        // Create co-change edges (threshold: 2+ co-occurrences) with temporal data
        let co_change_threshold = 2;
        let mut co_change_edges_created = 0;
        {
            let mut graph = self.lock_graph()?;

            for ((file_a, file_b), info) in &co_change_info {
                if info.count < co_change_threshold {
                    continue;
                }
                let src_id = format!("file:{file_a}");
                let dst_id = format!("file:{file_b}");

                if graph.get_node(&src_id).ok().flatten().is_none()
                    || graph.get_node(&dst_id).ok().flatten().is_none()
                {
                    continue;
                }

                let weight = if total_commits > 0 {
                    info.count as f64 / total_commits as f64
                } else {
                    0.0
                };

                let edge = Edge {
                    id: format!("cochange:{}:{}", file_a, file_b),
                    src: src_id,
                    dst: dst_id,
                    relationship: RelationshipType::CoChanged,
                    weight,
                    properties: HashMap::from([("commit_count".into(), json!(info.count))]),
                    created_at: chrono::Utc::now(),
                    valid_from: Some(info.earliest),
                    valid_to: Some(info.latest),
                };
                let _ = self.storage.insert_graph_edge(&edge);
                if graph.add_edge(edge).is_ok() {
                    co_change_edges_created += 1;
                }
            }
        }

        // Store insights
        let mut insights_stored = 0;

        // High-activity files
        for (file_path, stats) in &file_stats {
            if stats.commit_count > self.config.enrichment.git_min_commit_count {
                let mut sorted_authors: Vec<_> = stats.authors.iter().collect();
                sorted_authors.sort();
                let authors_str = sorted_authors
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                let content = format!(
                    "High activity: {} — {} commits in the last {} days by {}",
                    file_path, stats.commit_count, days, authors_str
                );
                let importance = (stats.commit_count as f64 / 100.0).clamp(0.2, 0.6);
                if self
                    .store_insight(
                        &content,
                        "activity",
                        &["git-history"],
                        importance,
                        namespace,
                        &[format!("file:{file_path}")],
                    )
                    .is_some()
                {
                    insights_stored += 1;
                }
            }
        }

        // Co-change patterns
        for ((file_a, file_b), info) in &co_change_info {
            if info.count >= self.config.enrichment.git_min_co_change_count {
                let content = format!(
                    "Co-change pattern: {} and {} change together in {} commits — likely coupled",
                    file_a, file_b, info.count
                );
                if self
                    .store_insight(
                        &content,
                        "activity",
                        &["git-history", "coupling"],
                        0.4,
                        namespace,
                        &[format!("file:{file_a}"), format!("file:{file_b}")],
                    )
                    .is_some()
                {
                    insights_stored += 1;
                }
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
            if self
                .store_insight(
                    &content,
                    "activity",
                    &["git-history", "contributor"],
                    0.5,
                    namespace,
                    &[],
                )
                .is_some()
            {
                insights_stored += 1;
            }
        }

        self.save_index();

        Ok(EnrichResult {
            insights_stored,
            details: json!({
                "total_commits": total_commits,
                "files_annotated": files_annotated,
                "co_change_edges_created": co_change_edges_created,
                "insights_stored": insights_stored,
                "unique_authors": author_commit_count.len(),
            }),
        })
    }
}
