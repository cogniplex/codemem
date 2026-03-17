//! Engine facade methods for graph algorithms.
//!
//! These wrap lock acquisition + graph algorithm calls so the MCP/API
//! transport layers don't interact with the graph mutex directly.

use crate::CodememEngine;
use chrono::{DateTime, Utc};
use codemem_core::{CodememError, Edge, GraphNode, NodeKind, RelationshipType};
use std::collections::{HashMap, HashSet};

// ── Result Types ─────────────────────────────────────────────────────────────

/// A node with its PageRank score.
#[derive(Debug, Clone)]
pub struct RankedNode {
    pub id: String,
    pub score: f64,
    pub kind: Option<String>,
    pub label: Option<String>,
}

/// In-memory graph statistics snapshot.
#[derive(Debug, Clone)]
pub struct GraphStats {
    pub node_count: usize,
    pub edge_count: usize,
    pub node_kind_counts: HashMap<String, usize>,
    pub relationship_type_counts: HashMap<String, usize>,
}

/// A commit entry in a temporal change report.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ChangeEntry {
    pub commit_id: String,
    pub hash: String,
    pub author: String,
    pub date: String,
    pub subject: String,
    pub changed_symbols: Vec<String>,
    pub changed_files: Vec<String>,
}

/// Snapshot of the graph at a point in time.
#[derive(Debug, Clone, serde::Serialize)]
pub struct TemporalSnapshot {
    pub at: String,
    pub live_nodes: usize,
    pub live_edges: usize,
    pub node_kind_counts: HashMap<String, usize>,
}

/// A file that may be stale (not modified recently but still referenced).
#[derive(Debug, Clone, serde::Serialize)]
pub struct StaleFile {
    pub file_path: String,
    pub centrality: f64,
    pub last_modified: Option<String>,
    pub incoming_edges: usize,
}

/// Report on architectural drift between two time periods.
#[derive(Debug, Clone, serde::Serialize)]
pub struct DriftReport {
    pub period: String,
    pub new_cross_module_edges: usize,
    pub removed_files: usize,
    pub added_files: usize,
    pub hotspot_files: Vec<String>,
    pub coupling_increases: Vec<(String, String, usize)>,
}

// ── Engine Methods ───────────────────────────────────────────────────────────

impl CodememEngine {
    /// BFS or DFS traversal from a start node, with optional kind/relationship filters.
    /// When `at_time` is provided, nodes/edges outside their validity window are excluded.
    ///
    /// Note: temporal filtering is post-hoc — the traversal runs on the full graph,
    /// then expired nodes are removed. This means depth counts may include hops through
    /// expired nodes. For precise temporal traversals, use `graph_at_time` instead.
    pub fn graph_traverse(
        &self,
        start_id: &str,
        depth: usize,
        algorithm: &str,
        exclude_kinds: &[NodeKind],
        include_relationships: Option<&[RelationshipType]>,
        at_time: Option<DateTime<Utc>>,
    ) -> Result<Vec<GraphNode>, CodememError> {
        let graph = self.lock_graph()?;
        let has_filters = !exclude_kinds.is_empty() || include_relationships.is_some();

        let mut nodes = if has_filters {
            match algorithm {
                "bfs" => graph.bfs_filtered(start_id, depth, exclude_kinds, include_relationships),
                "dfs" => graph.dfs_filtered(start_id, depth, exclude_kinds, include_relationships),
                _ => Err(CodememError::InvalidInput(format!(
                    "Unknown algorithm: {algorithm}"
                ))),
            }
        } else {
            match algorithm {
                "bfs" => graph.bfs(start_id, depth),
                "dfs" => graph.dfs(start_id, depth),
                _ => Err(CodememError::InvalidInput(format!(
                    "Unknown algorithm: {algorithm}"
                ))),
            }
        }?;

        // Filter by temporal validity if at_time is provided
        if let Some(at) = at_time {
            nodes.retain(|n| {
                n.valid_from.is_none_or(|vf| vf <= at) && n.valid_to.is_none_or(|vt| vt > at)
            });
        }

        Ok(nodes)
    }

    /// Get in-memory graph statistics.
    pub fn graph_stats(&self) -> Result<GraphStats, CodememError> {
        let graph = self.lock_graph()?;
        let stats = graph.stats();
        Ok(GraphStats {
            node_count: stats.node_count,
            edge_count: stats.edge_count,
            node_kind_counts: stats.node_kind_counts,
            relationship_type_counts: stats.relationship_type_counts,
        })
    }

    /// Get all edges for a node.
    pub fn get_node_edges(&self, node_id: &str) -> Result<Vec<Edge>, CodememError> {
        let graph = self.lock_graph()?;
        graph.get_edges(node_id)
    }

    /// Run Louvain community detection at the given resolution.
    pub fn louvain_communities(&self, resolution: f64) -> Result<Vec<Vec<String>>, CodememError> {
        let graph = self.lock_graph()?;
        Ok(graph.louvain_communities(resolution))
    }

    /// Compute PageRank and return the top-k nodes with their scores,
    /// kinds, and labels.
    pub fn find_important_nodes(
        &self,
        top_k: usize,
        damping: f64,
    ) -> Result<Vec<RankedNode>, CodememError> {
        let graph = self.lock_graph()?;
        let scores = graph.pagerank(damping, 100, 1e-6);

        let mut sorted: Vec<(String, f64)> = scores.into_iter().collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted.truncate(top_k);

        let results = sorted
            .into_iter()
            .map(|(id, score)| {
                let node = graph.get_node(&id).ok().flatten();
                RankedNode {
                    id,
                    score,
                    kind: node.as_ref().map(|n| n.kind.to_string()),
                    label: node.as_ref().map(|n| n.label.clone()),
                }
            })
            .collect();

        Ok(results)
    }

    // ── Temporal Queries ─────────────────────────────────────────────────

    /// Return what changed between two timestamps: commits, their files, and symbols.
    pub fn what_changed(
        &self,
        from: DateTime<Utc>,
        to: DateTime<Utc>,
        namespace: Option<&str>,
    ) -> Result<Vec<ChangeEntry>, CodememError> {
        let all_nodes = {
            let graph = self.lock_graph()?;
            graph.get_all_nodes()
        };
        let all_edges = self.storage.all_graph_edges()?;

        // Find commit nodes in the time range
        let commits: Vec<&GraphNode> = all_nodes
            .iter()
            .filter(|n| {
                n.kind == NodeKind::Commit
                    && !n.payload.contains_key("sentinel")
                    && n.valid_from.is_some_and(|vf| vf >= from && vf <= to)
                    && (namespace.is_none() || n.namespace.as_deref() == namespace)
            })
            .collect();

        // Index ModifiedBy edges by dst (commit ID) for O(1) lookup
        let mut edges_by_dst: HashMap<&str, Vec<&Edge>> = HashMap::new();
        for edge in &all_edges {
            if edge.relationship == RelationshipType::ModifiedBy {
                edges_by_dst.entry(&edge.dst).or_default().push(edge);
            }
        }

        let mut entries = Vec::new();
        for commit in &commits {
            let mut changed_files = Vec::new();
            let mut changed_symbols = Vec::new();

            if let Some(commit_edges) = edges_by_dst.get(commit.id.as_str()) {
                for edge in commit_edges {
                    if edge.src.starts_with("file:") {
                        changed_files.push(
                            edge.src
                                .strip_prefix("file:")
                                .unwrap_or(&edge.src)
                                .to_string(),
                        );
                    } else if edge.src.starts_with("sym:") {
                        changed_symbols.push(
                            edge.src
                                .strip_prefix("sym:")
                                .unwrap_or(&edge.src)
                                .to_string(),
                        );
                    }
                }
            }

            entries.push(ChangeEntry {
                commit_id: commit.id.clone(),
                hash: commit
                    .payload
                    .get("hash")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                author: commit
                    .payload
                    .get("author")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                date: commit
                    .payload
                    .get("date")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                subject: commit
                    .payload
                    .get("subject")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                changed_symbols,
                changed_files,
            });
        }

        // Sort by date descending (newest first)
        entries.sort_by(|a, b| b.date.cmp(&a.date));
        Ok(entries)
    }

    /// Snapshot of the graph at a point in time.
    ///
    /// Counts nodes/edges that were valid at `at`: `valid_from <= at` and
    /// (`valid_to` is None or `valid_to > at`). Nodes/edges with no temporal
    /// fields are always counted (they predate the temporal layer).
    pub fn graph_at_time(&self, at: DateTime<Utc>) -> Result<TemporalSnapshot, CodememError> {
        let all_nodes = {
            let graph = self.lock_graph()?;
            graph.get_all_nodes()
        };
        let all_edges = self.storage.all_graph_edges()?;

        let is_node_live = |n: &GraphNode| -> bool {
            let after_start = n.valid_from.is_none_or(|vf| vf <= at);
            let before_end = n.valid_to.is_none_or(|vt| vt > at);
            after_start && before_end
        };

        let is_edge_live = |e: &Edge| -> bool {
            let after_start = e.valid_from.is_none_or(|vf| vf <= at);
            let before_end = e.valid_to.is_none_or(|vt| vt > at);
            after_start && before_end
        };

        let live_nodes: Vec<&GraphNode> = all_nodes.iter().filter(|n| is_node_live(n)).collect();
        let live_edges = all_edges.iter().filter(|e| is_edge_live(e)).count();

        let mut kind_counts: HashMap<String, usize> = HashMap::new();
        for node in &live_nodes {
            *kind_counts.entry(node.kind.to_string()).or_default() += 1;
        }

        Ok(TemporalSnapshot {
            at: at.to_rfc3339(),
            live_nodes: live_nodes.len(),
            live_edges,
            node_kind_counts: kind_counts,
        })
    }

    /// Find files that haven't been modified recently but have high centrality
    /// or many incoming edges (potential stale code).
    pub fn find_stale_files(
        &self,
        namespace: Option<&str>,
        stale_days: u64,
    ) -> Result<Vec<StaleFile>, CodememError> {
        let all_nodes = {
            let graph = self.lock_graph()?;
            graph.get_all_nodes()
        };
        let all_edges = self.storage.all_graph_edges()?;
        let cutoff = Utc::now() - chrono::Duration::days(stale_days as i64);

        // Find the latest ModifiedBy edge date for each file
        let mut file_last_modified: HashMap<&str, DateTime<Utc>> = HashMap::new();
        for edge in &all_edges {
            if edge.relationship == RelationshipType::ModifiedBy && edge.src.starts_with("file:") {
                if let Some(vf) = edge.valid_from {
                    let entry = file_last_modified.entry(&edge.src).or_insert(vf);
                    if vf > *entry {
                        *entry = vf;
                    }
                }
            }
        }

        // Count incoming edges per file (how many things depend on it)
        let mut incoming: HashMap<&str, usize> = HashMap::new();
        for edge in &all_edges {
            if edge.dst.starts_with("file:") {
                *incoming.entry(&edge.dst).or_default() += 1;
            }
        }

        let mut stale = Vec::new();
        for node in &all_nodes {
            if node.kind != NodeKind::File {
                continue;
            }
            if node.valid_to.is_some() {
                continue; // Already deleted
            }
            if namespace.is_some() && node.namespace.as_deref() != namespace {
                continue;
            }

            let last_mod = file_last_modified.get(node.id.as_str()).copied();
            // Only flag as stale if we have temporal data — files with no
            // ModifiedBy edges haven't been through temporal ingestion yet,
            // so we can't determine staleness.
            let is_stale = last_mod.is_some_and(|lm| lm < cutoff);

            if is_stale
                && (node.centrality > 0.0
                    || incoming.get(node.id.as_str()).copied().unwrap_or(0) > 0)
            {
                let file_path = node
                    .id
                    .strip_prefix("file:")
                    .unwrap_or(&node.id)
                    .to_string();
                stale.push(StaleFile {
                    file_path,
                    centrality: node.centrality,
                    last_modified: last_mod.map(|d| d.to_rfc3339()),
                    incoming_edges: incoming.get(node.id.as_str()).copied().unwrap_or(0),
                });
            }
        }

        // Sort by centrality descending (most important stale files first)
        stale.sort_by(|a, b| {
            b.centrality
                .partial_cmp(&a.centrality)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(stale)
    }

    /// Detect architectural drift between two time periods.
    ///
    /// Compares the graph at `from` vs `to` to find: new cross-module edges,
    /// files added/removed, hotspot files (most commits), and coupling increases
    /// (module pairs with growing CoChanged counts).
    pub fn detect_drift(
        &self,
        from: DateTime<Utc>,
        to: DateTime<Utc>,
        namespace: Option<&str>,
    ) -> Result<DriftReport, CodememError> {
        let all_nodes = {
            let graph = self.lock_graph()?;
            graph.get_all_nodes()
        };
        let all_edges = self.storage.all_graph_edges()?;

        // Files alive at `from` vs `to`
        let files_at = |at: DateTime<Utc>| -> HashSet<String> {
            all_nodes
                .iter()
                .filter(|n| {
                    n.kind == NodeKind::File
                        && n.valid_from.is_none_or(|vf| vf <= at)
                        && n.valid_to.is_none_or(|vt| vt > at)
                        && (namespace.is_none() || n.namespace.as_deref() == namespace)
                })
                .map(|n| n.id.clone())
                .collect()
        };

        let files_before = files_at(from);
        let files_after = files_at(to);
        let added_files = files_after.difference(&files_before).count();
        let removed_files = files_before.difference(&files_after).count();

        // Count cross-module edges added in the period
        // A "cross-module" edge connects nodes in different top-level directories
        let module_of = |id: &str| -> String {
            let path = id
                .strip_prefix("file:")
                .or_else(|| id.strip_prefix("sym:"))
                .unwrap_or(id);
            path.split('/').next().unwrap_or("root").to_string()
        };

        let new_cross_module = all_edges
            .iter()
            .filter(|e| {
                e.valid_from.is_some_and(|vf| vf >= from && vf <= to)
                    && module_of(&e.src) != module_of(&e.dst)
                    && !matches!(
                        e.relationship,
                        RelationshipType::ModifiedBy | RelationshipType::PartOf
                    )
            })
            .count();

        // Hotspot files: most ModifiedBy edges in the period
        let mut file_commit_count: HashMap<String, usize> = HashMap::new();
        for edge in &all_edges {
            if edge.relationship == RelationshipType::ModifiedBy
                && edge.src.starts_with("file:")
                && edge.valid_from.is_some_and(|vf| vf >= from && vf <= to)
            {
                *file_commit_count
                    .entry(
                        edge.src
                            .strip_prefix("file:")
                            .unwrap_or(&edge.src)
                            .to_string(),
                    )
                    .or_default() += 1;
            }
        }
        let mut hotspots: Vec<(String, usize)> = file_commit_count.into_iter().collect();
        hotspots.sort_by(|a, b| b.1.cmp(&a.1));
        let hotspot_files: Vec<String> = hotspots.into_iter().take(10).map(|(f, _)| f).collect();

        // Coupling increases: CoChanged pairs with growing counts.
        // CoChanged edges from enrich_git_history use created_at (not valid_from),
        // so check both fields.
        let mut coupling: HashMap<(String, String), usize> = HashMap::new();
        for edge in &all_edges {
            if edge.relationship == RelationshipType::CoChanged
                && (edge.valid_from.is_some_and(|vf| vf >= from && vf <= to)
                    || (edge.valid_from.is_none()
                        && edge.created_at >= from
                        && edge.created_at <= to))
            {
                let pair = if edge.src < edge.dst {
                    (edge.src.clone(), edge.dst.clone())
                } else {
                    (edge.dst.clone(), edge.src.clone())
                };
                let count = edge
                    .properties
                    .get("commit_count")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(1) as usize;
                *coupling.entry(pair).or_default() += count;
            }
        }
        let mut coupling_vec: Vec<(String, String, usize)> =
            coupling.into_iter().map(|((a, b), c)| (a, b, c)).collect();
        coupling_vec.sort_by(|a, b| b.2.cmp(&a.2));
        coupling_vec.truncate(10);

        Ok(DriftReport {
            period: format!("{} to {}", from.to_rfc3339(), to.to_rfc3339()),
            new_cross_module_edges: new_cross_module,
            removed_files,
            added_files,
            hotspot_files,
            coupling_increases: coupling_vec,
        })
    }

    /// Get the history of commits that modified a specific symbol or file.
    pub fn symbol_history(&self, node_id: &str) -> Result<Vec<ChangeEntry>, CodememError> {
        let all_edges = self.storage.all_graph_edges()?;

        // Find all ModifiedBy edges from this node to commit nodes
        let commit_ids: HashSet<&str> = all_edges
            .iter()
            .filter(|e| e.src == node_id && e.relationship == RelationshipType::ModifiedBy)
            .map(|e| e.dst.as_str())
            .collect();

        if commit_ids.is_empty() {
            return Ok(Vec::new());
        }

        // Batch-load commit nodes from graph (single lock acquisition)
        let commit_nodes: Vec<GraphNode> = {
            let graph = self.lock_graph()?;
            graph
                .get_all_nodes()
                .into_iter()
                .filter(|n| commit_ids.contains(n.id.as_str()))
                .collect()
        };

        // Index ModifiedBy edges by dst to populate changed_files/symbols
        let mut edges_by_dst: HashMap<&str, Vec<&Edge>> = HashMap::new();
        for edge in &all_edges {
            if edge.relationship == RelationshipType::ModifiedBy
                && commit_ids.contains(edge.dst.as_str())
            {
                edges_by_dst.entry(&edge.dst).or_default().push(edge);
            }
        }

        let mut entries = Vec::new();
        for node in &commit_nodes {
            let mut changed_files = Vec::new();
            let mut changed_symbols = Vec::new();

            if let Some(commit_edges) = edges_by_dst.get(node.id.as_str()) {
                for edge in commit_edges {
                    if edge.src.starts_with("file:") {
                        changed_files.push(
                            edge.src
                                .strip_prefix("file:")
                                .unwrap_or(&edge.src)
                                .to_string(),
                        );
                    } else if edge.src.starts_with("sym:") {
                        changed_symbols.push(
                            edge.src
                                .strip_prefix("sym:")
                                .unwrap_or(&edge.src)
                                .to_string(),
                        );
                    }
                }
            }

            entries.push(ChangeEntry {
                commit_id: node.id.clone(),
                hash: node
                    .payload
                    .get("hash")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                author: node
                    .payload
                    .get("author")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                date: node
                    .payload
                    .get("date")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                subject: node
                    .payload
                    .get("subject")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                changed_symbols,
                changed_files,
            });
        }

        entries.sort_by(|a, b| b.date.cmp(&a.date));
        Ok(entries)
    }
}
