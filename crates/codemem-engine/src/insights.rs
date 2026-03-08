//! Insight aggregation domain logic.
//!
//! These methods compute graph-derived insight summaries (PageRank leaders,
//! Louvain communities, topology depth, security flags, coupling scores)
//! so the API/transport layer only formats results.

use crate::CodememEngine;
use codemem_core::{CodememError, MemoryNode, NodeKind};
use std::collections::HashSet;

// ── Result Types ─────────────────────────────────────────────────────────────

/// PageRank entry for a graph node.
#[derive(Debug, Clone)]
pub struct PagerankEntry {
    pub node_id: String,
    pub label: String,
    pub score: f64,
}

/// High-coupling node with its coupling score.
#[derive(Debug, Clone)]
pub struct CouplingNode {
    pub node_id: String,
    pub label: String,
    pub coupling_score: usize,
}

/// Git annotation summary from graph node payloads.
#[derive(Debug, Clone)]
pub struct GitSummary {
    pub total_annotated_files: usize,
    pub top_authors: Vec<String>,
}

/// Aggregated activity insights.
#[derive(Debug, Clone)]
pub struct ActivityInsights {
    pub insights: Vec<MemoryNode>,
    pub git_summary: GitSummary,
}

/// Aggregated code health insights.
#[derive(Debug, Clone)]
pub struct CodeHealthInsights {
    pub insights: Vec<MemoryNode>,
    pub file_hotspots: Vec<(String, usize, Vec<String>)>,
    pub decision_chains: Vec<(String, usize, Vec<String>)>,
    pub pagerank_leaders: Vec<PagerankEntry>,
    pub community_count: usize,
}

/// Aggregated security insights.
#[derive(Debug, Clone)]
pub struct SecurityInsights {
    pub insights: Vec<MemoryNode>,
    pub sensitive_file_count: usize,
    pub endpoint_count: usize,
    pub security_function_count: usize,
}

/// Aggregated performance insights.
#[derive(Debug, Clone)]
pub struct PerformanceInsights {
    pub insights: Vec<MemoryNode>,
    pub high_coupling_nodes: Vec<CouplingNode>,
    pub max_depth: usize,
    pub critical_path: Vec<PagerankEntry>,
}

// ── Engine Methods ───────────────────────────────────────────────────────────

impl CodememEngine {
    /// Aggregate activity insights: stored track:activity memories + git annotation summary.
    pub fn activity_insights(
        &self,
        namespace: Option<&str>,
        limit: usize,
    ) -> Result<ActivityInsights, CodememError> {
        let insights = self
            .storage
            .list_memories_by_tag("track:activity", namespace, limit)
            .unwrap_or_default();

        let git_summary = match self.lock_graph() {
            Ok(graph) => {
                let all_nodes = graph.get_all_nodes();
                let mut annotated = 0;
                let mut author_set: HashSet<String> = HashSet::new();
                for node in &all_nodes {
                    if node.payload.contains_key("git_commit_count") {
                        annotated += 1;
                        if let Some(authors) =
                            node.payload.get("git_authors").and_then(|a| a.as_array())
                        {
                            for a in authors {
                                if let Some(name) = a.as_str() {
                                    author_set.insert(name.to_string());
                                }
                            }
                        }
                    }
                }
                let mut top_authors: Vec<String> = author_set.into_iter().collect();
                top_authors.sort();
                top_authors.truncate(10);
                GitSummary {
                    total_annotated_files: annotated,
                    top_authors,
                }
            }
            Err(_) => GitSummary {
                total_annotated_files: 0,
                top_authors: Vec::new(),
            },
        };

        Ok(ActivityInsights {
            insights,
            git_summary,
        })
    }

    /// Aggregate code health insights: stored memories, file hotspots, decision chains,
    /// PageRank leaders, and Louvain community count.
    pub fn code_health_insights(
        &self,
        namespace: Option<&str>,
        limit: usize,
    ) -> Result<CodeHealthInsights, CodememError> {
        let mut insights: Vec<MemoryNode> = self
            .storage
            .list_memories_by_tag("track:code-health", namespace, limit)
            .unwrap_or_default();

        if insights.is_empty() {
            insights = self
                .storage
                .list_memories_by_tag("track:performance", namespace, limit)
                .unwrap_or_default();
        }

        let file_hotspots = self
            .storage
            .get_file_hotspots(2, namespace)
            .unwrap_or_default();

        let decision_chains = self
            .storage
            .get_decision_chains(2, namespace)
            .unwrap_or_default();

        let (pagerank_leaders, community_count) = match self.lock_graph() {
            Ok(graph) => {
                let all_nodes = graph.get_all_nodes();
                let mut file_pr: Vec<_> = all_nodes
                    .iter()
                    .filter(|n| n.kind == NodeKind::File)
                    .map(|n| PagerankEntry {
                        node_id: n.id.clone(),
                        label: n.label.clone(),
                        score: graph.get_pagerank(&n.id),
                    })
                    .filter(|e| e.score > 0.0)
                    .collect();
                file_pr.sort_by(|a, b| {
                    b.score
                        .partial_cmp(&a.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                file_pr.truncate(10);
                let communities = graph.louvain_communities(1.0).len();
                (file_pr, communities)
            }
            Err(_) => (Vec::new(), 0),
        };

        Ok(CodeHealthInsights {
            insights,
            file_hotspots,
            decision_chains,
            pagerank_leaders,
            community_count,
        })
    }

    /// Aggregate security insights: stored memories + security flag counts from graph nodes.
    pub fn security_insights(
        &self,
        namespace: Option<&str>,
        limit: usize,
    ) -> Result<SecurityInsights, CodememError> {
        let insights = self
            .storage
            .list_memories_by_tag("track:security", namespace, limit)
            .unwrap_or_default();

        let (sensitive_file_count, endpoint_count, security_function_count) = match self
            .lock_graph()
        {
            Ok(graph) => {
                let all_nodes = graph.get_all_nodes();
                let mut sensitive = 0;
                let mut endpoints = 0;
                let mut sec_fns = 0;
                for node in &all_nodes {
                    if let Some(flags) = node
                        .payload
                        .get("security_flags")
                        .and_then(|f| f.as_array())
                    {
                        let flag_strs: Vec<&str> =
                            flags.iter().filter_map(|f| f.as_str()).collect();
                        if flag_strs.contains(&"sensitive") || flag_strs.contains(&"auth_related") {
                            sensitive += 1;
                        }
                        if flag_strs.contains(&"exposed_endpoint") {
                            endpoints += 1;
                        }
                        if flag_strs.contains(&"security_function") {
                            sec_fns += 1;
                        }
                    }
                }
                (sensitive, endpoints, sec_fns)
            }
            Err(_) => (0, 0, 0),
        };

        Ok(SecurityInsights {
            insights,
            sensitive_file_count,
            endpoint_count,
            security_function_count,
        })
    }

    /// Aggregate performance insights: stored memories, coupling scores,
    /// topology depth, and PageRank critical path.
    pub fn performance_insights(
        &self,
        namespace: Option<&str>,
        limit: usize,
    ) -> Result<PerformanceInsights, CodememError> {
        let insights = self
            .storage
            .list_memories_by_tag("track:performance", namespace, limit)
            .unwrap_or_default();

        let (high_coupling_nodes, max_depth, critical_path) = match self.lock_graph() {
            Ok(graph) => {
                let all_nodes = graph.get_all_nodes();

                // Coupling scores from annotations
                let mut coupling_data: Vec<CouplingNode> = Vec::new();
                for node in &all_nodes {
                    if let Some(score) = node.payload.get("coupling_score").and_then(|v| v.as_u64())
                    {
                        if score > 15 {
                            coupling_data.push(CouplingNode {
                                node_id: node.id.clone(),
                                label: node.label.clone(),
                                coupling_score: score as usize,
                            });
                        }
                    }
                }
                coupling_data.sort_by(|a, b| b.coupling_score.cmp(&a.coupling_score));
                coupling_data.truncate(10);

                // Dependency depth from topological layers
                let depth = graph.topological_layers().len();

                // Critical path from PageRank
                let mut file_pr: Vec<_> = all_nodes
                    .iter()
                    .filter(|n| n.kind == NodeKind::File)
                    .map(|n| PagerankEntry {
                        node_id: n.id.clone(),
                        label: n.label.clone(),
                        score: graph.get_pagerank(&n.id),
                    })
                    .filter(|e| e.score > 0.0)
                    .collect();
                file_pr.sort_by(|a, b| {
                    b.score
                        .partial_cmp(&a.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                file_pr.truncate(10);

                (coupling_data, depth, file_pr)
            }
            Err(_) => (Vec::new(), 0, Vec::new()),
        };

        Ok(PerformanceInsights {
            insights,
            high_coupling_nodes,
            max_depth,
            critical_path,
        })
    }
}
