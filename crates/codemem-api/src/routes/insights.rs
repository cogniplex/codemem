//! Insight endpoints — read stored enrichment insights by track.

use crate::types::{
    ActivityInsightsResponse, CodeHealthInsightsResponse, CouplingNode, GitSummary, InsightsQuery,
    MemoryItem, PagerankEntry, PatternResponse, PerformanceInsightsResponse,
    SecurityInsightsResponse,
};
use crate::AppState;
use axum::{
    extract::{Query, State},
    Json,
};
use codemem_core::NodeKind;
use std::sync::Arc;

/// Convert a MemoryNode to the API MemoryItem type.
fn to_item(m: &codemem_core::MemoryNode) -> MemoryItem {
    MemoryItem {
        id: m.id.clone(),
        content: m.content.clone(),
        memory_type: m.memory_type.to_string(),
        importance: m.importance,
        confidence: m.confidence,
        access_count: m.access_count,
        tags: m.tags.clone(),
        namespace: m.namespace.clone(),
        created_at: m.created_at.to_rfc3339(),
        updated_at: m.updated_at.to_rfc3339(),
    }
}

/// GET /api/insights/activity
pub async fn get_activity_insights(
    State(state): State<Arc<AppState>>,
    Query(query): Query<InsightsQuery>,
) -> Json<ActivityInsightsResponse> {
    let limit = query.limit.unwrap_or(20);
    let ns = query.namespace.as_deref();

    let memories = state
        .storage_direct()
        .list_memories_by_tag("track:activity", ns, limit)
        .unwrap_or_default();

    let insights: Vec<MemoryItem> = memories.iter().map(to_item).collect();

    // Summarize git annotations from graph nodes
    let graph = state.server.lock_graph().ok();
    let (total_annotated, mut top_authors) = if let Some(ref g) = graph {
        let all_nodes = g.get_all_nodes();
        let mut annotated = 0;
        let mut author_set: std::collections::HashSet<String> = std::collections::HashSet::new();
        for node in &all_nodes {
            if node.payload.contains_key("git_commit_count") {
                annotated += 1;
                if let Some(authors) = node.payload.get("git_authors").and_then(|a| a.as_array()) {
                    for a in authors {
                        if let Some(name) = a.as_str() {
                            author_set.insert(name.to_string());
                        }
                    }
                }
            }
        }
        (annotated, author_set.into_iter().collect::<Vec<_>>())
    } else {
        (0, Vec::new())
    };

    top_authors.sort();
    top_authors.truncate(10);

    Json(ActivityInsightsResponse {
        insights,
        git_summary: GitSummary {
            total_annotated_files: total_annotated,
            top_authors,
        },
    })
}

/// GET /api/insights/code-health
pub async fn get_code_health_insights(
    State(state): State<Arc<AppState>>,
    Query(query): Query<InsightsQuery>,
) -> Json<CodeHealthInsightsResponse> {
    let limit = query.limit.unwrap_or(20);
    let ns = query.namespace.as_deref();

    // Get stored code-health insights (from pattern_insights and detect_patterns)
    let memories = state
        .storage_direct()
        .list_memories_by_tag("track:code-health", ns, limit)
        .unwrap_or_default();

    // Also include any Insight-type memories if code-health tag is sparse
    let mut insights: Vec<MemoryItem> = memories.iter().map(to_item).collect();

    if insights.is_empty() {
        // Fall back to general Insight type memories
        let fallback = state
            .storage_direct()
            .list_memories_by_tag("track:performance", ns, limit)
            .unwrap_or_default();
        insights = fallback.iter().map(to_item).collect();
    }

    // Use MCP to get file hotspots and decision chains
    let mut hotspots = Vec::new();
    let mut chains = Vec::new();

    if let Ok(file_results) = state.storage_direct().get_file_hotspots(2, ns) {
        for (fp, count, memory_ids) in file_results.iter().take(10) {
            hotspots.push(PatternResponse {
                pattern_type: "file_hotspot".into(),
                description: format!("{fp} — accessed {count} times"),
                frequency: *count,
                confidence: 0.0,
                related_memories: memory_ids.clone(),
            });
        }
    }

    if let Ok(chain_results) = state.storage_direct().get_decision_chains(2, ns) {
        for (fp, count, memory_ids) in chain_results.iter().take(10) {
            chains.push(PatternResponse {
                pattern_type: "decision_chain".into(),
                description: format!("{fp} — {count} edits"),
                frequency: *count,
                confidence: 0.0,
                related_memories: memory_ids.clone(),
            });
        }
    }

    // PageRank leaders
    let mut leaders = Vec::new();
    let community_count;

    if let Ok(graph) = state.server.lock_graph() {
        let all_nodes = graph.get_all_nodes();
        let mut file_pr: Vec<_> = all_nodes
            .iter()
            .filter(|n| n.kind == NodeKind::File)
            .map(|n| (n.id.clone(), n.label.clone(), graph.get_pagerank(&n.id)))
            .filter(|(_, _, pr)| *pr > 0.0)
            .collect();
        file_pr.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        for (id, label, score) in file_pr.into_iter().take(10) {
            leaders.push(PagerankEntry {
                node_id: id,
                label,
                score,
            });
        }

        community_count = graph.louvain_communities(1.0).len();
    } else {
        community_count = 0;
    }

    Json(CodeHealthInsightsResponse {
        insights,
        file_hotspots: hotspots,
        decision_chains: chains,
        pagerank_leaders: leaders,
        community_count,
    })
}

/// GET /api/insights/security
pub async fn get_security_insights(
    State(state): State<Arc<AppState>>,
    Query(query): Query<InsightsQuery>,
) -> Json<SecurityInsightsResponse> {
    let limit = query.limit.unwrap_or(20);
    let ns = query.namespace.as_deref();

    let memories = state
        .storage_direct()
        .list_memories_by_tag("track:security", ns, limit)
        .unwrap_or_default();

    let insights: Vec<MemoryItem> = memories.iter().map(to_item).collect();

    // Count nodes with security flags
    let (sensitive_count, endpoint_count, fn_count) = if let Ok(graph) = state.server.lock_graph() {
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
                let flag_strs: Vec<&str> = flags.iter().filter_map(|f| f.as_str()).collect();
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
    } else {
        (0, 0, 0)
    };

    Json(SecurityInsightsResponse {
        insights,
        sensitive_file_count: sensitive_count,
        endpoint_count,
        security_function_count: fn_count,
    })
}

/// GET /api/insights/performance
pub async fn get_performance_insights(
    State(state): State<Arc<AppState>>,
    Query(query): Query<InsightsQuery>,
) -> Json<PerformanceInsightsResponse> {
    let limit = query.limit.unwrap_or(20);
    let ns = query.namespace.as_deref();

    let memories = state
        .storage_direct()
        .list_memories_by_tag("track:performance", ns, limit)
        .unwrap_or_default();

    let insights: Vec<MemoryItem> = memories.iter().map(to_item).collect();

    let mut high_coupling_nodes = Vec::new();
    let mut max_depth = 0;
    let mut critical_path = Vec::new();

    if let Ok(graph) = state.server.lock_graph() {
        let all_nodes = graph.get_all_nodes();

        // Coupling scores from annotations
        let mut coupling_data: Vec<CouplingNode> = Vec::new();
        for node in &all_nodes {
            if let Some(score) = node.payload.get("coupling_score").and_then(|v| v.as_u64()) {
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
        high_coupling_nodes = coupling_data.into_iter().take(10).collect();

        // Dependency depth from topological layers
        let layers = graph.topological_layers();
        max_depth = layers.len();

        // Critical path from PageRank
        let mut file_pr: Vec<_> = all_nodes
            .iter()
            .filter(|n| n.kind == NodeKind::File)
            .map(|n| (n.id.clone(), n.label.clone(), graph.get_pagerank(&n.id)))
            .filter(|(_, _, pr)| *pr > 0.0)
            .collect();
        file_pr.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        critical_path = file_pr
            .into_iter()
            .take(10)
            .map(|(id, label, score)| PagerankEntry {
                node_id: id,
                label,
                score,
            })
            .collect();
    }

    Json(PerformanceInsightsResponse {
        insights,
        high_coupling_nodes,
        max_depth,
        critical_path,
    })
}
