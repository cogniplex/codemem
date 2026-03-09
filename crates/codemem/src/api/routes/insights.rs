//! Insight endpoints — read stored enrichment insights by track.

use crate::api::types::{
    ActivityInsightsResponse, CodeHealthInsightsResponse, CouplingNode, GitSummary, InsightsQuery,
    MemoryItem, PagerankEntry, PatternResponse, PerformanceInsightsResponse,
    SecurityInsightsResponse,
};
use crate::api::AppState;
use axum::{
    extract::{Query, State},
    Json,
};
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

    let result = state.server.engine.activity_insights(ns, limit);
    let data = result.unwrap_or_else(|_| codemem_engine::insights::ActivityInsights {
        insights: Vec::new(),
        git_summary: codemem_engine::insights::GitSummary {
            total_annotated_files: 0,
            top_authors: Vec::new(),
        },
    });

    Json(ActivityInsightsResponse {
        insights: data.insights.iter().map(to_item).collect(),
        git_summary: GitSummary {
            total_annotated_files: data.git_summary.total_annotated_files,
            top_authors: data.git_summary.top_authors,
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

    let result = state.server.engine.code_health_insights(ns, limit);
    let data = result.unwrap_or_else(|_| codemem_engine::insights::CodeHealthInsights {
        insights: Vec::new(),
        file_hotspots: Vec::new(),
        decision_chains: Vec::new(),
        pagerank_leaders: Vec::new(),
        community_count: 0,
    });

    let hotspots: Vec<PatternResponse> = data
        .file_hotspots
        .iter()
        .take(10)
        .map(|(fp, count, memory_ids)| PatternResponse {
            pattern_type: "file_hotspot".into(),
            description: format!("{fp} — accessed {count} times"),
            frequency: *count,
            confidence: 0.0,
            related_memories: memory_ids.clone(),
        })
        .collect();

    let chains: Vec<PatternResponse> = data
        .decision_chains
        .iter()
        .take(10)
        .map(|(fp, count, memory_ids)| PatternResponse {
            pattern_type: "decision_chain".into(),
            description: format!("{fp} — {count} edits"),
            frequency: *count,
            confidence: 0.0,
            related_memories: memory_ids.clone(),
        })
        .collect();

    let leaders: Vec<PagerankEntry> = data
        .pagerank_leaders
        .into_iter()
        .map(|e| PagerankEntry {
            node_id: e.node_id,
            label: e.label,
            score: e.score,
        })
        .collect();

    Json(CodeHealthInsightsResponse {
        insights: data.insights.iter().map(to_item).collect(),
        file_hotspots: hotspots,
        decision_chains: chains,
        pagerank_leaders: leaders,
        community_count: data.community_count,
    })
}

/// GET /api/insights/security
pub async fn get_security_insights(
    State(state): State<Arc<AppState>>,
    Query(query): Query<InsightsQuery>,
) -> Json<SecurityInsightsResponse> {
    let limit = query.limit.unwrap_or(20);
    let ns = query.namespace.as_deref();

    let result = state.server.engine.security_insights(ns, limit);
    let data = result.unwrap_or_else(|_| codemem_engine::insights::SecurityInsights {
        insights: Vec::new(),
        sensitive_file_count: 0,
        endpoint_count: 0,
        security_function_count: 0,
    });

    Json(SecurityInsightsResponse {
        insights: data.insights.iter().map(to_item).collect(),
        sensitive_file_count: data.sensitive_file_count,
        endpoint_count: data.endpoint_count,
        security_function_count: data.security_function_count,
    })
}

/// GET /api/insights/performance
pub async fn get_performance_insights(
    State(state): State<Arc<AppState>>,
    Query(query): Query<InsightsQuery>,
) -> Json<PerformanceInsightsResponse> {
    let limit = query.limit.unwrap_or(20);
    let ns = query.namespace.as_deref();

    let result = state.server.engine.performance_insights(ns, limit);
    let data = result.unwrap_or_else(|_| codemem_engine::insights::PerformanceInsights {
        insights: Vec::new(),
        high_coupling_nodes: Vec::new(),
        max_depth: 0,
        critical_path: Vec::new(),
    });

    let coupling: Vec<CouplingNode> = data
        .high_coupling_nodes
        .into_iter()
        .map(|c| CouplingNode {
            node_id: c.node_id,
            label: c.label,
            coupling_score: c.coupling_score,
        })
        .collect();

    let critical: Vec<PagerankEntry> = data
        .critical_path
        .into_iter()
        .map(|e| PagerankEntry {
            node_id: e.node_id,
            label: e.label,
            score: e.score,
        })
        .collect();

    Json(PerformanceInsightsResponse {
        insights: data.insights.iter().map(to_item).collect(),
        high_coupling_nodes: coupling,
        max_depth: data.max_depth,
        critical_path: critical,
    })
}
