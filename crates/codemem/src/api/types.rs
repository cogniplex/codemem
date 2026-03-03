//! REST API request/response types.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Stats & Health ──────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct StatsResponse {
    pub memory_count: usize,
    pub embedding_count: usize,
    pub node_count: usize,
    pub edge_count: usize,
    pub session_count: usize,
    pub namespace_count: usize,
}

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub storage: ComponentHealth,
    pub vector: ComponentHealth,
    pub graph: ComponentHealth,
    pub embeddings: ComponentHealth,
}

#[derive(Debug, Serialize)]
pub struct ComponentHealth {
    pub status: String,
    pub detail: Option<String>,
}

// ── Memories ────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct MemoryListQuery {
    pub namespace: Option<String>,
    #[serde(rename = "type")]
    pub memory_type: Option<String>,
    pub offset: Option<usize>,
    pub limit: Option<usize>,
    pub sort: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct MemoryListResponse {
    pub memories: Vec<MemoryItem>,
    pub total: usize,
    pub offset: usize,
    pub limit: usize,
}

#[derive(Debug, Serialize)]
pub struct MemoryItem {
    pub id: String,
    pub content: String,
    pub memory_type: String,
    pub importance: f64,
    pub confidence: f64,
    pub access_count: u32,
    pub tags: Vec<String>,
    pub namespace: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Deserialize)]
pub struct StoreMemoryRequest {
    pub content: String,
    pub memory_type: Option<String>,
    pub importance: Option<f64>,
    pub tags: Option<Vec<String>>,
    pub namespace: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct UpdateMemoryRequest {
    pub content: Option<String>,
    pub importance: Option<f64>,
}

// ── Search ──────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct SearchQuery {
    pub q: String,
    pub namespace: Option<String>,
    pub k: Option<usize>,
    #[serde(rename = "type")]
    pub memory_type: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResultItem>,
    pub query: String,
    pub k: usize,
}

#[derive(Debug, Serialize)]
pub struct SearchResultItem {
    pub id: String,
    pub content: String,
    pub memory_type: String,
    pub score: f64,
    pub score_breakdown: ScoreBreakdownResponse,
    pub tags: Vec<String>,
    pub namespace: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ScoreBreakdownResponse {
    pub vector_similarity: f64,
    pub graph_strength: f64,
    pub token_overlap: f64,
    pub temporal: f64,
    pub tag_matching: f64,
    pub importance: f64,
    pub confidence: f64,
    pub recency: f64,
}

// ── Graph ───────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct SubgraphQuery {
    pub namespace: Option<String>,
    pub max_nodes: Option<usize>,
    pub kinds: Option<String>,
    pub min_centrality: Option<f64>,
}

#[derive(Debug, Serialize)]
pub struct SubgraphResponse {
    pub nodes: Vec<GraphNodeResponse>,
    pub edges: Vec<GraphEdgeResponse>,
}

#[derive(Debug, Serialize)]
pub struct GraphNodeResponse {
    pub id: String,
    pub kind: String,
    pub label: String,
    pub centrality: f64,
    pub memory_id: Option<String>,
    pub namespace: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct GraphEdgeResponse {
    pub id: String,
    pub src: String,
    pub dst: String,
    pub relationship: String,
    pub weight: f64,
}

#[derive(Debug, Deserialize)]
pub struct NeighborsQuery {
    pub depth: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct CommunitiesQuery {
    pub namespace: Option<String>,
    pub resolution: Option<f64>,
}

#[derive(Debug, Serialize)]
pub struct CommunitiesResponse {
    pub communities: HashMap<String, usize>,
    pub num_communities: usize,
}

#[derive(Debug, Deserialize)]
pub struct PagerankQuery {
    pub namespace: Option<String>,
    pub top: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct PagerankResponse {
    pub scores: Vec<PagerankEntry>,
}

#[derive(Debug, Serialize)]
pub struct PagerankEntry {
    pub node_id: String,
    pub label: String,
    pub score: f64,
}

#[derive(Debug, Deserialize)]
pub struct ShortestPathQuery {
    pub from: String,
    pub to: String,
}

#[derive(Debug, Deserialize)]
pub struct BrowseQuery {
    pub namespace: Option<String>,
    pub kind: Option<String>,
    pub q: Option<String>,
    pub offset: Option<usize>,
    pub limit: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct BrowseNodeItem {
    pub id: String,
    pub kind: String,
    pub label: String,
    pub centrality: f64,
    pub namespace: Option<String>,
    pub degree: usize,
}

#[derive(Debug, Serialize)]
pub struct BrowseResponse {
    pub nodes: Vec<BrowseNodeItem>,
    pub total: usize,
    pub kinds: HashMap<String, usize>,
    pub edge_count: usize,
}

#[derive(Debug, Deserialize)]
pub struct VectorQuery {
    pub namespace: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct VectorPoint {
    pub id: String,
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub memory_type: String,
    pub importance: f64,
    pub namespace: Option<String>,
    pub label: String,
}

// ── Namespaces ──────────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct NamespaceItem {
    pub name: String,
    pub memory_count: usize,
}

#[derive(Debug, Serialize)]
pub struct NamespaceStatsResponse {
    pub namespace: String,
    pub memory_count: usize,
    pub type_distribution: HashMap<String, usize>,
}

// ── Repos ───────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct RegisterRepoRequest {
    pub path: String,
    pub name: Option<String>,
}

// ── Sessions ────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct SessionsQuery {
    pub namespace: Option<String>,
    pub limit: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct StartSessionRequest {
    pub namespace: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct EndSessionRequest {
    pub summary: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct SessionResponse {
    pub id: String,
    pub namespace: Option<String>,
    pub started_at: String,
    pub ended_at: Option<String>,
    pub memory_count: u32,
    pub summary: Option<String>,
}

// ── Timeline & Distribution ─────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct TimelineQuery {
    pub namespace: Option<String>,
    pub from: Option<String>,
    pub to: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct TimelineBucket {
    pub date: String,
    pub counts: HashMap<String, usize>,
    pub total: usize,
}

#[derive(Debug, Serialize)]
pub struct DistributionResponse {
    pub type_counts: HashMap<String, usize>,
    pub importance_histogram: Vec<usize>,
    pub total: usize,
}

// ── Patterns & Consolidation ────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct PatternResponse {
    pub pattern_type: String,
    pub description: String,
    pub frequency: usize,
    pub confidence: f64,
    pub related_memories: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct ConsolidationStatusResponse {
    pub cycles: Vec<ConsolidationCycleStatus>,
}

#[derive(Debug, Serialize)]
pub struct ConsolidationCycleStatus {
    pub cycle: String,
    pub last_run: Option<String>,
    pub affected_count: usize,
}

// ── Config ──────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct ScoringWeightsUpdate {
    pub vector_similarity: Option<f64>,
    pub graph_strength: Option<f64>,
    pub token_overlap: Option<f64>,
    pub temporal: Option<f64>,
    pub tag_matching: Option<f64>,
    pub importance: Option<f64>,
    pub confidence: Option<f64>,
    pub recency: Option<f64>,
}

// ── SSE Events ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct WatchEventResponse {
    pub path: String,
    pub event_type: String,
    pub timestamp: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct IndexingEventResponse {
    pub files_scanned: usize,
    pub files_parsed: usize,
    pub total_symbols: usize,
    pub current_file: String,
}

// ── Metrics ─────────────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct MetricsResponse {
    pub tool_calls_total: u64,
    pub latency_percentiles: HashMap<String, f64>,
}

// ── Agents / Recipes ───────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct RecipeListResponse {
    pub id: String,
    pub name: String,
    pub description: String,
    pub steps: Vec<RecipeStep>,
}

#[derive(Debug, Serialize)]
pub struct RecipeStep {
    pub tool: String,
    pub description: String,
}

#[derive(Debug, Deserialize)]
pub struct RunRecipeRequest {
    pub recipe: String,
    pub repo_id: Option<String>,
    pub namespace: Option<String>,
}

// ── Insights ────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct InsightsQuery {
    pub namespace: Option<String>,
    pub limit: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct ActivityInsightsResponse {
    pub insights: Vec<MemoryItem>,
    pub git_summary: GitSummary,
}

#[derive(Debug, Serialize)]
pub struct GitSummary {
    pub total_annotated_files: usize,
    pub top_authors: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct CodeHealthInsightsResponse {
    pub insights: Vec<MemoryItem>,
    pub file_hotspots: Vec<PatternResponse>,
    pub decision_chains: Vec<PatternResponse>,
    pub pagerank_leaders: Vec<PagerankEntry>,
    pub community_count: usize,
}

#[derive(Debug, Serialize)]
pub struct SecurityInsightsResponse {
    pub insights: Vec<MemoryItem>,
    pub sensitive_file_count: usize,
    pub endpoint_count: usize,
    pub security_function_count: usize,
}

#[derive(Debug, Serialize)]
pub struct PerformanceInsightsResponse {
    pub insights: Vec<MemoryItem>,
    pub high_coupling_nodes: Vec<CouplingNode>,
    pub max_depth: usize,
    pub critical_path: Vec<PagerankEntry>,
}

#[derive(Debug, Serialize)]
pub struct CouplingNode {
    pub node_id: String,
    pub label: String,
    pub coupling_score: usize,
}

// ── Generic ─────────────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct MessageResponse {
    pub message: String,
}

#[derive(Debug, Serialize)]
pub struct IdResponse {
    pub id: String,
}
