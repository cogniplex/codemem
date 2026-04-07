// TypeScript types mirroring Rust REST API structs

export interface StatsResponse {
  memory_count: number
  embedding_count: number
  node_count: number
  edge_count: number
  session_count: number
  namespace_count: number
}

export interface ComponentHealth {
  status: string
  detail?: string
}

export interface HealthResponse {
  storage: ComponentHealth
  vector: ComponentHealth
  graph: ComponentHealth
  embeddings: ComponentHealth
}

export interface MemoryItem {
  id: string
  content: string
  memory_type: string
  importance: number
  confidence: number
  access_count: number
  tags: string[]
  namespace?: string
  created_at: string
  updated_at: string
}

export interface MemoryListResponse {
  memories: MemoryItem[]
  total: number
  offset: number
  limit: number
}

export interface SearchResultItem {
  id: string
  content: string
  memory_type: string
  score: number
  score_breakdown: ScoreBreakdown
  tags: string[]
  namespace?: string
}

export interface ScoreBreakdown {
  vector_similarity: number
  graph_strength: number
  token_overlap: number
  temporal: number
  tag_matching: number
  importance: number
  confidence: number
  recency: number
}

export interface SearchResponse {
  results: SearchResultItem[]
  query: string
  k: number
}

export interface GraphNode {
  id: string
  kind: string
  label: string
  centrality: number
  memory_id?: string
  namespace?: string
  payload?: Record<string, unknown>
}

export interface TemporalChangesResponse {
  commits: number
  entries: TemporalEntry[]
}

export interface TemporalEntry {
  commit_id: string
  hash: string
  subject: string
  author: string
  date: string
  changed_files: string[]
  changed_symbols: string[]
}

export interface StaleFilesResponse {
  stale_days: number
  stale_files: number
  files: StaleFile[]
}

export interface StaleFile {
  file_path: string
  centrality: number
  last_modified: string | null
  incoming_edges: number
}

export interface DriftResponse {
  period: string
  new_cross_module_edges: number
  removed_files: number
  added_files: number
  hotspot_files: string[]
  coupling_increases: [string, string, number][]
}

export interface FileContentResponse {
  path: string
  content: string
  total_lines: number
  line_start: number
  line_end: number
  language: string
}

export interface GraphEdge {
  id: string
  src: string
  dst: string
  relationship: string
  weight: number
}

export interface SubgraphResponse {
  nodes: GraphNode[]
  edges: GraphEdge[]
}

export interface CommunitiesResponse {
  communities: Record<string, number>
  num_communities: number
}

export interface PagerankEntry {
  node_id: string
  label: string
  score: number
}

export interface NamespaceItem {
  name: string
  memory_count: number
}

export interface Repository {
  id: string
  path: string
  name?: string
  namespace?: string
  created_at: string
  last_indexed_at?: string
  status: string
}

export interface SessionResponse {
  id: string
  namespace?: string
  started_at: string
  ended_at?: string
  memory_count: number
  summary?: string
}

export interface TimelineBucket {
  date: string
  counts: Record<string, number>
  total: number
}

export interface DistributionResponse {
  type_counts: Record<string, number>
  importance_histogram: number[]
  total: number
}

export interface PatternResponse {
  pattern_type: string
  description: string
  frequency: number
  confidence: number
  related_memories: string[]
}

export interface ConsolidationStatus {
  cycles: { cycle: string; last_run?: string; affected_count: number }[]
}

export interface MessageResponse {
  message: string
}

export interface IdResponse {
  id: string
}

export interface MetricsResponse {
  tool_calls_total: number
  latency_percentiles: Record<string, number>
}

export interface Recipe {
  id: string
  name: string
  description: string
  steps: RecipeStep[]
}

export interface RecipeStep {
  tool: string
  description: string
}

// ── Insights ────────────────────────────────────────────────────────────────

export interface GitSummary {
  total_annotated_files: number
  top_authors: string[]
}

export interface ActivityInsightsResponse {
  insights: MemoryItem[]
  git_summary: GitSummary
}

export interface CodeHealthInsightsResponse {
  insights: MemoryItem[]
  file_hotspots: PatternResponse[]
  decision_chains: PatternResponse[]
  pagerank_leaders: PagerankEntry[]
  community_count: number
}

export interface SecurityInsightsResponse {
  insights: MemoryItem[]
  sensitive_file_count: number
  endpoint_count: number
  security_function_count: number
}

export interface CouplingNode {
  node_id: string
  label: string
  coupling_score: number
}

export interface PerformanceInsightsResponse {
  insights: MemoryItem[]
  high_coupling_nodes: CouplingNode[]
  max_depth: number
  critical_path: PagerankEntry[]
}

// ── Browse ────────────────────────────────────────────────────────────────

export interface BrowseNodeItem {
  id: string
  kind: string
  label: string
  centrality: number
  namespace?: string
  degree: number
}

export interface BrowseResponse {
  nodes: BrowseNodeItem[]
  total: number
  kinds: Record<string, number>
  edge_count: number
}

// ── Vectors ───────────────────────────────────────────────────────────────

export interface VectorPoint {
  id: string
  x: number
  y: number
  z: number
  memory_type: string
  importance: number
  namespace?: string
  label: string
}

