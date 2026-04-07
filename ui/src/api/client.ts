// REST API client (fetch wrapper)

const BASE_URL = ''

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { 'Content-Type': 'application/json', ...options?.headers },
    ...options,
  })
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`API error ${res.status}: ${text}`)
  }
  return res.json()
}

export const api = {
  // Stats & Health
  stats: () => request<import('./types').StatsResponse>('/api/stats'),
  health: () => request<import('./types').HealthResponse>('/api/health'),
  metrics: () => request<import('./types').MetricsResponse>('/api/metrics'),

  // Memories
  memories: (params?: { namespace?: string; type?: string; offset?: number; limit?: number }) => {
    const search = new URLSearchParams()
    if (params?.namespace) search.set('namespace', params.namespace)
    if (params?.type) search.set('type', params.type)
    if (params?.offset !== undefined) search.set('offset', String(params.offset))
    if (params?.limit !== undefined) search.set('limit', String(params.limit))
    return request<import('./types').MemoryListResponse>(`/api/memories?${search}`)
  },
  memory: (id: string) => request<import('./types').MemoryItem>(`/api/memories/${id}`),
  storeMemory: (body: { content: string; memory_type?: string; importance?: number; tags?: string[]; namespace?: string }) =>
    request<import('./types').IdResponse>('/api/memories', { method: 'POST', body: JSON.stringify(body) }),
  updateMemory: (id: string, body: { content?: string; importance?: number }) =>
    request<import('./types').MessageResponse>(`/api/memories/${id}`, { method: 'PUT', body: JSON.stringify(body) }),
  deleteMemory: (id: string) =>
    request<import('./types').MessageResponse>(`/api/memories/${id}`, { method: 'DELETE' }),

  // Search
  search: (params: { q: string; namespace?: string; k?: number; type?: string }) => {
    const search = new URLSearchParams({ q: params.q })
    if (params.namespace) search.set('namespace', params.namespace)
    if (params.k !== undefined) search.set('k', String(params.k))
    if (params.type) search.set('type', params.type)
    return request<import('./types').SearchResponse>(`/api/search?${search}`)
  },

  // Graph
  subgraph: (params?: { namespace?: string; max_nodes?: number; kinds?: string[]; min_centrality?: number }) => {
    const search = new URLSearchParams()
    if (params?.namespace) search.set('namespace', params.namespace)
    if (params?.max_nodes !== undefined) search.set('max_nodes', String(params.max_nodes))
    if (params?.kinds?.length) search.set('kinds', params.kinds.join(','))
    if (params?.min_centrality !== undefined) search.set('min_centrality', String(params.min_centrality))
    return request<import('./types').SubgraphResponse>(`/api/graph/subgraph?${search}`)
  },
  neighbors: (id: string, depth?: number) =>
    request<import('./types').SubgraphResponse>(`/api/graph/neighbors/${encodeURIComponent(id)}?depth=${depth ?? 1}`),
  communities: (params?: { resolution?: number }) =>
    request<import('./types').CommunitiesResponse>(`/api/graph/communities?resolution=${params?.resolution ?? 1.0}`),
  pagerank: (top?: number) =>
    request<{ scores: import('./types').PagerankEntry[] }>(`/api/graph/pagerank?top=${top ?? 20}`),
  graphBrowse: (params?: { namespace?: string; kind?: string; q?: string; offset?: number; limit?: number }) => {
    const search = new URLSearchParams()
    if (params?.namespace) search.set('namespace', params.namespace)
    if (params?.kind) search.set('kind', params.kind)
    if (params?.q) search.set('q', params.q)
    if (params?.offset !== undefined) search.set('offset', String(params.offset))
    if (params?.limit !== undefined) search.set('limit', String(params.limit))
    return request<import('./types').BrowseResponse>(`/api/graph/browse?${search}`)
  },
  fileContent: (params: { path: string; line_start?: number; line_end?: number; namespace?: string }) => {
    const search = new URLSearchParams({ path: params.path })
    if (params.line_start !== undefined) search.set('line_start', String(params.line_start))
    if (params.line_end !== undefined) search.set('line_end', String(params.line_end))
    if (params.namespace) search.set('namespace', params.namespace)
    return request<import('./types').FileContentResponse>(`/api/graph/file-content?${search}`)
  },
  vectors: (namespace?: string) => {
    const search = namespace ? `?namespace=${encodeURIComponent(namespace)}` : ''
    return request<import('./types').VectorPoint[]>(`/api/vectors${search}`)
  },

  // Namespaces
  namespaces: () => request<import('./types').NamespaceItem[]>('/api/namespaces'),

  // Repos
  repos: () => request<import('./types').Repository[]>('/api/repos'),
  registerRepo: (body: { path: string; name?: string }) =>
    request<import('./types').IdResponse>('/api/repos', { method: 'POST', body: JSON.stringify(body) }),
  indexRepo: (id: string) =>
    request<import('./types').MessageResponse>(`/api/repos/${id}/index`, { method: 'POST' }),
  repo: (id: string) => request<import('./types').Repository>(`/api/repos/${id}`),

  // Sessions
  sessions: (params?: { namespace?: string; limit?: number }) => {
    const search = new URLSearchParams()
    if (params?.namespace) search.set('namespace', params.namespace)
    if (params?.limit !== undefined) search.set('limit', String(params.limit))
    return request<import('./types').SessionResponse[]>(`/api/sessions?${search}`)
  },

  // Timeline & Distribution
  timeline: (params?: { namespace?: string; from?: string; to?: string }) => {
    const search = new URLSearchParams()
    if (params?.namespace) search.set('namespace', params.namespace)
    if (params?.from) search.set('from', params.from)
    if (params?.to) search.set('to', params.to)
    return request<import('./types').TimelineBucket[]>(`/api/timeline?${search}`)
  },
  distribution: (namespace?: string) => {
    const search = namespace ? `?namespace=${encodeURIComponent(namespace)}` : ''
    return request<import('./types').DistributionResponse>(`/api/distribution${search}`)
  },

  // Patterns & Consolidation
  patterns: (namespace?: string) => {
    const search = namespace ? `?namespace=${encodeURIComponent(namespace)}` : ''
    return request<import('./types').PatternResponse[]>(`/api/patterns${search}`)
  },
  patternInsights: (namespace?: string) => {
    const search = namespace ? `?namespace=${encodeURIComponent(namespace)}` : ''
    return request<import('./types').MessageResponse>(`/api/patterns/insights${search}`)
  },
  consolidationStatus: () => request<import('./types').ConsolidationStatus>('/api/consolidation/status'),
  runConsolidation: (cycle: string) =>
    request<import('./types').MessageResponse>(`/api/consolidation/${cycle}`, { method: 'POST' }),

  // Insights
  activityInsights: (params?: { namespace?: string; limit?: number }) => {
    const search = new URLSearchParams()
    if (params?.namespace) search.set('namespace', params.namespace)
    if (params?.limit !== undefined) search.set('limit', String(params.limit))
    return request<import('./types').ActivityInsightsResponse>(`/api/insights/activity?${search}`)
  },
  codeHealthInsights: (params?: { namespace?: string; limit?: number }) => {
    const search = new URLSearchParams()
    if (params?.namespace) search.set('namespace', params.namespace)
    if (params?.limit !== undefined) search.set('limit', String(params.limit))
    return request<import('./types').CodeHealthInsightsResponse>(`/api/insights/code-health?${search}`)
  },
  securityInsights: (params?: { namespace?: string; limit?: number }) => {
    const search = new URLSearchParams()
    if (params?.namespace) search.set('namespace', params.namespace)
    if (params?.limit !== undefined) search.set('limit', String(params.limit))
    return request<import('./types').SecurityInsightsResponse>(`/api/insights/security?${search}`)
  },
  performanceInsights: (params?: { namespace?: string; limit?: number }) => {
    const search = new URLSearchParams()
    if (params?.namespace) search.set('namespace', params.namespace)
    if (params?.limit !== undefined) search.set('limit', String(params.limit))
    return request<import('./types').PerformanceInsightsResponse>(`/api/insights/performance?${search}`)
  },

  // Agent Recipes
  recipes: () => request<import('./types').Recipe[]>('/api/agents/recipes'),

  // Config
  config: () => request<Record<string, unknown>>('/api/config'),
  updateScoringWeights: (weights: Partial<import('./types').ScoreBreakdown>) =>
    request<import('./types').MessageResponse>('/api/config/scoring', { method: 'PUT', body: JSON.stringify(weights) }),
}
