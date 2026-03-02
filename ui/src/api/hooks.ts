// TanStack Query hooks for all REST endpoints

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from './client'

// Stats & Health
export const useStats = () => useQuery({ queryKey: ['stats'], queryFn: api.stats, refetchInterval: 10000 })
export const useHealth = () => useQuery({ queryKey: ['health'], queryFn: api.health })
export const useMetrics = () => useQuery({ queryKey: ['metrics'], queryFn: api.metrics })

// Memories
export const useMemories = (params?: Parameters<typeof api.memories>[0]) =>
  useQuery({ queryKey: ['memories', params], queryFn: () => api.memories(params) })

export const useMemory = (id: string) =>
  useQuery({ queryKey: ['memory', id], queryFn: () => api.memory(id), enabled: !!id })

export const useStoreMemory = () => {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: api.storeMemory,
    onSuccess: () => qc.invalidateQueries({ queryKey: ['memories'] }),
  })
}

export const useUpdateMemory = () => {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ id, ...body }: { id: string; content?: string; importance?: number }) =>
      api.updateMemory(id, body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['memories'] })
      qc.invalidateQueries({ queryKey: ['memory'] })
    },
  })
}

export const useDeleteMemory = () => {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (id: string) => api.deleteMemory(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['memories'] }),
  })
}

// Search
export const useSearch = (params: Parameters<typeof api.search>[0], enabled = true) =>
  useQuery({ queryKey: ['search', params], queryFn: () => api.search(params), enabled })

// Graph
export const useSubgraph = (params?: Parameters<typeof api.subgraph>[0]) =>
  useQuery({ queryKey: ['subgraph', params], queryFn: () => api.subgraph(params) })

export const useNeighbors = (id: string, depth?: number) =>
  useQuery({ queryKey: ['neighbors', id, depth], queryFn: () => api.neighbors(id, depth), enabled: !!id })

export const useCommunities = (resolution?: number, enabled = true) =>
  useQuery({ queryKey: ['communities', resolution], queryFn: () => api.communities({ resolution }), enabled })

export const usePagerank = (top?: number) =>
  useQuery({ queryKey: ['pagerank', top], queryFn: () => api.pagerank(top) })

// Namespaces
export const useNamespaces = () => useQuery({ queryKey: ['namespaces'], queryFn: api.namespaces })

// Repos
export const useRepos = () =>
  useQuery({
    queryKey: ['repos'],
    queryFn: api.repos,
    refetchInterval: (query) => {
      const repos = query.state.data
      const indexing = repos?.some((r: { status: string }) => r.status === 'indexing')
      return indexing ? 2000 : false
    },
  })

export const useRegisterRepo = () => {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: api.registerRepo,
    onSuccess: () => qc.invalidateQueries({ queryKey: ['repos'] }),
  })
}

export const useIndexRepo = () => {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (id: string) => api.indexRepo(id),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['repos'] })
      qc.invalidateQueries({ queryKey: ['stats'] })
    },
  })
}

// Sessions
export const useSessions = (params?: Parameters<typeof api.sessions>[0]) =>
  useQuery({ queryKey: ['sessions', params], queryFn: () => api.sessions(params) })

// Timeline & Distribution
export const useTimeline = (params?: Parameters<typeof api.timeline>[0]) =>
  useQuery({ queryKey: ['timeline', params], queryFn: () => api.timeline(params) })

export const useDistribution = (namespace?: string) =>
  useQuery({ queryKey: ['distribution', namespace], queryFn: () => api.distribution(namespace) })

// Patterns
export const usePatterns = (namespace?: string) =>
  useQuery({ queryKey: ['patterns', namespace], queryFn: () => api.patterns(namespace) })

export const usePatternInsights = (namespace?: string) =>
  useQuery({ queryKey: ['pattern-insights', namespace], queryFn: () => api.patternInsights(namespace) })

// Consolidation
export const useConsolidationStatus = () =>
  useQuery({ queryKey: ['consolidation-status'], queryFn: api.consolidationStatus })

export const useRunConsolidation = () => {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (cycle: string) => api.runConsolidation(cycle),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['consolidation-status'] }),
  })
}

// Insights
export const useActivityInsights = (params?: Parameters<typeof api.activityInsights>[0]) =>
  useQuery({ queryKey: ['insights-activity', params], queryFn: () => api.activityInsights(params) })

export const useCodeHealthInsights = (params?: Parameters<typeof api.codeHealthInsights>[0]) =>
  useQuery({ queryKey: ['insights-code-health', params], queryFn: () => api.codeHealthInsights(params) })

export const useSecurityInsights = (params?: Parameters<typeof api.securityInsights>[0]) =>
  useQuery({ queryKey: ['insights-security', params], queryFn: () => api.securityInsights(params) })

export const usePerformanceInsights = (params?: Parameters<typeof api.performanceInsights>[0]) =>
  useQuery({ queryKey: ['insights-performance', params], queryFn: () => api.performanceInsights(params) })

// Agent Recipes
export const useRecipes = () => useQuery({ queryKey: ['recipes'], queryFn: api.recipes })

// Config
export const useConfig = () => useQuery({ queryKey: ['config'], queryFn: api.config })

export const useUpdateScoringWeights = () => {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (weights: Partial<import('./types').ScoreBreakdown>) =>
      api.updateScoringWeights(weights),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['config'] }),
  })
}
