// TanStack Query hooks for all REST endpoints

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from './client'

// Stats & Health
export const useStats = () => useQuery({ queryKey: ['stats'], queryFn: api.stats, refetchInterval: 10000 })
export const useHealth = () => useQuery({ queryKey: ['health'], queryFn: api.health })


// Memories
export const useMemories = (params?: Parameters<typeof api.memories>[0]) =>
  useQuery({ queryKey: ['memories', params], queryFn: () => api.memories(params) })

export const useMemory = (id: string) =>
  useQuery({ queryKey: ['memory', id], queryFn: () => api.memory(id), enabled: !!id })

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

export const useGraphBrowse = (params?: Parameters<typeof api.graphBrowse>[0]) =>
  useQuery({ queryKey: ['graph-browse', params], queryFn: () => api.graphBrowse(params) })

export const useVectors = (namespace?: string) =>
  useQuery({ queryKey: ['vectors', namespace], queryFn: () => api.vectors(namespace) })

export const useFileContent = (path: string | null, lineStart?: number, lineEnd?: number) =>
  useQuery({
    queryKey: ['file-content', path, lineStart, lineEnd],
    queryFn: () => api.fileContent({ path: path!, line_start: lineStart, line_end: lineEnd }),
    enabled: !!path,
  })

// Namespaces
export const useNamespaces = () => useQuery({ queryKey: ['namespaces'], queryFn: api.namespaces })

// Sessions
export const useSessions = (params?: Parameters<typeof api.sessions>[0]) =>
  useQuery({ queryKey: ['sessions', params], queryFn: () => api.sessions(params) })

export const useDistribution = (namespace?: string) =>
  useQuery({ queryKey: ['distribution', namespace], queryFn: () => api.distribution(namespace) })

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

