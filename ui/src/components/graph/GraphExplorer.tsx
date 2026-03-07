import { useState, useCallback, useMemo } from 'react'
import { Loader2, AlertTriangle } from 'lucide-react'
import { useSubgraph, useCommunities, useNeighbors } from '../../api/hooks'
import { useNamespaceStore } from '../../stores/namespace'
import { ALL_RELATIONSHIPS } from './constants'
import { SigmaGraph } from './SigmaGraph'
import { GraphControls } from './GraphControls'
import { NodeInspector } from './NodeInspector'
import { CommunityLegend } from './CommunityLegend'
import { RelationshipFilters } from './RelationshipFilters'
import { FocusToolbar } from './FocusToolbar'

const ALL_KINDS = new Set([
  'function', 'method', 'class', 'file', 'module', 'package',
  'variable', 'type', 'interface', 'trait', 'struct', 'enum',
  'memory', 'constant', 'endpoint', 'test',
])

export function GraphExplorer() {
  const namespace = useNamespaceStore((s) => s.active)

  const [maxNodes, setMaxNodes] = useState(1200)
  const [kinds, setKinds] = useState<Set<string>>(new Set(ALL_KINDS))
  const [showCommunities, setShowCommunities] = useState(false)
  const [showEdges, setShowEdges] = useState(true)
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null)
  const [searchLabel, setSearchLabel] = useState('')
  const [expandedNodeId, setExpandedNodeId] = useState<string | null>(null)
  const [layoutRunning, setLayoutRunning] = useState(false)
  // Hide noisy high-volume edge types by default — users can toggle them on
  const [activeRelationships, setActiveRelationships] = useState<Set<string>>(() => {
    const defaults = new Set(ALL_RELATIONSHIPS)
    defaults.delete('CO_CHANGED')
    defaults.delete('CALLS')
    return defaults
  })
  const [focusMode, setFocusMode] = useState<{ nodeId: string; depth: number } | null>(null)

  // Fetch neighbors for focus mode
  const focusNeighborId = focusMode?.nodeId ?? ''
  const focusDepth = focusMode?.depth ?? 1
  const { data: focusData } = useNeighbors(focusNeighborId, focusDepth)

  const subgraphParams = useMemo(
    () => ({
      namespace: namespace ?? undefined,
      max_nodes: maxNodes,
      kinds: kinds.size < ALL_KINDS.size ? [...kinds] : undefined,
    }),
    [namespace, maxNodes, kinds],
  )

  const { data: subgraph, isLoading, error } = useSubgraph(subgraphParams)
  const { data: communitiesData } = useCommunities(undefined, showCommunities)
  const { data: neighborsData } = useNeighbors(expandedNodeId ?? '', 2)

  // Merge neighbor data into the subgraph
  const subgraphNodes = subgraph?.nodes
  const subgraphEdges = subgraph?.edges
  const neighborNodes = neighborsData?.nodes
  const neighborEdges = neighborsData?.edges

  const mergedNodes = useMemo(() => {
    if (!subgraphNodes) return []
    if (!neighborNodes) return subgraphNodes
    const ids = new Set(subgraphNodes.map((n) => n.id))
    const extra = neighborNodes.filter((n) => !ids.has(n.id))
    return [...subgraphNodes, ...extra]
  }, [subgraphNodes, neighborNodes])

  const mergedEdges = useMemo(() => {
    if (!subgraphEdges) return []
    if (!neighborEdges) return subgraphEdges
    const ids = new Set(subgraphEdges.map((e) => e.id))
    const extra = neighborEdges.filter((e) => !ids.has(e.id))
    return [...subgraphEdges, ...extra]
  }, [subgraphEdges, neighborEdges])

  // When in focus mode, use focus data instead of merged subgraph
  const displayNodes = focusMode && focusData ? focusData.nodes : mergedNodes
  const displayEdges = focusMode && focusData ? focusData.edges : mergedEdges

  // Compute edge counts by relationship type for filters
  const edgeCounts = useMemo(() => {
    const counts: Record<string, number> = {}
    for (const edge of displayEdges) {
      counts[edge.relationship] = (counts[edge.relationship] ?? 0) + 1
    }
    return counts
  }, [displayEdges])

  const selectedNode = useMemo(
    () => displayNodes.find((n) => n.id === selectedNodeId) ?? null,
    [displayNodes, selectedNodeId],
  )

  const handleToggleKind = useCallback((kind: string) => {
    setKinds((prev) => {
      const next = new Set(prev)
      if (next.has(kind)) next.delete(kind)
      else next.add(kind)
      return next
    })
  }, [])

  const handleNodeClick = useCallback((nodeId: string) => {
    setSelectedNodeId(nodeId)
  }, [])

  const handleExpandNeighbors = useCallback((nodeId: string) => {
    setExpandedNodeId(nodeId)
  }, [])

  const handleToggleRelationship = useCallback((rel: string) => {
    setActiveRelationships((prev) => {
      const next = new Set(prev)
      if (next.has(rel)) next.delete(rel)
      else next.add(rel)
      return next
    })
  }, [])

  const handleFocus = useCallback((nodeId: string) => {
    setFocusMode({ nodeId, depth: 1 })
    setSelectedNodeId(null)
  }, [])

  const handleFocusDepthChange = useCallback((depth: number) => {
    setFocusMode((prev) => prev ? { ...prev, depth } : null)
  }, [])

  const handleExitFocus = useCallback(() => {
    setFocusMode(null)
  }, [])

  if (isLoading) {
    return (
      <div className="flex h-full items-center justify-center">
        <Loader2 size={24} className="animate-spin text-zinc-500" />
        <span className="ml-3 text-sm text-zinc-400">Loading graph...</span>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex h-full items-center justify-center">
        <AlertTriangle size={20} className="text-amber-400" />
        <span className="ml-2 text-sm text-zinc-400">Failed to load graph data</span>
      </div>
    )
  }

  if (!displayNodes.length) {
    return (
      <div className="flex h-full items-center justify-center">
        <p className="text-sm text-zinc-500">No graph nodes found. Index a repository first.</p>
      </div>
    )
  }

  // Find focus node label for toolbar
  const focusNodeLabel = focusMode
    ? displayNodes.find((n) => n.id === focusMode.nodeId)?.label ?? focusMode.nodeId
    : ''

  return (
    <div className="relative h-full w-full">
      <SigmaGraph
        nodes={displayNodes}
        edges={displayEdges}
        communities={focusMode ? null : (communitiesData?.communities ?? null)}
        showCommunities={!focusMode && showCommunities}
        showEdges={showEdges}
        onNodeClick={handleNodeClick}
        highlightNodeId={selectedNodeId}
        searchLabel={searchLabel}
        onLayoutRunning={setLayoutRunning}
        activeRelationships={activeRelationships}
        focusNodeId={focusMode?.nodeId ?? null}
      />

      {focusMode && (
        <FocusToolbar
          nodeLabel={focusNodeLabel}
          depth={focusMode.depth}
          onDepthChange={handleFocusDepthChange}
          onExit={handleExitFocus}
        />
      )}

      {layoutRunning && (
        <div className="pointer-events-none absolute bottom-4 left-1/2 z-10 -translate-x-1/2">
          <div className="flex items-center gap-2 rounded-full bg-zinc-900/80 px-3 py-1.5 text-xs text-zinc-400 backdrop-blur-sm">
            <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-violet-400" />
            Stabilizing layout...
          </div>
        </div>
      )}

      {!focusMode && (
        <GraphControls
          kinds={kinds}
          onToggleKind={handleToggleKind}
          maxNodes={maxNodes}
          onMaxNodesChange={setMaxNodes}
          showCommunities={showCommunities}
          onToggleCommunities={() => setShowCommunities((v) => !v)}
          showEdges={showEdges}
          onToggleEdges={() => setShowEdges((v) => !v)}
          searchLabel={searchLabel}
          onSearchChange={setSearchLabel}
        />
      )}

      {selectedNode && (
        <NodeInspector
          node={selectedNode}
          edges={displayEdges}
          allNodes={displayNodes}
          onClose={() => setSelectedNodeId(null)}
          onExpandNeighbors={handleExpandNeighbors}
          onFocus={handleFocus}
        />
      )}

      {showEdges && (
        <RelationshipFilters
          activeRelationships={activeRelationships}
          edgeCounts={edgeCounts}
          onToggle={handleToggleRelationship}
        />
      )}

      {!focusMode && showCommunities && communitiesData?.communities && (
        <CommunityLegend communities={communitiesData.communities} />
      )}
    </div>
  )
}
