import { useState, useCallback, useMemo } from 'react'
import { Loader2, AlertTriangle } from 'lucide-react'
import { useSubgraph, useCommunities, useNeighbors } from '../../api/hooks'
import { useNamespaceStore } from '../../stores/namespace'
import { SigmaGraph } from './SigmaGraph'
import { GraphControls } from './GraphControls'
import { NodeInspector } from './NodeInspector'
import { CommunityLegend } from './CommunityLegend'

const ALL_KINDS = new Set([
  'function', 'method', 'class', 'file', 'module', 'package',
  'variable', 'type', 'interface', 'trait', 'struct', 'enum',
  'memory', 'constant', 'endpoint', 'test',
])

export function GraphExplorer() {
  const namespace = useNamespaceStore((s) => s.active)

  const [maxNodes, setMaxNodes] = useState(500)
  const [kinds, setKinds] = useState<Set<string>>(new Set(ALL_KINDS))
  const [showCommunities, setShowCommunities] = useState(false)
  const [showEdges, setShowEdges] = useState(false)
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null)
  const [searchLabel, setSearchLabel] = useState('')
  const [expandedNodeId, setExpandedNodeId] = useState<string | null>(null)
  const [layoutRunning, setLayoutRunning] = useState(false)

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
  const mergedNodes = useMemo(() => {
    if (!subgraph?.nodes) return []
    if (!neighborsData?.nodes) return subgraph.nodes
    const ids = new Set(subgraph.nodes.map((n) => n.id))
    const extra = neighborsData.nodes.filter((n) => !ids.has(n.id))
    return [...subgraph.nodes, ...extra]
  }, [subgraph?.nodes, neighborsData?.nodes])

  const mergedEdges = useMemo(() => {
    if (!subgraph?.edges) return []
    if (!neighborsData?.edges) return subgraph.edges
    const ids = new Set(subgraph.edges.map((e) => e.id))
    const extra = neighborsData.edges.filter((e) => !ids.has(e.id))
    return [...subgraph.edges, ...extra]
  }, [subgraph?.edges, neighborsData?.edges])

  const selectedNode = useMemo(
    () => mergedNodes.find((n) => n.id === selectedNodeId) ?? null,
    [mergedNodes, selectedNodeId],
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

  if (!mergedNodes.length) {
    return (
      <div className="flex h-full items-center justify-center">
        <p className="text-sm text-zinc-500">No graph nodes found. Index a repository first.</p>
      </div>
    )
  }

  return (
    <div className="relative h-full w-full">
      <SigmaGraph
        nodes={mergedNodes}
        edges={mergedEdges}
        communities={communitiesData?.communities ?? null}
        showCommunities={showCommunities}
        showEdges={showEdges}
        onNodeClick={handleNodeClick}
        highlightNodeId={selectedNodeId}
        searchLabel={searchLabel}
        onLayoutRunning={setLayoutRunning}
      />

      {layoutRunning && (
        <div className="pointer-events-none absolute bottom-4 left-1/2 z-10 -translate-x-1/2">
          <div className="flex items-center gap-2 rounded-full bg-zinc-900/80 px-3 py-1.5 text-xs text-zinc-400 backdrop-blur-sm">
            <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-violet-400" />
            Stabilizing layout...
          </div>
        </div>
      )}

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

      {selectedNode && (
        <NodeInspector
          node={selectedNode}
          edges={mergedEdges}
          allNodes={mergedNodes}
          onClose={() => setSelectedNodeId(null)}
          onExpandNeighbors={handleExpandNeighbors}
        />
      )}

      {showCommunities && communitiesData?.communities && (
        <CommunityLegend communities={communitiesData.communities} />
      )}
    </div>
  )
}
