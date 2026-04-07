import { useState, useCallback, useMemo } from 'react'
import { Loader2, AlertTriangle } from 'lucide-react'
import { useSubgraph, useCommunities, useNeighbors } from '../../api/hooks'
import { useNamespaceStore } from '../../stores/namespace'
import { ALL_RELATIONSHIPS } from './constants'
import { SigmaGraph } from './SigmaGraph'
import { GraphToolbar } from './GraphToolbar'
import { NodeInspector } from './NodeInspector'
import { FocusToolbar } from './FocusToolbar'
import { FileTree } from './FileTree'
import { CodeTab } from './CodeTab'

const ALL_KINDS = new Set([
  'function', 'method', 'class', 'file', 'module', 'package',
  'type', 'interface', 'memory', 'constant', 'endpoint', 'test', 'chunk',
  'external', 'trait', 'enum', 'enum_variant', 'field', 'type_parameter', 'macro', 'property',
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
  const [showFileTree, setShowFileTree] = useState(true)
  const [showCode, setShowCode] = useState(false)
  const [activeRelationships, setActiveRelationships] = useState<Set<string>>(() => {
    const defaults = new Set(ALL_RELATIONSHIPS)
    defaults.delete('CO_CHANGED')
    return defaults
  })
  const [focusMode, setFocusMode] = useState<{ nodeId: string; depth: number } | null>(null)

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

  const mergedNodes = useMemo(() => {
    if (!subgraph?.nodes) return []
    if (!neighborsData?.nodes) return subgraph.nodes
    const ids = new Set(subgraph.nodes.map((n) => n.id))
    return [...subgraph.nodes, ...neighborsData.nodes.filter((n) => !ids.has(n.id))]
  }, [subgraph?.nodes, neighborsData?.nodes])

  const mergedEdges = useMemo(() => {
    if (!subgraph?.edges) return []
    if (!neighborsData?.edges) return subgraph.edges
    const ids = new Set(subgraph.edges.map((e) => e.id))
    return [...subgraph.edges, ...neighborsData.edges.filter((e) => !ids.has(e.id))]
  }, [subgraph?.edges, neighborsData?.edges])

  const displayNodes = focusMode && focusData ? focusData.nodes : mergedNodes
  const displayEdges = focusMode && focusData ? focusData.edges : mergedEdges

  const edgeCounts = useMemo(() => {
    const counts: Record<string, number> = {}
    for (const edge of displayEdges) counts[edge.relationship] = (counts[edge.relationship] ?? 0) + 1
    return counts
  }, [displayEdges])

  const selectedNode = useMemo(
    () => displayNodes.find((n) => n.id === selectedNodeId) ?? null,
    [displayNodes, selectedNodeId],
  )

  const hasSourceFile = selectedNode && (
    selectedNode.kind === 'file' || selectedNode.payload?.file_path || selectedNode.id.startsWith('file:')
  )

  const handleNodeClick = useCallback((nodeId: string) => {
    setSelectedNodeId(nodeId)
  }, [])

  const handleToggleKind = useCallback((kind: string) => {
    setKinds((prev) => { const next = new Set(prev); next.has(kind) ? next.delete(kind) : next.add(kind); return next })
  }, [])

  const handleToggleRelationship = useCallback((rel: string) => {
    setActiveRelationships((prev) => { const next = new Set(prev); next.has(rel) ? next.delete(rel) : next.add(rel); return next })
  }, [])

  const handleFocus = useCallback((nodeId: string) => {
    setFocusMode({ nodeId, depth: 1 }); setSelectedNodeId(null)
  }, [])

  if (isLoading) {
    return (
      <div className="flex h-full items-center justify-center">
        <Loader2 size={18} className="animate-spin text-zinc-600" />
        <span className="ml-3 text-[13px] text-zinc-500">Loading graph...</span>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex h-full items-center justify-center">
        <AlertTriangle size={16} className="text-amber-400" />
        <span className="ml-2 text-[13px] text-zinc-400">Failed to load graph data</span>
      </div>
    )
  }

  if (!displayNodes.length) {
    return (
      <div className="flex h-full items-center justify-center">
        <p className="text-[13px] text-zinc-500">
          No graph nodes. Run <code className="rounded bg-zinc-800 px-1.5 py-0.5 text-[12px] text-zinc-300">codemem analyze</code> first.
        </p>
      </div>
    )
  }

  return (
    <div className="flex h-full flex-col">
      {/* Toolbar */}
      <GraphToolbar
        searchLabel={searchLabel} onSearchChange={setSearchLabel}
        kinds={kinds} onToggleKind={handleToggleKind}
        maxNodes={maxNodes} onMaxNodesChange={setMaxNodes}
        showCommunities={showCommunities} onToggleCommunities={() => setShowCommunities((v) => !v)}
        showEdges={showEdges} onToggleEdges={() => setShowEdges((v) => !v)}
        activeRelationships={activeRelationships} edgeCounts={edgeCounts} onToggleRelationship={handleToggleRelationship}
        showFileTree={showFileTree} onToggleFileTree={() => setShowFileTree((v) => !v)}
      />

      {/* Main area: horizontal split — top: tree+graph+inspector, bottom: code */}
      <div className="flex min-h-0 flex-1 flex-col">
        {/* Top row */}
        <div className={`flex min-h-0 ${showCode ? 'flex-[3]' : 'flex-1'}`}>
          {/* File Tree */}
          {showFileTree && (
            <div className="w-56 shrink-0 border-r border-zinc-800/40 bg-zinc-950">
              <FileTree nodes={displayNodes} onSelectNode={handleNodeClick} selectedNodeId={selectedNodeId} />
            </div>
          )}

          {/* Graph */}
          <div className="relative min-w-0 flex-1 bg-zinc-950">
            <SigmaGraph
              nodes={displayNodes} edges={displayEdges}
              communities={focusMode ? null : (communitiesData?.communities ?? null)}
              showCommunities={!focusMode && showCommunities} showEdges={showEdges}
              onNodeClick={handleNodeClick} highlightNodeId={selectedNodeId}
              searchLabel={searchLabel} onLayoutRunning={setLayoutRunning}
              activeRelationships={activeRelationships} focusNodeId={focusMode?.nodeId ?? null}
            />
            {focusMode && (
              <FocusToolbar
                nodeLabel={displayNodes.find((n) => n.id === focusMode.nodeId)?.label ?? ''}
                depth={focusMode.depth}
                onDepthChange={(d) => setFocusMode((prev) => prev ? { ...prev, depth: d } : null)}
                onExit={() => setFocusMode(null)}
              />
            )}
            {layoutRunning && (
              <div className="pointer-events-none absolute bottom-3 left-1/2 z-10 -translate-x-1/2">
                <div className="flex items-center gap-2 rounded-full bg-zinc-900/90 px-3 py-1.5 text-[11px] text-zinc-500 backdrop-blur-sm">
                  <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-violet-500" />
                  Stabilizing...
                </div>
              </div>
            )}
          </div>

          {/* Right panel: Node context */}
          {selectedNode && (
            <div className="w-72 shrink-0 border-l border-zinc-800/40 bg-zinc-950">
              <NodeInspector
                node={selectedNode} edges={displayEdges} allNodes={displayNodes}
                onClose={() => setSelectedNodeId(null)}
                onExpandNeighbors={(id) => setExpandedNodeId(id)}
                onFocus={handleFocus}
                onToggleCode={hasSourceFile ? () => setShowCode((v) => !v) : undefined}
                showCode={showCode}
              />
            </div>
          )}
        </div>

        {/* Bottom: Code panel (horizontal split) */}
        {showCode && selectedNode && hasSourceFile && (
          <div className="flex-[2] border-t border-zinc-800/40 bg-zinc-950">
            <CodeTab node={selectedNode} />
          </div>
        )}
      </div>
    </div>
  )
}
